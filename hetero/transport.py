"""Activation transport: framed TCP, background I/O threads, optional shaping.

A pipeline stage should never block on the wire when it could be computing, so
sends and receives run on their own threads behind queues. The stage thread
touches only ``send()`` and ``recv()``.

Frames are ``[u32 header_len][json header][payload]``. Headers are small; the
payload is raw little-endian tensor bytes.
"""
from __future__ import annotations

import json
import queue
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

MAGIC = b"HTRO"


def precise_sleep(seconds: float) -> None:
    """Sleep accurately enough to emulate a link.

    macOS coalesces timers, so a single time.sleep(8ms) lands ~3ms late — which
    would silently inflate every shaped measurement. Halving the remaining time
    makes the overshoot shrink geometrically, and the last stretch is spun.
    """
    if seconds <= 0:
        return
    end = time.perf_counter() + seconds
    while True:
        rem = end - time.perf_counter()
        if rem <= 400e-6:
            break
        time.sleep(rem * 0.5 if rem > 2e-3 else rem - 300e-6)
    while time.perf_counter() < end:
        pass


@dataclass
class LinkProfile:
    """Emulated link characteristics. ``None`` fields mean 'do not shape'."""
    bandwidth_mbps: Optional[float] = None
    rtt_ms: Optional[float] = None

    @property
    def shaping(self) -> bool:
        return self.bandwidth_mbps is not None or self.rtt_ms is not None

    def delay_for(self, nbytes: int) -> float:
        """One-way delay a frame of this size would experience."""
        d = 0.0
        if self.rtt_ms:
            d += self.rtt_ms / 2000.0
        if self.bandwidth_mbps:
            d += (nbytes * 8.0) / (self.bandwidth_mbps * 1e6)
        return d


def tune(sock: socket.socket) -> None:
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 << 20)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
    except OSError:
        pass


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("peer closed")
        got += k
    return bytes(buf)


class Channel:
    """A duplex framed connection with background I/O threads."""

    def __init__(self, sock: socket.socket, link: Optional[LinkProfile] = None,
                 depth: int = 64):
        tune(sock)
        self.sock = sock
        self.link = link or LinkProfile()
        self._out: queue.Queue = queue.Queue(maxsize=depth)
        self._in: queue.Queue = queue.Queue(maxsize=depth)
        self._closed = threading.Event()
        self.bytes_sent = 0
        self.bytes_recv = 0
        self.frames_sent = 0
        self.shaped_seconds = 0.0
        self._lock = threading.Lock()
        self._tx = threading.Thread(target=self._tx_loop, daemon=True, name="tx")
        self._rx = threading.Thread(target=self._rx_loop, daemon=True, name="rx")
        self._tx.start()
        self._rx.start()

    # -- public API ------------------------------------------------------
    def send(self, header: dict, payload: bytes = b"") -> None:
        self._out.put((header, payload))

    def recv(self, timeout: Optional[float] = None) -> Tuple[dict, bytes]:
        item = self._in.get(timeout=timeout)
        if isinstance(item, Exception):
            raise item
        return item

    def poll(self) -> bool:
        return not self._in.empty()

    def close(self) -> None:
        self._closed.set()
        self._out.put(None)
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()

    def stats(self) -> dict:
        return {"bytes_sent": self.bytes_sent, "bytes_recv": self.bytes_recv,
                "frames_sent": self.frames_sent, "shaped_seconds": self.shaped_seconds}

    # -- internals -------------------------------------------------------
    def _tx_loop(self) -> None:
        while not self._closed.is_set():
            item = self._out.get()
            if item is None:
                return
            header, payload = item
            hb = json.dumps(header).encode()
            frame = MAGIC + struct.pack("<II", len(hb), len(payload)) + hb + payload
            if self.link.shaping:
                d = self.link.delay_for(len(frame))
                self.shaped_seconds += d
                precise_sleep(d)
            try:
                self.sock.sendall(frame)
            except OSError as e:
                self._in.put(ConnectionError(f"send failed: {e}"))
                return
            with self._lock:
                self.bytes_sent += len(frame)
                self.frames_sent += 1

    def _rx_loop(self) -> None:
        try:
            while not self._closed.is_set():
                head = _recv_exact(self.sock, 12)
                if head[:4] != MAGIC:
                    raise ConnectionError("frame desync")
                hlen, plen = struct.unpack("<II", head[4:])
                hb = _recv_exact(self.sock, hlen)
                payload = _recv_exact(self.sock, plen) if plen else b""
                with self._lock:
                    self.bytes_recv += 12 + hlen + plen
                self._in.put((json.loads(hb), payload))
        except Exception as e:  # surfaced to the reader
            if not self._closed.is_set():
                self._in.put(e)


def listen(host: str, port: int) -> socket.socket:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(8)
    return srv


def connect(host: str, port: int, retries: int = 200, delay: float = 0.05) -> socket.socket:
    last = None
    for _ in range(retries):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((host, port))
            return s
        except OSError as e:
            last = e
            time.sleep(delay)
    raise ConnectionError(f"cannot reach {host}:{port}: {last}")


# -- activation codec ----------------------------------------------------
# Hidden states move as raw fp16 by default. int8 halves the bytes on the wire
# at the cost of a per-row scale and a small amount of error; whether that is a
# win depends entirely on whether the link is bandwidth- or latency-bound.

def encode(arr, codec: str = "fp16") -> Tuple[dict, bytes]:
    import mlx.core as mx
    if codec == "fp16":
        a = np.array(arr.astype(mx.float16), copy=False)
        return {"codec": "fp16", "shape": list(a.shape)}, a.tobytes()
    if codec == "int8":
        a = np.array(arr.astype(mx.float32), copy=False)
        flat = a.reshape(-1, a.shape[-1])
        scale = np.abs(flat).max(axis=-1, keepdims=True) / 127.0
        scale[scale == 0] = 1.0
        q = np.clip(np.rint(flat / scale), -127, 127).astype(np.int8)
        return ({"codec": "int8", "shape": list(a.shape)},
                scale.astype(np.float32).tobytes() + q.tobytes())
    raise ValueError(f"unknown codec {codec}")


def decode(header: dict, payload: bytes):
    import mlx.core as mx
    shape = tuple(header["shape"])
    if header["codec"] == "fp16":
        a = np.frombuffer(payload, dtype=np.float16).reshape(shape)
        return mx.array(a)
    if header["codec"] == "int8":
        rows = int(np.prod(shape[:-1]))
        nsc = rows * 4
        scale = np.frombuffer(payload[:nsc], dtype=np.float32).reshape(rows, 1)
        q = np.frombuffer(payload[nsc:], dtype=np.int8).reshape(rows, shape[-1])
        return mx.array((q.astype(np.float32) * scale).reshape(shape).astype(np.float16))
    raise ValueError(f"unknown codec {header['codec']}")
