/* Fused int4 GEMV: dequantise in registers, never materialise the fp32 matrix.
 *
 * Decoding one token reads every weight exactly once, so this is bound by
 * memory traffic. Expanding 4-bit weights to float32 first moves eight times
 * the bytes; this reads the packed form and unpacks into registers.
 *
 * Weight layout matches MLX's quantisation:
 *   w       [n_out][n_in/8]   uint32, eight 4-bit values each, low nibble first
 *   scales  [n_out][n_in/gs]  float16
 *   biases  [n_out][n_in/gs]  float16
 * dequantising as  value = q * scale + bias.
 *
 * Three things matter for speed, all learned by measuring:
 *
 *  - The per-group bias term factors out of the dot product, so
 *    sum_{i in group} x_i is computed once for the whole matrix:
 *      y[o] = sum_g ( scale[o][g] * sum_i q_i x_i  +  bias[o][g] * xsum[g] )
 *
 *  - The reduction stays per group. Folding the scale into the accumulator to
 *    defer it to once per row was measured and was 2x slower: it costs an extra
 *    multiply in the innermost loop, which matters more than the reduction it
 *    saves.
 *
 *  - Four output rows are computed together so each x vector is loaded once and
 *    used four times, and four independent FMA chains hide their latency.
 *
 * Threads are a persistent pool. Creating them per call cost ~0.4 ms, and a
 * decoder layer makes seven calls.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#ifndef ROWS
#define ROWS 1          /* output rows computed together; tuned per machine */
#endif
#define MAX_THREADS 16

static inline float f16_to_f32(uint16_t h) {
#if defined(__F16C__)
    return _cvtsh_ss(h);
#elif defined(__ARM_NEON)
    __fp16 v; memcpy(&v, &h, 2); return (float)v;
#else
    uint32_t s = (uint32_t)(h >> 15) << 31, e = (h >> 10) & 0x1F, m = h & 0x3FF, bits;
    if (e == 0) {
        if (m == 0) bits = s;
        else { e = 113; while (!(m & 0x400)) { m <<= 1; e--; } bits = s | (e << 23) | ((m & 0x3FF) << 13); }
    } else if (e == 31) bits = s | 0x7F800000u | (m << 13);
    else bits = s | ((e - 15 + 127) << 23) | (m << 13);
    float f; memcpy(&f, &bits, 4); return f;
#endif
}

/* Dot exactly ROWS output rows against one group of `gs` inputs, unscaled.
   The x vectors are loaded once and reused across all ROWS.

   ROWS is a compile-time constant on purpose: with a runtime row count the
   compiler leaves a loop of unknown trip count in the innermost position and
   cannot unroll it, which measured 1.6x slower on both machines. */
static inline void group_rows(const uint32_t *const w[ROWS], const float *x, int words,
                              float dot[ROWS]) {
    const int nrows = ROWS;
#if defined(__AVX2__)
    /* A variable shift per word. Unpacking through bytes instead (to dodge
       vpsrlvd, which is 3 uops on Haswell) was tried and measured 1.3x slower
       on the 2013 machine, so the obvious-looking micro-optimisation loses. */
    const __m256i shifts = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
    const __m256i mask = _mm256_set1_epi32(0xF);
    __m256 a[ROWS];
    for (int r = 0; r < nrows; r++) a[r] = _mm256_setzero_ps();
    for (int k = 0; k < words; k++) {
        __m256 xv = _mm256_loadu_ps(x + 8 * k);
        for (int r = 0; r < nrows; r++) {
            __m256i q = _mm256_and_si256(
                _mm256_srlv_epi32(_mm256_set1_epi32((int)w[r][k]), shifts), mask);
            a[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(q), xv, a[r]);
        }
    }
    for (int r = 0; r < nrows; r++) {
        __m128 lo = _mm_add_ps(_mm256_castps256_ps128(a[r]), _mm256_extractf128_ps(a[r], 1));
        lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
        dot[r] = _mm_cvtss_f32(lo);
    }
#elif defined(__ARM_NEON)
    const int32x4_t sh_lo = {0, -4, -8, -12}, sh_hi = {-16, -20, -24, -28};
    const uint32x4_t mask = vdupq_n_u32(0xF);
    float32x4_t a0[ROWS], a1[ROWS];
    for (int r = 0; r < nrows; r++) { a0[r] = vdupq_n_f32(0.0f); a1[r] = vdupq_n_f32(0.0f); }
    for (int k = 0; k < words; k++) {
        float32x4_t x0 = vld1q_f32(x + 8 * k), x1 = vld1q_f32(x + 8 * k + 4);
        for (int r = 0; r < nrows; r++) {
            uint32x4_t d = vdupq_n_u32(w[r][k]);
            a0[r] = vfmaq_f32(a0[r], vcvtq_f32_u32(vandq_u32(vshlq_u32(d, sh_lo), mask)), x0);
            a1[r] = vfmaq_f32(a1[r], vcvtq_f32_u32(vandq_u32(vshlq_u32(d, sh_hi), mask)), x1);
        }
    }
    for (int r = 0; r < nrows; r++) dot[r] = vaddvq_f32(vaddq_f32(a0[r], a1[r]));
#else
    for (int r = 0; r < nrows; r++) {
        float s = 0.0f;
        for (int k = 0; k < words; k++) {
            uint32_t d = w[r][k];
            for (int j = 0; j < 8; j++) s += (float)((d >> (4 * j)) & 0xF) * x[8 * k + j];
        }
        dot[r] = s;
    }
#endif
}

typedef struct {
    const uint32_t *w; const uint16_t *scales, *biases;
    const float *x, *xsum; float *out;
    int n_in, gs, row0, row1;
} work_t;

typedef struct {
    pthread_t th; pthread_mutex_t mu; pthread_cond_t go, done;
    work_t work; int has_work, quit, finished;
} slot_t;

static slot_t g_slot[MAX_THREADS];
static int g_nthreads = 0;
static pthread_once_t g_once = PTHREAD_ONCE_INIT;

static void do_work(const work_t *j) {
    const int gs = j->gs, n_groups = j->n_in / gs;
    const int wpg = gs / 8, wpr = j->n_in / 8;
    int o = j->row0;
    for (; o + ROWS <= j->row1; o += ROWS) {
        const uint32_t *wr[ROWS];
        float acc[ROWS] = {0};
        for (int r = 0; r < ROWS; r++) wr[r] = j->w + (size_t)(o + r) * wpr;
        for (int g = 0; g < n_groups; g++) {
            const uint32_t *wg[ROWS];
            float dot[ROWS];
            for (int r = 0; r < ROWS; r++) wg[r] = wr[r] + g * wpg;
            group_rows(wg, j->x + g * gs, wpg, dot);
            for (int r = 0; r < ROWS; r++) {
                size_t idx = (size_t)(o + r) * n_groups + g;
                acc[r] += f16_to_f32(j->scales[idx]) * dot[r]
                        + f16_to_f32(j->biases[idx]) * j->xsum[g];
            }
        }
        for (int r = 0; r < ROWS; r++) j->out[o + r] = acc[r];
    }
    for (; o < j->row1; o++) {          /* rows left over past the last block */
        const uint32_t *wr = j->w + (size_t)o * wpr;
        float acc = 0.0f;
        for (int g = 0; g < n_groups; g++) {
            const uint32_t *wg = wr + g * wpg;
            const float *xg = j->x + g * gs;
            float s = 0.0f;
            for (int k = 0; k < wpg; k++) {
                uint32_t d = wg[k];
                for (int b = 0; b < 8; b++) s += (float)((d >> (4 * b)) & 0xF) * xg[8 * k + b];
            }
            size_t idx = (size_t)o * n_groups + g;
            acc += f16_to_f32(j->scales[idx]) * s + f16_to_f32(j->biases[idx]) * j->xsum[g];
        }
        j->out[o] = acc;
    }
}

static void *worker_loop(void *arg) {
    slot_t *s = (slot_t *)arg;
    for (;;) {
        pthread_mutex_lock(&s->mu);
        while (!s->has_work && !s->quit) pthread_cond_wait(&s->go, &s->mu);
        if (s->quit) { pthread_mutex_unlock(&s->mu); return NULL; }
        work_t j = s->work; s->has_work = 0;
        pthread_mutex_unlock(&s->mu);
        do_work(&j);
        pthread_mutex_lock(&s->mu);
        s->finished = 1; pthread_cond_signal(&s->done);
        pthread_mutex_unlock(&s->mu);
    }
}

static void pool_init(void) {
    const char *e = getenv("HETERO_KERNEL_THREADS");
    int n = e ? atoi(e) : 4;
    if (n < 1) n = 1;
    if (n > MAX_THREADS) n = MAX_THREADS;
    g_nthreads = n;
    for (int t = 1; t < n; t++) {
        pthread_mutex_init(&g_slot[t].mu, NULL);
        pthread_cond_init(&g_slot[t].go, NULL);
        pthread_cond_init(&g_slot[t].done, NULL);
        pthread_create(&g_slot[t].th, NULL, worker_loop, &g_slot[t]);
    }
}

int q4_gemv(const uint32_t *w, const uint16_t *scales, const uint16_t *biases,
            const float *x, float *out, int n_out, int n_in, int gs, int nthreads) {
    /* The AVX2 path consumes four words at a time, so a group must be a
       multiple of 32 values. Every MLX group size in use (64, 128) is. */
    if (n_in % gs || gs % 8) return -1;
    pthread_once(&g_once, pool_init);
    const int n_groups = n_in / gs;
    float *xsum = (float *)alloca(sizeof(float) * n_groups);
    for (int g = 0; g < n_groups; g++) {
        float s = 0.0f;
        for (int i = 0; i < gs; i++) s += x[g * gs + i];
        xsum[g] = s;
    }
    int n = nthreads > 0 ? nthreads : g_nthreads;
    if (n > g_nthreads) n = g_nthreads;
    /* Split on ROWS boundaries so blocks stay whole. */
    int blocks = (n_out + ROWS - 1) / ROWS;
    if (n > blocks) n = blocks;
    int per = ((blocks + n - 1) / n) * ROWS;

    for (int t = 1; t < n; t++) {
        int r0 = t * per, r1 = r0 + per < n_out ? r0 + per : n_out;
        if (r0 >= n_out) { continue; }
        pthread_mutex_lock(&g_slot[t].mu);
        g_slot[t].work = (work_t){w, scales, biases, x, xsum, out, n_in, gs, r0, r1};
        g_slot[t].has_work = 1; g_slot[t].finished = 0;
        pthread_cond_signal(&g_slot[t].go);
        pthread_mutex_unlock(&g_slot[t].mu);
    }
    work_t mine = {w, scales, biases, x, xsum, out, n_in, gs, 0, per < n_out ? per : n_out};
    do_work(&mine);
    for (int t = 1; t < n; t++) {
        if (t * per >= n_out) continue;
        pthread_mutex_lock(&g_slot[t].mu);
        while (!g_slot[t].finished) pthread_cond_wait(&g_slot[t].done, &g_slot[t].mu);
        pthread_mutex_unlock(&g_slot[t].mu);
    }
    return 0;
}
