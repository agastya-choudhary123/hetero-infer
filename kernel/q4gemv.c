/* Fused int4 GEMV: dequantise in registers, never materialise the fp32 matrix.
 *
 * Decoding one token reads every weight exactly once, so the whole operation is
 * memory-bandwidth-bound. Expanding 4-bit weights to float32 first means moving
 * eight times the bytes; this reads the packed form and unpacks into registers,
 * which is the difference between ~1 GB and ~130 MB of traffic per layer.
 *
 * Weight layout matches MLX's quantisation:
 *   w       [n_out][n_in/8]        uint32, eight 4-bit values each, low first
 *   scales  [n_out][n_in/gs]       float16
 *   biases  [n_out][n_in/gs]       float16
 * and a group dequantises as  value = q * scale + bias.
 *
 * The per-group bias term factors out of the dot product:
 *   y[o] = sum_g ( scale[o][g] * sum_{i in g} q_i * x_i  +  bias[o][g] * sum_{i in g} x_i )
 * so sum_{i in g} x_i is computed once for the whole matrix rather than per row.
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

static inline float f16_to_f32(uint16_t h) {
#if defined(__F16C__)
    return _cvtsh_ss(h);
#elif defined(__ARM_NEON)
    __fp16 v;
    memcpy(&v, &h, 2);
    return (float)v;
#else
    uint32_t s = (uint32_t)(h >> 15) << 31;
    uint32_t e = (h >> 10) & 0x1F;
    uint32_t m = h & 0x3FF;
    uint32_t bits;
    if (e == 0) {
        if (m == 0) { bits = s; }
        else {
            e = 127 - 15 + 1;
            while (!(m & 0x400)) { m <<= 1; e--; }
            m &= 0x3FF;
            bits = s | (e << 23) | (m << 13);
        }
    } else if (e == 31) {
        bits = s | 0x7F800000u | (m << 13);
    } else {
        bits = s | ((e - 15 + 127) << 23) | (m << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
#endif
}

/* Dot product of one group's 4-bit values with the matching slice of x. */
static inline float group_dot(const uint32_t *w, const float *x, int words) {
#if defined(__AVX2__)
    const __m256i shifts = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
    const __m256i mask = _mm256_set1_epi32(0xF);
    __m256 acc = _mm256_setzero_ps();
    for (int k = 0; k < words; k++) {
        __m256i q = _mm256_and_si256(
            _mm256_srlv_epi32(_mm256_set1_epi32((int)w[k]), shifts), mask);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(q), _mm256_loadu_ps(x + 8 * k), acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
#elif defined(__ARM_NEON)
    const int32x4_t sh_lo = {0, -4, -8, -12};
    const int32x4_t sh_hi = {-16, -20, -24, -28};
    const uint32x4_t mask = vdupq_n_u32(0xF);
    float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
    for (int k = 0; k < words; k++) {
        uint32x4_t d = vdupq_n_u32(w[k]);
        uint32x4_t q0 = vandq_u32(vshlq_u32(d, sh_lo), mask);
        uint32x4_t q1 = vandq_u32(vshlq_u32(d, sh_hi), mask);
        acc0 = vfmaq_f32(acc0, vcvtq_f32_u32(q0), vld1q_f32(x + 8 * k));
        acc1 = vfmaq_f32(acc1, vcvtq_f32_u32(q1), vld1q_f32(x + 8 * k + 4));
    }
    return vaddvq_f32(vaddq_f32(acc0, acc1));
#else
    float acc = 0.0f;
    for (int k = 0; k < words; k++) {
        uint32_t d = w[k];
        for (int j = 0; j < 8; j++) acc += (float)((d >> (4 * j)) & 0xF) * x[8 * k + j];
    }
    return acc;
#endif
}

typedef struct {
    const uint32_t *w;
    const uint16_t *scales, *biases;
    const float *x, *xsum;
    float *out;
    int n_out, n_in, gs, row0, row1;
} job_t;

static void *run_rows(void *arg) {
    job_t *j = (job_t *)arg;
    const int gs = j->gs;
    const int n_groups = j->n_in / gs;
    const int words_per_group = gs / 8;
    const int words_per_row = j->n_in / 8;
    for (int o = j->row0; o < j->row1; o++) {
        const uint32_t *wr = j->w + (size_t)o * words_per_row;
        const uint16_t *sr = j->scales + (size_t)o * n_groups;
        const uint16_t *br = j->biases + (size_t)o * n_groups;
        float acc = 0.0f;
        for (int g = 0; g < n_groups; g++) {
            float dq = group_dot(wr + g * words_per_group, j->x + g * gs, words_per_group);
            acc += f16_to_f32(sr[g]) * dq + f16_to_f32(br[g]) * j->xsum[g];
        }
        j->out[o] = acc;
    }
    return NULL;
}

/* y[n_out] = dequant(w) . x[n_in]   —  returns 0 on success. */
int q4_gemv(const uint32_t *w, const uint16_t *scales, const uint16_t *biases,
            const float *x, float *out, int n_out, int n_in, int gs, int nthreads) {
    if (n_in % gs || gs % 8) return -1;
    const int n_groups = n_in / gs;
    float *xsum = (float *)malloc(sizeof(float) * n_groups);
    if (!xsum) return -2;
    for (int g = 0; g < n_groups; g++) {
        float s = 0.0f;
        for (int i = 0; i < gs; i++) s += x[g * gs + i];
        xsum[g] = s;
    }
    if (nthreads < 1) nthreads = 1;
    if (nthreads > 16) nthreads = 16;
    if (nthreads > n_out) nthreads = n_out;

    pthread_t th[16];
    job_t jobs[16];
    int per = (n_out + nthreads - 1) / nthreads;
    for (int t = 0; t < nthreads; t++) {
        jobs[t] = (job_t){w, scales, biases, x, xsum, out, n_out, n_in, gs,
                          t * per, (t + 1) * per < n_out ? (t + 1) * per : n_out};
    }
    for (int t = 1; t < nthreads; t++) pthread_create(&th[t], NULL, run_rows, &jobs[t]);
    run_rows(&jobs[0]);
    for (int t = 1; t < nthreads; t++) pthread_join(th[t], NULL);
    free(xsum);
    return 0;
}
