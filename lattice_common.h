/*
 * Lattice Video Codec - Common Definitions
 */

#ifndef LATTICE_COMMON_H
#define LATTICE_COMMON_H

#include <stdint.h>
#include <stddef.h>

/* ============================================================================
 * Constants
 * ============================================================================ */

#define LATTICE_MAGIC 0x4C415454  /* "LATT" */
#define MAX_REF_FRAMES 8
#define TILE_SIZE 128
#define CELL_SIZE 8
#define CDF_PRECISION 16
#define CDF_TOTAL (1 << CDF_PRECISION)  /* 65536 */
#define ADAPT_RATE 5
#define FILTER_SHIFT 10
#define NUM_CONTEXTS 110

/* Context base indices - per spec §7.6 */
#define CTX_SHAPE 0          /* 9 slots */
#define CTX_MODE 9           /* 9 slots */
#define CTX_CBF 18           /* 3 slots */
#define CTX_QP_DELTA 21      /* 3 slots */
#define CTX_REF_IDX 24       /* 1 slot */
#define CTX_MVD_CLASS 25     /* 2 slots */
#define CTX_BAND_LUMA 27     /* 8 slots: band_index * 2 + prev_zero */
#define CTX_SIG_LUMA 35      /* 16 slots */
#define CTX_LEVEL_LUMA 51    /* 16 slots */
#define CTX_BAND_CHROMA 67   /* 8 slots */
#define CTX_SIG_CHROMA 75    /* 16 slots */
#define CTX_LEVEL_CHROMA 91  /* 16 slots */
#define CTX_FILTER_DELTA 107 /* 3 slots */

/* Block shapes: {width_cells, height_cells} */
static const int BLOCK_SHAPES[7][2] = {
    {1, 1}, {2, 1}, {1, 2}, {2, 2}, {4, 2}, {2, 4}, {4, 4}
};

/* QP to qstep table */
static const uint16_t QSTEP[52] = {
    26,29,32,36,40,45,52,58,64,72,80,90,104,116,128,144,160,180,
    208,232,256,288,320,360,416,464,512,576,640,720,832,928,1024,
    1152,1280,1440,1664,1856,2048,2304,2560,2880,3328,3712,4096,
    4608,5120,5760,6656,7424,8192,9216
};

/* Perceptual frequency weight (§6.3.2)
 * Models decreasing sensitivity to high spatial frequencies.
 * PW[i][j] = min(16 + i*i + j*j, 112)
 * DC (0,0) = 16 (baseline), high freq capped at 112 (7× coarser quantization)
 */
static inline int perceptual_weight(int i, int j) {
    int w = 16 + i * i + j * j;
    return w < 112 ? w : 112;
}

/* Compute effective qstep with perceptual weighting (§6.3.3)
 * effective_qstep = (qstep * PW[i][j] + 8) >> 4
 */
static inline int effective_qstep(int qstep, int i, int j) {
    int pw = perceptual_weight(i, j);
    return (qstep * pw + 8) >> 4;
}

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

static inline int clamp_i(int x, int lo, int hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static inline int round_shift(int x, int n) {
    return (x + (1 << (n - 1))) >> n;
}

static inline int round_div(int a, int d) {
    if (a >= 0)
        return (a + (d >> 1)) / d;
    else
        return -((-a + (d >> 1)) / d);
}

static inline int min_i(int a, int b) { return a < b ? a : b; }
static inline int max_i(int a, int b) { return a > b ? a : b; }

/* Mode category for prediction mode context (§5.2) */
static inline int mode_category(int mode) {
    if (mode == 0) return 0;    /* INTRA */
    if (mode == 1) return 1;    /* INTER */
    return 2;                    /* SKIP */
}

/* ============================================================================
 * CDF Context
 * ============================================================================ */

typedef struct {
    uint32_t cdf[17];  /* Max alphabet size is 16 (+1 for terminator) */
    int size;          /* cdf[size] = CDF_TOTAL = 65536 */
} CdfContext;

static inline void cdf_init_uniform(CdfContext *ctx, int size) {
    ctx->size = size;
    for (int i = 0; i <= size; i++)
        ctx->cdf[i] = (CDF_TOTAL * i) / size;
}

static inline void cdf_adapt(CdfContext *ctx, int symbol) {
    /*
     * Adaptation per spec §7.4: shift CDF boundaries to increase probability
     * of observed symbol. Boundaries at or below 'symbol' decrease (toward 0).
     * Boundaries above 'symbol' increase (toward CDF_TOTAL).
     *
     * Spec formula uses arithmetic right shift of signed values:
     *   cdf[i] += (0 - cdf[i]) >> ADAPT_RATE        (move toward 0)
     *   cdf[i] += (65536 - cdf[i]) >> ADAPT_RATE    (move toward 65536)
     */
    int32_t new_cdf[17];
    new_cdf[0] = 0;
    new_cdf[ctx->size] = CDF_TOTAL;

    for (int i = 1; i < ctx->size; i++) {
        int32_t val = (int32_t)ctx->cdf[i];
        if (i <= symbol) {
            /* Move toward 0: val += (0 - val) >> ADAPT_RATE */
            val += (0 - val) >> ADAPT_RATE;
        } else {
            /* Move toward CDF_TOTAL: val += (CDF_TOTAL - val) >> ADAPT_RATE */
            val += (CDF_TOTAL - val) >> ADAPT_RATE;
        }
        new_cdf[i] = val;
    }

    /* Enforce minimum frequency per §7.4.1: single-pass with dependency tracking */
    for (int i = 0; i < ctx->size - 1; i++) {
        int min_value = new_cdf[i] + 1;                      /* must be > previous entry */
        int max_value = CDF_TOTAL - (ctx->size - 1 - i);     /* must leave room for remaining symbols */
        new_cdf[i + 1] = clamp_i(new_cdf[i + 1], min_value, max_value);
    }

    /* Copy back */
    for (int i = 1; i < ctx->size; i++) {
        ctx->cdf[i] = (uint16_t)new_cdf[i];
    }
}

/* ============================================================================
 * Structures
 * ============================================================================ */

typedef struct {
    uint16_t frame_width;
    uint16_t frame_height;
    uint8_t bit_depth;
    uint8_t max_ref_frames;
    int tiles_wide;
    int tiles_high;
} SequenceHeader;

typedef struct {
    uint8_t frame_type;
    uint8_t base_qp;
    uint8_t filter_mode;
    int16_t luma_weights[4][4][4][3][3];
    int16_t luma_biases[4][4];
} FrameHeader;

typedef struct {
    int mode;
    int ref_idx;
    int16_t mv_x, mv_y;
    int qp_delta;
    int cbf;
    int x, y, w, h;
} BlockInfo;

typedef struct {
    uint16_t *y, *cb, *cr;
    int width, height;
} Frame;

typedef struct {
    Frame *frames[MAX_REF_FRAMES];
    int count;
} Dpb;

/* ============================================================================
 * DCT Coefficient Matrices
 * ============================================================================ */

static const int16_t DCT4[4][4] = {
    { 64,  64,  64,  64},
    { 84,  35, -35, -84},
    { 64, -64, -64,  64},
    { 35, -84,  84, -35}
};

static const int16_t DCT8[8][8] = {
    { 64,  64,  64,  64,  64,  64,  64,  64},
    { 89,  75,  50,  18, -18, -50, -75, -89},
    { 84,  35, -35, -84, -84, -35,  35,  84},
    { 75, -18, -89, -50,  50,  89,  18, -75},
    { 64, -64, -64,  64,  64, -64, -64,  64},
    { 50, -89,  18,  75, -75, -18,  89, -50},
    { 35, -84,  84, -35, -35,  84, -84,  35},
    { 18, -50,  75, -89,  89, -75,  50, -18}
};

static const int16_t DCT16[16][16] = {
    { 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64},
    { 90, 87, 80, 70, 57, 43, 26,  9, -9,-26,-43,-57,-70,-80,-87,-90},
    { 89, 75, 50, 18,-18,-50,-75,-89,-89,-75,-50,-18, 18, 50, 75, 89},
    { 87, 57,  9,-43,-80,-90,-70,-26, 26, 70, 90, 80, 43, -9,-57,-87},
    { 84, 35,-35,-84,-84,-35, 35, 84, 84, 35,-35,-84,-84,-35, 35, 84},
    { 80,  9,-70,-87,-26, 57, 90, 43,-43,-90,-57, 26, 87, 70, -9,-80},
    { 75,-18,-89,-50, 50, 89, 18,-75,-75, 18, 89, 50,-50,-89,-18, 75},
    { 70,-43,-87,  9, 90, 26,-80,-57, 57, 80,-26,-90, -9, 87, 43,-70},
    { 64,-64,-64, 64, 64,-64,-64, 64, 64,-64,-64, 64, 64,-64,-64, 64},
    { 57,-80,-26, 90, -9,-87, 43, 70,-70,-43, 87,  9,-90, 26, 80,-57},
    { 50,-89, 18, 75,-75,-18, 89,-50,-50, 89,-18,-75, 75, 18,-89, 50},
    { 43,-90, 57, 26,-87, 70,  9,-80, 80, -9,-70, 87,-26,-57, 90,-43},
    { 35,-84, 84,-35,-35, 84,-84, 35, 35,-84, 84,-35,-35, 84,-84, 35},
    { 26,-70, 90,-80, 43,  9,-57, 87,-87, 57, -9,-43, 80,-90, 70,-26},
    { 18,-50, 75,-89, 89,-75, 50,-18,-18, 50,-75, 89,-89, 75,-50, 18},
    {  9,-26, 43,-57, 70,-80, 87,-90, 90,-87, 80,-70, 57,-43, 26, -9}
};

static const int16_t DCT32[32][32] = {
    { 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64},
    { 90, 90, 88, 85, 82, 78, 73, 67, 61, 54, 47, 39, 30, 22, 13,  4, -4,-13,-22,-30,-39,-47,-54,-61,-67,-73,-78,-82,-85,-88,-90,-90},
    { 90, 87, 80, 70, 57, 43, 26,  9, -9,-26,-43,-57,-70,-80,-87,-90,-90,-87,-80,-70,-57,-43,-26, -9,  9, 26, 43, 57, 70, 80, 87, 90},
    { 90, 82, 67, 47, 22, -4,-30,-54,-73,-85,-90,-88,-78,-61,-39,-13, 13, 39, 61, 78, 88, 90, 85, 73, 54, 30,  4,-22,-47,-67,-82,-90},
    { 89, 75, 50, 18,-18,-50,-75,-89,-89,-75,-50,-18, 18, 50, 75, 89, 89, 75, 50, 18,-18,-50,-75,-89,-89,-75,-50,-18, 18, 50, 75, 89},
    { 88, 67, 30,-13,-54,-82,-90,-78,-47, -4, 39, 73, 90, 85, 61, 22,-22,-61,-85,-90,-73,-39,  4, 47, 78, 90, 82, 54, 13,-30,-67,-88},
    { 87, 57,  9,-43,-80,-90,-70,-26, 26, 70, 90, 80, 43, -9,-57,-87,-87,-57, -9, 43, 80, 90, 70, 26,-26,-70,-90,-80,-43,  9, 57, 87},
    { 85, 47,-13,-67,-90,-73,-22, 39, 82, 88, 54, -4,-61,-90,-78,-30, 30, 78, 90, 61,  4,-54,-88,-82,-39, 22, 73, 90, 67, 13,-47,-85},
    { 84, 35,-35,-84,-84,-35, 35, 84, 84, 35,-35,-84,-84,-35, 35, 84, 84, 35,-35,-84,-84,-35, 35, 84, 84, 35,-35,-84,-84,-35, 35, 84},
    { 82, 22,-54,-90,-61, 13, 78, 85, 30,-47,-90,-67,  4, 73, 88, 39,-39,-88,-73, -4, 67, 90, 47,-30,-85,-78,-13, 61, 90, 54,-22,-82},
    { 80,  9,-70,-87,-26, 57, 90, 43,-43,-90,-57, 26, 87, 70, -9,-80,-80, -9, 70, 87, 26,-57,-90,-43, 43, 90, 57,-26,-87,-70,  9, 80},
    { 78, -4,-82,-73, 13, 85, 67,-22,-88,-61, 30, 90, 54,-39,-90,-47, 47, 90, 39,-54,-90,-30, 61, 88, 22,-67,-85,-13, 73, 82,  4,-78},
    { 75,-18,-89,-50, 50, 89, 18,-75,-75, 18, 89, 50,-50,-89,-18, 75, 75,-18,-89,-50, 50, 89, 18,-75,-75, 18, 89, 50,-50,-89,-18, 75},
    { 73,-30,-90,-22, 78, 67,-39,-90,-13, 82, 61,-47,-88, -4, 85, 54,-54,-85,  4, 88, 47,-61,-82, 13, 90, 39,-67,-78, 22, 90, 30,-73},
    { 70,-43,-87,  9, 90, 26,-80,-57, 57, 80,-26,-90, -9, 87, 43,-70,-70, 43, 87, -9,-90,-26, 80, 57,-57,-80, 26, 90,  9,-87,-43, 70},
    { 67,-54,-78, 39, 85,-22,-90,  4, 90, 13,-88,-30, 82, 47,-73,-61, 61, 73,-47,-82, 30, 88,-13,-90, -4, 90, 22,-85,-39, 78, 54,-67},
    { 64,-64,-64, 64, 64,-64,-64, 64, 64,-64,-64, 64, 64,-64,-64, 64, 64,-64,-64, 64, 64,-64,-64, 64, 64,-64,-64, 64, 64,-64,-64, 64},
    { 61,-73,-47, 82, 30,-88,-13, 90, -4,-90, 22, 85,-39,-78, 54, 67,-67,-54, 78, 39,-85,-22, 90,  4,-90, 13, 88,-30,-82, 47, 73,-61},
    { 57,-80,-26, 90, -9,-87, 43, 70,-70,-43, 87,  9,-90, 26, 80,-57,-57, 80, 26,-90,  9, 87,-43,-70, 70, 43,-87, -9, 90,-26,-80, 57},
    { 54,-85, -4, 88,-47,-61, 82, 13,-90, 39, 67,-78,-22, 90,-30,-73, 73, 30,-90, 22, 78,-67,-39, 90,-13,-82, 61, 47,-88,  4, 85,-54},
    { 50,-89, 18, 75,-75,-18, 89,-50,-50, 89,-18,-75, 75, 18,-89, 50, 50,-89, 18, 75,-75,-18, 89,-50,-50, 89,-18,-75, 75, 18,-89, 50},
    { 47,-90, 39, 54,-90, 30, 61,-88, 22, 67,-85, 13, 73,-82,  4, 78,-78, -4, 82,-73,-13, 85,-67,-22, 88,-61,-30, 90,-54,-39, 90,-47},
    { 43,-90, 57, 26,-87, 70,  9,-80, 80, -9,-70, 87,-26,-57, 90,-43,-43, 90,-57,-26, 87,-70, -9, 80,-80,  9, 70,-87, 26, 57,-90, 43},
    { 39,-88, 73, -4,-67, 90,-47,-30, 85,-78, 13, 61,-90, 54, 22,-82, 82,-22,-54, 90,-61,-13, 78,-85, 30, 47,-90, 67,  4,-73, 88,-39},
    { 35,-84, 84,-35,-35, 84,-84, 35, 35,-84, 84,-35,-35, 84,-84, 35, 35,-84, 84,-35,-35, 84,-84, 35, 35,-84, 84,-35,-35, 84,-84, 35},
    { 30,-78, 90,-61,  4, 54,-88, 82,-39,-22, 73,-90, 67,-13,-47, 85,-85, 47, 13,-67, 90,-73, 22, 39,-82, 88,-54, -4, 61,-90, 78,-30},
    { 26,-70, 90,-80, 43,  9,-57, 87,-87, 57, -9,-43, 80,-90, 70,-26,-26, 70,-90, 80,-43, -9, 57,-87, 87,-57,  9, 43,-80, 90,-70, 26},
    { 22,-61, 85,-90, 73,-39, -4, 47,-78, 90,-82, 54,-13,-30, 67,-88, 88,-67, 30, 13,-54, 82,-90, 78,-47,  4, 39,-73, 90,-85, 61,-22},
    { 18,-50, 75,-89, 89,-75, 50,-18,-18, 50,-75, 89,-89, 75,-50, 18, 18,-50, 75,-89, 89,-75, 50,-18,-18, 50,-75, 89,-89, 75,-50, 18},
    { 13,-39, 61,-78, 88,-90, 85,-73, 54,-30,  4, 22,-47, 67,-82, 90,-90, 82,-67, 47,-22, -4, 30,-54, 73,-85, 90,-88, 78,-61, 39,-13},
    {  9,-26, 43,-57, 70,-80, 87,-90, 90,-87, 80,-70, 57,-43, 26, -9, -9, 26,-43, 57,-70, 80,-87, 90,-90, 87,-80, 70,-57, 43,-26,  9},
    {  4,-13, 22,-30, 39,-47, 54,-61, 67,-73, 78,-82, 85,-88, 90,-90, 90,-90, 88,-85, 82,-78, 73,-67, 61,-54, 47,-39, 30,-22, 13, -4}
};

/* ============================================================================
 * Default Filter Weights
 * ============================================================================ */

static inline void init_default_weights(int16_t weights[4][4][4][3][3], int16_t biases[4][4]) {
    for (int l = 0; l < 4; l++)
        for (int co = 0; co < 4; co++)
            for (int ci = 0; ci < 4; ci++)
                for (int ky = 0; ky < 3; ky++)
                    for (int kx = 0; kx < 3; kx++)
                        weights[l][co][ci][ky][kx] = 0;

    for (int l = 0; l < 4; l++)
        for (int c = 0; c < 4; c++)
            biases[l][c] = 0;

    /* Layer 1: w[0][0][1][1] = 1024 */
    weights[0][0][0][1][1] = 1024;
    /* Layer 2: w[c][c][1][1] = 1024 */
    for (int c = 0; c < 4; c++)
        weights[1][c][c][1][1] = 1024;
    /* Layer 3: w[c][c][1][1] = 1024 */
    for (int c = 0; c < 4; c++)
        weights[2][c][c][1][1] = 1024;
    /* Layer 4: w[0][0][1][1] = 1024 */
    weights[3][0][0][1][1] = 1024;
}

/* ============================================================================
 * Scan Order Utilities
 * ============================================================================ */

static inline void generate_scan(int W, int H, int *scan) {
    int pos = 0;
    for (int f = 0; f < W + H - 1; f++) {
        int v_min = max_i(0, f - W + 1);
        int v_max = min_i(f, H - 1);
        for (int v = v_min; v <= v_max; v++) {
            int u = f - v;
            scan[pos++] = v * W + u;
        }
    }
}

static inline void get_band_boundaries(int W, int H, int *bounds, int *num_bands) {
    int N = W * H;
    int cumulative[65] = {0};
    int idx = 1;
    for (int f = 0; f < W + H - 1; f++) {
        int v_min = max_i(0, f - W + 1);
        int v_max = min_i(f, H - 1);
        cumulative[idx] = cumulative[idx - 1] + (v_max - v_min + 1);
        idx++;
    }
    bounds[0] = 0;
    bounds[1] = cumulative[1];
    bounds[2] = cumulative[min_i(3, idx - 1)];
    if (W >= 8 && H >= 8) {
        bounds[3] = cumulative[min_i(7, idx - 1)];
        bounds[4] = N;
        *num_bands = 4;
    } else {
        bounds[3] = N;
        *num_bands = 3;
    }
}

#endif /* LATTICE_COMMON_H */
