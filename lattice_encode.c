/*
 * Lattice Video Codec - Full Encoder with Inter Prediction
 * Draft 0.4 - February 2026
 *
 * Usage: lattice_encode <input.yuv> <width> <height> <qp> <output.lat> [recon.yuv] [--psy]
 *
 * Features:
 * - Full RDO (Rate-Distortion Optimization) for block size and mode selection
 * - Variable block sizes: 8x8, 8x16, 16x8, 16x16, 8x32, 32x8, 32x32
 * - Intra prediction (DC + gradient)
 * - Inter prediction with motion compensation (bilinear, quarter-pel)
 * - SKIP mode for efficient coding of static/predicted regions
 * - Hierarchical motion search (diamond + sub-pel refinement)
 * - Checkpoint/rollback for trial encoding
 * - Optional perceptual RDO (--psy) that penalizes blocking artifacts
 * - Trellis quantization for improved R-D efficiency
 */

/* Global RDO mode: 0 = MSE, 1 = perceptual (boundary-aware) */
static int g_psy_rdo = 0;

/* Global trellis quantization flag: 0 = off, 1 = on */
static int g_trellis = 1;

/* Weight for boundary discontinuity penalty in perceptual RDO */
#define PSY_BOUNDARY_WEIGHT 2.0

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "lattice_common.h"
#include "lattice_bitstream.h"
#include "lattice_rans.h"

/* ============================================================================
 * Encoder Data Structures
 * ============================================================================ */

/* Prediction modes */
#define MODE_INTRA 0
#define MODE_INTER 1
#define MODE_SKIP  2

typedef struct {
    int16_t coeff_y[32*32];     /* Quantized luma coefficients */
    int16_t coeff_cb[16*16];    /* Quantized Cb coefficients */
    int16_t coeff_cr[16*16];    /* Quantized Cr coefficients */
    int cbf;                     /* Combined coded block flag */
    int qp_delta;
    int x, y, w, h;             /* Block position/size (luma pixels) */
    int mode;                    /* 0=INTRA, 1=INTER, 2=SKIP */
    int16_t mv_x, mv_y;         /* Motion vector (quarter-pel units) */
    int ref_idx;                 /* Reference frame index */
} EncodedBlock;

/* Reference frame in DPB */
typedef struct {
    uint16_t *y, *cb, *cr;
    int width, height;
} RefFrame;

/* Decoded Picture Buffer for encoder */
typedef struct {
    RefFrame frames[MAX_REF_FRAMES];
    int count;
    int max_refs;
} EncoderDPB;

typedef struct {
    int width, height;
    int chroma_w, chroma_h;     /* Chroma dimensions with ceiling division per §10.1 */
    int bit_depth;
    int base_qp;
    int frame_type;             /* 0=I-frame, 1=P-frame */

    uint16_t *orig_y, *orig_cb, *orig_cr;
    uint16_t *recon_y, *recon_cb, *recon_cr;

    EncodedBlock *blocks;
    int num_blocks;
    int *cell_to_block;         /* Maps cell position to block index */
    int cells_w, cells_h;

    EncoderDPB dpb;             /* Reference frames */
} EncoderContext;

/* Tile encoder state */
typedef struct {
    DualRansEncoder rans;
    BypassWriter bypass;
    CdfContext contexts[NUM_CONTEXTS];

    int tile_x, tile_y;
    int tile_w, tile_h;
    int cells_w, cells_h;

    EncodedBlock *blocks;
    int num_blocks;
    int *cell_to_block;
} TileEncoder;

/* ============================================================================
 * Encoder Checkpoint for RDO
 * ============================================================================ */

typedef struct {
    /* rANS symbol buffer state */
    size_t stream0_count;
    size_t stream1_count;
    int toggle;

    /* Bypass writer state */
    int bypass_byte_pos;
    int bypass_bit_idx;
    size_t bypass_data_size;
    uint8_t *bypass_data;

    /* Context state (CDFs) */
    CdfContext contexts[NUM_CONTEXTS];

    /* Cell-to-block mapping state */
    int *cell_to_block;
    int cells_size;

    /* Block count */
    int num_blocks;
} EncoderCheckpoint;

static void checkpoint_save(TileEncoder *te, EncoderCheckpoint *cp) {
    /* rANS state */
    cp->stream0_count = te->rans.stream0_syms.count;
    cp->stream1_count = te->rans.stream1_syms.count;
    cp->toggle = te->rans.toggle;

    /* Bypass state */
    cp->bypass_byte_pos = te->bypass.byte_pos;
    cp->bypass_bit_idx = te->bypass.bit_idx;
    cp->bypass_data_size = bypass_writer_size(&te->bypass);
    cp->bypass_data = malloc(cp->bypass_data_size);
    memcpy(cp->bypass_data, te->bypass.data, cp->bypass_data_size);

    /* Context state */
    memcpy(cp->contexts, te->contexts, sizeof(cp->contexts));

    /* Cell-to-block mapping */
    cp->cells_size = te->cells_w * te->cells_h;
    cp->cell_to_block = malloc(cp->cells_size * sizeof(int));
    memcpy(cp->cell_to_block, te->cell_to_block, cp->cells_size * sizeof(int));

    /* Block count */
    cp->num_blocks = te->num_blocks;
}

static void checkpoint_restore(TileEncoder *te, EncoderCheckpoint *cp) {
    /* rANS state */
    te->rans.stream0_syms.count = cp->stream0_count;
    te->rans.stream1_syms.count = cp->stream1_count;
    te->rans.toggle = cp->toggle;

    /* Bypass state */
    te->bypass.byte_pos = cp->bypass_byte_pos;
    te->bypass.bit_idx = cp->bypass_bit_idx;
    memcpy(te->bypass.data, cp->bypass_data, cp->bypass_data_size);

    /* Context state */
    memcpy(te->contexts, cp->contexts, sizeof(cp->contexts));

    /* Cell-to-block mapping */
    memcpy(te->cell_to_block, cp->cell_to_block, cp->cells_size * sizeof(int));

    /* Block count */
    te->num_blocks = cp->num_blocks;
}

static void checkpoint_free(EncoderCheckpoint *cp) {
    free(cp->bypass_data);
    free(cp->cell_to_block);
    cp->bypass_data = NULL;
    cp->cell_to_block = NULL;
}

/* Reconstruction buffer checkpoint (for a rectangular region) */
typedef struct {
    uint16_t *y_data;
    uint16_t *cb_data;
    uint16_t *cr_data;
    int x, y, w, h;
    int stride_y, stride_c;
} ReconCheckpoint;

static void recon_checkpoint_save(ReconCheckpoint *rcp,
                                   uint16_t *recon_y, uint16_t *recon_cb, uint16_t *recon_cr,
                                   int stride_y, int stride_c,
                                   int x, int y, int w, int h) {
    rcp->x = x;
    rcp->y = y;
    rcp->w = w;
    rcp->h = h;
    rcp->stride_y = stride_y;
    rcp->stride_c = stride_c;

    /* Save luma */
    rcp->y_data = malloc(w * h * sizeof(uint16_t));
    for (int row = 0; row < h; row++) {
        memcpy(&rcp->y_data[row * w], &recon_y[(y + row) * stride_y + x], w * sizeof(uint16_t));
    }

    /* Save chroma */
    int cw = w / 2, ch = h / 2, cx = x / 2, cy = y / 2;
    rcp->cb_data = malloc(cw * ch * sizeof(uint16_t));
    rcp->cr_data = malloc(cw * ch * sizeof(uint16_t));
    for (int row = 0; row < ch; row++) {
        memcpy(&rcp->cb_data[row * cw], &recon_cb[(cy + row) * stride_c + cx], cw * sizeof(uint16_t));
        memcpy(&rcp->cr_data[row * cw], &recon_cr[(cy + row) * stride_c + cx], cw * sizeof(uint16_t));
    }
}

static void recon_checkpoint_restore(ReconCheckpoint *rcp,
                                      uint16_t *recon_y, uint16_t *recon_cb, uint16_t *recon_cr) {
    /* Restore luma */
    for (int row = 0; row < rcp->h; row++) {
        memcpy(&recon_y[(rcp->y + row) * rcp->stride_y + rcp->x],
               &rcp->y_data[row * rcp->w], rcp->w * sizeof(uint16_t));
    }

    /* Restore chroma */
    int cw = rcp->w / 2, ch = rcp->h / 2, cx = rcp->x / 2, cy = rcp->y / 2;
    for (int row = 0; row < ch; row++) {
        memcpy(&recon_cb[(cy + row) * rcp->stride_c + cx], &rcp->cb_data[row * cw], cw * sizeof(uint16_t));
        memcpy(&recon_cr[(cy + row) * rcp->stride_c + cx], &rcp->cr_data[row * cw], cw * sizeof(uint16_t));
    }
}

static void recon_checkpoint_free(ReconCheckpoint *rcp) {
    free(rcp->y_data);
    free(rcp->cb_data);
    free(rcp->cr_data);
    rcp->y_data = rcp->cb_data = rcp->cr_data = NULL;
}

/* ============================================================================
 * Forward DCT Transform
 * ============================================================================ */

static void forward_transform(int16_t *residual, int16_t *coeff, int W, int H, int bit_depth) {
    const int16_t *CW = (W == 4) ? &DCT4[0][0] : (W == 8) ? &DCT8[0][0] : (W == 16) ? &DCT16[0][0] : &DCT32[0][0];
    const int16_t *CH = (H == 4) ? &DCT4[0][0] : (H == 8) ? &DCT8[0][0] : (H == 16) ? &DCT16[0][0] : &DCT32[0][0];
    (void)bit_depth;

    int16_t intermediate[32 * 32];
    int32_t temp[32];

    int total_shift;
    if (W == 4 && H == 4) total_shift = 9;
    else if (W == 8 && H == 8) total_shift = 11;
    else if (W == 16 && H == 16) total_shift = 13;
    else if (W == 32 && H == 32) total_shift = 15;
    else {
        int area = W * H;
        total_shift = (area <= 16) ? 9 : (area <= 64) ? 11 : (area <= 256) ? 13 : 15;
    }
    int shift1 = total_shift / 2;
    int shift2 = total_shift - shift1;

    for (int col = 0; col < W; col++) {
        for (int k = 0; k < H; k++) {
            int32_t acc = 0;
            for (int n = 0; n < H; n++)
                acc += CH[k * H + n] * residual[n * W + col];
            temp[k] = acc;
        }
        for (int k = 0; k < H; k++)
            intermediate[k * W + col] = clamp_i(round_shift(temp[k], shift1), -32768, 32767);
    }

    for (int row = 0; row < H; row++) {
        for (int k = 0; k < W; k++) {
            int32_t acc = 0;
            for (int n = 0; n < W; n++)
                acc += CW[k * W + n] * intermediate[row * W + n];
            temp[k] = acc;
        }
        for (int k = 0; k < W; k++)
            coeff[row * W + k] = round_shift(temp[k], shift2);
    }
}

/* ============================================================================
 * Inverse DCT (for reconstruction)
 * ============================================================================ */

static void idct_1d(const int16_t *C, int N, const int16_t *in, int32_t *out) {
    for (int n = 0; n < N; n++) {
        int32_t acc = 0;
        for (int k = 0; k < N; k++)
            acc += C[k * N + n] * in[k];
        out[n] = acc;
    }
}

static void inverse_transform(int16_t *coeff, int16_t *residual, int W, int H, int bit_depth) {
    const int16_t *CW = (W == 4) ? &DCT4[0][0] : (W == 8) ? &DCT8[0][0] : (W == 16) ? &DCT16[0][0] : &DCT32[0][0];
    const int16_t *CH = (H == 4) ? &DCT4[0][0] : (H == 8) ? &DCT8[0][0] : (H == 16) ? &DCT16[0][0] : &DCT32[0][0];

    int16_t intermediate[32 * 32];
    int32_t temp[32];

    for (int row = 0; row < H; row++) {
        idct_1d(CW, W, &coeff[row * W], temp);
        for (int n = 0; n < W; n++)
            intermediate[row * W + n] = clamp_i(round_shift(temp[n], 7), -32768, 32767);
    }

    int shift2 = 20 - bit_depth;
    for (int col = 0; col < W; col++) {
        int16_t col_in[32];
        for (int k = 0; k < H; k++)
            col_in[k] = intermediate[k * W + col];
        idct_1d(CH, H, col_in, temp);
        for (int m = 0; m < H; m++)
            residual[m * W + col] = round_shift(temp[m], shift2);
    }
}

/* ============================================================================
 * Quantization
 * ============================================================================ */

/* Quantize with perceptual frequency weighting (§6.3.3) */
static int quantize_block(int16_t *coeff, int16_t *quantized, int bw, int bh, int qstep) {
    int has_nonzero = 0;
    for (int row = 0; row < bh; row++) {
        for (int col = 0; col < bw; col++) {
            int eff_qs = effective_qstep(qstep, row, col);
            quantized[row * bw + col] = round_div(coeff[row * bw + col], eff_qs);
            if (quantized[row * bw + col] != 0)
                has_nonzero = 1;
        }
    }
    return has_nonzero;
}

/* Dequantize with perceptual frequency weighting (§6.3.3) */
static void dequantize_block(int16_t *quantized, int16_t *dequant, int bw, int bh, int qstep) {
    for (int row = 0; row < bh; row++) {
        for (int col = 0; col < bw; col++) {
            int eff_qs = effective_qstep(qstep, row, col);
            dequant[row * bw + col] = quantized[row * bw + col] * eff_qs;
        }
    }
}

/*
 * Trellis quantization optimization.
 *
 * After initial quantization, this function examines each non-zero coefficient
 * and decides whether to keep it or zero it based on rate-distortion cost.
 *
 * For each coefficient:
 *   - Distortion cost of zeroing = (level * eff_qstep)^2 (transform domain energy)
 *   - Rate savings from zeroing ≈ significance bit + level bits + sign bit
 *
 * We zero the coefficient if: lambda * rate_savings > distortion_cost
 *
 * This is a greedy approximation to full trellis - we process in reverse scan
 * order so that zeroing trailing coefficients can also eliminate band overhead.
 */
static int g_trellis_zeroed = 0;  /* Debug counter */
static int g_intra_blocks = 0, g_inter_blocks = 0, g_skip_blocks = 0;

/*
 * Trellis quantization optimization.
 *
 * Zeros out small, isolated coefficients where the rate cost exceeds
 * the perceptual benefit. This is a simplified "dead zone" approach
 * rather than full trellis optimization.
 *
 * Strategy:
 * 1. Zero level-1 coefficients that are isolated (no neighbors in scan order)
 * 2. Zero trailing level-1 coefficients (last non-zero in their band)
 * 3. Be more aggressive in higher frequency bands
 *
 * The R-D math shows that at typical QP values, the distortion cost of
 * zeroing any coefficient exceeds the rate savings when using the standard
 * lambda formula. So we use heuristics tuned for perceptual quality.
 */
static void trellis_optimize(int16_t *coeff, int16_t *quantized, int bw, int bh,
                             int qstep, double lambda) {
    (void)coeff;
    (void)qstep;

    if (!g_trellis)
        return;

    /* Generate scan order */
    int scan[1024];
    generate_scan(bw, bh, scan);
    int N = bw * bh;

    /* Get band boundaries */
    int bounds[5], num_bands;
    get_band_boundaries(bw, bh, bounds, &num_bands);

    /* Count non-zero coefficients per band */
    int band_nnz[4] = {0, 0, 0, 0};
    for (int band = 0; band < num_bands; band++) {
        for (int i = bounds[band]; i < bounds[band + 1]; i++) {
            if (quantized[scan[i]] != 0)
                band_nnz[band]++;
        }
    }

    /*
     * Aggressiveness based on lambda (higher QP = higher lambda = more zeroing)
     * lambda ~14 at QP=24, ~218 at QP=36, ~1400 at QP=44
     */
    int max_level_to_zero = 1;
    if (lambda > 100) max_level_to_zero = 2;
    if (lambda > 500) max_level_to_zero = 3;

    /* Process coefficients in reverse scan order */
    for (int band = num_bands - 1; band >= 0; band--) {
        /* Skip DC (band 0) - always preserve */
        if (band == 0)
            continue;

        for (int i = bounds[band + 1] - 1; i >= bounds[band]; i--) {
            int idx = scan[i];
            int level = abs(quantized[idx]);
            if (level == 0 || level > max_level_to_zero)
                continue;

            /* Check if this coefficient is "isolated" (no non-zero neighbors in scan) */
            int has_prev_nz = 0, has_next_nz = 0;

            /* Look at previous 2 coefficients in scan order */
            for (int j = i - 1; j >= bounds[band] && j >= i - 2; j--) {
                if (quantized[scan[j]] != 0) {
                    has_prev_nz = 1;
                    break;
                }
            }

            /* Look at next 2 coefficients in scan order */
            for (int j = i + 1; j < bounds[band + 1] && j <= i + 2; j++) {
                if (quantized[scan[j]] != 0) {
                    has_next_nz = 1;
                    break;
                }
            }

            /* Zero if isolated, or if it's the last one in the band */
            int is_isolated = !has_prev_nz && !has_next_nz;
            int is_last_in_band = (band_nnz[band] == 1);

            /* More aggressive zeroing for higher bands, scaled by lambda
             * lambda ~14 at QP=24, ~27 at QP=28, ~55 at QP=32, ~110 at QP=36
             */
            int should_zero = 0;
            if (band >= 3) {
                /* Highest frequency band: zero isolated level-1 always */
                should_zero = (level == 1 && is_isolated);
                /* At moderate compression, also zero last-in-band */
                if (lambda > 20 && level == 1 && is_last_in_band)
                    should_zero = 1;
                /* At higher compression, zero all level-1 */
                if (lambda > 50 && level == 1)
                    should_zero = 1;
            } else if (band == 2) {
                /* High-mid band: zero isolated level-1 at low+ compression */
                if (lambda > 10 && level == 1 && is_isolated)
                    should_zero = 1;
                /* At moderate compression, also zero last-in-band */
                if (lambda > 30 && level == 1 && is_last_in_band)
                    should_zero = 1;
            } else if (band == 1 && lambda > 50) {
                /* Low-mid band: only zero isolated trailing at moderate+ compression */
                should_zero = (level == 1 && is_isolated && is_last_in_band);
            }

            /* At very high lambda (QP>36), be more aggressive */
            if (lambda > 200 && level == 1 && is_isolated) {
                should_zero = 1;
            }

            if (should_zero) {
                quantized[idx] = 0;
                band_nnz[band]--;
                g_trellis_zeroed++;
            }
        }
    }

    /* Second pass: eliminate bands that have only 1-2 trailing coefficients */
    for (int band = num_bands - 1; band >= 1; band--) {
        if (band_nnz[band] > 0 && band_nnz[band] <= 2 && lambda > 20) {
            /* Find and zero the remaining coefficients */
            int remaining = band_nnz[band];
            for (int i = bounds[band + 1] - 1; i >= bounds[band] && remaining > 0; i--) {
                int idx = scan[i];
                if (quantized[idx] != 0 && abs(quantized[idx]) <= max_level_to_zero) {
                    quantized[idx] = 0;
                    band_nnz[band]--;
                    g_trellis_zeroed++;
                    remaining--;
                }
            }
        }
    }
    (void)N;
}

/* ============================================================================
 * Intra Prediction (matches decoder exactly)
 * ============================================================================ */

static void intra_predict(uint16_t *recon, int stride, int bx, int by, int bw, int bh,
                          int tile_x, int tile_y, int bit_depth, uint16_t *pred) {
    int max_val = (1 << bit_depth) - 1;
    int has_top = (by > tile_y);
    int has_left = (bx > tile_x);

    int dc, delta_h = 0, delta_v = 0;

    if (has_top && has_left) {
        int sum = 0;
        for (int x = 0; x < bw; x++)
            sum += recon[(by - 1) * stride + bx + x];
        for (int y = 0; y < bh; y++)
            sum += recon[(by + y) * stride + bx - 1];
        dc = round_div(sum, bw + bh);
        delta_h = recon[(by - 1) * stride + bx + bw - 1] - recon[(by - 1) * stride + bx];
        delta_v = recon[(by + bh - 1) * stride + bx - 1] - recon[by * stride + bx - 1];
    } else if (has_top) {
        int sum = 0;
        for (int x = 0; x < bw; x++)
            sum += recon[(by - 1) * stride + bx + x];
        dc = round_div(sum, bw);
        delta_h = recon[(by - 1) * stride + bx + bw - 1] - recon[(by - 1) * stride + bx];
    } else if (has_left) {
        int sum = 0;
        for (int y = 0; y < bh; y++)
            sum += recon[(by + y) * stride + bx - 1];
        dc = round_div(sum, bh);
        delta_v = recon[(by + bh - 1) * stride + bx - 1] - recon[by * stride + bx - 1];
    } else {
        dc = 1 << (bit_depth - 1);
    }

    for (int y = 0; y < bh; y++) {
        for (int x = 0; x < bw; x++) {
            int h_off = (bw > 1) ? round_div(delta_h * (2 * x - (bw - 1)), 2 * (bw - 1)) : 0;
            int v_off = (bh > 1) ? round_div(delta_v * (2 * y - (bh - 1)), 2 * (bh - 1)) : 0;
            pred[y * bw + x] = clamp_i(dc + h_off + v_off, 0, max_val);
        }
    }
}

/* ============================================================================
 * CNN Loop Filter (must match decoder for reference frame consistency)
 * ============================================================================ */

static void cnn_filter_plane(uint16_t *plane, int width, int height, int bit_depth,
                             int16_t weights[4][4][4][3][3], int16_t biases[4][4]) {
    int max_val = (1 << bit_depth) - 1;

    int16_t *feat[2];
    feat[0] = malloc(4 * width * height * sizeof(int16_t));
    feat[1] = malloc(4 * width * height * sizeof(int16_t));

    /* Layer 1: 1 -> 4 channels */
    for (int c_out = 0; c_out < 4; c_out++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int32_t acc = biases[0][c_out];
                for (int ky = 0; ky < 3; ky++) {
                    for (int kx = 0; kx < 3; kx++) {
                        int sy = clamp_i(y + ky - 1, 0, height - 1);
                        int sx = clamp_i(x + kx - 1, 0, width - 1);
                        acc += weights[0][c_out][0][ky][kx] * plane[sy * width + sx];
                    }
                }
                acc = round_shift(acc, FILTER_SHIFT);
                feat[0][c_out * width * height + y * width + x] = clamp_i(acc, 0, 2047);
            }
        }
    }

    /* Layers 2-3: 4 -> 4 channels */
    for (int layer = 1; layer <= 2; layer++) {
        int16_t *in_feat = feat[(layer - 1) & 1];
        int16_t *out_feat = feat[layer & 1];

        for (int c_out = 0; c_out < 4; c_out++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int32_t acc = biases[layer][c_out];
                    for (int c_in = 0; c_in < 4; c_in++) {
                        for (int ky = 0; ky < 3; ky++) {
                            for (int kx = 0; kx < 3; kx++) {
                                int sy = clamp_i(y + ky - 1, 0, height - 1);
                                int sx = clamp_i(x + kx - 1, 0, width - 1);
                                acc += weights[layer][c_out][c_in][ky][kx] *
                                       in_feat[c_in * width * height + sy * width + sx];
                            }
                        }
                    }
                    acc = round_shift(acc, FILTER_SHIFT);
                    out_feat[c_out * width * height + y * width + x] = clamp_i(acc, 0, 2047);
                }
            }
        }
    }

    /* Layer 4: 4 -> 1 channel */
    int16_t *in_feat = feat[0];

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int32_t acc = biases[3][0];
            for (int c_in = 0; c_in < 4; c_in++) {
                for (int ky = 0; ky < 3; ky++) {
                    for (int kx = 0; kx < 3; kx++) {
                        int sy = clamp_i(y + ky - 1, 0, height - 1);
                        int sx = clamp_i(x + kx - 1, 0, width - 1);
                        acc += weights[3][0][c_in][ky][kx] *
                               in_feat[c_in * width * height + sy * width + sx];
                    }
                }
            }
            acc = round_shift(acc, FILTER_SHIFT);
            plane[y * width + x] = clamp_i(acc, 0, max_val);
        }
    }

    free(feat[0]);
    free(feat[1]);
}

/* ============================================================================
 * Inter Prediction & Motion Search
 * ============================================================================ */

/* Reference sample fetch with boundary padding */
static inline uint16_t ref_sample(const uint16_t *ref, int w, int h, int ry, int rx) {
    rx = clamp_i(rx, 0, w - 1);
    ry = clamp_i(ry, 0, h - 1);
    return ref[ry * w + rx];
}

/* Bilinear interpolation at quarter-pel precision (matches decoder exactly) */
static void inter_predict(const uint16_t *ref, int ref_w, int ref_h,
                          int bx, int by, int bw, int bh,
                          int mvx, int mvy, int bit_depth, uint16_t *pred) {
    int max_val = (1 << bit_depth) - 1;
    int ix = mvx >> 2;
    int iy = mvy >> 2;
    int fx = mvx & 3;
    int fy = mvy & 3;

    for (int y = 0; y < bh; y++) {
        for (int x = 0; x < bw; x++) {
            int rx = bx + x + ix;
            int ry = by + y + iy;

            int s00 = ref_sample(ref, ref_w, ref_h, ry, rx);
            int s10 = ref_sample(ref, ref_w, ref_h, ry, rx + 1);
            int s01 = ref_sample(ref, ref_w, ref_h, ry + 1, rx);
            int s11 = ref_sample(ref, ref_w, ref_h, ry + 1, rx + 1);

            int h0 = s00 * (4 - fx) + s10 * fx;
            int h1 = s01 * (4 - fx) + s11 * fx;
            int val = round_shift(h0 * (4 - fy) + h1 * fy, 4);
            pred[y * bw + x] = clamp_i(val, 0, max_val);
        }
    }
}

/* Compute SAD (Sum of Absolute Differences) for motion estimation */
static int64_t compute_sad(const uint16_t *orig, int orig_stride,
                           const uint16_t *ref, int ref_w, int ref_h,
                           int bx, int by, int bw, int bh,
                           int mvx, int mvy) {
    int64_t sad = 0;
    int ix = mvx >> 2;
    int iy = mvy >> 2;
    int fx = mvx & 3;
    int fy = mvy & 3;

    for (int y = 0; y < bh; y++) {
        for (int x = 0; x < bw; x++) {
            int rx = bx + x + ix;
            int ry = by + y + iy;

            int pred;
            if (fx == 0 && fy == 0) {
                /* Integer position - no interpolation needed */
                pred = ref_sample(ref, ref_w, ref_h, ry, rx);
            } else {
                /* Sub-pel interpolation */
                int s00 = ref_sample(ref, ref_w, ref_h, ry, rx);
                int s10 = ref_sample(ref, ref_w, ref_h, ry, rx + 1);
                int s01 = ref_sample(ref, ref_w, ref_h, ry + 1, rx);
                int s11 = ref_sample(ref, ref_w, ref_h, ry + 1, rx + 1);

                int h0 = s00 * (4 - fx) + s10 * fx;
                int h1 = s01 * (4 - fx) + s11 * fx;
                pred = round_shift(h0 * (4 - fy) + h1 * fy, 4);
            }

            sad += abs((int)orig[(by + y) * orig_stride + bx + x] - pred);
        }
    }
    return sad;
}

/*
 * Hierarchical Motion Search
 *
 * Budget: ~100 candidates max per block
 *
 * Strategy based on EPZS (Enhanced Predictive Zonal Search) and UMHexagonS:
 * 1. Predictor check (~8 candidates): Zero, spatial MVP, neighbor MVs
 * 2. Diamond search (~50 candidates): Multi-scale, starting from best predictor
 * 3. Sub-pel refinement (~40 candidates): Quarter-pel search around best int pos
 *
 * References:
 * - Tourapis, "Enhanced Predictive Zonal Search" (EPZS), 2002
 * - Chen et al, "UMHexagonS" (x264/JM reference software)
 */

/* Diamond search pattern offsets (quarter-pel units) */
static const int diamond_small[][2] = {
    {0, -4}, {-4, 0}, {4, 0}, {0, 4}  /* 1 pixel = 4 quarter-pels */
};
static const int diamond_large[][2] = {
    {0, -8}, {-8, 0}, {8, 0}, {0, 8},     /* 2 pixels */
    {-4, -4}, {4, -4}, {-4, 4}, {4, 4}    /* diagonals */
};

typedef struct {
    int16_t mv_x, mv_y;
    int64_t cost;
} MVCandidate;

static void motion_search(const uint16_t *orig, int orig_stride,
                          const uint16_t *ref, int ref_w, int ref_h,
                          int bx, int by, int bw, int bh,
                          int16_t mvp_x, int16_t mvp_y,
                          int16_t *best_mv_x, int16_t *best_mv_y,
                          EncodedBlock *left_blk, EncodedBlock *above_blk) {
    MVCandidate candidates[16];
    int num_candidates = 0;
    int candidates_tested = 0;
    const int MAX_CANDIDATES = 100;

    /* Search range in quarter-pel units (±32 pixels = ±128 qpel) */
    const int SEARCH_RANGE = 128;

    /* === Phase 1: Predictor candidates (~8) === */

    /* Zero MV */
    candidates[num_candidates++] = (MVCandidate){0, 0, 0};

    /* Spatial predictor (MVP) */
    if (mvp_x != 0 || mvp_y != 0) {
        candidates[num_candidates++] = (MVCandidate){mvp_x, mvp_y, 0};
    }

    /* Left neighbor's MV */
    if (left_blk && left_blk->mode == MODE_INTER) {
        candidates[num_candidates++] = (MVCandidate){left_blk->mv_x, left_blk->mv_y, 0};
    }

    /* Above neighbor's MV */
    if (above_blk && above_blk->mode == MODE_INTER) {
        candidates[num_candidates++] = (MVCandidate){above_blk->mv_x, above_blk->mv_y, 0};
    }

    /* Small offsets from zero */
    candidates[num_candidates++] = (MVCandidate){4, 0, 0};   /* 1 pixel right */
    candidates[num_candidates++] = (MVCandidate){-4, 0, 0};  /* 1 pixel left */
    candidates[num_candidates++] = (MVCandidate){0, 4, 0};   /* 1 pixel down */
    candidates[num_candidates++] = (MVCandidate){0, -4, 0};  /* 1 pixel up */

    /* Evaluate predictor candidates */
    int16_t best_x = 0, best_y = 0;
    int64_t best_sad = INT64_MAX;

    for (int i = 0; i < num_candidates && candidates_tested < MAX_CANDIDATES; i++) {
        int16_t mx = candidates[i].mv_x;
        int16_t my = candidates[i].mv_y;

        /* Clamp to search range */
        mx = clamp_i(mx, -SEARCH_RANGE, SEARCH_RANGE);
        my = clamp_i(my, -SEARCH_RANGE, SEARCH_RANGE);

        int64_t sad = compute_sad(orig, orig_stride, ref, ref_w, ref_h,
                                   bx, by, bw, bh, mx, my);
        candidates_tested++;

        if (sad < best_sad) {
            best_sad = sad;
            best_x = mx;
            best_y = my;
        }
    }

    /* === Phase 2: Diamond search at integer positions (~50) === */

    /* Large diamond first */
    int improved = 1;
    while (improved && candidates_tested < MAX_CANDIDATES - 40) {
        improved = 0;
        for (int i = 0; i < 8; i++) {
            int16_t mx = best_x + diamond_large[i][0];
            int16_t my = best_y + diamond_large[i][1];

            mx = clamp_i(mx, -SEARCH_RANGE, SEARCH_RANGE);
            my = clamp_i(my, -SEARCH_RANGE, SEARCH_RANGE);

            /* Round to integer position for this phase */
            mx = (mx >> 2) << 2;
            my = (my >> 2) << 2;

            int64_t sad = compute_sad(orig, orig_stride, ref, ref_w, ref_h,
                                       bx, by, bw, bh, mx, my);
            candidates_tested++;

            if (sad < best_sad) {
                best_sad = sad;
                best_x = mx;
                best_y = my;
                improved = 1;
            }
        }
    }

    /* Small diamond refinement */
    improved = 1;
    while (improved && candidates_tested < MAX_CANDIDATES - 20) {
        improved = 0;
        for (int i = 0; i < 4; i++) {
            int16_t mx = best_x + diamond_small[i][0];
            int16_t my = best_y + diamond_small[i][1];

            mx = clamp_i(mx, -SEARCH_RANGE, SEARCH_RANGE);
            my = clamp_i(my, -SEARCH_RANGE, SEARCH_RANGE);

            int64_t sad = compute_sad(orig, orig_stride, ref, ref_w, ref_h,
                                       bx, by, bw, bh, mx, my);
            candidates_tested++;

            if (sad < best_sad) {
                best_sad = sad;
                best_x = mx;
                best_y = my;
                improved = 1;
            }
        }
    }

    /* === Phase 3: Sub-pel refinement (~20-40) === */

    /* Half-pel search */
    int16_t int_best_x = best_x, int_best_y = best_y;
    for (int dy = -2; dy <= 2; dy += 2) {
        for (int dx = -2; dx <= 2; dx += 2) {
            if (dx == 0 && dy == 0) continue;
            if (candidates_tested >= MAX_CANDIDATES) break;

            int16_t mx = int_best_x + dx;
            int16_t my = int_best_y + dy;

            int64_t sad = compute_sad(orig, orig_stride, ref, ref_w, ref_h,
                                       bx, by, bw, bh, mx, my);
            candidates_tested++;

            if (sad < best_sad) {
                best_sad = sad;
                best_x = mx;
                best_y = my;
            }
        }
    }

    /* Quarter-pel search */
    int16_t half_best_x = best_x, half_best_y = best_y;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            if (candidates_tested >= MAX_CANDIDATES) break;

            int16_t mx = half_best_x + dx;
            int16_t my = half_best_y + dy;

            int64_t sad = compute_sad(orig, orig_stride, ref, ref_w, ref_h,
                                       bx, by, bw, bh, mx, my);
            candidates_tested++;

            if (sad < best_sad) {
                best_sad = sad;
                best_x = mx;
                best_y = my;
            }
        }
    }

    *best_mv_x = best_x;
    *best_mv_y = best_y;
}

/* Get MV predictor (matches decoder exactly) */
static void get_mv_predictor(TileEncoder *te, int bx, int by,
                             int16_t *mvp_x, int16_t *mvp_y,
                             EncodedBlock **left_blk, EncodedBlock **above_blk) {
    int16_t ax = 0, ay = 0, bx_mv = 0, by_mv = 0;
    int has_a = 0, has_b = 0;
    *left_blk = NULL;
    *above_blk = NULL;

    /* Left neighbor (A) */
    if (bx > te->tile_x) {
        int cx = (bx - te->tile_x) / CELL_SIZE - 1;
        int cy = (by - te->tile_y) / CELL_SIZE;
        int left_idx = te->cell_to_block[cy * te->cells_w + cx];
        if (left_idx >= 0 && te->blocks[left_idx].mode != MODE_INTRA) {
            ax = te->blocks[left_idx].mv_x;
            ay = te->blocks[left_idx].mv_y;
            has_a = 1;
            *left_blk = &te->blocks[left_idx];
        }
    }

    /* Above neighbor (B) */
    if (by > te->tile_y) {
        int cx = (bx - te->tile_x) / CELL_SIZE;
        int cy = (by - te->tile_y) / CELL_SIZE - 1;
        int above_idx = te->cell_to_block[cy * te->cells_w + cx];
        if (above_idx >= 0 && te->blocks[above_idx].mode != MODE_INTRA) {
            bx_mv = te->blocks[above_idx].mv_x;
            by_mv = te->blocks[above_idx].mv_y;
            has_b = 1;
            *above_blk = &te->blocks[above_idx];
        }
    }

    /* Compute predictor - truncating division toward zero per spec §5.4.3 */
    if (has_a && has_b) {
        int32_t sum_x = (int32_t)ax + (int32_t)bx_mv;
        int32_t sum_y = (int32_t)ay + (int32_t)by_mv;
        *mvp_x = (sum_x + ((sum_x >> 31) & 1)) >> 1;
        *mvp_y = (sum_y + ((sum_y >> 31) & 1)) >> 1;
    } else if (has_a) {
        *mvp_x = ax;
        *mvp_y = ay;
    } else if (has_b) {
        *mvp_x = bx_mv;
        *mvp_y = by_mv;
    } else {
        *mvp_x = 0;
        *mvp_y = 0;
    }
}

/* ============================================================================
 * Coefficient Encoding
 * ============================================================================ */

static void encode_coefficients(TileEncoder *te, int16_t *coeff, int W, int H, int is_chroma) {
    int N = W * H;

    int scan[1024];
    generate_scan(W, H, scan);

    /* Reorder coefficients to scan order */
    int16_t scan_coeff[1024];
    for (int i = 0; i < N; i++)
        scan_coeff[i] = coeff[scan[i]];

    int bounds[5], num_bands;
    get_band_boundaries(W, H, bounds, &num_bands);

    int ctx_band = is_chroma ? CTX_BAND_CHROMA : CTX_BAND_LUMA;
    int ctx_sig = is_chroma ? CTX_SIG_CHROMA : CTX_SIG_LUMA;
    int ctx_level = is_chroma ? CTX_LEVEL_CHROMA : CTX_LEVEL_LUMA;

    int prev_band_zero = 0;  /* Track if previous band was ALL_ZERO */
    for (int band = 0; band < num_bands; band++) {
        int band_start = bounds[band];
        int band_end = bounds[band + 1];

        /* Check if band has any non-zero coefficients */
        int has_nonzero = 0;
        for (int i = band_start; i < band_end; i++) {
            if (scan_coeff[i] != 0) {
                has_nonzero = 1;
                break;
            }
        }

        /* Band status context per §9.4: band_index * 2 + prev_zero */
        int prev_zero = (band > 0 && prev_band_zero) ? 1 : 0;
        int status_ctx = ctx_band + min_i(band, 3) * 2 + prev_zero;
        dual_rans_enc_put(&te->rans, has_nonzero, &te->contexts[status_ctx]);

        prev_band_zero = !has_nonzero;

        if (!has_nonzero)
            continue;

        /* Encode significance flags with sliding window per §9.4 */
        int last4[4] = {0, 0, 0, 0};
        int last4_idx = 0;
        for (int i = band_start; i < band_end; i++) {
            int sig = (scan_coeff[i] != 0) ? 1 : 0;
            /* Count non-zero flags in last 4, saturated to 3 */
            int density = last4[0] + last4[1] + last4[2] + last4[3];
            if (density > 3) density = 3;
            int sig_ctx = ctx_sig + (min_i(band, 3) * 4 + density);
            dual_rans_enc_put(&te->rans, sig, &te->contexts[sig_ctx]);
            /* Update sliding window */
            last4[last4_idx] = sig;
            last4_idx = (last4_idx + 1) & 3;
        }

        /* Encode levels */
        int prev_level = 0;
        for (int i = band_start; i < band_end; i++) {
            if (scan_coeff[i] == 0) continue;

            int level = abs(scan_coeff[i]);
            int prev_cat = (prev_level <= 1) ? 0 : (prev_level <= 4) ? 1 : (prev_level <= 7) ? 2 : 3;
            int lev_ctx = ctx_level + ((band < 4 ? band : 3) * 4 + prev_cat);

            int token;
            if (level <= 7) {
                token = level - 1;
            } else {
                token = 7;
            }
            dual_rans_enc_put(&te->rans, token, &te->contexts[lev_ctx]);

            if (level > 7) {
                bypass_write_expgolomb(&te->bypass, level - 8);
            }

            prev_level = level;
        }
    }

    /* Encode signs (bypass) */
    for (int i = 0; i < N; i++) {
        if (scan_coeff[i] != 0) {
            bypass_write_bit(&te->bypass, scan_coeff[i] < 0 ? 1 : 0);
        }
    }
}

/* ============================================================================
 * Block Map Encoding
 * ============================================================================ */

static int size_category(int area) {
    if (area <= 64) return 0;
    if (area <= 256) return 1;
    return 2;
}

static int get_valid_shapes(int cx, int cy, int cells_w, int cells_h, int *cell_to_block, int *valid) {
    int count = 0;
    for (int s = 0; s < 7; s++) {
        int w_cells = BLOCK_SHAPES[s][0];
        int h_cells = BLOCK_SHAPES[s][1];
        if (cx + w_cells <= cells_w && cy + h_cells <= cells_h) {
            int overlap = 0;
            for (int dy = 0; dy < h_cells && !overlap; dy++) {
                for (int dx = 0; dx < w_cells && !overlap; dx++) {
                    if (cell_to_block[(cy + dy) * cells_w + (cx + dx)] >= 0)
                        overlap = 1;
                }
            }
            if (!overlap)
                valid[count++] = s;
        }
    }
    return count;
}

__attribute__((unused))
static int find_shape_index(int w_cells, int h_cells) {
    for (int s = 0; s < 7; s++) {
        if (BLOCK_SHAPES[s][0] == w_cells && BLOCK_SHAPES[s][1] == h_cells)
            return s;
    }
    return 3; /* Default to 2x2 = 16x16 */
}

/* Encode shape symbol for a block
 * Per spec Draft 0.3: always use full 7-symbol CDF, no alphabet reduction */
static void encode_shape(TileEncoder *te, int cx, int cy, int shape_idx) {
    int above_cat = 0, left_cat = 0;
    if (cy > 0) {
        int above_idx = te->cell_to_block[(cy - 1) * te->cells_w + cx];
        if (above_idx >= 0) {
            EncodedBlock *ab = &te->blocks[above_idx];
            above_cat = size_category(ab->w * ab->h);
        }
    }
    if (cx > 0) {
        int left_idx = te->cell_to_block[cy * te->cells_w + (cx - 1)];
        if (left_idx >= 0) {
            EncodedBlock *lb = &te->blocks[left_idx];
            left_cat = size_category(lb->w * lb->h);
        }
    }
    int ctx = above_cat * 3 + left_cat;

    /* Always encode shape_idx directly using full 7-symbol CDF */
    dual_rans_enc_put(&te->rans, shape_idx, &te->contexts[CTX_SHAPE + ctx]);
}

/* ============================================================================
 * MV Component Encoding (matches decoder exactly)
 * ============================================================================ */

static void encode_mv_component(TileEncoder *te, int16_t mvd, int comp) {
    int mag = abs(mvd);
    int cls;

    if (mag == 0) {
        cls = 0;
    } else if (mag == 1) {
        cls = 1;
    } else if (mag <= 3) {
        cls = 2;
    } else if (mag <= 7) {
        cls = 3;
    } else if (mag <= 15) {
        cls = 4;
    } else if (mag <= 31) {
        cls = 5;
    } else {
        cls = 6;
    }

    dual_rans_enc_put(&te->rans, cls, &te->contexts[CTX_MVD_CLASS + comp]);

    if (cls == 2) {
        bypass_write_bits(&te->bypass, mag - 2, 1);
    } else if (cls == 3) {
        bypass_write_bits(&te->bypass, mag - 4, 2);
    } else if (cls == 4) {
        bypass_write_bits(&te->bypass, mag - 8, 3);
    } else if (cls == 5) {
        bypass_write_bits(&te->bypass, mag - 16, 4);
    } else if (cls == 6) {
        bypass_write_expgolomb(&te->bypass, mag - 32);
    }

    if (mag > 0) {
        bypass_write_bit(&te->bypass, mvd < 0 ? 1 : 0);
    }
}

/* ============================================================================
 * Block Data Encoding (mode, MV, QP delta, CBF, coefficients)
 * ============================================================================ */

static void encode_block_data(TileEncoder *te, EncodedBlock *blk, int base_qp,
                              int is_inter_frame, int dpb_count) {
    /* For inter frames, encode prediction mode */
    if (is_inter_frame) {
        /* Get context from mode categories per §5.2 */
        int above_cat = 1, left_cat = 1;  /* Default to INTER (1) for missing neighbors */
        int bx = blk->x, by = blk->y;

        if (by > te->tile_y) {
            int cx = (bx - te->tile_x) / CELL_SIZE;
            int cy = (by - te->tile_y) / CELL_SIZE - 1;
            int above_idx = te->cell_to_block[cy * te->cells_w + cx];
            if (above_idx >= 0)
                above_cat = mode_category(te->blocks[above_idx].mode);
        }
        if (bx > te->tile_x) {
            int cx = (bx - te->tile_x) / CELL_SIZE - 1;
            int cy = (by - te->tile_y) / CELL_SIZE;
            int left_idx = te->cell_to_block[cy * te->cells_w + cx];
            if (left_idx >= 0)
                left_cat = mode_category(te->blocks[left_idx].mode);
        }

        int ctx = above_cat * 3 + left_cat;

        /* Encode mode: 0=INTRA, 1=INTER, 2=SKIP (direct 3-way symbol) */
        dual_rans_enc_put(&te->rans, blk->mode, &te->contexts[CTX_MODE + ctx]);

        /* For INTER mode, encode reference index and MV delta */
        if (blk->mode == MODE_INTER) {
            /* Reference index (if DPB has multiple frames) */
            if (dpb_count > 1) {
                dual_rans_enc_put(&te->rans, blk->ref_idx, &te->contexts[CTX_REF_IDX]);
            }

            /* MV delta from predictor */
            int16_t mvp_x, mvp_y;
            EncodedBlock *left_blk, *above_blk;
            get_mv_predictor(te, blk->x, blk->y, &mvp_x, &mvp_y, &left_blk, &above_blk);

            int16_t mvd_x = blk->mv_x - mvp_x;
            int16_t mvd_y = blk->mv_y - mvp_y;

            encode_mv_component(te, mvd_x, 0);
            encode_mv_component(te, mvd_y, 1);
        }

        /* SKIP mode: no QP delta, CBF, or coefficients */
        if (blk->mode == MODE_SKIP) {
            return;
        }
    }

    /* QP delta */
    {
        int above_nz = 0, left_nz = 0;
        int bx = blk->x, by = blk->y;

        if (by > te->tile_y) {
            int cx = (bx - te->tile_x) / CELL_SIZE;
            int cy = (by - te->tile_y) / CELL_SIZE - 1;
            int above_idx = te->cell_to_block[cy * te->cells_w + cx];
            if (above_idx >= 0 && te->blocks[above_idx].qp_delta != 0)
                above_nz = 1;
        }
        if (bx > te->tile_x) {
            int cx = (bx - te->tile_x) / CELL_SIZE - 1;
            int cy = (by - te->tile_y) / CELL_SIZE;
            int left_idx = te->cell_to_block[cy * te->cells_w + cx];
            if (left_idx >= 0 && te->blocks[left_idx].qp_delta != 0)
                left_nz = 1;
        }

        int ctx = above_nz + left_nz;
        int sym = blk->qp_delta + 2;  /* Map [-2,2] to [0,4] */
        dual_rans_enc_put(&te->rans, sym, &te->contexts[CTX_QP_DELTA + ctx]);
    }

    /* CBF */
    {
        int above_cbf = 0, left_cbf = 0;
        int bx = blk->x, by = blk->y;

        if (by > te->tile_y) {
            int cx = (bx - te->tile_x) / CELL_SIZE;
            int cy = (by - te->tile_y) / CELL_SIZE - 1;
            int above_idx = te->cell_to_block[cy * te->cells_w + cx];
            if (above_idx >= 0)
                above_cbf = te->blocks[above_idx].cbf;
        }
        if (bx > te->tile_x) {
            int cx = (bx - te->tile_x) / CELL_SIZE - 1;
            int cy = (by - te->tile_y) / CELL_SIZE;
            int left_idx = te->cell_to_block[cy * te->cells_w + cx];
            if (left_idx >= 0)
                left_cbf = te->blocks[left_idx].cbf;
        }

        int ctx = above_cbf + left_cbf;
        dual_rans_enc_put(&te->rans, blk->cbf, &te->contexts[CTX_CBF + ctx]);
    }

    /* Coefficients */
    if (blk->cbf) {
        (void)base_qp;
        encode_coefficients(te, blk->coeff_y, blk->w, blk->h, 0);
        encode_coefficients(te, blk->coeff_cb, blk->w / 2, blk->h / 2, 1);
        encode_coefficients(te, blk->coeff_cr, blk->w / 2, blk->h / 2, 1);
    }
}

/* ============================================================================
 * Pixel Domain Encoding (transform, quantize, reconstruct)
 * ============================================================================ */

static int encode_pixel_block(uint16_t *orig, uint16_t *recon,
                              int stride, int frame_w, int frame_h,
                              int bx, int by, int bw, int bh,
                              int tile_x, int tile_y, int bit_depth, int qp,
                              double lambda, int16_t *coeff_out) {
    int max_val = (1 << bit_depth) - 1;
    int qstep = QSTEP[qp];

    uint16_t pred[32 * 32];
    intra_predict(recon, stride, bx, by, bw, bh, tile_x, tile_y, bit_depth, pred);

    /* Compute residual - use edge extension for pixels beyond frame */
    int16_t residual[32 * 32];
    for (int y = 0; y < bh; y++) {
        for (int x = 0; x < bw; x++) {
            int sy = min_i(by + y, frame_h - 1);
            int sx = min_i(bx + x, frame_w - 1);
            residual[y * bw + x] = (int16_t)orig[sy * stride + sx] -
                                   (int16_t)pred[y * bw + x];
        }
    }

    int16_t coeff[32 * 32];
    forward_transform(residual, coeff, bw, bh, bit_depth);

    quantize_block(coeff, coeff_out, bw, bh, qstep);

    /* Trellis optimization: zero out coefficients where rate cost exceeds distortion */
    trellis_optimize(coeff, coeff_out, bw, bh, qstep, lambda);

    /* Recompute CBF after trellis optimization */
    int cbf = 0;
    for (int i = 0; i < bw * bh; i++) {
        if (coeff_out[i] != 0) {
            cbf = 1;
            break;
        }
    }

    int16_t dequant[32 * 32];
    dequantize_block(coeff_out, dequant, bw, bh, qstep);

    int16_t recon_residual[32 * 32];
    inverse_transform(dequant, recon_residual, bw, bh, bit_depth);

    /* Reconstruct - clip to frame boundaries */
    int y_end = min_i(bh, frame_h - by);
    int x_end = min_i(bw, frame_w - bx);
    for (int y = 0; y < y_end; y++) {
        for (int x = 0; x < x_end; x++) {
            int val = pred[y * bw + x] + recon_residual[y * bw + x];
            recon[(by + y) * stride + bx + x] = clamp_i(val, 0, max_val);
        }
    }

    return cbf;
}

/* Encode block using inter prediction (motion compensation) */
static int encode_inter_block(uint16_t *orig, uint16_t *recon,
                              const uint16_t *ref, int ref_w, int ref_h,
                              int stride, int frame_w, int frame_h,
                              int bx, int by, int bw, int bh,
                              int mvx, int mvy, int bit_depth, int qp,
                              double lambda, int16_t *coeff_out) {
    int max_val = (1 << bit_depth) - 1;
    int qstep = QSTEP[qp];

    /* Generate inter prediction */
    uint16_t pred[32 * 32];
    inter_predict(ref, ref_w, ref_h, bx, by, bw, bh, mvx, mvy, bit_depth, pred);

    /* Compute residual */
    int16_t residual[32 * 32];
    for (int y = 0; y < bh; y++) {
        for (int x = 0; x < bw; x++) {
            int sy = min_i(by + y, frame_h - 1);
            int sx = min_i(bx + x, frame_w - 1);
            residual[y * bw + x] = (int16_t)orig[sy * stride + sx] -
                                   (int16_t)pred[y * bw + x];
        }
    }

    int16_t coeff[32 * 32];
    forward_transform(residual, coeff, bw, bh, bit_depth);

    quantize_block(coeff, coeff_out, bw, bh, qstep);

    /* Trellis optimization */
    trellis_optimize(coeff, coeff_out, bw, bh, qstep, lambda);

    /* Recompute CBF after trellis */
    int cbf = 0;
    for (int i = 0; i < bw * bh; i++) {
        if (coeff_out[i] != 0) {
            cbf = 1;
            break;
        }
    }

    int16_t dequant[32 * 32];
    dequantize_block(coeff_out, dequant, bw, bh, qstep);

    int16_t recon_residual[32 * 32];
    inverse_transform(dequant, recon_residual, bw, bh, bit_depth);

    /* Reconstruct - clip to frame boundaries */
    int y_end = min_i(bh, frame_h - by);
    int x_end = min_i(bw, frame_w - bx);
    for (int y = 0; y < y_end; y++) {
        for (int x = 0; x < x_end; x++) {
            int val = pred[y * bw + x] + recon_residual[y * bw + x];
            recon[(by + y) * stride + bx + x] = clamp_i(val, 0, max_val);
        }
    }

    return cbf;
}

/* Encode block using SKIP mode (copy from reference, no residual) */
static void encode_skip_block(uint16_t *recon, const uint16_t *ref, int ref_w, int ref_h,
                              int stride, int frame_w, int frame_h,
                              int bx, int by, int bw, int bh,
                              int mvx, int mvy, int bit_depth) {
    int max_val = (1 << bit_depth) - 1;

    /* Generate prediction and use directly as reconstruction - clip to frame boundaries */
    int y_end = min_i(bh, frame_h - by);
    int x_end = min_i(bw, frame_w - bx);
    for (int y = 0; y < y_end; y++) {
        for (int x = 0; x < x_end; x++) {
            int ix = mvx >> 2;
            int iy = mvy >> 2;
            int fx = mvx & 3;
            int fy = mvy & 3;

            int rx = bx + x + ix;
            int ry = by + y + iy;

            int s00 = ref_sample(ref, ref_w, ref_h, ry, rx);
            int s10 = ref_sample(ref, ref_w, ref_h, ry, rx + 1);
            int s01 = ref_sample(ref, ref_w, ref_h, ry + 1, rx);
            int s11 = ref_sample(ref, ref_w, ref_h, ry + 1, rx + 1);

            int h0 = s00 * (4 - fx) + s10 * fx;
            int h1 = s01 * (4 - fx) + s11 * fx;
            int val = round_shift(h0 * (4 - fy) + h1 * fy, 4);

            recon[(by + y) * stride + bx + x] = clamp_i(val, 0, max_val);
        }
    }
}

/* Compute SSE distortion for a block */
static int64_t compute_block_distortion(uint16_t *orig, uint16_t *recon,
                                         int stride, int frame_w, int frame_h,
                                         int bx, int by, int bw, int bh) {
    int64_t sse = 0;
    for (int y = 0; y < bh; y++) {
        for (int x = 0; x < bw; x++) {
            int sy = min_i(by + y, frame_h - 1);
            int sx = min_i(bx + x, frame_w - 1);
            int orig_val = orig[sy * stride + sx];
            int recon_val = recon[(by + y) * stride + bx + x];
            int diff = orig_val - recon_val;
            sse += diff * diff;
        }
    }
    return sse;
}

/*
 * Compute boundary discontinuity penalty for perceptual RDO.
 *
 * Measures the squared gradient at block boundaries - how much the
 * reconstructed block's edge pixels differ from their neighbors in
 * already-reconstructed areas (above and left).
 *
 * High discontinuity = visible blocking artifact = high penalty.
 */
static int64_t compute_boundary_discontinuity(uint16_t *recon, int stride,
                                               int bx, int by, int bw, int bh,
                                               int tile_x, int tile_y) {
    int64_t penalty = 0;

    /* Top boundary: compare first row of block with row above */
    if (by > tile_y) {
        for (int x = 0; x < bw; x++) {
            int block_val = recon[by * stride + bx + x];
            int above_val = recon[(by - 1) * stride + bx + x];
            int diff = block_val - above_val;
            penalty += diff * diff;
        }
    }

    /* Left boundary: compare first column of block with column to left */
    if (bx > tile_x) {
        for (int y = 0; y < bh; y++) {
            int block_val = recon[(by + y) * stride + bx];
            int left_val = recon[(by + y) * stride + bx - 1];
            int diff = block_val - left_val;
            penalty += diff * diff;
        }
    }

    return penalty;
}

/*
 * Compute perceptual distortion = SSE + weighted boundary penalty.
 *
 * The boundary penalty encourages the encoder to choose block sizes
 * that create smoother transitions, reducing visible blocking artifacts.
 */
static int64_t compute_perceptual_distortion(uint16_t *orig, uint16_t *recon,
                                              int stride, int frame_w, int frame_h,
                                              int bx, int by, int bw, int bh,
                                              int tile_x, int tile_y) {
    int64_t sse = compute_block_distortion(orig, recon, stride, frame_w, frame_h,
                                            bx, by, bw, bh);

    int64_t boundary = compute_boundary_discontinuity(recon, stride,
                                                       bx, by, bw, bh,
                                                       tile_x, tile_y);

    /* Scale boundary penalty by weight and normalize by boundary length */
    /* More boundary pixels = more chance for artifacts = more penalty */
    return sse + (int64_t)(PSY_BOUNDARY_WEIGHT * (double)boundary);
}

/* ============================================================================
 * RDO Lambda Calculation
 * ============================================================================ */

static double compute_lambda(int qp) {
    /* Standard lambda formula: λ = 0.85 * 2^((QP-12)/3) */
    return 0.85 * pow(2.0, (qp - 12) / 3.0);
}

/* ============================================================================
 * Rate Estimation (count symbols and bypass bits)
 * ============================================================================ */

static size_t estimate_rate(TileEncoder *te, EncoderCheckpoint *cp_before) {
    /* Rate = (symbols added to rANS) + (bypass bits added) */
    size_t rans_symbols = (te->rans.stream0_syms.count - cp_before->stream0_count) +
                          (te->rans.stream1_syms.count - cp_before->stream1_count);

    /* Bypass bits: compare positions */
    size_t bypass_bits_before = cp_before->bypass_byte_pos * 8 + (7 - cp_before->bypass_bit_idx);
    size_t bypass_bits_after = te->bypass.byte_pos * 8 + (7 - te->bypass.bit_idx);
    size_t bypass_bits = bypass_bits_after - bypass_bits_before;

    /* Rough estimate: each rANS symbol is ~10-12 bits on average for this codec */
    /* More accurate would be to use actual CDF probabilities, but this is a reasonable approximation */
    return rans_symbols * 11 + bypass_bits;
}

/* ============================================================================
 * Tile Context Initialization
 * ============================================================================ */

static void init_tile_contexts(TileEncoder *te) {
    int ctx_sizes[NUM_CONTEXTS] = {0};
    for (int i = CTX_SHAPE; i < CTX_SHAPE + 9; i++) ctx_sizes[i] = 7;
    for (int i = CTX_MODE; i < CTX_MODE + 9; i++) ctx_sizes[i] = 3;
    for (int i = CTX_CBF; i < CTX_CBF + 3; i++) ctx_sizes[i] = 2;
    for (int i = CTX_QP_DELTA; i < CTX_QP_DELTA + 3; i++) ctx_sizes[i] = 5;
    ctx_sizes[CTX_REF_IDX] = 1;
    for (int i = CTX_MVD_CLASS; i < CTX_MVD_CLASS + 2; i++) ctx_sizes[i] = 7;
    for (int i = CTX_BAND_LUMA; i < CTX_BAND_LUMA + 8; i++) ctx_sizes[i] = 2;
    for (int i = CTX_SIG_LUMA; i < CTX_SIG_LUMA + 16; i++) ctx_sizes[i] = 2;
    for (int i = CTX_LEVEL_LUMA; i < CTX_LEVEL_LUMA + 16; i++) ctx_sizes[i] = 8;
    for (int i = CTX_BAND_CHROMA; i < CTX_BAND_CHROMA + 8; i++) ctx_sizes[i] = 2;
    for (int i = CTX_SIG_CHROMA; i < CTX_SIG_CHROMA + 16; i++) ctx_sizes[i] = 2;
    for (int i = CTX_LEVEL_CHROMA; i < CTX_LEVEL_CHROMA + 16; i++) ctx_sizes[i] = 8;
    for (int i = CTX_FILTER_DELTA; i < CTX_FILTER_DELTA + 3; i++) ctx_sizes[i] = 9;

    for (int i = 0; i < NUM_CONTEXTS; i++) {
        if (ctx_sizes[i] > 0)
            cdf_init_uniform(&te->contexts[i], ctx_sizes[i]);
    }
}

/* ============================================================================
 * Tile Encoding with RDO
 * ============================================================================ */

static uint8_t *encode_tile_rdo(EncoderContext *enc, TileEncoder *te, int base_qp,
                                 size_t *out_size, uint16_t *bypass_offset_out) {
    dual_rans_enc_init(&te->rans);
    bypass_writer_init(&te->bypass, 4096);
    init_tile_contexts(te);

    /* Initialize cell_to_block mapping */
    te->cell_to_block = malloc(te->cells_w * te->cells_h * sizeof(int));
    for (int i = 0; i < te->cells_w * te->cells_h; i++)
        te->cell_to_block[i] = -1;

    /* Allocate blocks array */
    int max_blocks = te->cells_w * te->cells_h;
    te->blocks = malloc(max_blocks * sizeof(EncodedBlock));
    te->num_blocks = 0;

    double lambda = compute_lambda(base_qp);

    /* Process cells in raster scan order with RDO */
    for (int cy = 0; cy < te->cells_h; cy++) {
        for (int cx = 0; cx < te->cells_w; cx++) {
            /* Skip if already covered by a previous block */
            if (te->cell_to_block[cy * te->cells_w + cx] >= 0)
                continue;

            /* Get valid shapes for this position */
            int valid_shapes[7];
            int num_valid = get_valid_shapes(cx, cy, te->cells_w, te->cells_h,
                                              te->cell_to_block, valid_shapes);

            if (num_valid == 0) continue;

            /* RDO: try each valid shape and pick the best */
            int best_shape = valid_shapes[0];
            double best_rd_cost = 1e30;
            EncodedBlock best_block;
            memset(&best_block, 0, sizeof(best_block));

            /* Save checkpoint before trying shapes */
            EncoderCheckpoint cp;
            checkpoint_save(te, &cp);

            /* Compute max block extent for reconstruction checkpoint */
            int max_w_cells = 0, max_h_cells = 0;
            for (int v = 0; v < num_valid; v++) {
                int s = valid_shapes[v];
                if (BLOCK_SHAPES[s][0] > max_w_cells) max_w_cells = BLOCK_SHAPES[s][0];
                if (BLOCK_SHAPES[s][1] > max_h_cells) max_h_cells = BLOCK_SHAPES[s][1];
            }
            int recon_x = te->tile_x + cx * CELL_SIZE;
            int recon_y = te->tile_y + cy * CELL_SIZE;
            int recon_w = max_w_cells * CELL_SIZE;
            int recon_h = max_h_cells * CELL_SIZE;

            ReconCheckpoint rcp;
            recon_checkpoint_save(&rcp, enc->recon_y, enc->recon_cb, enc->recon_cr,
                                  enc->width, enc->chroma_w,
                                  recon_x, recon_y, recon_w, recon_h);

            for (int v = 0; v < num_valid; v++) {
                int shape_idx = valid_shapes[v];
                int w_cells = BLOCK_SHAPES[shape_idx][0];
                int h_cells = BLOCK_SHAPES[shape_idx][1];
                int bw = w_cells * CELL_SIZE;
                int bh = h_cells * CELL_SIZE;
                int bx = te->tile_x + cx * CELL_SIZE;
                int by = te->tile_y + cy * CELL_SIZE;

                /* Restore checkpoint for this trial */
                if (v > 0) {
                    checkpoint_restore(te, &cp);
                    recon_checkpoint_restore(&rcp, enc->recon_y, enc->recon_cb, enc->recon_cr);
                }

                /* Create trial block */
                EncodedBlock trial_blk;
                trial_blk.x = bx;
                trial_blk.y = by;
                trial_blk.w = bw;
                trial_blk.h = bh;
                trial_blk.qp_delta = 0;
                trial_blk.mode = MODE_INTRA;
                trial_blk.mv_x = 0;
                trial_blk.mv_y = 0;
                trial_blk.ref_idx = 0;

                /* For P-frames, decide between INTRA, INTER, and SKIP modes */
                int is_inter_frame = (enc->frame_type == 1 && enc->dpb.count > 0);

                if (is_inter_frame) {
                    /* Get MV predictor and neighbor info */
                    int16_t mvp_x, mvp_y;
                    EncodedBlock *left_blk, *above_blk;
                    get_mv_predictor(te, bx, by, &mvp_x, &mvp_y, &left_blk, &above_blk);

                    /* Reference frame (use most recent) */
                    RefFrame *ref = &enc->dpb.frames[0];

                    /* Motion search */
                    int16_t best_mv_x, best_mv_y;
                    motion_search(enc->orig_y, enc->width,
                                  ref->y, ref->width, ref->height,
                                  bx, by, bw, bh,
                                  mvp_x, mvp_y,
                                  &best_mv_x, &best_mv_y,
                                  left_blk, above_blk);

                    /* Try SKIP mode (if MV == predictor, minimal cost) */
                    int64_t skip_dist = compute_sad(enc->orig_y, enc->width,
                                                    ref->y, ref->width, ref->height,
                                                    bx, by, bw, bh, mvp_x, mvp_y);
                    /* SKIP has very low rate: just mode symbol */
                    double skip_cost = (double)skip_dist * skip_dist / (bw * bh) + lambda * 2.0;

                    /* Try INTER mode with searched MV */
                    int64_t inter_dist = compute_sad(enc->orig_y, enc->width,
                                                     ref->y, ref->width, ref->height,
                                                     bx, by, bw, bh, best_mv_x, best_mv_y);
                    /* INTER rate: mode + MV delta (estimate ~10-20 bits) */
                    int mv_bits = 4;
                    if (best_mv_x != mvp_x) mv_bits += 5 + (abs(best_mv_x - mvp_x) > 4 ? 3 : 0);
                    if (best_mv_y != mvp_y) mv_bits += 5 + (abs(best_mv_y - mvp_y) > 4 ? 3 : 0);
                    double inter_cost = (double)inter_dist * inter_dist / (bw * bh) + lambda * mv_bits;

                    /* Try INTRA mode (estimate cost) */
                    uint16_t intra_pred[32 * 32];
                    intra_predict(enc->recon_y, enc->width, bx, by, bw, bh,
                                  te->tile_x, te->tile_y, enc->bit_depth, intra_pred);
                    int64_t intra_dist = 0;
                    for (int y = 0; y < bh; y++) {
                        for (int x = 0; x < bw; x++) {
                            int diff = (int)enc->orig_y[(by + y) * enc->width + bx + x] -
                                       (int)intra_pred[y * bw + x];
                            intra_dist += abs(diff);
                        }
                    }
                    double intra_cost = (double)intra_dist * intra_dist / (bw * bh) + lambda * 3.0;

                    /* Choose best mode */
                    if (skip_cost <= inter_cost && skip_cost <= intra_cost) {
                        trial_blk.mode = MODE_SKIP;
                        trial_blk.mv_x = mvp_x;
                        trial_blk.mv_y = mvp_y;
                    } else if (inter_cost <= intra_cost) {
                        trial_blk.mode = MODE_INTER;
                        trial_blk.mv_x = best_mv_x;
                        trial_blk.mv_y = best_mv_y;
                    } else {
                        trial_blk.mode = MODE_INTRA;
                    }
                }

                /* Encode based on selected mode */
                int cbf_y, cbf_cb, cbf_cr;

                if (trial_blk.mode == MODE_SKIP) {
                    RefFrame *ref = &enc->dpb.frames[trial_blk.ref_idx];
                    encode_skip_block(enc->recon_y, ref->y, ref->width, ref->height,
                                      enc->width, enc->width, enc->height, bx, by, bw, bh,
                                      trial_blk.mv_x, trial_blk.mv_y, enc->bit_depth);
                    encode_skip_block(enc->recon_cb, ref->cb, (ref->width + 1) / 2, (ref->height + 1) / 2,
                                      enc->chroma_w, enc->chroma_w, enc->chroma_h, bx / 2, by / 2, bw / 2, bh / 2,
                                      trial_blk.mv_x / 2, trial_blk.mv_y / 2, enc->bit_depth);
                    encode_skip_block(enc->recon_cr, ref->cr, (ref->width + 1) / 2, (ref->height + 1) / 2,
                                      enc->chroma_w, enc->chroma_w, enc->chroma_h, bx / 2, by / 2, bw / 2, bh / 2,
                                      trial_blk.mv_x / 2, trial_blk.mv_y / 2, enc->bit_depth);
                    cbf_y = cbf_cb = cbf_cr = 0;
                    memset(trial_blk.coeff_y, 0, sizeof(trial_blk.coeff_y));
                    memset(trial_blk.coeff_cb, 0, sizeof(trial_blk.coeff_cb));
                    memset(trial_blk.coeff_cr, 0, sizeof(trial_blk.coeff_cr));
                } else if (trial_blk.mode == MODE_INTER) {
                    RefFrame *ref = &enc->dpb.frames[trial_blk.ref_idx];
                    cbf_y = encode_inter_block(enc->orig_y, enc->recon_y,
                                               ref->y, ref->width, ref->height,
                                               enc->width, enc->width, enc->height,
                                               bx, by, bw, bh,
                                               trial_blk.mv_x, trial_blk.mv_y,
                                               enc->bit_depth, base_qp, lambda, trial_blk.coeff_y);
                    cbf_cb = encode_inter_block(enc->orig_cb, enc->recon_cb,
                                                ref->cb, (ref->width + 1) / 2, (ref->height + 1) / 2,
                                                enc->chroma_w, enc->chroma_w, enc->chroma_h,
                                                bx / 2, by / 2, bw / 2, bh / 2,
                                                trial_blk.mv_x / 2, trial_blk.mv_y / 2,
                                                enc->bit_depth, base_qp, lambda, trial_blk.coeff_cb);
                    cbf_cr = encode_inter_block(enc->orig_cr, enc->recon_cr,
                                                ref->cr, (ref->width + 1) / 2, (ref->height + 1) / 2,
                                                enc->chroma_w, enc->chroma_w, enc->chroma_h,
                                                bx / 2, by / 2, bw / 2, bh / 2,
                                                trial_blk.mv_x / 2, trial_blk.mv_y / 2,
                                                enc->bit_depth, base_qp, lambda, trial_blk.coeff_cr);
                } else {
                    /* INTRA mode */
                    cbf_y = encode_pixel_block(enc->orig_y, enc->recon_y,
                                               enc->width, enc->width, enc->height,
                                               bx, by, bw, bh,
                                               te->tile_x, te->tile_y, enc->bit_depth, base_qp,
                                               lambda, trial_blk.coeff_y);
                    cbf_cb = encode_pixel_block(enc->orig_cb, enc->recon_cb,
                                                enc->chroma_w, enc->chroma_w, enc->chroma_h,
                                                bx / 2, by / 2, bw / 2, bh / 2,
                                                te->tile_x / 2, te->tile_y / 2, enc->bit_depth, base_qp,
                                                lambda, trial_blk.coeff_cb);
                    cbf_cr = encode_pixel_block(enc->orig_cr, enc->recon_cr,
                                                enc->chroma_w, enc->chroma_w, enc->chroma_h,
                                                bx / 2, by / 2, bw / 2, bh / 2,
                                                te->tile_x / 2, te->tile_y / 2, enc->bit_depth, base_qp,
                                                lambda, trial_blk.coeff_cr);
                }

                trial_blk.cbf = (cbf_y || cbf_cb || cbf_cr) ? 1 : 0;

                /* Add block to tile encoder state temporarily */
                int trial_block_idx = te->num_blocks;
                te->blocks[trial_block_idx] = trial_blk;
                te->num_blocks++;

                /* Encode shape symbol BEFORE marking cells (get_valid_shapes needs cells unmarked) */
                EncoderCheckpoint cp_before_encode;
                checkpoint_save(te, &cp_before_encode);

                encode_shape(te, cx, cy, shape_idx);

                /* Mark cells as occupied AFTER encoding shape */
                for (int dy = 0; dy < h_cells; dy++) {
                    for (int dx = 0; dx < w_cells; dx++) {
                        te->cell_to_block[(cy + dy) * te->cells_w + (cx + dx)] = trial_block_idx;
                    }
                }

                encode_block_data(te, &trial_blk, base_qp,
                                  enc->frame_type == 1, enc->dpb.count);

                /* Compute rate */
                size_t rate_bits = estimate_rate(te, &cp_before_encode);
                checkpoint_free(&cp_before_encode);

                /* Compute distortion (SSE or perceptual) */
                int64_t dist_y, dist_cb, dist_cr;
                if (g_psy_rdo) {
                    /* Perceptual RDO: SSE + boundary discontinuity penalty */
                    dist_y = compute_perceptual_distortion(enc->orig_y, enc->recon_y,
                                                            enc->width, enc->width, enc->height,
                                                            bx, by, bw, bh,
                                                            te->tile_x, te->tile_y);
                    dist_cb = compute_perceptual_distortion(enc->orig_cb, enc->recon_cb,
                                                             enc->chroma_w, enc->chroma_w, enc->chroma_h,
                                                             bx / 2, by / 2, bw / 2, bh / 2,
                                                             te->tile_x / 2, te->tile_y / 2);
                    dist_cr = compute_perceptual_distortion(enc->orig_cr, enc->recon_cr,
                                                             enc->chroma_w, enc->chroma_w, enc->chroma_h,
                                                             bx / 2, by / 2, bw / 2, bh / 2,
                                                             te->tile_x / 2, te->tile_y / 2);
                } else {
                    /* Standard RDO: pure SSE */
                    dist_y = compute_block_distortion(enc->orig_y, enc->recon_y,
                                                       enc->width, enc->width, enc->height,
                                                       bx, by, bw, bh);
                    dist_cb = compute_block_distortion(enc->orig_cb, enc->recon_cb,
                                                        enc->chroma_w, enc->chroma_w, enc->chroma_h,
                                                        bx / 2, by / 2, bw / 2, bh / 2);
                    dist_cr = compute_block_distortion(enc->orig_cr, enc->recon_cr,
                                                        enc->chroma_w, enc->chroma_w, enc->chroma_h,
                                                        bx / 2, by / 2, bw / 2, bh / 2);
                }

                /* Weight chroma (4:2:0 has 1/4 the samples) */
                int64_t total_dist = dist_y + (dist_cb + dist_cr) * 4;

                /* RD cost = D + λ * R */
                double rd_cost = (double)total_dist + lambda * (double)rate_bits;

                if (rd_cost < best_rd_cost) {
                    best_rd_cost = rd_cost;
                    best_shape = shape_idx;
                    best_block = trial_blk;
                }
            }

            /* Restore to checkpoint and apply best choice */
            checkpoint_restore(te, &cp);
            recon_checkpoint_restore(&rcp, enc->recon_y, enc->recon_cb, enc->recon_cr);
            checkpoint_free(&cp);
            recon_checkpoint_free(&rcp);

            /* Now actually encode with best shape */
            int w_cells = BLOCK_SHAPES[best_shape][0];
            int h_cells = BLOCK_SHAPES[best_shape][1];
            int bw = w_cells * CELL_SIZE;
            int bh = h_cells * CELL_SIZE;
            int bx = te->tile_x + cx * CELL_SIZE;
            int by = te->tile_y + cy * CELL_SIZE;

            /* Re-encode pixels with best shape and mode */
            EncodedBlock *blk = &te->blocks[te->num_blocks];
            blk->x = bx;
            blk->y = by;
            blk->w = bw;
            blk->h = bh;
            blk->qp_delta = 0;
            blk->mode = best_block.mode;
            blk->mv_x = best_block.mv_x;
            blk->mv_y = best_block.mv_y;
            blk->ref_idx = best_block.ref_idx;

            int cbf_y, cbf_cb, cbf_cr;

            if (blk->mode == MODE_SKIP) {
                RefFrame *ref = &enc->dpb.frames[blk->ref_idx];
                encode_skip_block(enc->recon_y, ref->y, ref->width, ref->height,
                                  enc->width, enc->width, enc->height, bx, by, bw, bh,
                                  blk->mv_x, blk->mv_y, enc->bit_depth);
                encode_skip_block(enc->recon_cb, ref->cb, (ref->width + 1) / 2, (ref->height + 1) / 2,
                                  enc->chroma_w, enc->chroma_w, enc->chroma_h, bx / 2, by / 2, bw / 2, bh / 2,
                                  blk->mv_x / 2, blk->mv_y / 2, enc->bit_depth);
                encode_skip_block(enc->recon_cr, ref->cr, (ref->width + 1) / 2, (ref->height + 1) / 2,
                                  enc->chroma_w, enc->chroma_w, enc->chroma_h, bx / 2, by / 2, bw / 2, bh / 2,
                                  blk->mv_x / 2, blk->mv_y / 2, enc->bit_depth);
                cbf_y = cbf_cb = cbf_cr = 0;
                memset(blk->coeff_y, 0, sizeof(blk->coeff_y));
                memset(blk->coeff_cb, 0, sizeof(blk->coeff_cb));
                memset(blk->coeff_cr, 0, sizeof(blk->coeff_cr));
            } else if (blk->mode == MODE_INTER) {
                RefFrame *ref = &enc->dpb.frames[blk->ref_idx];
                cbf_y = encode_inter_block(enc->orig_y, enc->recon_y,
                                           ref->y, ref->width, ref->height,
                                           enc->width, enc->width, enc->height,
                                           bx, by, bw, bh,
                                           blk->mv_x, blk->mv_y,
                                           enc->bit_depth, base_qp, lambda, blk->coeff_y);
                cbf_cb = encode_inter_block(enc->orig_cb, enc->recon_cb,
                                            ref->cb, (ref->width + 1) / 2, (ref->height + 1) / 2,
                                            enc->chroma_w, enc->chroma_w, enc->chroma_h,
                                            bx / 2, by / 2, bw / 2, bh / 2,
                                            blk->mv_x / 2, blk->mv_y / 2,
                                            enc->bit_depth, base_qp, lambda, blk->coeff_cb);
                cbf_cr = encode_inter_block(enc->orig_cr, enc->recon_cr,
                                            ref->cr, (ref->width + 1) / 2, (ref->height + 1) / 2,
                                            enc->chroma_w, enc->chroma_w, enc->chroma_h,
                                            bx / 2, by / 2, bw / 2, bh / 2,
                                            blk->mv_x / 2, blk->mv_y / 2,
                                            enc->bit_depth, base_qp, lambda, blk->coeff_cr);
            } else {
                /* INTRA mode */
                cbf_y = encode_pixel_block(enc->orig_y, enc->recon_y,
                                           enc->width, enc->width, enc->height,
                                           bx, by, bw, bh,
                                           te->tile_x, te->tile_y, enc->bit_depth, base_qp,
                                           lambda, blk->coeff_y);
                cbf_cb = encode_pixel_block(enc->orig_cb, enc->recon_cb,
                                            enc->chroma_w, enc->chroma_w, enc->chroma_h,
                                            bx / 2, by / 2, bw / 2, bh / 2,
                                            te->tile_x / 2, te->tile_y / 2, enc->bit_depth, base_qp,
                                            lambda, blk->coeff_cb);
                cbf_cr = encode_pixel_block(enc->orig_cr, enc->recon_cr,
                                            enc->chroma_w, enc->chroma_w, enc->chroma_h,
                                            bx / 2, by / 2, bw / 2, bh / 2,
                                            te->tile_x / 2, te->tile_y / 2, enc->bit_depth, base_qp,
                                            lambda, blk->coeff_cr);
            }

            blk->cbf = (cbf_y || cbf_cb || cbf_cr) ? 1 : 0;

            /* Encode shape BEFORE marking cells (get_valid_shapes needs cells unmarked) */
            encode_shape(te, cx, cy, best_shape);

            /* Mark cells as occupied AFTER encoding shape */
            for (int dy = 0; dy < h_cells; dy++) {
                for (int dx = 0; dx < w_cells; dx++) {
                    te->cell_to_block[(cy + dy) * te->cells_w + (cx + dx)] = te->num_blocks;
                }
            }

            /* Track mode statistics */
            if (blk->mode == MODE_INTRA) g_intra_blocks++;
            else if (blk->mode == MODE_INTER) g_inter_blocks++;
            else g_skip_blocks++;

            te->num_blocks++;
        }
    }

    /* Second pass: encode block data (matching decoder's decode_blocks) */
    for (int i = 0; i < te->num_blocks; i++) {
        EncodedBlock *blk = &te->blocks[i];
        encode_block_data(te, blk, base_qp, enc->frame_type == 1, enc->dpb.count);
    }

    /* Finalize rANS streams */
    size_t rans_size;
    size_t rans_bypass_offset;
    uint8_t *rans_data = dual_rans_enc_finish(&te->rans, &rans_size, &rans_bypass_offset);

    /* Get bypass data */
    size_t bypass_size = bypass_writer_size(&te->bypass);

    /* Combine: [rANS data][bypass data] */
    *out_size = rans_size + bypass_size;
    uint8_t *output = malloc(*out_size);
    memcpy(output, rans_data, rans_size);
    memcpy(output + rans_size, te->bypass.data, bypass_size);

    *bypass_offset_out = (uint16_t)rans_size;

    free(rans_data);
    bypass_writer_free(&te->bypass);
    dual_rans_enc_free(&te->rans);
    free(te->cell_to_block);
    free(te->blocks);

    return output;
}

/* ============================================================================
 * Frame Encoding
 * ============================================================================ */

/* Add already-filtered frame to DPB for reference */
static void dpb_add_frame_filtered(EncoderDPB *dpb, uint16_t *filtered_y, uint16_t *filtered_cb, uint16_t *filtered_cr,
                                   int width, int height) {
    /* Shift existing frames if DPB is full */
    if (dpb->count >= dpb->max_refs) {
        /* Free oldest frame */
        free(dpb->frames[dpb->max_refs - 1].y);
        free(dpb->frames[dpb->max_refs - 1].cb);
        free(dpb->frames[dpb->max_refs - 1].cr);

        /* Shift frames */
        for (int i = dpb->max_refs - 1; i > 0; i--) {
            dpb->frames[i] = dpb->frames[i - 1];
        }
    } else {
        /* Shift existing frames */
        for (int i = dpb->count; i > 0; i--) {
            dpb->frames[i] = dpb->frames[i - 1];
        }
        dpb->count++;
    }

    /* Allocate and copy new frame at position 0 */
    /* Chroma dimensions use ceiling division per spec §10.1 */
    int luma_size = width * height;
    int chroma_w = (width + 1) / 2;
    int chroma_h = (height + 1) / 2;
    int chroma_size = chroma_w * chroma_h;

    dpb->frames[0].y = malloc(luma_size * sizeof(uint16_t));
    dpb->frames[0].cb = malloc(chroma_size * sizeof(uint16_t));
    dpb->frames[0].cr = malloc(chroma_size * sizeof(uint16_t));
    dpb->frames[0].width = width;
    dpb->frames[0].height = height;

    memcpy(dpb->frames[0].y, filtered_y, luma_size * sizeof(uint16_t));
    memcpy(dpb->frames[0].cb, filtered_cb, chroma_size * sizeof(uint16_t));
    memcpy(dpb->frames[0].cr, filtered_cr, chroma_size * sizeof(uint16_t));
}

static void encode_frame(EncoderContext *enc, BitstreamWriter *bs) {
    int tiles_wide = (enc->width + TILE_SIZE - 1) / TILE_SIZE;
    int tiles_high = (enc->height + TILE_SIZE - 1) / TILE_SIZE;

    /* Write frame header */
    bs_write_u8(bs, enc->frame_type);  /* frame_type: 0=I, 1=P */
    bs_write_u8(bs, enc->base_qp);     /* base_qp */
    bs_write_u8(bs, 0);                /* filter_mode = 0 (default) */

    /* Process each tile with RDO */
    for (int ty = 0; ty < tiles_high; ty++) {
        for (int tx = 0; tx < tiles_wide; tx++) {
            int tile_x = tx * TILE_SIZE;
            int tile_y = ty * TILE_SIZE;
            int tile_w = min_i(TILE_SIZE, enc->width - tile_x);
            int tile_h = min_i(TILE_SIZE, enc->height - tile_y);
            int cells_w = (tile_w + CELL_SIZE - 1) / CELL_SIZE;
            int cells_h = (tile_h + CELL_SIZE - 1) / CELL_SIZE;

            TileEncoder te = {0};
            te.tile_x = tile_x;
            te.tile_y = tile_y;
            te.tile_w = tile_w;
            te.tile_h = tile_h;
            te.cells_w = cells_w;
            te.cells_h = cells_h;

            size_t tile_size;
            uint16_t bypass_offset;
            uint8_t *tile_data = encode_tile_rdo(enc, &te, enc->base_qp,
                                                  &tile_size, &bypass_offset);

            /* Write tile header */
            bs_write_u24(bs, (uint32_t)tile_size);
            bs_write_u16(bs, bypass_offset);

            /* Write tile data */
            bs_write_bytes(bs, tile_data, tile_size);

            free(tile_data);
        }
    }
}

/* ============================================================================
 * PSNR Calculation
 * ============================================================================ */

static double compute_psnr(uint16_t *orig, uint16_t *recon, int N, int bit_depth) {
    double mse = 0.0;
    for (int i = 0; i < N; i++) {
        double diff = (double)orig[i] - (double)recon[i];
        mse += diff * diff;
    }
    mse /= N;

    if (mse < 1e-10)
        return 99.99;

    double max_val = (1 << bit_depth) - 1;
    return 10.0 * log10((max_val * max_val) / mse);
}

/* ============================================================================
 * YUV I/O
 * ============================================================================ */

static int read_yuv_frame(FILE *f, EncoderContext *enc) {
    int luma_size = enc->width * enc->height;
    /* Chroma dimensions use ceiling division per spec §10.1 */
    int chroma_size = ((enc->width + 1) / 2) * ((enc->height + 1) / 2);

    uint8_t *buf = malloc(luma_size);

    if (fread(buf, 1, luma_size, f) != (size_t)luma_size) {
        free(buf);
        return 0;
    }
    for (int i = 0; i < luma_size; i++)
        enc->orig_y[i] = buf[i];

    if (fread(buf, 1, chroma_size, f) != (size_t)chroma_size) {
        free(buf);
        return 0;
    }
    for (int i = 0; i < chroma_size; i++)
        enc->orig_cb[i] = buf[i];

    if (fread(buf, 1, chroma_size, f) != (size_t)chroma_size) {
        free(buf);
        return 0;
    }
    for (int i = 0; i < chroma_size; i++)
        enc->orig_cr[i] = buf[i];

    free(buf);
    return 1;
}

__attribute__((unused))
static void write_yuv_frame(FILE *f, EncoderContext *enc) {
    int luma_size = enc->width * enc->height;
    /* Chroma dimensions use ceiling division per spec §10.1 */
    int chroma_size = ((enc->width + 1) / 2) * ((enc->height + 1) / 2);

    uint8_t *buf = malloc(luma_size);

    for (int i = 0; i < luma_size; i++)
        buf[i] = (uint8_t)enc->recon_y[i];
    fwrite(buf, 1, luma_size, f);

    for (int i = 0; i < chroma_size; i++)
        buf[i] = (uint8_t)enc->recon_cb[i];
    fwrite(buf, 1, chroma_size, f);

    for (int i = 0; i < chroma_size; i++)
        buf[i] = (uint8_t)enc->recon_cr[i];
    fwrite(buf, 1, chroma_size, f);

    free(buf);
}

/* ============================================================================
 * Main Entry Point
 * ============================================================================ */

int main(int argc, char **argv) {
    /* Parse --psy flag from any position */
    int positional_argc = 0;
    char *positional_argv[10];
    for (int i = 0; i < argc && positional_argc < 10; i++) {
        if (strcmp(argv[i], "--psy") == 0) {
            g_psy_rdo = 1;
        } else {
            positional_argv[positional_argc++] = argv[i];
        }
    }

    if (positional_argc < 6 || positional_argc > 7) {
        fprintf(stderr, "Usage: %s <input.yuv> <width> <height> <qp> <output.lat> [recon.yuv] [--psy]\n", argv[0]);
        fprintf(stderr, "\n");
        fprintf(stderr, "  input.yuv   - Raw YUV 4:2:0 planar, 8-bit\n");
        fprintf(stderr, "  width       - Frame width in pixels\n");
        fprintf(stderr, "  height      - Frame height in pixels\n");
        fprintf(stderr, "  qp          - Quantization parameter (0-51)\n");
        fprintf(stderr, "  output.lat  - Encoded bitstream output\n");
        fprintf(stderr, "  recon.yuv   - Optional: reconstructed YUV output\n");
        fprintf(stderr, "  --psy       - Enable perceptual RDO (reduces blocking artifacts)\n");
        return 1;
    }

    int width = atoi(positional_argv[2]);
    int height = atoi(positional_argv[3]);
    int qp = atoi(positional_argv[4]);

    if (width <= 0 || height <= 0) {
        fprintf(stderr, "Error: Invalid dimensions %dx%d (must be positive)\n", width, height);
        return 1;
    }

    if (qp < 0 || qp > 51) {
        fprintf(stderr, "Error: QP must be in range 0-51\n");
        return 1;
    }

    FILE *fin = fopen(positional_argv[1], "rb");
    if (!fin) {
        fprintf(stderr, "Error: Cannot open input file '%s'\n", positional_argv[1]);
        return 1;
    }

    FILE *fout_recon = NULL;
    if (positional_argc == 7) {
        fout_recon = fopen(positional_argv[6], "wb");
        if (!fout_recon) {
            fprintf(stderr, "Error: Cannot open recon file '%s'\n", positional_argv[6]);
            fclose(fin);
            return 1;
        }
    }

    /* Initialize encoder context */
    EncoderContext enc = {0};
    enc.width = width;
    enc.height = height;
    /* Chroma dimensions use ceiling division per spec §10.1 */
    enc.chroma_w = (width + 1) / 2;
    enc.chroma_h = (height + 1) / 2;
    enc.bit_depth = 8;
    enc.base_qp = qp;

    /* Chroma dimensions use ceiling division per spec §10.1 */
    int luma_size = width * height;
    int chroma_w = (width + 1) / 2;
    int chroma_h = (height + 1) / 2;
    int chroma_size = chroma_w * chroma_h;

    /* Allocate padded buffers for reconstruction */
    int padded_w = ((width + CELL_SIZE - 1) / CELL_SIZE) * CELL_SIZE;
    int padded_h = ((height + CELL_SIZE - 1) / CELL_SIZE) * CELL_SIZE;
    int padded_luma_size = (padded_h - 1) * width + padded_w;
    int padded_cw = ((chroma_w + CELL_SIZE - 1) / CELL_SIZE) * CELL_SIZE;
    int padded_ch = ((chroma_h + CELL_SIZE - 1) / CELL_SIZE) * CELL_SIZE;
    int padded_chroma_size = (padded_ch - 1) * chroma_w + padded_cw;

    enc.orig_y = malloc(luma_size * sizeof(uint16_t));
    enc.orig_cb = malloc(chroma_size * sizeof(uint16_t));
    enc.orig_cr = malloc(chroma_size * sizeof(uint16_t));
    enc.recon_y = calloc(padded_luma_size, sizeof(uint16_t));
    enc.recon_cb = calloc(padded_chroma_size, sizeof(uint16_t));
    enc.recon_cr = calloc(padded_chroma_size, sizeof(uint16_t));

    /* Initialize bitstream writer */
    BitstreamWriter bs;
    bs_writer_init(&bs, 1024 * 1024);

    /* Initialize DPB */
    enc.dpb.count = 0;
    enc.dpb.max_refs = 1;  /* Use 1 reference frame for now */

    /* Write sequence header */
    bs_write_u32(&bs, LATTICE_MAGIC);
    bs_write_u16(&bs, (uint16_t)width);
    bs_write_u16(&bs, (uint16_t)height);
    bs_write_u8(&bs, 8);   /* bit_depth */
    bs_write_u8(&bs, enc.dpb.max_refs);  /* max_ref_frames */

    printf("Lattice encoder (RDO%s): %dx%d, QP=%d\n",
           g_psy_rdo ? "+psy" : "", width, height, qp);

    double total_psnr_y = 0.0, total_psnr_cb = 0.0, total_psnr_cr = 0.0;
    int frame_count = 0;

    while (read_yuv_frame(fin, &enc)) {
        memset(enc.recon_y, 0, padded_luma_size * sizeof(uint16_t));
        memset(enc.recon_cb, 0, padded_chroma_size * sizeof(uint16_t));
        memset(enc.recon_cr, 0, padded_chroma_size * sizeof(uint16_t));

        /* Set frame type: first frame is I, rest are P */
        enc.frame_type = (frame_count == 0) ? 0 : 1;

        encode_frame(&enc, &bs);

        /* Apply loop filter to reconstruction for output and PSNR calculation.
         * This matches what the decoder outputs. */
        int16_t weights[4][4][4][3][3];
        int16_t biases[4][4];
        init_default_weights(weights, biases);

        /* Create filtered copy for output (don't modify enc.recon_* in place) */
        uint16_t *filtered_y = malloc(luma_size * sizeof(uint16_t));
        uint16_t *filtered_cb = malloc(chroma_size * sizeof(uint16_t));
        uint16_t *filtered_cr = malloc(chroma_size * sizeof(uint16_t));
        memcpy(filtered_y, enc.recon_y, luma_size * sizeof(uint16_t));
        memcpy(filtered_cb, enc.recon_cb, chroma_size * sizeof(uint16_t));
        memcpy(filtered_cr, enc.recon_cr, chroma_size * sizeof(uint16_t));

        cnn_filter_plane(filtered_y, enc.width, enc.height, enc.bit_depth, weights, biases);
        cnn_filter_plane(filtered_cb, chroma_w, chroma_h, enc.bit_depth, weights, biases);
        cnn_filter_plane(filtered_cr, chroma_w, chroma_h, enc.bit_depth, weights, biases);

        /* Compute PSNR against filtered output (what viewer sees) */
        double psnr_y = compute_psnr(enc.orig_y, filtered_y, luma_size, enc.bit_depth);
        double psnr_cb = compute_psnr(enc.orig_cb, filtered_cb, chroma_size, enc.bit_depth);
        double psnr_cr = compute_psnr(enc.orig_cr, filtered_cr, chroma_size, enc.bit_depth);

        total_psnr_y += psnr_y;
        total_psnr_cb += psnr_cb;
        total_psnr_cr += psnr_cr;

        /* Write filtered reconstruction (matches decoder output) */
        if (fout_recon) {
            uint8_t *buf = malloc(luma_size);
            for (int i = 0; i < luma_size; i++)
                buf[i] = (uint8_t)filtered_y[i];
            fwrite(buf, 1, luma_size, fout_recon);
            for (int i = 0; i < chroma_size; i++)
                buf[i] = (uint8_t)filtered_cb[i];
            fwrite(buf, 1, chroma_size, fout_recon);
            for (int i = 0; i < chroma_size; i++)
                buf[i] = (uint8_t)filtered_cr[i];
            fwrite(buf, 1, chroma_size, fout_recon);
            free(buf);
        }

        /* Add filtered reconstruction to DPB for reference by subsequent frames */
        dpb_add_frame_filtered(&enc.dpb, filtered_y, filtered_cb, filtered_cr,
                               enc.width, enc.height);

        free(filtered_y);
        free(filtered_cb);
        free(filtered_cr);

        frame_count++;
        fprintf(stderr, "\rEncoding frame %d...", frame_count);
        fflush(stderr);
    }
    fprintf(stderr, "\n");

    /* Write bitstream to file */
    FILE *fout = fopen(positional_argv[5], "wb");
    if (!fout) {
        fprintf(stderr, "Error: Cannot open output file '%s'\n", positional_argv[5]);
        return 1;
    }
    fwrite(bs.data, 1, bs.pos, fout);
    fclose(fout);

    /* Print statistics */
    if (frame_count > 0) {
        double avg_psnr_y = total_psnr_y / frame_count;
        double avg_psnr_cb = total_psnr_cb / frame_count;
        double avg_psnr_cr = total_psnr_cr / frame_count;
        double avg_combined = (4.0 * avg_psnr_y + avg_psnr_cb + avg_psnr_cr) / 6.0;

        printf("Encoded %d frame%s\n", frame_count, frame_count == 1 ? "" : "s");
        printf("PSNR Y: %.2f dB  Cb: %.2f dB  Cr: %.2f dB\n", avg_psnr_y, avg_psnr_cb, avg_psnr_cr);
        printf("Average: %.2f dB\n", avg_combined);
        printf("Bitstream size: %zu bytes (%.2f bpp)\n", bs.pos,
               (double)bs.pos * 8.0 / (width * height * frame_count));
        if (g_trellis && g_trellis_zeroed > 0) {
            printf("Trellis: zeroed %d coefficients\n", g_trellis_zeroed);
        }
        int total_blocks = g_intra_blocks + g_inter_blocks + g_skip_blocks;
        if (total_blocks > 0 && (g_inter_blocks > 0 || g_skip_blocks > 0)) {
            printf("Modes: INTRA=%d (%.1f%%), INTER=%d (%.1f%%), SKIP=%d (%.1f%%)\n",
                   g_intra_blocks, 100.0 * g_intra_blocks / total_blocks,
                   g_inter_blocks, 100.0 * g_inter_blocks / total_blocks,
                   g_skip_blocks, 100.0 * g_skip_blocks / total_blocks);
        }
    }

    /* Cleanup */
    fclose(fin);
    if (fout_recon) fclose(fout_recon);
    free(enc.orig_y);
    free(enc.orig_cb);
    free(enc.orig_cr);
    free(enc.recon_y);
    free(enc.recon_cb);
    free(enc.recon_cr);
    bs_writer_free(&bs);

    return 0;
}
