/*
 * Lattice Video Codec - Reference Decoder
 * Draft 0.3 - February 2026
 *
 * Usage: lattice_decode <input.lat> <output.yuv>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lattice_common.h"
#include "lattice_bitstream.h"
#include "lattice_rans.h"

/* ============================================================================
 * Tile Decoder State
 * ============================================================================ */

typedef struct {
    DualRansDecoder rans;
    BypassReader bypass;
    CdfContext contexts[NUM_CONTEXTS];

    BlockInfo *blocks;
    int num_blocks;
    int *cell_to_block;

    int tile_x, tile_y;
    int tile_w, tile_h;
    int cells_w, cells_h;

    uint16_t *recon_y, *recon_cb, *recon_cr;
} TileDecoder;

static int tile_rans_decode(TileDecoder *td, int ctx_idx) {
    return dual_rans_dec_get(&td->rans, &td->contexts[ctx_idx]);
}

/* ============================================================================
 * Inverse DCT
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
 * Intra Prediction
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
 * Inter Prediction
 * ============================================================================ */

static inline uint16_t ref_sample(const uint16_t *ref, int w, int h, int ry, int rx) {
    rx = clamp_i(rx, 0, w - 1);
    ry = clamp_i(ry, 0, h - 1);
    return ref[ry * w + rx];
}

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

/* ============================================================================
 * CNN Loop Filter
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
 * Block Map Decoding
 * ============================================================================ */

static int size_category(int area) {
    if (area <= 64) return 0;
    if (area <= 256) return 1;
    return 2;
}

/* Check if a shape is valid at given cell position (no overlap, fits in tile) */
static int shape_is_valid(int shape_idx, int cx, int cy, int cells_w, int cells_h, int *cell_to_block) {
    int w_cells = BLOCK_SHAPES[shape_idx][0];
    int h_cells = BLOCK_SHAPES[shape_idx][1];

    /* Check if shape fits in tile */
    if (cx + w_cells > cells_w || cy + h_cells > cells_h)
        return 0;

    /* Check for overlap with existing blocks */
    for (int dy = 0; dy < h_cells; dy++) {
        for (int dx = 0; dx < w_cells; dx++) {
            if (cell_to_block[(cy + dy) * cells_w + (cx + dx)] >= 0)
                return 0;
        }
    }
    return 1;
}

static void decode_block_map(TileDecoder *td) {
    td->num_blocks = 0;
    td->blocks = malloc(td->cells_w * td->cells_h * sizeof(BlockInfo));
    td->cell_to_block = malloc(td->cells_w * td->cells_h * sizeof(int));
    for (int i = 0; i < td->cells_w * td->cells_h; i++)
        td->cell_to_block[i] = -1;

    for (int cy = 0; cy < td->cells_h; cy++) {
        for (int cx = 0; cx < td->cells_w; cx++) {
            if (td->cell_to_block[cy * td->cells_w + cx] >= 0)
                continue;

            /* Compute context from neighbor block sizes per §4.3 */
            int above_cat = 0, left_cat = 0;
            if (cy > 0) {
                int above_idx = td->cell_to_block[(cy - 1) * td->cells_w + cx];
                if (above_idx >= 0) {
                    BlockInfo *ab = &td->blocks[above_idx];
                    above_cat = size_category(ab->w * ab->h);
                }
            }
            if (cx > 0) {
                int left_idx = td->cell_to_block[cy * td->cells_w + (cx - 1)];
                if (left_idx >= 0) {
                    BlockInfo *lb = &td->blocks[left_idx];
                    left_cat = size_category(lb->w * lb->h);
                }
            }
            int ctx = above_cat * 3 + left_cat;

            /*
             * Per spec §4.3: Always decode from full 7-symbol CDF.
             * No alphabet reduction - the same 7-entry CDF is used at every
             * cell position regardless of which shapes fit.
             */
            int shape_idx = tile_rans_decode(td, CTX_SHAPE + ctx);

            /* Validate: conformant bitstream shall not specify invalid shapes */
            if (!shape_is_valid(shape_idx, cx, cy, td->cells_w, td->cells_h, td->cell_to_block)) {
                fprintf(stderr, "Error: Invalid shape %d at cell (%d,%d)\n", shape_idx, cx, cy);
                /* In a production decoder, this would be a bitstream error */
            }

            int w_cells = BLOCK_SHAPES[shape_idx][0];
            int h_cells = BLOCK_SHAPES[shape_idx][1];

            BlockInfo *blk = &td->blocks[td->num_blocks];
            blk->x = td->tile_x + cx * CELL_SIZE;
            blk->y = td->tile_y + cy * CELL_SIZE;
            blk->w = w_cells * CELL_SIZE;
            blk->h = h_cells * CELL_SIZE;
            blk->mode = 0;
            blk->ref_idx = 0;
            blk->mv_x = 0;
            blk->mv_y = 0;
            blk->qp_delta = 0;
            blk->cbf = 0;

            for (int dy = 0; dy < h_cells; dy++) {
                for (int dx = 0; dx < w_cells; dx++) {
                    td->cell_to_block[(cy + dy) * td->cells_w + (cx + dx)] = td->num_blocks;
                }
            }

            td->num_blocks++;
        }
    }
}

/* ============================================================================
 * MV Decoding
 * ============================================================================ */

static int16_t decode_mv_component(TileDecoder *td, int comp) {
    int cls = tile_rans_decode(td, CTX_MVD_CLASS + comp);

    int mag = 0;
    if (cls == 0) {
        return 0;
    } else if (cls == 1) {
        mag = 1;
    } else if (cls == 2) {
        mag = 2 + bypass_read_bits(&td->bypass, 1);
    } else if (cls == 3) {
        mag = 4 + bypass_read_bits(&td->bypass, 2);
    } else if (cls == 4) {
        mag = 8 + bypass_read_bits(&td->bypass, 3);
    } else if (cls == 5) {
        mag = 16 + bypass_read_bits(&td->bypass, 4);
    } else {
        int residual = bypass_read_expgolomb(&td->bypass);
        mag = 32 + residual;
    }

    int sign = bypass_read_bit(&td->bypass);
    return sign ? -mag : mag;
}

static void get_mv_predictor(TileDecoder *td, int blk_idx, int16_t *mvp_x, int16_t *mvp_y) {
    BlockInfo *blk = &td->blocks[blk_idx];
    int bx = blk->x, by = blk->y;

    int16_t ax = 0, ay = 0, bx_mv = 0, by_mv = 0;
    int has_a = 0, has_b = 0;

    if (bx > td->tile_x) {
        int cx = (bx - td->tile_x) / CELL_SIZE - 1;
        int cy = (by - td->tile_y) / CELL_SIZE;
        int left_idx = td->cell_to_block[cy * td->cells_w + cx];
        if (left_idx >= 0 && td->blocks[left_idx].mode != 0) {
            ax = td->blocks[left_idx].mv_x;
            ay = td->blocks[left_idx].mv_y;
            has_a = 1;
        }
    }

    if (by > td->tile_y) {
        int cx = (bx - td->tile_x) / CELL_SIZE;
        int cy = (by - td->tile_y) / CELL_SIZE - 1;
        int above_idx = td->cell_to_block[cy * td->cells_w + cx];
        if (above_idx >= 0 && td->blocks[above_idx].mode != 0) {
            bx_mv = td->blocks[above_idx].mv_x;
            by_mv = td->blocks[above_idx].mv_y;
            has_b = 1;
        }
    }

    if (has_a && has_b) {
        int sum_x = ax + bx_mv;
        int sum_y = ay + by_mv;
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
 * Coefficient Decoding
 * ============================================================================ */

static void decode_coefficients(TileDecoder *td, int16_t *coeff, int W, int H, int is_chroma) {
    int N = W * H;
    memset(coeff, 0, N * sizeof(int16_t));

    int scan[1024];
    generate_scan(W, H, scan);

    int bounds[5], num_bands;
    get_band_boundaries(W, H, bounds, &num_bands);

    int ctx_band = is_chroma ? CTX_BAND_CHROMA : CTX_BAND_LUMA;
    int ctx_sig = is_chroma ? CTX_SIG_CHROMA : CTX_SIG_LUMA;
    int ctx_level = is_chroma ? CTX_LEVEL_CHROMA : CTX_LEVEL_LUMA;

    int prev_band_zero = 0;  /* Track if previous band was ALL_ZERO */
    for (int band = 0; band < num_bands; band++) {
        int band_start = bounds[band];
        int band_end = bounds[band + 1];

        /* Band status context per §9.4: band_index * 2 + prev_zero */
        int prev_zero = (band > 0 && prev_band_zero) ? 1 : 0;
        int status_ctx = ctx_band + min_i(band, 3) * 2 + prev_zero;
        int band_status = tile_rans_decode(td, status_ctx);

        prev_band_zero = (band_status == 0);

        if (band_status == 0) {
            continue;
        }

        int sig_flags[1024];
        /* Sliding window of last 4 flags per §9.4 */
        int last4[4] = {0, 0, 0, 0};
        int last4_idx = 0;
        for (int i = band_start; i < band_end; i++) {
            /* Count non-zero flags in last 4, saturated to 3 */
            int density = last4[0] + last4[1] + last4[2] + last4[3];
            if (density > 3) density = 3;
            int sig_ctx = ctx_sig + (min_i(band, 3) * 4 + density);
            sig_flags[i] = tile_rans_decode(td, sig_ctx);
            /* Update sliding window */
            last4[last4_idx] = sig_flags[i] ? 1 : 0;
            last4_idx = (last4_idx + 1) & 3;
        }

        int prev_level = 0;
        for (int i = band_start; i < band_end; i++) {
            if (!sig_flags[i]) continue;

            int prev_cat = (prev_level <= 1) ? 0 : (prev_level <= 4) ? 1 : (prev_level <= 7) ? 2 : 3;
            int lev_ctx = ctx_level + ((band < 4 ? band : 3) * 4 + prev_cat);
            int token = tile_rans_decode(td, lev_ctx);

            int level;
            if (token < 7) {
                level = token + 1;
            } else {
                int escape_val = bypass_read_expgolomb(&td->bypass);
                level = 8 + escape_val;
            }

            coeff[scan[i]] = level;
            prev_level = level;
        }
    }

    int sign_count = 0;
    for (int i = 0; i < N; i++) {
        if (coeff[scan[i]] != 0) {
            int sign = bypass_read_bit(&td->bypass);
            if (sign)
                coeff[scan[i]] = -coeff[scan[i]];
            sign_count++;
        }
    }
    (void)sign_count;
}

/* ============================================================================
 * Block Decoding
 * ============================================================================ */

static void decode_blocks(TileDecoder *td, FrameHeader *fh, Dpb *dpb,
                          int frame_w, int frame_h, int bit_depth) {
    int max_val = (1 << bit_depth) - 1;
    /* Chroma stride uses ceiling division per spec §10.1 */
    int chroma_stride = (frame_w + 1) / 2;
    int chroma_w = (frame_w + 1) / 2;
    int chroma_h = (frame_h + 1) / 2;

    for (int i = 0; i < td->num_blocks; i++) {
        BlockInfo *blk = &td->blocks[i];

        if (fh->frame_type == 0) {
            blk->mode = 0;
        } else {
            /* Context based on mode categories per §5.2 */
            int above_cat = 1, left_cat = 1;  /* Default to INTER (1) for missing neighbors */
            int bx = blk->x, by = blk->y;

            if (by > td->tile_y) {
                int cx = (bx - td->tile_x) / CELL_SIZE;
                int cy = (by - td->tile_y) / CELL_SIZE - 1;
                int above_idx = td->cell_to_block[cy * td->cells_w + cx];
                if (above_idx >= 0)
                    above_cat = mode_category(td->blocks[above_idx].mode);
            }
            if (bx > td->tile_x) {
                int cx = (bx - td->tile_x) / CELL_SIZE - 1;
                int cy = (by - td->tile_y) / CELL_SIZE;
                int left_idx = td->cell_to_block[cy * td->cells_w + cx];
                if (left_idx >= 0)
                    left_cat = mode_category(td->blocks[left_idx].mode);
            }

            int ctx = above_cat * 3 + left_cat;
            blk->mode = tile_rans_decode(td, CTX_MODE + ctx);
        }

        if (blk->mode == 1 && dpb->count > 1) {
            blk->ref_idx = tile_rans_decode(td, CTX_REF_IDX);
        } else if (blk->mode == 1 || blk->mode == 2) {
            blk->ref_idx = 0;
        }

        if (blk->mode == 1) {
            int16_t mvp_x, mvp_y;
            get_mv_predictor(td, i, &mvp_x, &mvp_y);
            int16_t mvd_x = decode_mv_component(td, 0);
            int16_t mvd_y = decode_mv_component(td, 1);
            blk->mv_x = mvp_x + mvd_x;
            blk->mv_y = mvp_y + mvd_y;
        } else if (blk->mode == 2) {
            int16_t mvp_x, mvp_y;
            get_mv_predictor(td, i, &mvp_x, &mvp_y);
            blk->mv_x = mvp_x;
            blk->mv_y = mvp_y;
        }

        if (blk->mode != 2) {
            int above_nz = 0, left_nz = 0;
            int bx = blk->x, by = blk->y;

            if (by > td->tile_y) {
                int cx = (bx - td->tile_x) / CELL_SIZE;
                int cy = (by - td->tile_y) / CELL_SIZE - 1;
                int above_idx = td->cell_to_block[cy * td->cells_w + cx];
                if (above_idx >= 0 && td->blocks[above_idx].qp_delta != 0)
                    above_nz = 1;
            }
            if (bx > td->tile_x) {
                int cx = (bx - td->tile_x) / CELL_SIZE - 1;
                int cy = (by - td->tile_y) / CELL_SIZE;
                int left_idx = td->cell_to_block[cy * td->cells_w + cx];
                if (left_idx >= 0 && td->blocks[left_idx].qp_delta != 0)
                    left_nz = 1;
            }

            int ctx = above_nz + left_nz;
            int sym = tile_rans_decode(td, CTX_QP_DELTA + ctx);
            blk->qp_delta = sym - 2;
        }

        if (blk->mode != 2) {
            int above_cbf = 0, left_cbf = 0;
            int bx = blk->x, by = blk->y;

            if (by > td->tile_y) {
                int cx = (bx - td->tile_x) / CELL_SIZE;
                int cy = (by - td->tile_y) / CELL_SIZE - 1;
                int above_idx = td->cell_to_block[cy * td->cells_w + cx];
                if (above_idx >= 0)
                    above_cbf = td->blocks[above_idx].cbf;
            }
            if (bx > td->tile_x) {
                int cx = (bx - td->tile_x) / CELL_SIZE - 1;
                int cy = (by - td->tile_y) / CELL_SIZE;
                int left_idx = td->cell_to_block[cy * td->cells_w + cx];
                if (left_idx >= 0)
                    left_cbf = td->blocks[left_idx].cbf;
            }

            int ctx = above_cbf + left_cbf;
            blk->cbf = tile_rans_decode(td, CTX_CBF + ctx);
        }

        uint16_t pred_y[32 * 32], pred_cb[16 * 16], pred_cr[16 * 16];
        int bx = blk->x, by = blk->y, bw = blk->w, bh = blk->h;

        if (blk->mode == 0) {
            intra_predict(td->recon_y, frame_w, bx, by, bw, bh,
                         td->tile_x, td->tile_y, bit_depth, pred_y);
            intra_predict(td->recon_cb, chroma_stride, bx / 2, by / 2, bw / 2, bh / 2,
                         td->tile_x / 2, td->tile_y / 2, bit_depth, pred_cb);
            intra_predict(td->recon_cr, chroma_stride, bx / 2, by / 2, bw / 2, bh / 2,
                         td->tile_x / 2, td->tile_y / 2, bit_depth, pred_cr);
        } else {
            Frame *ref = dpb->frames[blk->ref_idx];
            inter_predict(ref->y, ref->width, ref->height,
                         bx, by, bw, bh, blk->mv_x, blk->mv_y, bit_depth, pred_y);

            int16_t chroma_mvx = blk->mv_x / 2;
            int16_t chroma_mvy = blk->mv_y / 2;
            /* Reference chroma dimensions use ceiling division per spec §10.1 */
            int ref_chroma_w = (ref->width + 1) / 2;
            int ref_chroma_h = (ref->height + 1) / 2;
            inter_predict(ref->cb, ref_chroma_w, ref_chroma_h,
                         bx / 2, by / 2, bw / 2, bh / 2,
                         chroma_mvx, chroma_mvy, bit_depth, pred_cb);
            inter_predict(ref->cr, ref_chroma_w, ref_chroma_h,
                         bx / 2, by / 2, bw / 2, bh / 2,
                         chroma_mvx, chroma_mvy, bit_depth, pred_cr);
        }

        if (blk->cbf) {
            int qp = clamp_i(fh->base_qp + blk->qp_delta, 0, 51);
            int qstep = QSTEP[qp];

            int16_t coeff_y[32 * 32], residual_y[32 * 32];
            decode_coefficients(td, coeff_y, bw, bh, 0);
            /* Dequantize with perceptual frequency weighting (§6.3.3) */
            for (int row = 0; row < bh; row++) {
                for (int col = 0; col < bw; col++) {
                    int eff_qs = effective_qstep(qstep, row, col);
                    coeff_y[row * bw + col] = coeff_y[row * bw + col] * eff_qs;
                }
            }
            inverse_transform(coeff_y, residual_y, bw, bh, bit_depth);

            int cw = bw / 2, ch = bh / 2;
            int16_t coeff_cb[16 * 16], coeff_cr[16 * 16];
            int16_t residual_cb[16 * 16], residual_cr[16 * 16];
            decode_coefficients(td, coeff_cb, cw, ch, 1);
            decode_coefficients(td, coeff_cr, cw, ch, 1);
            /* Dequantize chroma with perceptual frequency weighting */
            for (int row = 0; row < ch; row++) {
                for (int col = 0; col < cw; col++) {
                    int eff_qs = effective_qstep(qstep, row, col);
                    coeff_cb[row * cw + col] = coeff_cb[row * cw + col] * eff_qs;
                    coeff_cr[row * cw + col] = coeff_cr[row * cw + col] * eff_qs;
                }
            }
            inverse_transform(coeff_cb, residual_cb, cw, ch, bit_depth);
            inverse_transform(coeff_cr, residual_cr, cw, ch, bit_depth);

            /* Clip luma block to frame boundaries */
            int y_end = min_i(bh, frame_h - by);
            int x_end = min_i(bw, frame_w - bx);
            for (int y = 0; y < y_end; y++) {
                for (int x = 0; x < x_end; x++) {
                    int val = pred_y[y * bw + x] + residual_y[y * bw + x];
                    td->recon_y[(by + y) * frame_w + bx + x] = clamp_i(val, 0, max_val);
                }
            }
            /* Clip chroma block to frame boundaries */
            int cy_end = min_i(ch, chroma_h - by / 2);
            int cx_end = min_i(cw, chroma_w - bx / 2);
            for (int y = 0; y < cy_end; y++) {
                for (int x = 0; x < cx_end; x++) {
                    int val_cb = pred_cb[y * cw + x] + residual_cb[y * cw + x];
                    int val_cr = pred_cr[y * cw + x] + residual_cr[y * cw + x];
                    td->recon_cb[(by / 2 + y) * chroma_stride + bx / 2 + x] = clamp_i(val_cb, 0, max_val);
                    td->recon_cr[(by / 2 + y) * chroma_stride + bx / 2 + x] = clamp_i(val_cr, 0, max_val);
                }
            }
        } else {
            /* Clip luma block to frame boundaries */
            int y_end = min_i(bh, frame_h - by);
            int x_end = min_i(bw, frame_w - bx);
            for (int y = 0; y < y_end; y++) {
                for (int x = 0; x < x_end; x++) {
                    td->recon_y[(by + y) * frame_w + bx + x] = pred_y[y * bw + x];
                }
            }
            int cw = bw / 2, ch = bh / 2;
            /* Clip chroma block to frame boundaries */
            int cy_end = min_i(ch, chroma_h - by / 2);
            int cx_end = min_i(cw, chroma_w - bx / 2);
            for (int y = 0; y < cy_end; y++) {
                for (int x = 0; x < cx_end; x++) {
                    td->recon_cb[(by / 2 + y) * chroma_stride + bx / 2 + x] = pred_cb[y * cw + x];
                    td->recon_cr[(by / 2 + y) * chroma_stride + bx / 2 + x] = pred_cr[y * cw + x];
                }
            }
        }
    }
}

/* ============================================================================
 * Tile Decoding
 * ============================================================================ */

static void decode_tile(BitstreamReader *bs, TileDecoder *td, FrameHeader *fh, Dpb *dpb,
                        SequenceHeader *sh, uint16_t *recon_y, uint16_t *recon_cb, uint16_t *recon_cr) {
    uint32_t tile_data_size = bs_read_u24(bs);
    uint16_t bypass_offset = bs_read_u16(bs);

    const uint8_t *payload = bs_read_bytes(bs, tile_data_size);

    dual_rans_dec_init(&td->rans, payload, bypass_offset);
    bypass_reader_init(&td->bypass, payload, bypass_offset);

    int ctx_sizes[NUM_CONTEXTS] = {0};
    for (int i = CTX_SHAPE; i < CTX_SHAPE + 9; i++) ctx_sizes[i] = 7;
    for (int i = CTX_MODE; i < CTX_MODE + 9; i++) ctx_sizes[i] = 3;
    for (int i = CTX_CBF; i < CTX_CBF + 3; i++) ctx_sizes[i] = 2;
    for (int i = CTX_QP_DELTA; i < CTX_QP_DELTA + 3; i++) ctx_sizes[i] = 5;
    ctx_sizes[CTX_REF_IDX] = (dpb->count > 0) ? dpb->count : 1;
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
            cdf_init_uniform(&td->contexts[i], ctx_sizes[i]);
    }

    td->recon_y = recon_y;
    td->recon_cb = recon_cb;
    td->recon_cr = recon_cr;

    decode_block_map(td);
    decode_blocks(td, fh, dpb, sh->frame_width, sh->frame_height, sh->bit_depth);

    free(td->blocks);
    free(td->cell_to_block);
}

/* ============================================================================
 * Frame Decoding
 * ============================================================================ */

static Frame *alloc_frame(int width, int height) {
    Frame *f = malloc(sizeof(Frame));
    f->width = width;
    f->height = height;

    /* Allocate padded buffers to handle edge blocks that extend beyond frame
     * boundaries. Blocks are cell-aligned (multiples of CELL_SIZE), so for
     * non-cell-aligned frame dimensions, edge blocks extend beyond the frame.
     * These extended writes use stride=width, so they wrap to subsequent rows.
     * Buffer must be large enough to hold all such extended writes. */
    int padded_w = ((width + CELL_SIZE - 1) / CELL_SIZE) * CELL_SIZE;
    int padded_h = ((height + CELL_SIZE - 1) / CELL_SIZE) * CELL_SIZE;
    int padded_luma_size = (padded_h - 1) * width + padded_w;

    /* Chroma dimensions use ceiling division per spec §10.1 */
    int chroma_w = (width + 1) / 2;
    int chroma_h = (height + 1) / 2;
    int padded_cw = ((chroma_w + CELL_SIZE - 1) / CELL_SIZE) * CELL_SIZE;
    int padded_ch = ((chroma_h + CELL_SIZE - 1) / CELL_SIZE) * CELL_SIZE;
    int padded_chroma_size = (padded_ch - 1) * chroma_w + padded_cw;

    f->y = calloc(padded_luma_size, sizeof(uint16_t));
    f->cb = calloc(padded_chroma_size, sizeof(uint16_t));
    f->cr = calloc(padded_chroma_size, sizeof(uint16_t));
    return f;
}

static void free_frame(Frame *f) {
    if (f) {
        free(f->y);
        free(f->cb);
        free(f->cr);
        free(f);
    }
}

/* Decode custom filter weights per §8.5 */
static void decode_filter_weights(BitstreamReader *bs,
                                   int16_t weights[4][4][4][3][3],
                                   int16_t biases[4][4],
                                   const int16_t default_weights[4][4][4][3][3],
                                   const int16_t default_biases[4][4]) {
    /* Read rANS stream size and initialize decoder */
    uint16_t rans_size = bs_read_u16(bs);
    const uint8_t *payload = bs_read_bytes(bs, rans_size);

    RansDecoder rd;
    rans_dec_init_forward(&rd, payload);

    /* Initialize 3 context slots for filter weight deltas (9-symbol alphabet) */
    CdfContext filter_ctx[3];
    for (int i = 0; i < 3; i++)
        cdf_init_uniform(&filter_ctx[i], 9);

    int param_idx = 0;

    /* Layer 1 weights: 1 input channel, 4 output channels, 3x3 kernel = 36 */
    for (int c_out = 0; c_out < 4; c_out++) {
        for (int ky = 0; ky < 3; ky++) {
            for (int kx = 0; kx < 3; kx++) {
                int ctx_slot = param_idx % 3;
                int delta_sym = rans_decode(&rd, &filter_ctx[ctx_slot]);
                cdf_adapt(&filter_ctx[ctx_slot], delta_sym);
                int delta = delta_sym - 4;  /* Map [0,8] to [-4,+4] */
                weights[0][c_out][0][ky][kx] = clamp_i(default_weights[0][c_out][0][ky][kx] + delta, -2048, 2047);
                param_idx++;
            }
        }
    }

    /* Layer 1 biases: 4 */
    for (int c_out = 0; c_out < 4; c_out++) {
        int ctx_slot = param_idx % 3;
        int delta_sym = rans_decode(&rd, &filter_ctx[ctx_slot]);
        cdf_adapt(&filter_ctx[ctx_slot], delta_sym);
        int delta = delta_sym - 4;
        biases[0][c_out] = clamp_i(default_biases[0][c_out] + delta, -2048, 2047);
        param_idx++;
    }

    /* Layers 2-3: 4 input channels, 4 output channels, 3x3 kernel = 144 each */
    for (int layer = 1; layer <= 2; layer++) {
        for (int c_out = 0; c_out < 4; c_out++) {
            for (int c_in = 0; c_in < 4; c_in++) {
                for (int ky = 0; ky < 3; ky++) {
                    for (int kx = 0; kx < 3; kx++) {
                        int ctx_slot = param_idx % 3;
                        int delta_sym = rans_decode(&rd, &filter_ctx[ctx_slot]);
                        cdf_adapt(&filter_ctx[ctx_slot], delta_sym);
                        int delta = delta_sym - 4;
                        weights[layer][c_out][c_in][ky][kx] = clamp_i(default_weights[layer][c_out][c_in][ky][kx] + delta, -2048, 2047);
                        param_idx++;
                    }
                }
            }
        }
        /* Layer biases: 4 */
        for (int c_out = 0; c_out < 4; c_out++) {
            int ctx_slot = param_idx % 3;
            int delta_sym = rans_decode(&rd, &filter_ctx[ctx_slot]);
            cdf_adapt(&filter_ctx[ctx_slot], delta_sym);
            int delta = delta_sym - 4;
            biases[layer][c_out] = clamp_i(default_biases[layer][c_out] + delta, -2048, 2047);
            param_idx++;
        }
    }

    /* Layer 4 weights: 4 input channels, 1 output channel, 3x3 kernel = 36 */
    for (int c_in = 0; c_in < 4; c_in++) {
        for (int ky = 0; ky < 3; ky++) {
            for (int kx = 0; kx < 3; kx++) {
                int ctx_slot = param_idx % 3;
                int delta_sym = rans_decode(&rd, &filter_ctx[ctx_slot]);
                cdf_adapt(&filter_ctx[ctx_slot], delta_sym);
                int delta = delta_sym - 4;
                weights[3][0][c_in][ky][kx] = clamp_i(default_weights[3][0][c_in][ky][kx] + delta, -2048, 2047);
                param_idx++;
            }
        }
    }

    /* Layer 4 bias: 1 */
    {
        int ctx_slot = param_idx % 3;
        int delta_sym = rans_decode(&rd, &filter_ctx[ctx_slot]);
        cdf_adapt(&filter_ctx[ctx_slot], delta_sym);
        int delta = delta_sym - 4;
        biases[3][0] = clamp_i(default_biases[3][0] + delta, -2048, 2047);
    }
}

static Frame *decode_frame(BitstreamReader *bs, SequenceHeader *sh, Dpb *dpb) {
    FrameHeader fh;

    fh.frame_type = bs_read_u8(bs);
    fh.base_qp = bs_read_u8(bs);
    fh.filter_mode = bs_read_u8(bs);

    int16_t default_weights[4][4][4][3][3];
    int16_t default_biases[4][4];
    init_default_weights(default_weights, default_biases);

    if (fh.filter_mode == 1) {
        /* Decode custom luma weights per §8.5 */
        decode_filter_weights(bs, fh.luma_weights, fh.luma_biases,
                             default_weights, default_biases);
    } else {
        memcpy(fh.luma_weights, default_weights, sizeof(fh.luma_weights));
        memcpy(fh.luma_biases, default_biases, sizeof(fh.luma_biases));
    }

    Frame *frame = alloc_frame(sh->frame_width, sh->frame_height);

    for (int ty = 0; ty < sh->tiles_high; ty++) {
        for (int tx = 0; tx < sh->tiles_wide; tx++) {
            TileDecoder td = {0};

            td.tile_x = tx * TILE_SIZE;
            td.tile_y = ty * TILE_SIZE;
            td.tile_w = (tx == sh->tiles_wide - 1) ?
                        (sh->frame_width - tx * TILE_SIZE) : TILE_SIZE;
            td.tile_h = (ty == sh->tiles_high - 1) ?
                        (sh->frame_height - ty * TILE_SIZE) : TILE_SIZE;
            td.cells_w = (td.tile_w + CELL_SIZE - 1) / CELL_SIZE;
            td.cells_h = (td.tile_h + CELL_SIZE - 1) / CELL_SIZE;

            decode_tile(bs, &td, &fh, dpb, sh, frame->y, frame->cb, frame->cr);
        }
    }

    /* Apply loop filter - filtered frames are used for both output and DPB reference */
    cnn_filter_plane(frame->y, sh->frame_width, sh->frame_height,
                     sh->bit_depth, fh.luma_weights, fh.luma_biases);
    /* Chroma dimensions use ceiling division per spec §10.1 */
    int chroma_w = (sh->frame_width + 1) / 2;
    int chroma_h = (sh->frame_height + 1) / 2;
    cnn_filter_plane(frame->cb, chroma_w, chroma_h,
                     sh->bit_depth, default_weights, default_biases);
    cnn_filter_plane(frame->cr, chroma_w, chroma_h,
                     sh->bit_depth, default_weights, default_biases);

    return frame;
}

/* ============================================================================
 * DPB Management
 * ============================================================================ */

static void dpb_insert(Dpb *dpb, Frame *frame, int max_ref) {
    if (dpb->count >= max_ref) {
        free_frame(dpb->frames[dpb->count - 1]);
        dpb->count--;
    }
    for (int i = dpb->count; i > 0; i--)
        dpb->frames[i] = dpb->frames[i - 1];
    dpb->frames[0] = frame;
    dpb->count++;
}

/* ============================================================================
 * Main Entry Point
 * ============================================================================ */

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.lat> <output.yuv>\n", argv[0]);
        return 1;
    }

    FILE *fin = fopen(argv[1], "rb");
    if (!fin) {
        fprintf(stderr, "Error: Cannot open input file '%s'\n", argv[1]);
        return 1;
    }

    fseek(fin, 0, SEEK_END);
    size_t file_size = ftell(fin);
    fseek(fin, 0, SEEK_SET);

    uint8_t *data = malloc(file_size);
    fread(data, 1, file_size, fin);
    fclose(fin);

    BitstreamReader bs;
    bs_reader_init(&bs, data, file_size);

    SequenceHeader sh;
    uint32_t magic = bs_read_u32(&bs);
    if (magic != LATTICE_MAGIC) {
        fprintf(stderr, "Error: Invalid magic number (expected 0x%08X, got 0x%08X)\n", LATTICE_MAGIC, magic);
        free(data);
        return 1;
    }

    sh.frame_width = bs_read_u16(&bs);
    sh.frame_height = bs_read_u16(&bs);
    sh.bit_depth = bs_read_u8(&bs);
    sh.max_ref_frames = bs_read_u8(&bs);
    sh.tiles_wide = (sh.frame_width + 127) / 128;
    sh.tiles_high = (sh.frame_height + 127) / 128;

    printf("Lattice decoder: %dx%d, %d-bit, max %d refs\n",
           sh.frame_width, sh.frame_height, sh.bit_depth, sh.max_ref_frames);

    FILE *fout = fopen(argv[2], "wb");
    if (!fout) {
        fprintf(stderr, "Error: Cannot open output file '%s'\n", argv[2]);
        free(data);
        return 1;
    }

    Dpb dpb = {0};

    int frame_num = 0;
    while (bs.pos < bs.size) {
        Frame *frame = decode_frame(&bs, &sh, &dpb);

        int luma_size = sh.frame_width * sh.frame_height;
        /* Chroma dimensions use ceiling division per spec §10.1 */
        int chroma_size = ((sh.frame_width + 1) / 2) * ((sh.frame_height + 1) / 2);

        if (sh.bit_depth == 8) {
            uint8_t *out = malloc(luma_size);
            for (int i = 0; i < luma_size; i++)
                out[i] = frame->y[i];
            fwrite(out, 1, luma_size, fout);
            for (int i = 0; i < chroma_size; i++)
                out[i] = frame->cb[i];
            fwrite(out, 1, chroma_size, fout);
            for (int i = 0; i < chroma_size; i++)
                out[i] = frame->cr[i];
            fwrite(out, 1, chroma_size, fout);
            free(out);
        } else {
            fwrite(frame->y, sizeof(uint16_t), luma_size, fout);
            fwrite(frame->cb, sizeof(uint16_t), chroma_size, fout);
            fwrite(frame->cr, sizeof(uint16_t), chroma_size, fout);
        }

        dpb_insert(&dpb, frame, sh.max_ref_frames);

        frame_num++;
        printf("Decoded frame %d\n", frame_num);
    }

    for (int i = 0; i < dpb.count; i++)
        free_frame(dpb.frames[i]);

    fclose(fout);
    free(data);

    printf("Decoded %d frames successfully.\n", frame_num);
    return 0;
}
