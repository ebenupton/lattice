/*
 * Lattice Video Codec - Bitstream Reader/Writer
 */

#ifndef LATTICE_BITSTREAM_H
#define LATTICE_BITSTREAM_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Bitstream Reader
 * ============================================================================ */

typedef struct {
    const uint8_t *data;
    size_t size;
    size_t pos;
} BitstreamReader;

static inline void bs_reader_init(BitstreamReader *bs, const uint8_t *data, size_t size) {
    bs->data = data;
    bs->size = size;
    bs->pos = 0;
}

static inline uint32_t bs_read_u8(BitstreamReader *bs) {
    return bs->data[bs->pos++];
}

static inline uint32_t bs_read_u16(BitstreamReader *bs) {
    uint32_t v = ((uint32_t)bs->data[bs->pos] << 8) | bs->data[bs->pos + 1];
    bs->pos += 2;
    return v;
}

static inline uint32_t bs_read_u24(BitstreamReader *bs) {
    uint32_t v = ((uint32_t)bs->data[bs->pos] << 16) |
                 ((uint32_t)bs->data[bs->pos + 1] << 8) |
                 bs->data[bs->pos + 2];
    bs->pos += 3;
    return v;
}

static inline uint32_t bs_read_u32(BitstreamReader *bs) {
    uint32_t v = ((uint32_t)bs->data[bs->pos] << 24) |
                 ((uint32_t)bs->data[bs->pos + 1] << 16) |
                 ((uint32_t)bs->data[bs->pos + 2] << 8) |
                 bs->data[bs->pos + 3];
    bs->pos += 4;
    return v;
}

static inline const uint8_t *bs_read_bytes(BitstreamReader *bs, size_t n) {
    const uint8_t *ptr = bs->data + bs->pos;
    bs->pos += n;
    return ptr;
}

/* ============================================================================
 * Bitstream Writer
 * ============================================================================ */

typedef struct {
    uint8_t *data;
    size_t capacity;
    size_t pos;
} BitstreamWriter;

static inline void bs_writer_init(BitstreamWriter *bs, size_t initial_capacity) {
    bs->data = (uint8_t *)malloc(initial_capacity);
    bs->capacity = initial_capacity;
    bs->pos = 0;
}

static inline void bs_writer_free(BitstreamWriter *bs) {
    free(bs->data);
    bs->data = NULL;
    bs->capacity = 0;
    bs->pos = 0;
}

static inline void bs_writer_ensure(BitstreamWriter *bs, size_t extra) {
    if (bs->pos + extra > bs->capacity) {
        size_t new_cap = bs->capacity * 2;
        while (new_cap < bs->pos + extra) new_cap *= 2;
        bs->data = (uint8_t *)realloc(bs->data, new_cap);
        bs->capacity = new_cap;
    }
}

static inline void bs_write_u8(BitstreamWriter *bs, uint8_t v) {
    bs_writer_ensure(bs, 1);
    bs->data[bs->pos++] = v;
}

static inline void bs_write_u16(BitstreamWriter *bs, uint16_t v) {
    bs_writer_ensure(bs, 2);
    bs->data[bs->pos++] = (v >> 8) & 0xFF;
    bs->data[bs->pos++] = v & 0xFF;
}

static inline void bs_write_u24(BitstreamWriter *bs, uint32_t v) {
    bs_writer_ensure(bs, 3);
    bs->data[bs->pos++] = (v >> 16) & 0xFF;
    bs->data[bs->pos++] = (v >> 8) & 0xFF;
    bs->data[bs->pos++] = v & 0xFF;
}

static inline void bs_write_u32(BitstreamWriter *bs, uint32_t v) {
    bs_writer_ensure(bs, 4);
    bs->data[bs->pos++] = (v >> 24) & 0xFF;
    bs->data[bs->pos++] = (v >> 16) & 0xFF;
    bs->data[bs->pos++] = (v >> 8) & 0xFF;
    bs->data[bs->pos++] = v & 0xFF;
}

static inline void bs_write_bytes(BitstreamWriter *bs, const uint8_t *data, size_t n) {
    bs_writer_ensure(bs, n);
    memcpy(bs->data + bs->pos, data, n);
    bs->pos += n;
}

/* Patch a value at a specific position (for backpatching sizes) */
static inline void bs_patch_u24(BitstreamWriter *bs, size_t offset, uint32_t v) {
    bs->data[offset]     = (v >> 16) & 0xFF;
    bs->data[offset + 1] = (v >> 8) & 0xFF;
    bs->data[offset + 2] = v & 0xFF;
}

static inline void bs_patch_u16(BitstreamWriter *bs, size_t offset, uint16_t v) {
    bs->data[offset]     = (v >> 8) & 0xFF;
    bs->data[offset + 1] = v & 0xFF;
}

/* ============================================================================
 * Bypass Bit Reader (for entropy bypass bits)
 * ============================================================================ */

typedef struct {
    const uint8_t *data;
    int byte_pos;
    int bit_idx;  /* 7 = MSB, 0 = LSB */
} BypassReader;

static inline void bypass_reader_init(BypassReader *br, const uint8_t *data, int offset) {
    br->data = data;
    br->byte_pos = offset;
    br->bit_idx = 7;
}

static inline int bypass_read_bit(BypassReader *br) {
    int bit = (br->data[br->byte_pos] >> br->bit_idx) & 1;
    br->bit_idx--;
    if (br->bit_idx < 0) {
        br->bit_idx = 7;
        br->byte_pos++;
    }
    return bit;
}

static inline int bypass_read_bits(BypassReader *br, int n) {
    int v = 0;
    for (int i = 0; i < n; i++)
        v = (v << 1) | bypass_read_bit(br);
    return v;
}

static inline int bypass_read_expgolomb(BypassReader *br) {
    int n = 0;
    while (bypass_read_bit(br) == 0) n++;
    int v = 0;
    for (int i = n - 1; i >= 0; i--)
        v = (v << 1) | bypass_read_bit(br);
    v += (1 << n) - 1;
    return v;
}

/* ============================================================================
 * Bypass Bit Writer
 * ============================================================================ */

typedef struct {
    uint8_t *data;
    size_t capacity;
    int byte_pos;
    int bit_idx;  /* 7 = MSB, 0 = LSB */
} BypassWriter;

static inline void bypass_writer_init(BypassWriter *bw, size_t initial_capacity) {
    bw->data = (uint8_t *)calloc(initial_capacity, 1);
    bw->capacity = initial_capacity;
    bw->byte_pos = 0;
    bw->bit_idx = 7;
}

static inline void bypass_writer_free(BypassWriter *bw) {
    free(bw->data);
    bw->data = NULL;
}

static inline void bypass_writer_ensure(BypassWriter *bw) {
    if ((size_t)bw->byte_pos >= bw->capacity) {
        size_t new_cap = bw->capacity * 2;
        bw->data = (uint8_t *)realloc(bw->data, new_cap);
        memset(bw->data + bw->capacity, 0, new_cap - bw->capacity);
        bw->capacity = new_cap;
    }
}

static inline void bypass_write_bit(BypassWriter *bw, int bit) {
    bypass_writer_ensure(bw);
    if (bit)
        bw->data[bw->byte_pos] |= (1 << bw->bit_idx);
    else
        bw->data[bw->byte_pos] &= ~(1 << bw->bit_idx);
    bw->bit_idx--;
    if (bw->bit_idx < 0) {
        bw->bit_idx = 7;
        bw->byte_pos++;
    }
}

static inline void bypass_write_bits(BypassWriter *bw, int v, int n) {
    for (int i = n - 1; i >= 0; i--)
        bypass_write_bit(bw, (v >> i) & 1);
}

static inline void bypass_write_expgolomb(BypassWriter *bw, int v) {
    /* Exp-Golomb order 0: prefix of n zeros then 1, followed by n bits */
    int val_plus_1 = v + 1;
    int n = 0;
    int tmp = val_plus_1;
    while (tmp > 1) {
        tmp >>= 1;
        n++;
    }
    /* Write n zeros followed by 1 */
    for (int i = 0; i < n; i++)
        bypass_write_bit(bw, 0);
    bypass_write_bit(bw, 1);
    /* Write n suffix bits */
    int suffix = val_plus_1 - (1 << n);
    for (int i = n - 1; i >= 0; i--)
        bypass_write_bit(bw, (suffix >> i) & 1);
}

/* Get the number of bytes used (rounded up) */
static inline size_t bypass_writer_size(BypassWriter *bw) {
    return bw->byte_pos + (bw->bit_idx < 7 ? 1 : 0);
}

/* Flush and get data (copies to new buffer) */
static inline uint8_t *bypass_writer_get_data(BypassWriter *bw, size_t *size) {
    *size = bypass_writer_size(bw);
    uint8_t *out = (uint8_t *)malloc(*size);
    memcpy(out, bw->data, *size);
    return out;
}

#endif /* LATTICE_BITSTREAM_H */
