/*
 * Lattice Video Codec - rANS Encoder/Decoder
 *
 * rANS (range Asymmetric Numeral Systems) implementation with:
 * - Dual-stream interleaving for throughput
 * - Forward-reading stream 0, backward-reading stream 1
 * - CDF-based symbol coding with adaptation
 */

#ifndef LATTICE_RANS_H
#define LATTICE_RANS_H

#include "lattice_common.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ============================================================================
 * rANS Decoder
 * ============================================================================ */

typedef struct {
    const uint8_t *payload;
    uint32_t x;          /* 32-bit state */
    int pos;             /* Current byte position */
    int direction;       /* 1 = forward, -1 = backward */
} RansDecoder;

static inline void rans_dec_init_forward(RansDecoder *rd, const uint8_t *payload) {
    rd->payload = payload;
    rd->x = ((uint32_t)payload[0] << 24) | ((uint32_t)payload[1] << 16) |
            ((uint32_t)payload[2] << 8) | payload[3];
    rd->pos = 4;
    rd->direction = 1;
}

static inline void rans_dec_init_backward(RansDecoder *rd, const uint8_t *payload, int bypass_offset) {
    rd->payload = payload;
    int p = bypass_offset - 1;
    rd->x = ((uint32_t)payload[p] << 24) | ((uint32_t)payload[p-1] << 16) |
            ((uint32_t)payload[p-2] << 8) | payload[p-3];
    rd->pos = p - 4;
    rd->direction = -1;
}

static inline int rans_decode(RansDecoder *rd, CdfContext *ctx) {
    uint32_t q = rd->x >> 16;
    uint32_t r = rd->x & 0xFFFF;

    /* Linear search for symbol where cdf[s] <= r < cdf[s+1] */
    int s = 0;
    while (s < ctx->size - 1 && ctx->cdf[s + 1] <= r) s++;

    uint32_t freq = ctx->cdf[s + 1] - ctx->cdf[s];
    rd->x = q * freq + (r - ctx->cdf[s]);

    /* Renormalise */
    while (rd->x < (1u << 16)) {
        if (rd->direction > 0) {
            rd->x = (rd->x << 8) | rd->payload[rd->pos++];
        } else {
            rd->x = (rd->x << 8) | rd->payload[rd->pos--];
        }
    }

    return s;
}

/* ============================================================================
 * rANS Encoder
 *
 * ANS encoding is a LIFO process - we encode symbols in reverse order.
 * The encoder collects symbols, then processes them backwards to produce
 * the byte stream.
 *
 * For dual-stream encoding:
 * - Stream 0: decoder reads forward, so encoder output is reversed
 * - Stream 1: decoder reads backward, so encoder output is natural order
 * ============================================================================ */

/* Recorded symbol for later encoding */
typedef struct {
    int symbol;
    uint16_t cdf_start;  /* cdf[symbol] at time of encoding */
    uint16_t cdf_freq;   /* cdf[symbol+1] - cdf[symbol] at time of encoding */
} RansSymbol;

typedef struct {
    RansSymbol *symbols;
    size_t count;
    size_t capacity;
} RansSymbolBuffer;

static inline void rans_sym_init(RansSymbolBuffer *buf, size_t initial_capacity) {
    buf->symbols = (RansSymbol *)malloc(initial_capacity * sizeof(RansSymbol));
    buf->capacity = initial_capacity;
    buf->count = 0;
}

static inline void rans_sym_free(RansSymbolBuffer *buf) {
    free(buf->symbols);
    buf->symbols = NULL;
    buf->count = buf->capacity = 0;
}

static inline void rans_sym_push(RansSymbolBuffer *buf, int symbol, uint16_t cdf_start, uint16_t cdf_freq) {
    if (buf->count >= buf->capacity) {
        buf->capacity *= 2;
        buf->symbols = (RansSymbol *)realloc(buf->symbols, buf->capacity * sizeof(RansSymbol));
    }
    buf->symbols[buf->count].symbol = symbol;
    buf->symbols[buf->count].cdf_start = cdf_start;
    buf->symbols[buf->count].cdf_freq = cdf_freq;
    buf->count++;
}

static inline void rans_sym_clear(RansSymbolBuffer *buf) {
    buf->count = 0;
}

/* Encoder state for a single stream */
typedef struct {
    uint32_t x;          /* 32-bit state */
    uint8_t *bytes;      /* Output bytes (in encoding order) */
    size_t byte_count;
    size_t byte_capacity;
} RansEncoder;

static inline void rans_enc_init(RansEncoder *re) {
    re->x = (1u << 23);  /* Initial state in valid range [2^16, 2^24) */
    re->bytes = (uint8_t *)malloc(4096);
    re->byte_count = 0;
    re->byte_capacity = 4096;
}

static inline void rans_enc_free(RansEncoder *re) {
    free(re->bytes);
    re->bytes = NULL;
}

static inline void rans_enc_emit_byte(RansEncoder *re, uint8_t b) {
    if (re->byte_count >= re->byte_capacity) {
        re->byte_capacity *= 2;
        re->bytes = (uint8_t *)realloc(re->bytes, re->byte_capacity);
    }
    re->bytes[re->byte_count++] = b;
}

/*
 * Encode a single symbol (must be called in REVERSE order of decoding)
 *
 * The encoding operation is the inverse of decoding:
 * Decode: x' = (x >> 16) * freq + (x & 0xFFFF) - start
 * Encode: x = ((x' / freq) << 16) + (x' % freq) + start
 *
 * For byte-at-a-time rANS with L=2^16 and state in [L, L*256):
 * Renormalize by emitting bytes while x >= freq * 256
 */
static inline void rans_enc_put(RansEncoder *re, uint16_t cdf_start, uint16_t cdf_freq) {
    uint32_t x = re->x;

    /* Renormalize: emit bytes while x would overflow after encoding
     * For state range [2^16, 2^24), renorm when x >= freq * 256 */
    uint32_t x_max = (uint32_t)cdf_freq << 8;  /* freq * 256 */
    while (x >= x_max) {
        rans_enc_emit_byte(re, x & 0xFF);
        x >>= 8;
    }

    /* Encode: x = (x / freq) << 16 + x % freq + start */
    re->x = ((x / cdf_freq) << 16) + (x % cdf_freq) + cdf_start;
}

/* Flush final state (4 bytes) - emits LSB first since encoding is LIFO */
static inline void rans_enc_flush(RansEncoder *re) {
    uint32_t x = re->x;
    /* Emit all 4 bytes of state - will be reversed for forward-reading decoder */
    rans_enc_emit_byte(re, x & 0xFF);
    rans_enc_emit_byte(re, (x >> 8) & 0xFF);
    rans_enc_emit_byte(re, (x >> 16) & 0xFF);
    rans_enc_emit_byte(re, (x >> 24) & 0xFF);
}

/* Get the encoder's byte buffer (still in encoding order, not reversed) */
static inline const uint8_t *rans_enc_get_bytes(RansEncoder *re, size_t *count) {
    *count = re->byte_count;
    return re->bytes;
}

/* ============================================================================
 * Dual-Stream rANS Encoder
 *
 * Manages interleaved encoding to two streams, matching decoder's alternation.
 * ============================================================================ */

typedef struct {
    RansSymbolBuffer stream0_syms;  /* Symbols for stream 0 */
    RansSymbolBuffer stream1_syms;  /* Symbols for stream 1 */
    int toggle;                      /* 0 or 1, which stream gets next symbol */
} DualRansEncoder;

static int _rans_enc_sym_counter = 0;

static inline void dual_rans_enc_init(DualRansEncoder *dre) {
    rans_sym_init(&dre->stream0_syms, 4096);
    rans_sym_init(&dre->stream1_syms, 4096);
    dre->toggle = 0;
    _rans_enc_sym_counter = 0;  /* Reset debug counter */
}

static inline void dual_rans_enc_free(DualRansEncoder *dre) {
    rans_sym_free(&dre->stream0_syms);
    rans_sym_free(&dre->stream1_syms);
}

static inline void dual_rans_enc_reset(DualRansEncoder *dre) {
    rans_sym_clear(&dre->stream0_syms);
    rans_sym_clear(&dre->stream1_syms);
    dre->toggle = 0;
}

/* Record a symbol for encoding (call in forward order, same as decoding) */
static inline void dual_rans_enc_put(DualRansEncoder *dre, int symbol, CdfContext *ctx) {
    uint16_t cdf_start = ctx->cdf[symbol];
    uint16_t cdf_freq = ctx->cdf[symbol + 1] - ctx->cdf[symbol];

    /* fprintf(stderr, "  ENC[%d]: sym=%d stream=%d cdf_start=%d freq=%d\n",
            _rans_enc_sym_counter++, symbol, dre->toggle, cdf_start, cdf_freq); */
    (void)_rans_enc_sym_counter;

    if (dre->toggle == 0) {
        rans_sym_push(&dre->stream0_syms, symbol, cdf_start, cdf_freq);
    } else {
        rans_sym_push(&dre->stream1_syms, symbol, cdf_start, cdf_freq);
    }
    dre->toggle ^= 1;

    /* Adapt the CDF (same as decoder) */
    cdf_adapt(ctx, symbol);
}

/*
 * Finalize encoding and produce the interleaved byte stream.
 * Returns allocated buffer that caller must free.
 *
 * Layout:
 *   [stream0 bytes, reversed] [stream1 bytes, natural order]
 *   |<-- stream0 reads fwd -->|<-- stream1 reads backward --|
 *
 * The bypass_offset returned is where stream1 data begins (= stream0 size).
 */
static inline uint8_t *dual_rans_enc_finish(DualRansEncoder *dre,
                                             size_t *out_size,
                                             size_t *bypass_offset) {
    RansEncoder enc0, enc1;
    rans_enc_init(&enc0);
    rans_enc_init(&enc1);

    /* Encode stream 0 symbols in reverse order */
    for (int i = (int)dre->stream0_syms.count - 1; i >= 0; i--) {
        RansSymbol *sym = &dre->stream0_syms.symbols[i];
        rans_enc_put(&enc0, sym->cdf_start, sym->cdf_freq);
    }
    rans_enc_flush(&enc0);

    /* Encode stream 1 symbols in reverse order */
    for (int i = (int)dre->stream1_syms.count - 1; i >= 0; i--) {
        RansSymbol *sym = &dre->stream1_syms.symbols[i];
        rans_enc_put(&enc1, sym->cdf_start, sym->cdf_freq);
    }
    rans_enc_flush(&enc1);

    /* Stream 0: decoder reads forward, so reverse the encoder output
     * Stream 1: decoder reads backward, so keep encoder output as-is */

    size_t s0_size = enc0.byte_count;
    size_t s1_size = enc1.byte_count;

    uint8_t *output = (uint8_t *)malloc(s0_size + s1_size);

    /* Copy stream 0 reversed (encoder emits LSB first, decoder expects big-endian init) */
    for (size_t i = 0; i < s0_size; i++) {
        output[i] = enc0.bytes[s0_size - 1 - i];
    }

    /* Copy stream 1 as-is (decoder reads backward starting from end) */
    memcpy(output + s0_size, enc1.bytes, s1_size);

    *out_size = s0_size + s1_size;
    *bypass_offset = s0_size + s1_size;  /* Bypass starts after both streams */

    rans_enc_free(&enc0);
    rans_enc_free(&enc1);

    return output;
}

/* ============================================================================
 * Dual-Stream rANS Decoder (convenience wrapper)
 * ============================================================================ */

typedef struct {
    RansDecoder stream[2];
    int toggle;
} DualRansDecoder;

static int _rans_dec_sym_counter = 0;

static inline void dual_rans_dec_init(DualRansDecoder *drd,
                                       const uint8_t *payload,
                                       int bypass_offset) {
    rans_dec_init_forward(&drd->stream[0], payload);
    rans_dec_init_backward(&drd->stream[1], payload, bypass_offset);
    drd->toggle = 0;
    _rans_dec_sym_counter = 0;  /* Reset debug counter */
}

static inline int dual_rans_dec_get(DualRansDecoder *drd, CdfContext *ctx) {
    int sym = rans_decode(&drd->stream[drd->toggle], ctx);
    /* uint16_t cdf_start = ctx->cdf[sym];
    uint16_t cdf_freq = ctx->cdf[sym + 1] - ctx->cdf[sym];
    fprintf(stderr, "  DEC[%d]: sym=%d stream=%d cdf_start=%d freq=%d\n",
            _rans_dec_sym_counter++, sym, drd->toggle, cdf_start, cdf_freq); */
    (void)_rans_dec_sym_counter;
    cdf_adapt(ctx, sym);
    drd->toggle ^= 1;
    return sym;
}

#endif /* LATTICE_RANS_H */
