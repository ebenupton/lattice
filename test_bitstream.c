/*
 * Lattice Video Codec - Bitstream Encoder/Decoder Soak Test
 *
 * Tests:
 * 1. Bitstream writer/reader round-trip for fixed-width fields
 * 2. Bypass bit writer/reader round-trip
 * 3. rANS single-stream encode/decode round-trip
 * 4. Dual-stream rANS encode/decode round-trip with CDF adaptation
 * 5. Mixed rANS + bypass encoding (simulating real tile encoding)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>

#include "lattice_common.h"
#include "lattice_bitstream.h"
#include "lattice_rans.h"

/* ============================================================================
 * Test Utilities
 * ============================================================================ */

static int g_test_count = 0;
static int g_pass_count = 0;

#define TEST_START(name) \
    do { g_test_count++; printf("  [%d] %s... ", g_test_count, name); fflush(stdout); } while(0)

#define TEST_PASS() \
    do { g_pass_count++; printf("PASS\n"); } while(0)

#define TEST_FAIL(msg) \
    do { printf("FAIL: %s\n", msg); } while(0)

#define ASSERT_EQ(a, b, msg) \
    do { if ((a) != (b)) { printf("FAIL: %s (expected %d, got %d)\n", msg, (int)(b), (int)(a)); return 0; } } while(0)

/* ============================================================================
 * Test 1: Bitstream Writer/Reader
 * ============================================================================ */

static int test_bitstream_basic(void) {
    TEST_START("Bitstream basic read/write");

    BitstreamWriter bsw;
    bs_writer_init(&bsw, 64);

    /* Write various sizes */
    bs_write_u8(&bsw, 0xAB);
    bs_write_u16(&bsw, 0xCDEF);
    bs_write_u24(&bsw, 0x123456);
    bs_write_u32(&bsw, 0xDEADBEEF);
    bs_write_u8(&bsw, 0x42);

    /* Read back */
    BitstreamReader bsr;
    bs_reader_init(&bsr, bsw.data, bsw.pos);

    ASSERT_EQ(bs_read_u8(&bsr), 0xAB, "u8");
    ASSERT_EQ(bs_read_u16(&bsr), 0xCDEF, "u16");
    ASSERT_EQ(bs_read_u24(&bsr), 0x123456, "u24");
    ASSERT_EQ(bs_read_u32(&bsr), 0xDEADBEEF, "u32");
    ASSERT_EQ(bs_read_u8(&bsr), 0x42, "u8 final");

    bs_writer_free(&bsw);
    TEST_PASS();
    return 1;
}

static int test_bitstream_patch(void) {
    TEST_START("Bitstream backpatching");

    BitstreamWriter bsw;
    bs_writer_init(&bsw, 64);

    /* Write placeholder, then patch */
    size_t patch_pos = bsw.pos;
    bs_write_u24(&bsw, 0);  /* Placeholder */
    bs_write_u16(&bsw, 0);  /* Another placeholder */
    bs_write_u32(&bsw, 0x12345678);

    /* Patch the placeholders */
    bs_patch_u24(&bsw, patch_pos, 0xABCDEF);
    bs_patch_u16(&bsw, patch_pos + 3, 0x9876);

    /* Verify */
    BitstreamReader bsr;
    bs_reader_init(&bsr, bsw.data, bsw.pos);

    ASSERT_EQ(bs_read_u24(&bsr), 0xABCDEF, "patched u24");
    ASSERT_EQ(bs_read_u16(&bsr), 0x9876, "patched u16");
    ASSERT_EQ(bs_read_u32(&bsr), 0x12345678, "unpatched u32");

    bs_writer_free(&bsw);
    TEST_PASS();
    return 1;
}

/* ============================================================================
 * Test 2: Bypass Bit Writer/Reader
 * ============================================================================ */

static int test_bypass_bits(void) {
    TEST_START("Bypass single bits");

    BypassWriter bpw;
    bypass_writer_init(&bpw, 64);

    /* Write specific bit pattern: 1,0,1,1,0,0,1,0 = 0xB2 */
    bypass_write_bit(&bpw, 1);
    bypass_write_bit(&bpw, 0);
    bypass_write_bit(&bpw, 1);
    bypass_write_bit(&bpw, 1);
    bypass_write_bit(&bpw, 0);
    bypass_write_bit(&bpw, 0);
    bypass_write_bit(&bpw, 1);
    bypass_write_bit(&bpw, 0);

    /* Add a few more bits across byte boundary */
    bypass_write_bit(&bpw, 1);
    bypass_write_bit(&bpw, 1);
    bypass_write_bit(&bpw, 0);
    bypass_write_bit(&bpw, 1);

    size_t size;
    uint8_t *data = bypass_writer_get_data(&bpw, &size);

    /* Read back */
    BypassReader bpr;
    bypass_reader_init(&bpr, data, 0);

    ASSERT_EQ(bypass_read_bit(&bpr), 1, "bit 0");
    ASSERT_EQ(bypass_read_bit(&bpr), 0, "bit 1");
    ASSERT_EQ(bypass_read_bit(&bpr), 1, "bit 2");
    ASSERT_EQ(bypass_read_bit(&bpr), 1, "bit 3");
    ASSERT_EQ(bypass_read_bit(&bpr), 0, "bit 4");
    ASSERT_EQ(bypass_read_bit(&bpr), 0, "bit 5");
    ASSERT_EQ(bypass_read_bit(&bpr), 1, "bit 6");
    ASSERT_EQ(bypass_read_bit(&bpr), 0, "bit 7");
    ASSERT_EQ(bypass_read_bit(&bpr), 1, "bit 8");
    ASSERT_EQ(bypass_read_bit(&bpr), 1, "bit 9");
    ASSERT_EQ(bypass_read_bit(&bpr), 0, "bit 10");
    ASSERT_EQ(bypass_read_bit(&bpr), 1, "bit 11");

    free(data);
    bypass_writer_free(&bpw);
    TEST_PASS();
    return 1;
}

static int test_bypass_multi_bits(void) {
    TEST_START("Bypass multi-bit fields");

    BypassWriter bpw;
    bypass_writer_init(&bpw, 64);

    bypass_write_bits(&bpw, 0x5, 3);   /* 101 */
    bypass_write_bits(&bpw, 0xAB, 8);  /* 10101011 */
    bypass_write_bits(&bpw, 0x3, 2);   /* 11 */
    bypass_write_bits(&bpw, 0x1234, 13); /* Full 13-bit value */

    size_t size;
    uint8_t *data = bypass_writer_get_data(&bpw, &size);

    BypassReader bpr;
    bypass_reader_init(&bpr, data, 0);

    ASSERT_EQ(bypass_read_bits(&bpr, 3), 0x5, "3-bit");
    ASSERT_EQ(bypass_read_bits(&bpr, 8), 0xAB, "8-bit");
    ASSERT_EQ(bypass_read_bits(&bpr, 2), 0x3, "2-bit");
    ASSERT_EQ(bypass_read_bits(&bpr, 13), 0x1234, "13-bit");

    free(data);
    bypass_writer_free(&bpw);
    TEST_PASS();
    return 1;
}

static int test_bypass_expgolomb(void) {
    TEST_START("Bypass exp-golomb coding");

    BypassWriter bpw;
    bypass_writer_init(&bpw, 64);

    /* Test various values */
    int test_values[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 31, 32, 63, 100, 255, 1000};
    int num_values = sizeof(test_values) / sizeof(test_values[0]);

    for (int i = 0; i < num_values; i++) {
        bypass_write_expgolomb(&bpw, test_values[i]);
    }

    size_t size;
    uint8_t *data = bypass_writer_get_data(&bpw, &size);

    BypassReader bpr;
    bypass_reader_init(&bpr, data, 0);

    for (int i = 0; i < num_values; i++) {
        int v = bypass_read_expgolomb(&bpr);
        if (v != test_values[i]) {
            printf("FAIL: exp-golomb value %d (expected %d, got %d)\n", i, test_values[i], v);
            free(data);
            bypass_writer_free(&bpw);
            return 0;
        }
    }

    free(data);
    bypass_writer_free(&bpw);
    TEST_PASS();
    return 1;
}

/* ============================================================================
 * Test 3: Single-Stream rANS
 * ============================================================================ */

static int test_rans_single_uniform(void) {
    TEST_START("rANS single stream, uniform CDF");

    /* Encode symbols with uniform distribution */
    CdfContext ctx_enc, ctx_dec;
    cdf_init_uniform(&ctx_enc, 8);
    cdf_init_uniform(&ctx_dec, 8);

    int test_symbols[] = {0, 1, 2, 3, 4, 5, 6, 7, 0, 7, 3, 3, 3, 5, 2, 1};
    int num_symbols = sizeof(test_symbols) / sizeof(test_symbols[0]);

    /* Collect symbols with their CDF state */
    RansSymbolBuffer syms;
    rans_sym_init(&syms, 64);

    for (int i = 0; i < num_symbols; i++) {
        int s = test_symbols[i];
        rans_sym_push(&syms, s, ctx_enc.cdf[s], ctx_enc.cdf[s+1] - ctx_enc.cdf[s]);
        cdf_adapt(&ctx_enc, s);
    }

    /* Encode in reverse */
    RansEncoder enc;
    rans_enc_init(&enc);

    for (int i = num_symbols - 1; i >= 0; i--) {
        rans_enc_put(&enc, syms.symbols[i].cdf_start, syms.symbols[i].cdf_freq);
    }
    rans_enc_flush(&enc);

    /* Reverse output for forward-reading decoder */
    uint8_t *encoded = (uint8_t *)malloc(enc.byte_count);
    for (size_t i = 0; i < enc.byte_count; i++) {
        encoded[i] = enc.bytes[enc.byte_count - 1 - i];
    }

    /* Decode */
    RansDecoder dec;
    rans_dec_init_forward(&dec, encoded);

    for (int i = 0; i < num_symbols; i++) {
        int s = rans_decode(&dec, &ctx_dec);
        cdf_adapt(&ctx_dec, s);
        if (s != test_symbols[i]) {
            printf("FAIL: symbol %d (expected %d, got %d)\n", i, test_symbols[i], s);
            free(encoded);
            rans_enc_free(&enc);
            rans_sym_free(&syms);
            return 0;
        }
    }

    free(encoded);
    rans_enc_free(&enc);
    rans_sym_free(&syms);
    TEST_PASS();
    return 1;
}

/* ============================================================================
 * Test 4: Dual-Stream rANS
 * ============================================================================ */

static int test_dual_rans_basic(void) {
    TEST_START("Dual rANS, basic interleaved");

    CdfContext ctx_enc, ctx_dec;
    cdf_init_uniform(&ctx_enc, 4);
    cdf_init_uniform(&ctx_dec, 4);

    int test_symbols[] = {0, 1, 2, 3, 0, 1, 2, 3, 3, 2, 1, 0, 0, 0, 1, 1};
    int num_symbols = sizeof(test_symbols) / sizeof(test_symbols[0]);

    DualRansEncoder dre;
    dual_rans_enc_init(&dre);

    for (int i = 0; i < num_symbols; i++) {
        dual_rans_enc_put(&dre, test_symbols[i], &ctx_enc);
    }

    size_t rans_size, bypass_offset;
    uint8_t *encoded = dual_rans_enc_finish(&dre, &rans_size, &bypass_offset);

    /* Decode */
    DualRansDecoder drd;
    dual_rans_dec_init(&drd, encoded, bypass_offset);

    for (int i = 0; i < num_symbols; i++) {
        int s = dual_rans_dec_get(&drd, &ctx_dec);
        if (s != test_symbols[i]) {
            printf("FAIL: symbol %d (expected %d, got %d)\n", i, test_symbols[i], s);
            free(encoded);
            dual_rans_enc_free(&dre);
            return 0;
        }
    }

    free(encoded);
    dual_rans_enc_free(&dre);
    TEST_PASS();
    return 1;
}

static int test_dual_rans_random(void) {
    TEST_START("Dual rANS, random stress test");

    srand(12345);  /* Deterministic seed */

    for (int trial = 0; trial < 100; trial++) {
        int alphabet_size = 2 + (rand() % 15);  /* 2-16 symbols */
        int num_symbols = 100 + (rand() % 1000);

        CdfContext ctx_enc, ctx_dec;
        cdf_init_uniform(&ctx_enc, alphabet_size);
        cdf_init_uniform(&ctx_dec, alphabet_size);

        int *test_symbols = (int *)malloc(num_symbols * sizeof(int));
        for (int i = 0; i < num_symbols; i++) {
            test_symbols[i] = rand() % alphabet_size;
        }

        DualRansEncoder dre;
        dual_rans_enc_init(&dre);

        for (int i = 0; i < num_symbols; i++) {
            dual_rans_enc_put(&dre, test_symbols[i], &ctx_enc);
        }

        size_t rans_size, bypass_offset;
        uint8_t *encoded = dual_rans_enc_finish(&dre, &rans_size, &bypass_offset);

        /* Decode */
        DualRansDecoder drd;
        dual_rans_dec_init(&drd, encoded, bypass_offset);

        for (int i = 0; i < num_symbols; i++) {
            int s = dual_rans_dec_get(&drd, &ctx_dec);
            if (s != test_symbols[i]) {
                printf("FAIL: trial %d, symbol %d (expected %d, got %d)\n",
                       trial, i, test_symbols[i], s);
                free(test_symbols);
                free(encoded);
                dual_rans_enc_free(&dre);
                return 0;
            }
        }

        free(test_symbols);
        free(encoded);
        dual_rans_enc_free(&dre);
    }

    TEST_PASS();
    return 1;
}

/* ============================================================================
 * Test 5: Multiple Context Types (simulating real encoding)
 * ============================================================================ */

static int test_multi_context(void) {
    TEST_START("Multiple contexts (simulating tile)");

    /* Simulate encoding with different context types like real tiles */
    CdfContext shape_ctx_enc, mode_ctx_enc, cbf_ctx_enc;
    CdfContext shape_ctx_dec, mode_ctx_dec, cbf_ctx_dec;

    cdf_init_uniform(&shape_ctx_enc, 7);
    cdf_init_uniform(&mode_ctx_enc, 3);
    cdf_init_uniform(&cbf_ctx_enc, 2);
    cdf_init_uniform(&shape_ctx_dec, 7);
    cdf_init_uniform(&mode_ctx_dec, 3);
    cdf_init_uniform(&cbf_ctx_dec, 2);

    /* Generate test data */
    int num_blocks = 50;
    int *shapes = (int *)malloc(num_blocks * sizeof(int));
    int *modes = (int *)malloc(num_blocks * sizeof(int));
    int *cbfs = (int *)malloc(num_blocks * sizeof(int));

    srand(54321);
    for (int i = 0; i < num_blocks; i++) {
        shapes[i] = rand() % 7;
        modes[i] = rand() % 3;
        cbfs[i] = rand() % 2;
    }

    /* Encode */
    DualRansEncoder dre;
    dual_rans_enc_init(&dre);

    for (int i = 0; i < num_blocks; i++) {
        dual_rans_enc_put(&dre, shapes[i], &shape_ctx_enc);
        dual_rans_enc_put(&dre, modes[i], &mode_ctx_enc);
        dual_rans_enc_put(&dre, cbfs[i], &cbf_ctx_enc);
    }

    size_t rans_size, bypass_offset;
    uint8_t *encoded = dual_rans_enc_finish(&dre, &rans_size, &bypass_offset);

    /* Decode */
    DualRansDecoder drd;
    dual_rans_dec_init(&drd, encoded, bypass_offset);

    for (int i = 0; i < num_blocks; i++) {
        int shape = dual_rans_dec_get(&drd, &shape_ctx_dec);
        int mode = dual_rans_dec_get(&drd, &mode_ctx_dec);
        int cbf = dual_rans_dec_get(&drd, &cbf_ctx_dec);

        if (shape != shapes[i] || mode != modes[i] || cbf != cbfs[i]) {
            printf("FAIL: block %d mismatch\n", i);
            free(shapes);
            free(modes);
            free(cbfs);
            free(encoded);
            dual_rans_enc_free(&dre);
            return 0;
        }
    }

    free(shapes);
    free(modes);
    free(cbfs);
    free(encoded);
    dual_rans_enc_free(&dre);
    TEST_PASS();
    return 1;
}

/* ============================================================================
 * Test 6: Simulated Tile Payload (rANS + bypass combined)
 * ============================================================================ */

typedef struct {
    int shape;
    int mode;
    int cbf;
    int mv_sign_x;  /* bypass */
    int mv_sign_y;  /* bypass */
    int escape_val; /* bypass exp-golomb */
} TestBlock;

static int test_tile_simulation(void) {
    TEST_START("Simulated tile payload (rANS + bypass)");

    srand(99999);

    int num_blocks = 100;
    TestBlock *blocks = (TestBlock *)malloc(num_blocks * sizeof(TestBlock));

    for (int i = 0; i < num_blocks; i++) {
        blocks[i].shape = rand() % 7;
        blocks[i].mode = rand() % 3;
        blocks[i].cbf = rand() % 2;
        blocks[i].mv_sign_x = rand() % 2;
        blocks[i].mv_sign_y = rand() % 2;
        blocks[i].escape_val = rand() % 256;
    }

    /* === ENCODE === */
    CdfContext shape_ctx_enc, mode_ctx_enc, cbf_ctx_enc;
    cdf_init_uniform(&shape_ctx_enc, 7);
    cdf_init_uniform(&mode_ctx_enc, 3);
    cdf_init_uniform(&cbf_ctx_enc, 2);

    DualRansEncoder dre;
    dual_rans_enc_init(&dre);

    BypassWriter bpw;
    bypass_writer_init(&bpw, 1024);

    for (int i = 0; i < num_blocks; i++) {
        dual_rans_enc_put(&dre, blocks[i].shape, &shape_ctx_enc);
        dual_rans_enc_put(&dre, blocks[i].mode, &mode_ctx_enc);
        dual_rans_enc_put(&dre, blocks[i].cbf, &cbf_ctx_enc);

        /* Bypass data */
        bypass_write_bit(&bpw, blocks[i].mv_sign_x);
        bypass_write_bit(&bpw, blocks[i].mv_sign_y);
        bypass_write_expgolomb(&bpw, blocks[i].escape_val);
    }

    /* Finalize rANS */
    size_t rans_size, rans_bypass_offset;
    uint8_t *rans_data = dual_rans_enc_finish(&dre, &rans_size, &rans_bypass_offset);

    /* Get bypass data */
    size_t bypass_size;
    uint8_t *bypass_data = bypass_writer_get_data(&bpw, &bypass_size);

    /* Build tile payload: [rans_data][bypass_data]
     * bypass_offset = rans_size */
    size_t payload_size = rans_size + bypass_size;
    uint8_t *payload = (uint8_t *)malloc(payload_size);
    memcpy(payload, rans_data, rans_size);
    memcpy(payload + rans_size, bypass_data, bypass_size);
    size_t actual_bypass_offset = rans_size;

    /* === DECODE === */
    CdfContext shape_ctx_dec, mode_ctx_dec, cbf_ctx_dec;
    cdf_init_uniform(&shape_ctx_dec, 7);
    cdf_init_uniform(&mode_ctx_dec, 3);
    cdf_init_uniform(&cbf_ctx_dec, 2);

    DualRansDecoder drd;
    dual_rans_dec_init(&drd, payload, actual_bypass_offset);

    BypassReader bpr;
    bypass_reader_init(&bpr, payload, actual_bypass_offset);

    for (int i = 0; i < num_blocks; i++) {
        int shape = dual_rans_dec_get(&drd, &shape_ctx_dec);
        int mode = dual_rans_dec_get(&drd, &mode_ctx_dec);
        int cbf = dual_rans_dec_get(&drd, &cbf_ctx_dec);

        int mv_sign_x = bypass_read_bit(&bpr);
        int mv_sign_y = bypass_read_bit(&bpr);
        int escape_val = bypass_read_expgolomb(&bpr);

        if (shape != blocks[i].shape ||
            mode != blocks[i].mode ||
            cbf != blocks[i].cbf ||
            mv_sign_x != blocks[i].mv_sign_x ||
            mv_sign_y != blocks[i].mv_sign_y ||
            escape_val != blocks[i].escape_val) {
            printf("FAIL: block %d mismatch\n", i);
            printf("  shape: %d vs %d\n", shape, blocks[i].shape);
            printf("  mode: %d vs %d\n", mode, blocks[i].mode);
            printf("  cbf: %d vs %d\n", cbf, blocks[i].cbf);
            printf("  mv_sign_x: %d vs %d\n", mv_sign_x, blocks[i].mv_sign_x);
            printf("  mv_sign_y: %d vs %d\n", mv_sign_y, blocks[i].mv_sign_y);
            printf("  escape_val: %d vs %d\n", escape_val, blocks[i].escape_val);
            free(blocks);
            free(rans_data);
            free(bypass_data);
            free(payload);
            dual_rans_enc_free(&dre);
            bypass_writer_free(&bpw);
            return 0;
        }
    }

    free(blocks);
    free(rans_data);
    free(bypass_data);
    free(payload);
    dual_rans_enc_free(&dre);
    bypass_writer_free(&bpw);
    TEST_PASS();
    return 1;
}

/* ============================================================================
 * Test 7: CDF Adaptation Consistency
 * ============================================================================ */

static int test_cdf_adaptation(void) {
    TEST_START("CDF adaptation consistency");

    /* Verify encoder and decoder CDF states stay in sync */
    CdfContext ctx1, ctx2;
    cdf_init_uniform(&ctx1, 8);
    cdf_init_uniform(&ctx2, 8);

    /* Apply same sequence of adaptations */
    int symbols[] = {0, 0, 0, 1, 1, 2, 7, 7, 7, 7, 3, 3, 4, 5, 6, 0, 0, 0};
    int n = sizeof(symbols) / sizeof(symbols[0]);

    for (int i = 0; i < n; i++) {
        cdf_adapt(&ctx1, symbols[i]);
        cdf_adapt(&ctx2, symbols[i]);
    }

    /* Verify CDFs match */
    for (int i = 0; i <= ctx1.size; i++) {
        if (ctx1.cdf[i] != ctx2.cdf[i]) {
            printf("FAIL: CDF[%d] mismatch (%d vs %d)\n", i, ctx1.cdf[i], ctx2.cdf[i]);
            return 0;
        }
    }

    /* Verify minimum frequency is maintained */
    for (int i = 0; i < ctx1.size; i++) {
        int freq = ctx1.cdf[i+1] - ctx1.cdf[i];
        if (freq < 1) {
            printf("FAIL: frequency for symbol %d is %d (< 1)\n", i, freq);
            return 0;
        }
    }

    TEST_PASS();
    return 1;
}

/* ============================================================================
 * Test 8: Edge Cases
 * ============================================================================ */

static int test_edge_cases(void) {
    TEST_START("Edge cases");

    /* Test with binary alphabet (very common case) */
    CdfContext ctx_enc, ctx_dec;
    cdf_init_uniform(&ctx_enc, 2);
    cdf_init_uniform(&ctx_dec, 2);

    int test_bits[64];
    for (int i = 0; i < 64; i++) {
        test_bits[i] = (i * 7 + 3) % 2;  /* Pseudo-random pattern */
    }

    DualRansEncoder dre;
    dual_rans_enc_init(&dre);

    for (int i = 0; i < 64; i++) {
        dual_rans_enc_put(&dre, test_bits[i], &ctx_enc);
    }

    size_t size, offset;
    uint8_t *encoded = dual_rans_enc_finish(&dre, &size, &offset);

    DualRansDecoder drd;
    dual_rans_dec_init(&drd, encoded, offset);

    for (int i = 0; i < 64; i++) {
        int b = dual_rans_dec_get(&drd, &ctx_dec);
        if (b != test_bits[i]) {
            printf("FAIL: bit %d mismatch\n", i);
            free(encoded);
            dual_rans_enc_free(&dre);
            return 0;
        }
    }

    free(encoded);
    dual_rans_enc_free(&dre);

    /* Test single symbol */
    cdf_init_uniform(&ctx_enc, 5);
    cdf_init_uniform(&ctx_dec, 5);
    dual_rans_enc_init(&dre);
    dual_rans_enc_put(&dre, 3, &ctx_enc);
    encoded = dual_rans_enc_finish(&dre, &size, &offset);
    dual_rans_dec_init(&drd, encoded, offset);
    int s = dual_rans_dec_get(&drd, &ctx_dec);
    if (s != 3) {
        printf("FAIL: single symbol mismatch\n");
        free(encoded);
        dual_rans_enc_free(&dre);
        return 0;
    }
    free(encoded);
    dual_rans_enc_free(&dre);

    TEST_PASS();
    return 1;
}

/* ============================================================================
 * Test 9: Large Payload Stress Test
 * ============================================================================ */

static int test_large_payload(void) {
    TEST_START("Large payload stress test");

    srand(77777);

    int num_symbols = 100000;
    int *symbols = (int *)malloc(num_symbols * sizeof(int));

    CdfContext ctx_enc, ctx_dec;
    cdf_init_uniform(&ctx_enc, 16);
    cdf_init_uniform(&ctx_dec, 16);

    /* Generate biased distribution (more realistic) */
    for (int i = 0; i < num_symbols; i++) {
        int r = rand() % 100;
        if (r < 50) symbols[i] = rand() % 4;       /* 50% low symbols */
        else if (r < 80) symbols[i] = 4 + rand() % 4; /* 30% mid symbols */
        else symbols[i] = 8 + rand() % 8;          /* 20% high symbols */
    }

    DualRansEncoder dre;
    dual_rans_enc_init(&dre);

    for (int i = 0; i < num_symbols; i++) {
        dual_rans_enc_put(&dre, symbols[i], &ctx_enc);
    }

    size_t size, offset;
    uint8_t *encoded = dual_rans_enc_finish(&dre, &size, &offset);

    /* Report compression ratio */
    double raw_bits = num_symbols * 4.0;  /* 4 bits per symbol without compression */
    double compressed_bits = size * 8.0;
    /* printf("    (%.1f%% of raw) ", 100.0 * compressed_bits / raw_bits); */

    DualRansDecoder drd;
    dual_rans_dec_init(&drd, encoded, offset);

    for (int i = 0; i < num_symbols; i++) {
        int s = dual_rans_dec_get(&drd, &ctx_dec);
        if (s != symbols[i]) {
            printf("FAIL: symbol %d mismatch (expected %d, got %d)\n", i, symbols[i], s);
            free(symbols);
            free(encoded);
            dual_rans_enc_free(&dre);
            return 0;
        }
    }

    free(symbols);
    free(encoded);
    dual_rans_enc_free(&dre);
    TEST_PASS();
    return 1;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    printf("Lattice Bitstream Encoder/Decoder Tests\n");
    printf("========================================\n\n");

    printf("Bitstream tests:\n");
    test_bitstream_basic();
    test_bitstream_patch();

    printf("\nBypass bit tests:\n");
    test_bypass_bits();
    test_bypass_multi_bits();
    test_bypass_expgolomb();

    printf("\nSingle-stream rANS tests:\n");
    test_rans_single_uniform();

    printf("\nDual-stream rANS tests:\n");
    test_dual_rans_basic();
    test_dual_rans_random();

    printf("\nIntegration tests:\n");
    test_multi_context();
    test_tile_simulation();
    test_cdf_adaptation();
    test_edge_cases();
    test_large_payload();

    printf("\n========================================\n");
    printf("Results: %d/%d tests passed\n", g_pass_count, g_test_count);

    return (g_pass_count == g_test_count) ? 0 : 1;
}
