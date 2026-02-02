#!/bin/bash
#
# Lattice Codec Test Suite
#
# Combined test suite covering:
#   - Conformance tests: edge cases with odd dimensions, partial tiles, chroma boundaries
#   - Comprehensive tests: various sizes and content patterns
#
# Usage:
#   ./test_codec.sh              Run all tests
#   ./test_codec.sh conformance  Run only conformance tests
#   ./test_codec.sh comprehensive Run only comprehensive tests
#   ./test_codec.sh --keep       Keep test output files
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENCODER="$SCRIPT_DIR/lattice_encode"
DECODER="$SCRIPT_DIR/lattice_decode"
TEST_DIR="$SCRIPT_DIR/test_output"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Parse arguments
TEST_MODE="all"
KEEP_FILES=0

for arg in "$@"; do
    case "$arg" in
        conformance) TEST_MODE="conformance" ;;
        comprehensive) TEST_MODE="comprehensive" ;;
        --keep) KEEP_FILES=1 ;;
    esac
done

# Create test directory
rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"

# =============================================================================
# Conformance Tests - Edge cases with odd dimensions, partial tiles, etc.
# Uses ceiling division for chroma per spec §10.1
# =============================================================================
CONFORMANCE_TESTS=(
    # Category: Odd luma dimensions (tests ceiling division for chroma)
    "odd_width_129 129 64 3 gradient Odd width: chroma_w=(129+1)/2=65"
    "odd_height_65 64 65 3 gradient Odd height: chroma_h=(65+1)/2=33"
    "odd_both_129x65 129 65 3 gradient Both odd: 65x33 chroma"
    "odd_prime_127x131 127 131 3 noise Prime dimensions: 64x66 chroma"
    "odd_small_17x19 17 19 5 gradient Small odd: 9x10 chroma"
    "odd_one_off_255x255 255 255 2 checkerboard One less than 256: 128x128 chroma"

    # Category: Partial tiles (frame not multiple of 128)
    "partial_tile_130x130 130 130 3 gradient Partial: 2 tiles, edge tile is 2x2"
    "partial_tile_129x129 129 129 3 noise Partial: edge tile is 1x1"
    "partial_tile_200x150 200 150 3 stripes Partial: 72x22 edge tile"
    "partial_tile_256x130 256 130 3 gradient 2x2 tiles, bottom row partial"
    "partial_tile_130x256 130 256 3 gradient 2x2 tiles, right column partial"

    # Category: Edge cases combining odd dimensions and partial tiles
    "combo_131x133 131 133 3 noise Odd dims + partial tiles"
    "combo_255x257 255 257 2 gradient Max partial + 1 pixel over tile"
    "combo_257x255 257 255 2 gradient 1 pixel into second tile"

    # Category: Minimum and boundary sizes
    "min_size_16x16 16 16 5 flat Minimum tile size"
    "min_odd_17x17 17 17 5 gradient Minimum odd both dims"
    "single_cell_9x9 9 9 5 noise Single 8x8 cell + 1 pixel partial"
    "two_cells_wide_17x8 17 8 5 gradient Two cells wide, one partial"
    "two_cells_high_8x17 8 17 5 stripes Two cells high, one partial"

    # Category: Specific chroma boundary cases
    "chroma_boundary_126x126 126 126 3 gradient Luma 126, chroma 63 (floor=63, ceil=63)"
    "chroma_boundary_127x127 127 127 3 gradient Luma 127, chroma ceil=64 vs floor=63"
    "chroma_boundary_1x1 1 1 2 flat Minimum: 1x1 luma, 1x1 chroma"
    "chroma_boundary_2x2 2 2 3 flat 2x2 luma, 1x1 chroma"
    "chroma_boundary_3x3 3 3 3 gradient 3x3 luma, 2x2 chroma (ceil)"
)

# =============================================================================
# Comprehensive Tests - Various sizes and patterns
# =============================================================================
COMPREHENSIVE_TESTS=(
    # Standard small sizes
    "standard_64x64 64 64 5 gradient Standard 64x64"
    "standard_128x128 128 128 8 noise Standard 128x128"
    "standard_256x256 256 256 4 checkerboard Standard 256x256"
    "standard_128x64 128 64 6 gradient Wide aspect ratio"
    "standard_64x128 64 128 6 stripes Tall aspect ratio"

    # Even dimensions that exercise chroma subsampling
    "even_66x66 66 66 5 gradient 66x66 (chroma 33x33)"
    "even_130x130 130 130 4 noise 130x130 (chroma 65x65)"
    "even_62x126 62 126 5 checkerboard 62x126 tall"
    "even_126x62 126 62 5 stripes 126x62 wide"
    "even_254x254 254 254 3 gradient Near tile boundary"

    # Small sizes
    "small_34x34 34 34 8 noise 34x34 small"
    "small_18x18 18 18 9 gradient 18x18 very small"
    "small_50x26 50 26 7 checkerboard 50x26 small wide"
    "small_26x50 26 50 7 stripes 26x50 small tall"
    "small_10x10 10 10 5 flat 10x10 tiny"

    # Non-square aspect ratios
    "aspect_194x66 194 66 4 gradient Wide 3:1"
    "aspect_66x194 66 194 4 noise Tall 1:3"
    "aspect_242x98 242 98 3 checkerboard Wide 2.5:1"
    "aspect_98x242 98 242 3 stripes Tall 1:2.5"

    # Minimum standard size
    "min_standard_16x16 16 16 6 noise Minimum 16x16"
)

# =============================================================================
# Test video generation (uses ceiling division for chroma per spec)
# =============================================================================
generate_test_video() {
    local output="$1"
    local width="$2"
    local height="$3"
    local frames="$4"
    local pattern="$5"

    python3 << EOF
import struct
import math
import random

width, height, frames = $width, $height, $frames
pattern = "$pattern"
random.seed(42)  # Reproducible

# Chroma dimensions use ceiling division per spec §10.1
chroma_w = (width + 1) // 2
chroma_h = (height + 1) // 2

with open("$output", "wb") as f:
    for frame_idx in range(frames):
        # Generate luma plane
        y_plane = bytearray(width * height)
        for row in range(height):
            for col in range(width):
                if pattern == "gradient":
                    val = int((col + row + frame_idx * 10) * 255 / max(1, width + height - 2)) % 256
                elif pattern == "noise":
                    base = int((col * 0.3 + row * 0.3) * 255 / max(1, width + height - 2))
                    val = (base + random.randint(-30, 30)) % 256
                elif pattern == "checkerboard":
                    block = max(4, min(width, height) // 8)
                    cx = (col + frame_idx * 2) // block
                    cy = row // block
                    val = 200 if (cx + cy) % 2 == 0 else 50
                elif pattern == "stripes":
                    stripe_width = max(2, height // 8)
                    val = 180 if ((row + frame_idx * 3) // stripe_width) % 2 == 0 else 60
                elif pattern == "flat":
                    val = 128 + frame_idx * 5
                else:
                    val = 128
                y_plane[row * width + col] = max(0, min(255, val))

        # Generate Cb plane (ceiling division for dimensions)
        cb_plane = bytearray(chroma_w * chroma_h)
        for row in range(chroma_h):
            for col in range(chroma_w):
                if pattern == "noise":
                    val = 128 + random.randint(-20, 20)
                else:
                    val = 128 + int(20 * math.sin(frame_idx * 0.5 + col * 0.1))
                cb_plane[row * chroma_w + col] = max(0, min(255, val))

        # Generate Cr plane (ceiling division for dimensions)
        cr_plane = bytearray(chroma_w * chroma_h)
        for row in range(chroma_h):
            for col in range(chroma_w):
                if pattern == "noise":
                    val = 128 + random.randint(-20, 20)
                else:
                    val = 128 + int(20 * math.cos(frame_idx * 0.5 + row * 0.1))
                cr_plane[row * chroma_w + col] = max(0, min(255, val))

        f.write(y_plane)
        f.write(cb_plane)
        f.write(cr_plane)
EOF
}

# =============================================================================
# PSNR computation (uses ceiling division for chroma)
# =============================================================================
compute_psnr() {
    local file1="$1"
    local file2="$2"
    local width="$3"
    local height="$4"
    local frames="$5"

    python3 << EOF
import struct
import math

width, height, frames = $width, $height, $frames
luma_size = width * height
# Ceiling division for chroma per spec §10.1
chroma_w = (width + 1) // 2
chroma_h = (height + 1) // 2
chroma_size = chroma_w * chroma_h

with open("$file1", "rb") as f1, open("$file2", "rb") as f2:
    total_mse_y = 0
    total_mse_cb = 0
    total_mse_cr = 0
    total_y_samples = 0
    total_c_samples = 0

    for f_idx in range(frames):
        # Read luma
        y1 = f1.read(luma_size)
        y2 = f2.read(luma_size)
        if len(y1) != luma_size or len(y2) != luma_size:
            print(f"ERROR: Truncated luma at frame {f_idx}")
            break

        mse = sum((a - b) ** 2 for a, b in zip(y1, y2))
        total_mse_y += mse
        total_y_samples += luma_size

        # Read Cb
        cb1 = f1.read(chroma_size)
        cb2 = f2.read(chroma_size)
        if len(cb1) != chroma_size or len(cb2) != chroma_size:
            print(f"ERROR: Truncated Cb at frame {f_idx} (expected {chroma_size}, got {len(cb1)}/{len(cb2)})")
            break
        mse = sum((a - b) ** 2 for a, b in zip(cb1, cb2))
        total_mse_cb += mse

        # Read Cr
        cr1 = f1.read(chroma_size)
        cr2 = f2.read(chroma_size)
        if len(cr1) != chroma_size or len(cr2) != chroma_size:
            print(f"ERROR: Truncated Cr at frame {f_idx}")
            break
        mse = sum((a - b) ** 2 for a, b in zip(cr1, cr2))
        total_mse_cr += mse
        total_c_samples += chroma_size

    if total_y_samples == 0:
        print("PSNR: ERROR")
    else:
        mse_y = total_mse_y / total_y_samples
        mse_cb = total_mse_cb / total_c_samples if total_c_samples > 0 else 0
        mse_cr = total_mse_cr / total_c_samples if total_c_samples > 0 else 0

        psnr_y = 10 * math.log10(255**2 / mse_y) if mse_y > 0 else 99.99
        psnr_cb = 10 * math.log10(255**2 / mse_cb) if mse_cb > 0 else 99.99
        psnr_cr = 10 * math.log10(255**2 / mse_cr) if mse_cr > 0 else 99.99

        psnr_avg = (6 * psnr_y + psnr_cb + psnr_cr) / 8

        print(f"PSNR Y={psnr_y:.2f} Cb={psnr_cb:.2f} Cr={psnr_cr:.2f} Avg={psnr_avg:.2f}")
EOF
}

# =============================================================================
# Verify file sizes match expected dimensions
# =============================================================================
verify_file_size() {
    local file="$1"
    local width="$2"
    local height="$3"
    local frames="$4"

    python3 << EOF
import os
width, height, frames = $width, $height, $frames
luma_size = width * height
chroma_w = (width + 1) // 2
chroma_h = (height + 1) // 2
chroma_size = chroma_w * chroma_h
expected = (luma_size + 2 * chroma_size) * frames
actual = os.path.getsize("$file")
if actual == expected:
    print("SIZE_OK")
else:
    print(f"SIZE_MISMATCH expected={expected} actual={actual}")
EOF
}

# =============================================================================
# Compare two files for exact match
# =============================================================================
files_identical() {
    local file1="$1"
    local file2="$2"

    if cmp -s "$file1" "$file2"; then
        return 0
    else
        return 1
    fi
}

# =============================================================================
# Show diff details when files mismatch
# =============================================================================
show_diff_details() {
    local recon_yuv="$1"
    local decoded_yuv="$2"

    recon_size=$(stat -f%z "$recon_yuv" 2>/dev/null || stat -c%s "$recon_yuv")
    decoded_size=$(stat -f%z "$decoded_yuv" 2>/dev/null || stat -c%s "$decoded_yuv")
    echo "    Recon size: $recon_size, Decoded size: $decoded_size"

    if [ "$recon_size" -eq "$decoded_size" ]; then
        python3 << EOF
with open("$recon_yuv", "rb") as f1, open("$decoded_yuv", "rb") as f2:
    pos = 0
    diffs = 0
    while True:
        b1 = f1.read(1)
        b2 = f2.read(1)
        if not b1 or not b2:
            break
        if b1 != b2:
            if diffs < 5:
                print(f"    Diff at byte {pos}: recon={ord(b1)}, decoded={ord(b2)}")
            diffs += 1
        pos += 1
    print(f"    Total differing bytes: {diffs}")
EOF
    fi
}

# =============================================================================
# Run a single test - returns result via TEST_RESULT global variable
# =============================================================================
run_test() {
    local test_config="$1"
    local is_conformance="$2"
    local min_psnr="$3"

    read -r name width height frames pattern description <<< "$test_config"

    echo "----------------------------------------"
    echo "Test: $name"
    if [ -n "$description" ]; then
        echo "  $description"
    fi
    echo "  Dimensions: ${width}x${height}, Frames: $frames, Pattern: $pattern"

    # Calculate and display chroma dimensions for conformance tests
    if [ "$is_conformance" = "1" ]; then
        chroma_w=$(( (width + 1) / 2 ))
        chroma_h=$(( (height + 1) / 2 ))
        echo "  Chroma: ${chroma_w}x${chroma_h} (ceiling division)"
    fi

    input_yuv="$TEST_DIR/${name}_input.yuv"
    encoded_lat="$TEST_DIR/${name}.lat"
    recon_yuv="$TEST_DIR/${name}_recon.yuv"
    decoded_yuv="$TEST_DIR/${name}_decoded.yuv"

    # Generate test video
    echo -n "  Generating... "
    if ! generate_test_video "$input_yuv" "$width" "$height" "$frames" "$pattern" 2>/dev/null; then
        echo -e "${RED}FAILED${NC}"
        TEST_RESULT="GENERATE_FAILED"
        return
    fi
    echo "OK"

    # Verify input file size (conformance tests only)
    if [ "$is_conformance" = "1" ]; then
        echo -n "  Verifying input size... "
        size_check=$(verify_file_size "$input_yuv" "$width" "$height" "$frames")
        if [[ "$size_check" == "SIZE_OK" ]]; then
            echo "OK"
        else
            echo -e "${RED}$size_check${NC}"
            TEST_RESULT="INPUT_SIZE_MISMATCH"
            return
        fi
    fi

    # Encode
    echo -n "  Encoding (QP=25)... "
    if ! "$ENCODER" "$input_yuv" "$width" "$height" 25 "$encoded_lat" "$recon_yuv" 2>/dev/null; then
        echo -e "${RED}FAILED${NC}"
        TEST_RESULT="ENCODE_FAILED"
        return
    fi
    lat_size=$(stat -f%z "$encoded_lat" 2>/dev/null || stat -c%s "$encoded_lat")
    echo "OK (${lat_size} bytes)"

    # Verify recon file size (conformance tests only)
    if [ "$is_conformance" = "1" ]; then
        echo -n "  Verifying recon size... "
        size_check=$(verify_file_size "$recon_yuv" "$width" "$height" "$frames")
        if [[ "$size_check" == "SIZE_OK" ]]; then
            echo "OK"
        else
            echo -e "${RED}$size_check${NC}"
            TEST_RESULT="RECON_SIZE_MISMATCH"
            return
        fi
    fi

    # Decode
    echo -n "  Decoding... "
    if ! "$DECODER" "$encoded_lat" "$decoded_yuv" 2>/dev/null; then
        echo -e "${RED}FAILED${NC}"
        TEST_RESULT="DECODE_FAILED"
        return
    fi
    echo "OK"

    # Verify decoded file size (conformance tests only)
    if [ "$is_conformance" = "1" ]; then
        echo -n "  Verifying decoded size... "
        size_check=$(verify_file_size "$decoded_yuv" "$width" "$height" "$frames")
        if [[ "$size_check" == "SIZE_OK" ]]; then
            echo "OK"
        else
            echo -e "${RED}$size_check${NC}"
            TEST_RESULT="DECODED_SIZE_MISMATCH"
            return
        fi
    fi

    # Check recon vs decoded (should be identical)
    echo -n "  Checking recon == decoded... "
    if files_identical "$recon_yuv" "$decoded_yuv"; then
        echo -e "${GREEN}MATCH${NC}"
        recon_match="MATCH"
    else
        echo -e "${RED}MISMATCH${NC}"
        recon_match="MISMATCH"
        show_diff_details "$recon_yuv" "$decoded_yuv"
    fi

    # Compute PSNR (decoded vs original)
    echo -n "  PSNR (decoded vs original): "
    psnr_result=$(compute_psnr "$input_yuv" "$decoded_yuv" "$width" "$height" "$frames")
    echo "$psnr_result"

    # Extract average PSNR for summary
    avg_psnr=$(echo "$psnr_result" | grep -oE 'Avg=[0-9.]+' | cut -d= -f2)

    # Determine pass/fail
    if [ "$recon_match" = "MATCH" ] && [ -n "$avg_psnr" ]; then
        psnr_ok=$(python3 -c "print('OK' if float('$avg_psnr') > $min_psnr else 'LOW')")
        if [ "$psnr_ok" = "OK" ]; then
            echo -e "  Result: ${GREEN}PASS${NC}"
            TEST_RESULT="PASS:$avg_psnr"
        else
            echo -e "  Result: ${YELLOW}WARN${NC} (low PSNR)"
            TEST_RESULT="WARN:$avg_psnr"
        fi
    else
        echo -e "  Result: ${RED}FAIL${NC}"
        TEST_RESULT="FAIL:$recon_match"
    fi

    echo ""
}

# =============================================================================
# Main test runner
# =============================================================================
echo "========================================"
echo "Lattice Codec Test Suite"
echo "========================================"
echo ""

PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0
RESULTS=()

# Run conformance tests
if [ "$TEST_MODE" = "all" ] || [ "$TEST_MODE" = "conformance" ]; then
    echo -e "${CYAN}=== Conformance Tests (${#CONFORMANCE_TESTS[@]} tests) ===${NC}"
    echo "Edge cases: odd dimensions, partial tiles, chroma boundaries"
    echo ""

    for test_config in "${CONFORMANCE_TESTS[@]}"; do
        TEST_RESULT=""
        run_test "$test_config" "1" "15"

        name=$(echo "$test_config" | cut -d' ' -f1)

        if [[ "$TEST_RESULT" == PASS:* ]]; then
            psnr=$(echo "$TEST_RESULT" | cut -d: -f2)
            PASS_COUNT=$((PASS_COUNT + 1))
            RESULTS+=("conformance/$name: PASS (PSNR=${psnr}dB)")
        elif [[ "$TEST_RESULT" == WARN:* ]]; then
            psnr=$(echo "$TEST_RESULT" | cut -d: -f2)
            WARN_COUNT=$((WARN_COUNT + 1))
            PASS_COUNT=$((PASS_COUNT + 1))
            RESULTS+=("conformance/$name: WARN (PSNR=${psnr}dB - low)")
        else
            FAIL_COUNT=$((FAIL_COUNT + 1))
            RESULTS+=("conformance/$name: FAIL ($TEST_RESULT)")
        fi
    done
fi

# Run comprehensive tests
if [ "$TEST_MODE" = "all" ] || [ "$TEST_MODE" = "comprehensive" ]; then
    echo -e "${CYAN}=== Comprehensive Tests (${#COMPREHENSIVE_TESTS[@]} tests) ===${NC}"
    echo "Various sizes and content patterns"
    echo ""

    for test_config in "${COMPREHENSIVE_TESTS[@]}"; do
        TEST_RESULT=""
        run_test "$test_config" "0" "25"

        name=$(echo "$test_config" | cut -d' ' -f1)

        if [[ "$TEST_RESULT" == PASS:* ]]; then
            psnr=$(echo "$TEST_RESULT" | cut -d: -f2)
            PASS_COUNT=$((PASS_COUNT + 1))
            RESULTS+=("comprehensive/$name: PASS (PSNR=${psnr}dB)")
        elif [[ "$TEST_RESULT" == WARN:* ]]; then
            psnr=$(echo "$TEST_RESULT" | cut -d: -f2)
            WARN_COUNT=$((WARN_COUNT + 1))
            PASS_COUNT=$((PASS_COUNT + 1))
            RESULTS+=("comprehensive/$name: WARN (PSNR=${psnr}dB - low)")
        else
            FAIL_COUNT=$((FAIL_COUNT + 1))
            RESULTS+=("comprehensive/$name: FAIL ($TEST_RESULT)")
        fi
    done
fi

# Calculate totals
if [ "$TEST_MODE" = "all" ]; then
    TOTAL=$((${#CONFORMANCE_TESTS[@]} + ${#COMPREHENSIVE_TESTS[@]}))
elif [ "$TEST_MODE" = "conformance" ]; then
    TOTAL=${#CONFORMANCE_TESTS[@]}
else
    TOTAL=${#COMPREHENSIVE_TESTS[@]}
fi

# Summary
echo "========================================"
echo "TEST SUMMARY"
echo "========================================"
echo ""
echo "Mode: $TEST_MODE"
echo "Total: $TOTAL tests"
echo -e "Passed: ${GREEN}${PASS_COUNT}${NC}"
if [ $WARN_COUNT -gt 0 ]; then
    echo -e "Warnings: ${YELLOW}${WARN_COUNT}${NC}"
fi
echo -e "Failed: ${RED}${FAIL_COUNT}${NC}"
echo ""
echo "Individual Results:"
for result in "${RESULTS[@]}"; do
    if [[ "$result" == *"PASS"* ]]; then
        echo -e "  ${GREEN}$result${NC}"
    elif [[ "$result" == *"WARN"* ]]; then
        echo -e "  ${YELLOW}$result${NC}"
    else
        echo -e "  ${RED}$result${NC}"
    fi
done
echo ""

# Cleanup option
if [ $KEEP_FILES -eq 0 ]; then
    echo "Cleaning up test files... (use --keep to preserve)"
    rm -rf "$TEST_DIR"
fi

# Exit with appropriate code
if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi
