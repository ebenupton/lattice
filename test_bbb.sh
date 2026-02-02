#!/bin/bash
#
# Big Buck Bunny End-to-End Test
#
# Downloads BBB (if not present), encodes a clip with Lattice, decodes it,
# verifies encoder reconstruction matches decoder output, and produces
# a high-quality H.264 for visual inspection.
#
# Usage: ./test_bbb.sh [options]
#
# Options:
#   --start TIME    Start time in seconds or HH:MM:SS (default: 300 = 5 minutes)
#   --duration SEC  Duration in seconds (default: 10)
#   --qp QP         Quantization parameter 0-51 (default: 25)
#   --psy           Enable perceptual RDO
#   --keep          Keep intermediate YUV files
#   --help          Show this help
#
# Output files (in current directory):
#   big_buck_bunny_480p_h264_qp<QP>.lat       - Lattice encoded bitstream
#   big_buck_bunny_480p_h264_qp<QP>_h264.mp4  - Re-encoded H.264 for viewing
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENCODER="$SCRIPT_DIR/lattice_encode"
DECODER="$SCRIPT_DIR/lattice_decode"

# Default parameters
START=300      # 5 minutes in
DURATION=10    # 10 seconds
QP=25
PSY_FLAG=""
KEEP_FILES=0

# BBB source
BBB_URL="https://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_480p_h264.mov"
BBB_FILE="$SCRIPT_DIR/big_buck_bunny_480p_h264.mov"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Big Buck Bunny End-to-End Test"
    echo ""
    echo "Options:"
    echo "  --start TIME    Start time in seconds or HH:MM:SS (default: 300 = 5 minutes)"
    echo "  --duration SEC  Duration in seconds (default: 10)"
    echo "  --qp QP         Quantization parameter 0-51 (default: 25)"
    echo "  --psy           Enable perceptual RDO"
    echo "  --keep          Keep intermediate YUV files"
    echo "  --help          Show this help"
    echo ""
    echo "Output files:"
    echo "  big_buck_bunny_480p_h264_qp<QP>.lat       - Lattice bitstream"
    echo "  big_buck_bunny_480p_h264_qp<QP>_h264.mp4  - H.264 for viewing"
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --start)
            START="$2"
            shift 2
            ;;
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --qp)
            QP="$2"
            shift 2
            ;;
        --psy)
            PSY_FLAG="--psy"
            shift
            ;;
        --keep)
            KEEP_FILES=1
            shift
            ;;
        --help|-h)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

echo "========================================"
echo "Big Buck Bunny End-to-End Test"
echo "========================================"
echo ""
echo "Parameters:"
echo "  Start:    $START"
echo "  Duration: $DURATION seconds"
echo "  QP:       $QP"
[[ -n "$PSY_FLAG" ]] && echo "  PSY:      enabled"
echo ""

# Step 1: Download BBB if not present
echo -e "${CYAN}[1/5] Checking for Big Buck Bunny...${NC}"
if [[ -f "$BBB_FILE" ]]; then
    echo "  Found: $BBB_FILE"
    # Verify it's a valid video
    if ! ffprobe -v error "$BBB_FILE" >/dev/null 2>&1; then
        echo -e "  ${YELLOW}File appears corrupted, re-downloading...${NC}"
        rm -f "$BBB_FILE"
    fi
fi

if [[ ! -f "$BBB_FILE" ]]; then
    echo "  Downloading from Blender Foundation..."
    echo "  URL: $BBB_URL"
    echo "  (This is ~160MB, may take a moment)"

    if command -v curl &> /dev/null; then
        curl -L --progress-bar -o "$BBB_FILE" "$BBB_URL"
    elif command -v wget &> /dev/null; then
        wget --show-progress -O "$BBB_FILE" "$BBB_URL"
    else
        echo -e "${RED}Error: Neither curl nor wget found${NC}"
        exit 1
    fi

    echo -e "  ${GREEN}Download complete${NC}"
fi

# Get video info
DIMS=$(ffprobe -v error -select_streams v:0 \
    -show_entries stream=width,height -of csv=p=0 "$BBB_FILE")
WIDTH=$(echo "$DIMS" | cut -d',' -f1)
HEIGHT=$(echo "$DIMS" | cut -d',' -f2)
# Ensure even dimensions
WIDTH=$((WIDTH / 2 * 2))
HEIGHT=$((HEIGHT / 2 * 2))

TOTAL_DURATION=$(ffprobe -v error -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 "$BBB_FILE" | cut -d. -f1)

echo "  Video: ${WIDTH}x${HEIGHT}, ${TOTAL_DURATION}s total"
echo ""

# Create temp directory for intermediate files
TMPDIR=$(mktemp -d)
if [[ $KEEP_FILES -eq 0 ]]; then
    trap "rm -rf $TMPDIR" EXIT
fi

# File paths
BASENAME="big_buck_bunny_480p_h264"
YUV_FILE="$TMPDIR/${BASENAME}.yuv"
LAT_FILE="${BASENAME}_qp${QP}.lat"
RECON_FILE="$TMPDIR/${BASENAME}_qp${QP}_recon.yuv"
DECODED_FILE="$TMPDIR/${BASENAME}_decoded.yuv"
OUTPUT_MP4="${BASENAME}_qp${QP}_h264.mp4"

# Step 2: Extract YUV from source
echo -e "${CYAN}[2/5] Extracting clip to YUV...${NC}"
FFMPEG_CMD=(ffmpeg -y -hide_banner -loglevel warning)
FFMPEG_CMD+=(-ss "$START" -t "$DURATION")
FFMPEG_CMD+=(-i "$BBB_FILE")
FFMPEG_CMD+=(-vf "scale=${WIDTH}:${HEIGHT}")
FFMPEG_CMD+=(-pix_fmt yuv420p -f rawvideo "$YUV_FILE")
"${FFMPEG_CMD[@]}"

YUV_SIZE=$(stat -f%z "$YUV_FILE" 2>/dev/null || stat -c%s "$YUV_FILE")
FRAME_SIZE=$((WIDTH * HEIGHT * 3 / 2))
NUM_FRAMES=$((YUV_SIZE / FRAME_SIZE))
echo "  Extracted $NUM_FRAMES frames ($(echo "scale=2; $YUV_SIZE/1024/1024" | bc) MB)"
echo ""

# Step 3: Encode with Lattice
echo -e "${CYAN}[3/5] Encoding with Lattice (QP=$QP)...${NC}"
"$ENCODER" "$YUV_FILE" "$WIDTH" "$HEIGHT" "$QP" "$LAT_FILE" "$RECON_FILE" $PSY_FLAG

LAT_SIZE=$(stat -f%z "$LAT_FILE" 2>/dev/null || stat -c%s "$LAT_FILE")
RECON_SIZE=$(stat -f%z "$RECON_FILE" 2>/dev/null || stat -c%s "$RECON_FILE")
echo "  Output: $LAT_FILE ($(echo "scale=2; $LAT_SIZE/1024" | bc) KB)"
echo "  Compression: $(echo "scale=2; $YUV_SIZE/$LAT_SIZE" | bc):1"
echo ""

# Step 4: Decode with Lattice
echo -e "${CYAN}[4/5] Decoding with Lattice...${NC}"
"$DECODER" "$LAT_FILE" "$DECODED_FILE" > "$TMPDIR/decoder_output.txt" 2>&1
cat "$TMPDIR/decoder_output.txt"

DECODED_SIZE=$(stat -f%z "$DECODED_FILE" 2>/dev/null || stat -c%s "$DECODED_FILE")
echo ""

# Verify recon == decoded
echo -n "  Verifying recon == decoded... "
if cmp -s "$RECON_FILE" "$DECODED_FILE"; then
    echo -e "${GREEN}MATCH${NC}"
    VERIFY_RESULT="PASS"
else
    echo -e "${RED}MISMATCH${NC}"
    VERIFY_RESULT="FAIL"

    # Show details
    echo "    Recon size:   $RECON_SIZE bytes"
    echo "    Decoded size: $DECODED_SIZE bytes"

    if [[ "$RECON_SIZE" -eq "$DECODED_SIZE" ]]; then
        # Find differences
        python3 << EOF
with open("$RECON_FILE", "rb") as f1, open("$DECODED_FILE", "rb") as f2:
    pos = 0
    diffs = 0
    first_diffs = []
    while True:
        b1 = f1.read(1)
        b2 = f2.read(1)
        if not b1 or not b2:
            break
        if b1 != b2:
            if len(first_diffs) < 5:
                first_diffs.append(f"byte {pos}: recon={ord(b1)}, decoded={ord(b2)}")
            diffs += 1
        pos += 1
    print(f"    Total differing bytes: {diffs}")
    for d in first_diffs:
        print(f"      {d}")
EOF
    fi
fi
echo ""

# Step 5: Re-encode to H.264 for viewing
echo -e "${CYAN}[5/5] Re-encoding to H.264 for viewing...${NC}"
ffmpeg -y -hide_banner -loglevel warning \
    -f rawvideo -pix_fmt yuv420p -s "${WIDTH}x${HEIGHT}" -r 24 \
    -i "$DECODED_FILE" \
    -c:v libx264 -crf 8 -preset slow \
    -pix_fmt yuv420p \
    "$OUTPUT_MP4"

MP4_SIZE=$(stat -f%z "$OUTPUT_MP4" 2>/dev/null || stat -c%s "$OUTPUT_MP4")
echo "  Output: $OUTPUT_MP4 ($(echo "scale=2; $MP4_SIZE/1024" | bc) KB)"
echo ""

# Calculate PSNR between original and decoded
echo -e "${CYAN}Quality metrics (decoded vs original):${NC}"
PSNR_OUTPUT=$(ffmpeg -f rawvideo -pix_fmt yuv420p -s "${WIDTH}x${HEIGHT}" -i "$YUV_FILE" \
                     -f rawvideo -pix_fmt yuv420p -s "${WIDTH}x${HEIGHT}" -i "$DECODED_FILE" \
                     -lavfi "psnr" -f null - 2>&1 | grep "PSNR")
PSNR_Y=$(echo "$PSNR_OUTPUT" | sed -n 's/.*y:\([0-9.]*\).*/\1/p')
PSNR_U=$(echo "$PSNR_OUTPUT" | sed -n 's/.*u:\([0-9.]*\).*/\1/p')
PSNR_V=$(echo "$PSNR_OUTPUT" | sed -n 's/.*v:\([0-9.]*\).*/\1/p')
PSNR_AVG=$(echo "$PSNR_OUTPUT" | sed -n 's/.*average:\([0-9.]*\).*/\1/p')

echo "  PSNR Y:  $PSNR_Y dB"
echo "  PSNR U:  $PSNR_U dB"
echo "  PSNR V:  $PSNR_V dB"
echo "  PSNR avg: $PSNR_AVG dB"
echo ""

# Summary
echo "========================================"
echo "SUMMARY"
echo "========================================"
echo ""
echo "Input:       Big Buck Bunny 480p"
echo "Clip:        ${DURATION}s starting at ${START}s ($NUM_FRAMES frames)"
echo "Dimensions:  ${WIDTH}x${HEIGHT}"
echo "QP:          $QP"
echo ""
echo "File sizes:"
echo "  Raw YUV:   $(echo "scale=2; $YUV_SIZE/1024/1024" | bc) MB"
echo "  Lattice:   $(echo "scale=2; $LAT_SIZE/1024" | bc) KB ($(echo "scale=2; $YUV_SIZE/$LAT_SIZE" | bc):1 compression)"
echo "  H.264 out: $(echo "scale=2; $MP4_SIZE/1024" | bc) KB"
echo ""
echo "Quality:     PSNR avg = $PSNR_AVG dB"
echo ""

if [[ "$VERIFY_RESULT" == "PASS" ]]; then
    echo -e "Verification: ${GREEN}PASS${NC} (encoder recon == decoder output)"
else
    echo -e "Verification: ${RED}FAIL${NC} (encoder recon != decoder output)"
fi

echo ""
echo "Output files:"
echo "  $LAT_FILE"
echo "  $OUTPUT_MP4"

if [[ $KEEP_FILES -eq 1 ]]; then
    echo ""
    echo "Intermediate files kept in: $TMPDIR"
    echo "  $YUV_FILE"
    echo "  $RECON_FILE"
    echo "  $DECODED_FILE"
fi

echo ""

# Exit with appropriate code
if [[ "$VERIFY_RESULT" == "PASS" ]]; then
    echo -e "${GREEN}Test passed!${NC}"
    exit 0
else
    echo -e "${RED}Test failed!${NC}"
    exit 1
fi
