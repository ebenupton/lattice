#!/bin/bash
#
# Decode Lattice file and encode as high-rate H.264
#
# Usage: ./decode_video.sh <input.lat> [output.mp4] [--crf N]
#
# Examples:
#   ./decode_video.sh video.lat
#   ./decode_video.sh video.lat output.mp4
#   ./decode_video.sh video.lat output.mp4 --crf 10

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DECODER="$SCRIPT_DIR/lattice_decode"

usage() {
    echo "Usage: $0 <input.lat> [output.mp4] [--crf N]"
    echo ""
    echo "Arguments:"
    echo "  input.lat   Lattice encoded file"
    echo "  output.mp4  Output H.264 file (default: input_h264.mp4)"
    echo "  --crf N     H.264 quality (0-51, default: 8 for high quality)"
    exit 1
}

if [[ $# -lt 1 ]]; then
    usage
fi

INPUT="$1"
CRF=8  # High quality default

# Parse arguments
shift
OUTPUT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --crf)
            CRF="$2"
            shift 2
            ;;
        *)
            if [[ -z "$OUTPUT" ]]; then
                OUTPUT="$1"
            fi
            shift
            ;;
    esac
done

if [[ ! -f "$INPUT" ]]; then
    echo "Error: Input file '$INPUT' not found"
    exit 1
fi

if [[ ! -x "$DECODER" ]]; then
    echo "Error: Decoder not found at '$DECODER'"
    exit 1
fi

# Default output filename
BASENAME=$(basename "$INPUT" .lat)
OUTPUT="${OUTPUT:-${BASENAME}_h264.mp4}"

# Create temp directory
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

YUV_FILE="$TMPDIR/decoded.yuv"

echo "Input: $INPUT"
echo "Output: $OUTPUT"
echo "H.264 CRF: $CRF"
echo ""

# Decode and capture dimensions from output
echo "Decoding Lattice file..."
DECODER_OUTPUT=$("$DECODER" "$INPUT" "$YUV_FILE" 2>&1)
echo "$DECODER_OUTPUT" | tail -5

# Parse dimensions from decoder output
DIMS_LINE=$(echo "$DECODER_OUTPUT" | head -1)
WIDTH=$(echo "$DIMS_LINE" | sed -n 's/.*: \([0-9]*\)x.*/\1/p')
HEIGHT=$(echo "$DIMS_LINE" | sed -n 's/.*x\([0-9]*\),.*/\1/p')

if [[ -z "$WIDTH" || -z "$HEIGHT" ]]; then
    echo "Error: Could not parse dimensions from decoder output"
    exit 1
fi

echo ""
echo "Dimensions: ${WIDTH}x${HEIGHT}"

# Count frames
YUV_SIZE=$(stat -f%z "$YUV_FILE" 2>/dev/null || stat -c%s "$YUV_FILE")
FRAME_SIZE=$((WIDTH * HEIGHT * 3 / 2))
NUM_FRAMES=$((YUV_SIZE / FRAME_SIZE))
echo "Frames: $NUM_FRAMES"

# Encode to H.264 with high quality settings
echo ""
echo "Encoding to H.264..."
ffmpeg -y -hide_banner -loglevel warning \
    -f rawvideo -pix_fmt yuv420p -s "${WIDTH}x${HEIGHT}" -r 24 \
    -i "$YUV_FILE" \
    -c:v libx264 -crf "$CRF" -preset slow \
    -pix_fmt yuv420p \
    "$OUTPUT"

# Report results
LAT_SIZE=$(stat -f%z "$INPUT" 2>/dev/null || stat -c%s "$INPUT")
MP4_SIZE=$(stat -f%z "$OUTPUT" 2>/dev/null || stat -c%s "$OUTPUT")

echo ""
echo "Done!"
echo "  Lattice:  $(echo "scale=2; $LAT_SIZE/1024" | bc) KB"
echo "  H.264:    $(echo "scale=2; $MP4_SIZE/1024" | bc) KB"
echo "  Output:   $OUTPUT"
