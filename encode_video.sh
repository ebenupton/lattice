#!/bin/bash
#
# Extract video from H.264 file and encode with Lattice encoder
#
# Usage: ./encode_video.sh <input.mov> <qp> [start_time] [duration] [--psy]
#
# Examples:
#   ./encode_video.sh big_buck_bunny_480p_h264.mov 25
#   ./encode_video.sh big_buck_bunny_480p_h264.mov 25 300 10
#   ./encode_video.sh big_buck_bunny_480p_h264.mov 25 5:00 10 --psy

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENCODER="$SCRIPT_DIR/lattice_encode"

usage() {
    echo "Usage: $0 <input_video> <qp> [start_time] [duration] [--psy]"
    echo ""
    echo "Arguments:"
    echo "  input_video  Input video file (H.264/MP4/MOV/etc)"
    echo "  qp           Quantization parameter (0-51, lower = better quality)"
    echo "  start_time   Start time in seconds or HH:MM:SS format (default: 0)"
    echo "  duration     Duration in seconds (default: all)"
    echo "  --psy        Enable perceptual RDO"
    exit 1
}

if [[ $# -lt 2 ]]; then
    usage
fi

INPUT="$1"
QP="$2"
START="${3:-0}"
DURATION="${4:-}"
PSY_FLAG=""

# Check for --psy flag
for arg in "$@"; do
    if [[ "$arg" == "--psy" ]]; then
        PSY_FLAG="--psy"
    fi
done

if [[ ! -f "$INPUT" ]]; then
    echo "Error: Input file '$INPUT' not found"
    exit 1
fi

if [[ ! -x "$ENCODER" ]]; then
    echo "Error: Encoder not found at '$ENCODER'"
    exit 1
fi

# Get video dimensions
DIMS=$(ffprobe -v error -select_streams v:0 \
    -show_entries stream=width,height -of csv=p=0 "$INPUT")
ORIG_WIDTH=$(echo "$DIMS" | cut -d',' -f1)
ORIG_HEIGHT=$(echo "$DIMS" | cut -d',' -f2)

# Ensure even dimensions (required by encoder)
WIDTH=$((ORIG_WIDTH / 2 * 2))
HEIGHT=$((ORIG_HEIGHT / 2 * 2))

echo "Input: $INPUT"
echo "Original dimensions: ${ORIG_WIDTH}x${ORIG_HEIGHT}"
if [[ "$WIDTH" != "$ORIG_WIDTH" || "$HEIGHT" != "$ORIG_HEIGHT" ]]; then
    echo "Adjusted to even: ${WIDTH}x${HEIGHT}"
fi
echo "QP: $QP"
echo "Start: $START"
[[ -n "$DURATION" ]] && echo "Duration: $DURATION seconds"
[[ -n "$PSY_FLAG" ]] && echo "Perceptual RDO: enabled"

# Create temp directory for intermediate files
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

BASENAME=$(basename "$INPUT" | sed 's/\.[^.]*$//')
YUV_FILE="$TMPDIR/${BASENAME}.yuv"
OUTPUT_FILE="${BASENAME}_qp${QP}.lat"
RECON_FILE="${BASENAME}_qp${QP}_recon.yuv"

# Build ffmpeg command
FFMPEG_CMD=(ffmpeg -y -hide_banner -loglevel warning)
FFMPEG_CMD+=(-ss "$START")
[[ -n "$DURATION" ]] && FFMPEG_CMD+=(-t "$DURATION")
FFMPEG_CMD+=(-i "$INPUT")
# Scale to even dimensions if needed
if [[ "$WIDTH" != "$ORIG_WIDTH" || "$HEIGHT" != "$ORIG_HEIGHT" ]]; then
    FFMPEG_CMD+=(-vf "scale=${WIDTH}:${HEIGHT}")
fi
FFMPEG_CMD+=(-pix_fmt yuv420p -f rawvideo "$YUV_FILE")

echo ""
echo "Extracting video to YUV..."
"${FFMPEG_CMD[@]}"

YUV_SIZE=$(stat -f%z "$YUV_FILE" 2>/dev/null || stat -c%s "$YUV_FILE")
FRAME_SIZE=$((WIDTH * HEIGHT * 3 / 2))
NUM_FRAMES=$((YUV_SIZE / FRAME_SIZE))
echo "Extracted $NUM_FRAMES frames ($(echo "scale=2; $YUV_SIZE/1024/1024" | bc) MB)"

echo ""
echo "Encoding with Lattice encoder..."
"$ENCODER" "$YUV_FILE" "$WIDTH" "$HEIGHT" "$QP" "$OUTPUT_FILE" "$RECON_FILE" $PSY_FLAG

echo ""
echo "Output: $OUTPUT_FILE"
echo "Reconstruction: $RECON_FILE"

# Show file sizes
LAT_SIZE=$(stat -f%z "$OUTPUT_FILE" 2>/dev/null || stat -c%s "$OUTPUT_FILE")
echo ""
echo "Compression results:"
echo "  Raw YUV:  $(echo "scale=2; $YUV_SIZE/1024/1024" | bc) MB"
echo "  Encoded:  $(echo "scale=2; $LAT_SIZE/1024" | bc) KB"
echo "  Ratio:    $(echo "scale=2; $YUV_SIZE/$LAT_SIZE" | bc):1"
