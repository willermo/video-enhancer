#!/bin/bash

# ---
# Video Splitting Script for Colab Workflow
# ---
# This script splits a large video file into smaller, fixed-duration
# chunks without re-encoding, which is very fast.
#
# Usage: ./scripts/split_video.sh [INPUT_FILE] [CHUNK_DURATION_SECONDS]
# Example: ./scripts/split_video.sh "my_video.mp4" 240
# ---

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null
then
    echo "Error: ffmpeg is not installed. Please install it."
    echo "On Ubuntu: sudo apt install ffmpeg"
    echo "On macOS: brew install ffmpeg"
    exit 1
fi

# Check for correct number of arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 [INPUT_FILE] [CHUNK_DURATION_SECONDS]"
    echo "Example: $0 'my_video.mp4' 240"
    exit 1
fi

INPUT_FILE="$1"
DURATION="$2"
BASENAME=$(basename -- "$INPUT_FILE")
DIRNAME=$(dirname -- "$INPUT_FILE")
EXTENSION="${BASENAME##*.}"
FILENAME="${BASENAME%.*}"
OUTPUT_TEMPLATE="${DIRNAME}/${FILENAME}_chunk_%02d.${EXTENSION}"

echo "Input video: $INPUT_FILE"
echo "Chunk duration: $DURATION seconds"
echo "Output template: $OUTPUT_TEMPLATE"
echo "Splitting..."

# -i: Input file
# -c copy: Copy codecs (no re-encoding, fast)
# -f segment: Use the segmenter
# -segment_time: Create a new segment at this time
# -reset_timestamps 1: Fixes timestamp issues for concatenation
# -map 0: Map all streams (video and audio)
ffmpeg -i "$INPUT_FILE" \
       -c copy \
       -f segment \
       -segment_time "$DURATION" \
       -reset_timestamps 1 \
       -map 0 \
       "$OUTPUT_TEMPLATE"

echo "Done. Chunks created in $DIRNAME"

