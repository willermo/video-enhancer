#!/bin/bash

# ---
# Video Reassembly Script for Colab Workflow
# ---
# This script finds all files matching 'restored_chunk_*.mp4' in the
# current directory, creates a temporary file list, and uses ffmpeg's
# concat demuxer to join them into a single, final video.
#
# Usage: ./scripts/reassemble_chunks.sh [OUTPUT_FILENAME]
# Example: ./scripts/reassemble_chunks.sh "final_restored_video.mp4"
# ---

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null
then
    echo "Error: ffmpeg is not installed. Please install it."
    exit 1
fi

# Set output filename
OUTPUT_FILE="${1:-final_video.mp4}"
LIST_FILE="temp_filelist.txt"

# Find all restored chunks in the current directory and sort them
# Create the temporary file list for ffmpeg
echo "Creating file list..."
find . -maxdepth 1 -name "restored_chunk_*.mp4" | sort | while read -r line; do
    # 'file' keyword is required by ffmpeg's concat demuxer
    echo "file '$line'" >> "$LIST_FILE"
done

if [ ! -s "$LIST_FILE" ]; then
    echo "Error: No files found matching 'restored_chunk_*.mp4' in this directory."
    rm -f "$LIST_FILE"
    exit 1
fi

echo "Found files to concatenate:"
cat "$LIST_FILE"

# -f concat: Use the concat demuxer
# -safe 0: Required to allow file paths from the list file
# -i: Input file (our list)
# -c copy: Copy codecs (no re-encoding, fast)
echo "Concatenating files into $OUTPUT_FILE..."
ffmpeg -f concat -safe 0 -i "$LIST_FILE" -c copy "$OUTPUT_FILE"

# Clean up the temporary file
rm "$LIST_FILE"

echo "Done. Final video saved as $OUTPUT_FILE"
