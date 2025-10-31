# AI Video Enhancement Pipeline

This repository contains a complete Python-based pipeline to upscale and enhance low-quality videos, specifically designed to restore blurry faces from compressed recordings.

The workflow uses a 2-stage AI process:
1.  **Real-ESRGAN:** A powerful upscaler to increase video resolution 4x and remove general compression artifacts.
2.  **GFPGAN:** A face-specific model that restores realistic details to blurry or "waxy" faces from the upscaled video.

This project is designed to be run on a powerful NVIDIA GPU, either on a local workstation or via Google Colab.



---



##  Workflow 1: Local (NVIDIA GPU) Workstation Setup

This is the recommended workflow for fast, automated batch processing.

### 1. Initial Setup (One Time Only)

#### A. Install `pyenv` (Python Version Manager)

This is critical to ensure you use the correct Python version (3.10).

**On Ubuntu/Debian:**
```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev libncurses5-dev \
libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl

# Install pyenv
curl https://pyenv.run | bash

# Add pyenv to your shell (run this and restart your terminal)
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
```

**On macOS (using Homebrew):**
```bash
brew install pyenv pyenv-virtualenv
```

#### B. Install Python 3.10

Restart your terminal after installing `pyenv` (or source the appropriate environment files), then run:
```bash
# Installs Python 3.10.19 (takes a few minutes)
pyenv install 3.10.19
```

### 2. Project Setup (For This Project)

```bash
# 1. Clone this repository
git clone git@github.com:willermo/video-enhancer.git
cd video-enhancer-pipeline

# 2. Set the local Python version for this folder
pyenv local 3.10.19

# 3. Create a virtual environment
python -m venv venv
source venv/bin/activate

# 4. Install all dependencies
pip install -r requirements.txt
```

### Running the Enhancement

You are now ready to process your videos.

```bash
# Activate the environment (if you haven't already)
source venv/bin/activate

# Run the master script on a single file
python enhance.py \
    --input "/path/to/my/video.mp4" \
    --output "/path/to/my/restored_video.mp4"

# --- OR ---

# Run the script on an entire directory
python enhance.py \
    --input "/path/to/videos_folder/" \
    --output "/path/to/restored_folder/"
```



---



## Workflow 2: Google Colab (Free Tier)

This workflow is for users without an NVIDIA GPU. It is **much more manual** and requires you to split your video into small chunks to avoid Colab's free-tier time limits.

### 1. Local Prep: Split Your Video

Before uploading, you must split your video into 4-minute (240-second) chunks.

```bash
# Run the split script
# Usage: ./scripts/split_video.sh [INPUT_FILE] [CHUNK_DURATION_SECONDS]
./scripts/split_video.sh "/path/to/my/video.mp4" 240
```

This will create `chunk_01.mp4`, `chunk_02.mp4`, etc., in the same directory.

### 2. Google Colab Workflow

1. **Open the Colab Notebook**: [Link to your Colab Notebook - we will create this]
2. **Runtime**: In the Colab menu, select **Runtime > Change runtime type > T4 GPU**.
3. **Upload**: In the Colab "Files" sidebar, upload `requirements.txt` and your first chunk (e.g., `chunk_01.mp4`).
4. **Run Cells**: Follow the notebook instructions to:
  - Install Python 3.10.
  - Install dependencies from `requirements.txt`.
  - Run the enhancement on `chunk_01.mp4`.
5. **Download**: Download the resulting `restored_chunk_01.mp4`.
6. **Repeat**: Delete the processed files from Colab to save space, then upload `chunk_02.mp4` and repeat.

### Local Reassembly: Stitch Chunks

Once you have downloaded all your restored chunks, use this script to join them back together:

```bash
# This script finds all 'restored_chunk_*.mp4' files in the
# current directory and stitches them into 'final_video.mp4'
./scripts/reassemble_chunks.sh
```



---

