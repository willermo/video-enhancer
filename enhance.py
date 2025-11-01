#!/usr/bin/env python
"""
Author: Davide Oriani <willermo@gmail.com>
Date: 2025-11-01

Automated AI Video Enhancement and Upscaling Pipeline.

This script provides a complete, end-to-end solution for restoring low-quality 
videos. It is designed to fix common issues like blurriness, compression 
artifacts, and low-resolution faces by using a 2-stage AI process:

1.  **Real-ESRGAN:** Upscales the video resolution (default 4x), enhancing 
    general texture and detail.
2.  **GFPGAN:** Performs a second pass to specifically identify and restore 
    human faces, fixing "waxy" or blurry features into a realistic result.

Features:
-   **Batch Processing:** Can process a single video file or an entire directory.
-   **Hardware Acceleration:** Automatically uses an NVIDIA GPU (CUDA) if 
    available for maximum speed.
-   **Parallel CPU Mode:** Includes a `--parallel` flag for CPU-only machines 
    to use all available cores, speeding up processing.
-   **All-in-One:** Handles frame extraction, AI processing, and re-assembly 
    into a high-quality, streamable MP4 file with its original audio.
"""

import warnings
# --- Filter Harmless Warnings ---
# Silence all UserWarnings from torchvision, which is noisy
# about deprecated APIs used by our dependencies.
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.*")
import os
import sys
import subprocess
import shutil
import argparse
import glob
import torch
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import traceback

### Fast fail section - start ###

# --- Import AI Modules ---
try:
    import cv2
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan.utils import RealESRGANer
    from gfpgan import GFPGANer
except ImportError as e:
    print("\n--- DETAILED IMPORT ERROR ---")
    print(f"A required AI library failed to import.")
    print(f"The original error is: {e}")
    traceback.print_exc()
    print("\nThis is likely a dependency conflict.")
    print("Please check the full traceback above.")
    sys.exit(1)
except Exception as e:
    print(f"\n--- UNEXPECTED ERROR DURING IMPORT ---")
    print(f"An error occurred: {e}")
    traceback.print_exc()
    sys.exit(1)
    
### Fast fail section - end ###


### GLOBAL VARS - start ###

# print("DEBUG: All AI libraries imported successfully.")

# --- FFMPEG Configuration ---
FFMPEG_CMD = "ffmpeg"  # Assumes ffmpeg is in your system's PATH

# --- AI Model Configuration ---
REALESRGAN_MODEL_NAME = "RealESRGAN_x4plus"
GFPGAN_VERSION = "1.4"
UPSCALE_RATIO = 4

# --- PARALLEL: Global variables for worker processes ---
# These will be initialized *once* per process
worker_upsampler = None
worker_face_enhancer = None
worker_args = None
worker_device = None

### GLOBAL VARS - end ###


def get_args():
    """Gets CLI arguments"""
    parser = argparse.ArgumentParser(description="AI Video Enhancement Pipeline")
    parser.add_argument("-i",
                        "--input",
                        type=str,
                        required=True, 
                        help="Path to the input video file or a directory of videos.")
    parser.add_argument("-o",
                        "--output",
                        type=str,
                        required=True, 
                        help="Path for the output restored video or directory.")
    parser.add_argument("--skip_upscale",
                        action="store_true",
                        help="Skip the RealESRGAN upscaling stage.")
    parser.add_argument("--skip_face",
                        action="store_true",
                        help="Skip the GFPGAN face restoration stage.")
    parser.add_argument("--parallel",
                        action="store_true", 
                        help="Enable parallel CPU processing. Ignored if a GPU is detected.")
    return parser.parse_args()

def init_worker(args, device):
    """
    Initializer function for each worker process in the pool.
    Loads the AI models into the worker's global memory.
    """
    global worker_upsampler, worker_face_enhancer, worker_args, worker_device
    
    # Store args and device for worker functions
    worker_args = args
    worker_device = device
    
    print(f"DEBUG: Initializing worker process {os.getpid()}...")
    try:
        # Initialize RealESRGAN
        if not args.skip_upscale:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            worker_upsampler = RealESRGANer(
                scale=UPSCALE_RATIO,
                model_path=f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/{REALESRGAN_MODEL_NAME}.pth",
                model=model,
                device=device
            )

        # Initialize GFPGAN
        if not args.skip_face:
            worker_face_enhancer = GFPGANer(
                model_path=f"https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv{GFPGAN_VERSION}.pth",
                upscale=UPSCALE_RATIO,
                arch="clean",
                channel_multiplier=2,
                # Note: bg_upsampler is not easily usable in parallel mode, setting to None
                bg_upsampler=None, 
                device=device
            )
    except Exception as e:
        print(f"Error initializing AI models in worker: {e}")
        
def worker_upscale(frame_path):
    """Worker function for RealESRGAN upscaling."""
    global worker_upsampler
    if worker_upsampler is None:
        return (frame_path, None) # Worker init failed

    try:
        basename = os.path.basename(frame_path)
        img = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
        if img is None: 
            return (frame_path, False) # Read error
        
        output, _ = worker_upsampler.enhance(img, outscale=UPSCALE_RATIO)
        return (frame_path, output)
    
    except Exception as e:
        print(f"Error upscaling {frame_path}: {e}. Skipping frame.")
        return (frame_path, None) # Processing error

def worker_face_restore(frame_path):
    """Worker function for GFPGAN face restoration."""
    global worker_face_enhancer
    if worker_face_enhancer is None:
        return (frame_path, None) # Worker init failed
        
    try:
        basename = os.path.basename(frame_path)
        img = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
        if img is None: 
            return (frame_path, False) # Read error

        _, _, restored_img = worker_face_enhancer.enhance(
            img, has_aligned=False, only_center_face=False, paste_back=True
        )
        return (frame_path, restored_img)
    
    except Exception as e:
        print(f"Error restoring face on {frame_path}: {e}. Skipping frame.")
        return (frame_path, None) # Processing error

def main():
    args = get_args()

    # --- Step 1: Find Video Files ---
    input_files = []
    if os.path.isdir(args.input):
        input_files = sorted(glob.glob(os.path.join(args.input, "*.*")))
        if not os.path.isdir(args.output):
            os.makedirs(args.output, exist_ok=True)
            print(f"Created output directory: {args.output}")
    elif os.path.isfile(args.input):
        input_files = [args.input]
    else:
        print(f"Error: Input path not found: {args.input}")
        sys.exit(1)
        
    if not input_files:
        print(f"Error: No video files found in {args.input}")
        sys.exit(1)

    # --- Step 2: Initialize AI Models (or configure for parallel) ---
    print("Initializing AI models...")
    
    use_gpu = torch.cuda.is_available()
    use_parallel = args.parallel and not use_gpu
    num_cores = cpu_count()
    
    # Main (single-thread) models
    upsampler = None
    face_enhancer = None
    
    if use_gpu:
        device = torch.device("cuda")
        print(f"NVIDIA GPU detected: {torch.cuda.get_device_name(0)}. Running in single-threaded GPU mode.")
    elif use_parallel:
        device = torch.device("cpu")
        print(f"Warning: No GPU detected. Running in PARALLEL CPU mode using {num_cores} cores.")
    else:
        device = torch.device("cpu")
        print("Warning: No NVIDIA GPU detected. Running in single-threaded CPU mode.")
        print("This will be EXTREMELY slow. Consider adding the --parallel flag.")

    if not use_parallel:
        # --- Single-Threaded Path: Initialize models now ---
        try:
            if not args.skip_upscale:
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                upsampler = RealESRGANer(
                    scale=UPSCALE_RATIO,
                    model_path=f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/{REALESRGAN_MODEL_NAME}.pth",
                    model=model,
                    device=device
                )
            if not args.skip_face:
                face_enhancer = GFPGANer(
                    model_path=f"https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv{GFPGAN_VERSION}.pth",
                    upscale=UPSCALE_RATIO,
                    arch="clean",
                    channel_multiplier=2,
                    bg_upsampler=upsampler if args.skip_upscale else None,
                    device=device # --- BUGFIX: Added device ---
                )
            print("AI models initialized successfully.")
        except Exception as e:
            print(f"Error initializing AI models: {e}")
            sys.exit(1)
    else:
        # --- Parallel Path: Models will be initialized in workers ---
        print(f"Models will be initialized in {num_cores} parallel worker processes.")
        pass # No models to load here

    # --- Step 3: Process Each Video File ---
    for video_file in input_files:
        video_name = Path(video_file).stem
        print(f"\n--- Processing Video: {video_name} ---")

        # Define output path
        if os.path.isdir(args.output):
            output_video_path = os.path.join(args.output, f"{video_name}_restored.mp4")
        else:
            output_video_path = args.output
            
        temp_dir = Path(f"./temp_{video_name}").resolve()

        try:
            # --- Pipeline Step 3.1: Setup Directories ---
            shutil.rmtree(temp_dir, ignore_errors=True)
            temp_frames_dir = temp_dir / "temp_frames"
            temp_upscaled_dir = temp_dir / "temp_upscaled"
            temp_final_dir = temp_dir / "temp_final_restored"
            os.makedirs(temp_frames_dir, exist_ok=True)
            os.makedirs(temp_upscaled_dir, exist_ok=True)
            os.makedirs(temp_final_dir, exist_ok=True)

            print(f"1/6: Extracting frames to {temp_frames_dir}")
            run_ffmpeg([
                "-i", str(video_file),
                str(temp_frames_dir / "frame_%06d.png")
            ])
            
            frame_list = sorted(glob.glob(str(temp_frames_dir / "*.png")))
            if not frame_list:
                raise Exception("No frames were extracted from the video.")
            
            # --- Pipeline Step 3.2: RealESRGAN Upscaling ---
            if not args.skip_upscale:
                print("2/6: Upscaling (RealESRGAN)...")
                
                if use_parallel:
                    # --- PARALLEL CPU PATH ---
                    pbar = tqdm(total=len(frame_list))
                    with Pool(processes=num_cores, initializer=init_worker, initargs=(args, device)) as pool:
                        for frame_path, output in pool.imap_unordered(worker_upscale, frame_list):
                            if output is not None:
                                basename = os.path.basename(frame_path)
                                cv2.imwrite(str(temp_upscaled_dir / basename), output)
                            pbar.update(1)
                else:
                    # --- SINGLE-THREAD (GPU/CPU) PATH ---
                    pbar = tqdm(frame_list)
                    pbar.set_description("2/6: Upscaling (RealESRGAN)")
                    for frame_path in pbar:
                        basename = os.path.basename(frame_path)
                        img = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
                        if img is None: continue
                        
                        try:
                            output, _ = upsampler.enhance(img, outscale=UPSCALE_RATIO)
                            cv2.imwrite(str(temp_upscaled_dir / basename), output)
                        except Exception as e:
                            print(f"Error upscaling {basename}: {e}. Skipping frame.")
                
                process_dir = temp_upscaled_dir
            else:
                print("2/6: Skipping Upscaling (RealESRGAN)")
                process_dir = temp_frames_dir

            # --- Pipeline Step 3.3: GFPGAN Face Restoration ---
            if not args.skip_face:
                print("3/6: Restoring Faces (GFPGAN)...")
                
                # Get new frame list from the (potentially) upscaled dir
                frame_list = sorted(glob.glob(str(process_dir / "*.png")))
                
                if use_parallel:
                    # --- PARALLEL CPU PATH ---
                    pbar = tqdm(total=len(frame_list))
                    with Pool(processes=num_cores, initializer=init_worker, initargs=(args, device)) as pool:
                        for frame_path, restored_img in pool.imap_unordered(worker_face_restore, frame_list):
                            if restored_img is not None:
                                basename = os.path.basename(frame_path)
                                cv2.imwrite(str(temp_final_dir / basename), restored_img)
                            pbar.update(1)
                else:
                    # --- SINGLE-THREAD (GPU/CPU) PATH ---
                    pbar = tqdm(frame_list)
                    pbar.set_description("3/6: Restoring Faces (GFPGAN)")
                    for frame_path in pbar:
                        basename = os.path.basename(frame_path)
                        img = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
                        if img is None: continue

                        try:
                            _, _, restored_img = face_enhancer.enhance(
                                img, has_aligned=False, only_center_face=False, paste_back=True
                            )
                            cv2.imwrite(str(temp_final_dir / basename), restored_img)
                        except Exception as e:
                            print(f"Error restoring face on {basename}: {e}. Skipping frame.")
                
                final_frames_path = str(temp_final_dir / "frame_%06d.png")
            else:
                print("3/6: Skipping Face Restoration (GFPGAN)")
                final_frames_path = str(process_dir / "frame_%06d.png")

            # --- Pipeline Step 3.4: Re-assemble Video ---
            print("4/6: Re-assembling video...")
            fps = get_video_fps(video_file)
            run_ffmpeg([
                "-framerate", fps,
                "-i", final_frames_path,
                "-i", str(video_file),
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-c:a", "aac", "-b:a", "192k",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                "-movflags", "+faststart",
                str(output_video_path)
            ])

        except Exception as e:
            print(f"An error occurred while processing {video_name}: {e}")
        
        finally:
            # --- Pipeline Step 3.5 & 3.6: Cleanup ---
            print("5/6: Cleaning up temporary files...")
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"6/6: Finished. Restored video saved to:\n{output_video_path}\n")

def run_ffmpeg(cmd):
    """Helper function to run an FFMPEG command."""
    full_cmd = [FFMPEG_CMD] + cmd
    result = subprocess.run(full_cmd, capture_output=True, text=True, check=True)
    return result

def get_video_fps(video_file):
    """Helper function to get the framerate of a video."""
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_file)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        fps = str(eval(result.stdout.strip()))
        if not fps: return "25"
        return fps
    except Exception:
        print("Warning: Could not detect framerate. Defaulting to 25.")
        return "25"

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()

