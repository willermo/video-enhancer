#!/usr/bin/env python

import os
import sys
import subprocess
import shutil
import argparse
import glob
import torch
from pathlib import Path
from tqdm import tqdm

# --- Import AI Modules ---
# We'll import them inside a function to check for dependencies
try:
    from realesrgan.inference_realesrgan import RealESRGANer
    from gfpgan.inference_gfpgan import GFPGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
except ImportError:
    print("Error: Required AI libraries (realesrgan, gfpgan) not found.")
    print("Please activate your virtual environment and install dependencies from requirements.txt")
    sys.exit(1)

# --- FFMPEG Configuration ---
FFMPEG_CMD = "ffmpeg"  # Assumes ffmpeg is in your system's PATH

# --- AI Model Configuration ---
# These models will be auto-downloaded by the libraries on first run
REALESRGAN_MODEL_NAME = "RealESRGAN_x4plus"
GFPGAN_VERSION = "1.4"
UPSCALE_RATIO = 4

def main():
    parser = argparse.ArgumentParser(description="AI Video Enhancement Pipeline")
    parser.add_argument("-i", "--input", type=str, required=True, 
                        help="Path to the input video file or a directory of videos.")
    parser.add_argument("-o", "--output", type=str, required=True, 
                        help="Path for the output restored video or directory.")
    parser.add_argument("--skip_upscale", action="store_true", help="Skip the RealESRGAN upscaling stage.")
    parser.add_argument("--skip_face", action="store_true", help="Skip the GFPGAN face restoration stage.")
    args = parser.parse_args()

    # --- Step 1: Find Video Files ---
    input_files = []
    if os.path.isdir(args.input):
        input_files = sorted(glob.glob(os.path.join(args.input, "*.*")))
        # Ensure output is also a directory
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

    # --- Step 2: Initialize AI Models ---
    # This is a bit slow, so we do it once at the start
    print("Initializing AI models... (This may take a moment on first run to download models)")
    
    # Check for NVIDIA GPU
    if not torch.cuda.is_available():
        print("Warning: No NVIDIA GPU detected. This process will be EXTREMELY slow.")
        print("Using CPU. This is not recommended.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print(f"NVIDIA GPU detected: {torch.cuda.get_device_name(0)}")

    try:
        # Initialize RealESRGAN
        upsampler = None
        if not args.skip_upscale:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            upsampler = RealESRGANer(
                scale=UPSCALE_RATIO,
                model_path=f"httpsE://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/{REALESRGAN_MODEL_NAME}.pth",
                model=model,
                device=device
            )

        # Initialize GFPGAN
        face_enhancer = None
        if not args.skip_face:
            face_enhancer = GFPGANer(
                model_path=f"httpsE://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv{GFPGAN_VERSION}.pth",
                upscale=UPSCALE_RATIO,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=upsampler if args.skip_upscale else None, # Use upsampler as BG upsampler ONLY if we skip main upscaling
                device=device
            )
    except Exception as e:
        print(f"Error initializing AI models: {e}")
        print("Please check your internet connection and dependencies.")
        sys.exit(1)
        
    print("AI models initialized successfully.")

    # --- Step 3: Process Each Video File ---
    for video_file in input_files:
        video_name = Path(video_file).stem
        print(f"\n--- Processing Video: {video_name} ---")

        # Define output path
        if os.path.isdir(args.output):
            output_video_path = os.path.join(args.output, f"{video_name}_restored.mp4")
        else:
            output_video_path = args.output
            
        # Define unique temp directory
        temp_dir = Path(f"./temp_{video_name}").resolve()

        try:
            # --- Pipeline Step 3.1: Setup Directories ---
            shutil.rmtree(temp_dir, ignore_errors=True) # Clean up old runs
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
            
            # Use TQDM for progress bars
            pbar = tqdm(frame_list)

            # --- Pipeline Step 3.2: RealESRGAN Upscaling ---
            if not args.skip_upscale:
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
                
                # Point the next stage to the upscaled frames
                process_dir = temp_upscaled_dir
            else:
                print("2/6: Skipping Upscaling (RealESRGAN)")
                # Point the next stage to the original frames
                process_dir = temp_frames_dir

            # --- Pipeline Step 3.3: GFPGAN Face Restoration ---
            frame_list = sorted(glob.glob(str(process_dir / "*.png")))
            pbar = tqdm(frame_list)

            if not args.skip_face:
                pbar.set_description("3/6: Restoring Faces (GFPGAN)")
                for frame_path in pbar:
                    basename = os.path.basename(frame_path)
                    img = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
                    if img is None: continue

                    try:
                        # Enhance returns _, _, restored_img
                        _, _, restored_img = face_enhancer.enhance(
                            img, has_aligned=False, only_center_face=False, paste_back=True
                        )
                        cv2.imwrite(str(temp_final_dir / basename), restored_img)
                    except Exception as e:
                        print(f"Error restoring face on {basename}: {e}. Skipping frame.")
                
                # Point the final stage to the restored frames
                final_frames_path = str(temp_final_dir / "frame_%06d.png")
            else:
                print("3/6: Skipping Face Restoration (GFPGAN)")
                # Point the final stage to the (maybe) upscaled frames
                final_frames_path = str(process_dir / "frame_%06d.png")

            # --- Pipeline Step 3.4: Re-assemble Video ---
            print("4/6: Re-assembling video...")
            # Get framerate from original video
            fps = get_video_fps(video_file)
            run_ffmpeg([
                "-framerate", fps,
                "-i", final_frames_path,
                "-i", str(video_file), # Use original video as audio source
                "-map", "0:v:0", # Map video from 1st input (frames)
                "-map", "1:a:0?", # Map audio from 2nd input (original video), '?' ignores error if no audio
                "-c:a", "aac", "-b:a", "192k", # Re-encode audio for compatibility
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                "-movflags", "+faststart", # Make streamable
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
    try:
        # We'll import cv2 here, as it's only needed for the AI steps
        import cv2 
    except ImportError:
        pass # Will be handled by main import block
        
    full_cmd = [FFMPEG_CMD] + cmd
    # Use subprocess.run, capture output to hide ffmpeg's verbose logging
    result = subprocess.run(full_cmd, capture_output=True, text=True, check=True)
    return result

def get_video_fps(video_file):
    """Helper function to get the framerate of a video."""
    try:
        # Use ffprobe (part of ffmpeg) to get framerate
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_file)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Output is like "25/1", so we eval it
        fps = str(eval(result.stdout.strip()))
        if not fps: return "25" # Default
        return fps
    except Exception:
        print("Warning: Could not detect framerate. Defaulting to 25.")
        return "25"

if __name__ == "__main__":
    # This is needed for multiprocessing in GFPGAN on some systems
    torch.multiprocessing.set_start_method("spawn", force=True)
    
    # We need to import cv2 here for the AI models
    try:
        import cv2
    except ImportError:
        print("Error: 'opencv-python' is not installed.")
        print("Please activate your virtual environment and install dependencies.")
        sys.exit(1)
        
    main()

