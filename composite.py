

# python composite.py --data_path /root/Data/objects_insert/coast_road --shadow_factor 1
# python composite.py --data_path /root/Data/objects_insert/corridor --shadow_factor 1
# python composite.py --data_path /root/Data/objects_insert/forest --shadow_factor 0.5
# python composite.py --data_path /root/Data/objects_insert/snow --shadow_factor 1
# python composite.py --data_path /root/Data/objects_insert/stairs --shadow_factor 1
# python composite.py --data_path /root/Data/objects_insert/light_switch --shadow_factor 1
# python composite.py --data_path /root/Data/objects_insert/train_tunnel --shadow_factor 1
# python composite.py --data_path /root/Data/objects_insert/corridor --shadow_factor 1
# python composite.py --data_path /root/Data/objects_insert/corridor_square --shadow_factor 1
# python composite.py --data_path /root/Data/objects_insert/spotlight --shadow_factor 1
# python composite.py --data_path /root/Data/objects_insert/christmas_tree --shadow_factor 1
# python composite.py --data_path /root/Data/objects_insert/korea --shadow_factor 1
# python composite.py --data_path /root/Data/objects_insert/rotate --shadow_factor 1
# python composite.py --data_path /root/Data/objects_insert/road --shadow_factor 1
# python composite.py --data_path /root/Data/objects_insert/rotate_81_no_temp --shadow_factor 1
# python composite.py --data_path /root/Data/objects_insert/rotate_81 --shadow_factor 1
# python composite.py --data_path /root/Data/objects_insert/fire --shadow_factor 1
# python composite.py --data_path /root/Data/objects_insert/wall --shadow_factor 1

import numpy as np
import argparse
import cv2
import subprocess
from pathlib import Path
from ezexr import imread, imsave

def linear_to_srgb(image):
    """Convert linear RGB image to sRGB color space."""
    threshold = 0.0031308
    srgb_image = np.where(image <= threshold,
                          image * 12.92,
                          1.055 * np.power(image, 1/2.4) - 0.055)
    return srgb_image

def srgb_to_linear(image):
    """Convert sRGB image to linear RGB color space."""
    threshold = 0.04045
    linear_image = np.where(image <= threshold,
                            image / 12.92,
                            np.power((image + 0.055) / 1.055, 2.4))
    return linear_image


def composite_frame(original_path, render_path, shadow_path, output_path, shadow_factor=1.0):
    """
    Composite a single frame with original background, rendered object, and shadow catcher.
    
    Args:
        original_path: Path to original frame
        render_path: Path to rendered object with alpha
        shadow_path: Path to shadow catcher render
        output_path: Path to save composite
        shadow_factor: Multiplier for shadow intensity (0-1)
    """
    # Read images
    original = cv2.imread(str(original_path), cv2.IMREAD_UNCHANGED)
    render = cv2.imread(str(render_path), cv2.IMREAD_UNCHANGED)
    #shadow = cv2.imread(str(shadow_path), cv2.IMREAD_UNCHANGED)
    shadow = imread(str(shadow_path))
    
    if original is None:
        raise FileNotFoundError(f"Original frame not found: {original_path}")
    if render is None:
        raise FileNotFoundError(f"Render not found: {render_path}")
    if shadow is None:
        raise FileNotFoundError(f"Shadow not found: {shadow_path}")
    
    # Convert to float for processing
    original = original.astype(np.float32) / 255.0
    render = render.astype(np.float32) / 255.0
    shadow = shadow.astype(np.float32)
    
    # Ensure all images have the same dimensions
    h, w = original.shape[:2]
    if render.shape[:2] != (h, w):
        render = cv2.resize(render, (w, h), interpolation=cv2.INTER_LINEAR)
    if shadow.shape[:2] != (h, w):
        shadow = cv2.resize(shadow, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Extract alpha channels
    if render.shape[2] == 4:
        render_rgb = render[:, :, :3]
        render_alpha = render[:, :, 3:4]
    else:
        render_rgb = render[:, :, :3]
        render_alpha = np.ones((h, w, 1), dtype=np.float32)
    
    # Process shadow (assuming shadow catcher gives us shadow information)
    if shadow.shape[2] >= 3:
        # Use luminance of shadow as shadow mask
        shadow_mask = np.mean(shadow[:, :, :3], axis=2, keepdims=True)
        # Invert if needed (darker = more shadow)
        shadow_mask = np.clip(shadow_mask * shadow_factor, 0, 1)
    else:
        shadow_mask = np.zeros((h, w, 1), dtype=np.float32)
    
    #bring original and render to linear space
    original[:, :, :3] = srgb_to_linear(original[:, :, :3])
    render_rgb = srgb_to_linear(render_rgb)

    # Apply shadow to original
    shadowed_original = original[:, :, :3] * shadow_mask
    
    # Composite: blend rendered object over shadowed original
    composite = shadowed_original * (1.0 - render_alpha) + render_rgb * render_alpha
    
    # Convert back to uint8 and save
    composite = linear_to_srgb(composite)
    composite = np.clip(composite * 255.0, 0, 255).astype(np.uint8)

    #Make sure divisible by 2
    composite = composite[:,:-1,:] if composite.shape[1] %2 !=0 else composite
    composite = composite[:-1,:,:] if composite.shape[0] %2 !=0 else composite

    cv2.imwrite(str(output_path), composite)


def create_video(frames_dir, output_video_path, fps=30, crf=18):
    """
    Create a video from composited frames using FFmpeg.
    
    Args:
        frames_dir: Directory containing frame_*.png files
        output_video_path: Path for output video file
        fps: Frames per second (default=30)
        crf: Constant Rate Factor for quality, lower=better (default=18)
    """
    frames_pattern = frames_dir / "frame_%04d.png"
    
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file
        '-framerate', str(fps),
        '-i', str(frames_pattern),
        '-c:v', 'libx264',
        '-crf', str(crf),
        '-pix_fmt', 'yuv420p',
        str(output_video_path)
    ]
    
    print(f"\nCreating video: {output_video_path}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Video created successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e}")
        print(f"stderr: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Composite original frames with rendered objects and shadows')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Base path containing original_frames, renders, and composite directories')
    parser.add_argument('--shadow_factor', '-c', type=float, default=1.0,
                        help='Shadow intensity factor (0-1, default=1.0)')
    parser.add_argument('--start_frame', type=int, default=0,
                        help='Starting frame number (default=0)')
    parser.add_argument('--end_frame', type=int, default=None,
                        help='Ending frame number (default=auto-detect)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Video frame rate (default=30)')
    parser.add_argument('--crf', type=int, default=18,
                        help='Video quality CRF value, lower=better (default=18)')
    parser.add_argument('--output_video', type=str, default=None,
                        help='Output video filename (default=composite.mp4)')
    
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    original_dir = data_path / "original_frames"
    render_dir = data_path / "renders"
    output_dir = data_path / "composite"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine frame range
    if args.end_frame is None:
        # Auto-detect by counting original frames
        original_frames = sorted(original_dir.glob("frame_*.jpg"))
        if not original_frames:
            print(f"No frames found in {original_dir}")
            return
        args.end_frame = len(original_frames)
    
    print(f"Compositing frames {args.start_frame} to {args.end_frame - 1}")
    print(f"Shadow factor: {args.shadow_factor}")
    
    # Process each frame
    for frame in range(args.start_frame, args.end_frame):
        original_path = original_dir / f"frame_{frame:04d}.jpg"
        render_path = render_dir / f"Image{frame+1:04d}.png"
        shadow_path = render_dir / f"Shadow{frame+1:04d}.exr"
        output_path = output_dir / f"frame_{frame:04d}.png"
        
        try:
            composite_frame(original_path, render_path, shadow_path, output_path, args.shadow_factor)
            print(f"Processed frame {frame:04d}")
        except FileNotFoundError as e:
            print(f"Warning: Skipping frame {frame:04d} - {e}")
        except Exception as e:
            print(f"Error processing frame {frame:04d}: {e}")
    
    print(f"Compositing complete! Output saved to {output_dir}")
    
    # Create video if requested:
    if args.output_video is None:
        video_path = data_path / "composite.mp4"
    else:
        video_path = Path(args.output_video)
    
    create_video(output_dir, video_path, fps=args.fps, crf=args.crf)


if __name__ == "__main__":
    main()


