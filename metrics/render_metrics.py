#!/usr/bin/env python3

# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/synthetic_preds_150K/classroom_anim
# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/synthetic_4dlighting/classroom_anim
# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/synthetic_pred_flux/classroom_anim_mirror_only
# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/synthetic_pred_flux/classroom_anim_better_thresh
# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/synthetic_pred_flux/classroom_anim_fixed
# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/synthetic_preds_6oct_70K/classroom_anim
# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/synthetic_preds_14oct_150K/classroom_anim
# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/synthetic/classroom_anim --is_gt

# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/outputs/infinigen/ours_flux
# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/outputs/infinigen/ours_flux_EV0
# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/outputs/infinigen/ours_flux_gt_pos
# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/outputs/infinigen/ours_flux_no_geo
# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/outputs/infinigen/ours_flux_no_geo_correct
# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/outputs/infinigen/ours_flux_no_geo_no_diffuse
# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/outputs/infinigen/ours_flux_no_diffuse
# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/outputs/infinigen/ours19oct
# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/outputs/infinigen/4DLighting
# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/outputs/infinigen/diffusionlight
# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/infinigen/final_gt_pos --is_gt

# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/real/garon_processed --is_gt
# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/real/garon_processed_reflectance --is_gt
# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/outputs/laval/ours_flux
# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/outputs/laval/diffusionlight
# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/outputs/laval/4DLighting
# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/outputs/laval/ours19oct

# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/outputs/video/dynamic_object/diffusionlight
# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/outputs/video/dynamic_object/ours19oct
# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/outputs/video/dynamic_object/ours19oct_time
# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/outputs/video/dynamic_object/ours19oct_time_next_prev
# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/outputs/video/dynamic_object/4DLighting
# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/outputs/video/dynamic_object/ours_flux
# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/video/dynamic_object

# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/outputs/video/dynamic_lighting/4DLighting
# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/outputs/video/dynamic_lighting/ours19oct
# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/outputs/video/dynamic_lighting/ours19oct_time
# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/outputs/video/dynamic_lighting/ours_flux
# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/video/dynamic_lighting

# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/outputs/video/dynamic_camera/4DLighting
# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/outputs/video/dynamic_camera/ours19oct
# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/outputs/video/dynamic_camera/ours19oct_time
# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/outputs/video/dynamic_camera/ours_flux
# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/video/dynamic_camera

# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/outputs/video/combination/4DLighting
# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/outputs/video/combination/ours19oct
# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/outputs/video/combination/ours19oct_time
# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/outputs/video/combination/ours_flux
# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/test_sets/video/combination


# python render_metrics.py --blender_path /root/blender-4.3.2-linux-x64/blender --data_path /root/Data/olat/1_out_4_time

"""
Render Metrics Script - Takes HDR predictions and renders with existing test_scene.blend setup

This script:
1. Scans the output directory for HDR predictions
2. Loads the render_test.blend file
3. Updates the existing Environment Texture node with predicted HDR lighting
4. Renders with 128 samples and saves to render_metrics subdirectories
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from tqdm import tqdm
from glob import glob

def find_blender_executable(blender_path=None):
    """Find Blender executable"""
    if blender_path and os.path.exists(blender_path):
        return blender_path
    
    # Common Blender locations
    common_paths = [
        "/root/blender-4.4.0-linux-x64/blender",
        "/usr/bin/blender",
        "/usr/local/bin/blender",
        "blender"  # In PATH
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError("Blender executable not found. Please specify with --blender_path")

def create_blender_script(blend_file, hdr_path, output_path, frame_number, camera_id):
    """Create a Blender Python script for rendering with existing setup and HDR lighting"""
    
    script_content = f'''
import bpy
import os
import sys

try:
    # Clear existing scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Load the render_test.blend file
    bpy.ops.wm.open_mainfile(filepath="{blend_file}")

    # Set frame
    bpy.context.scene.frame_set({frame_number})

    # Get existing cameras in the same order as renderer_test.py
    existing_cameras = []
    for camera in bpy.data.objects:
        if camera.type == 'CAMERA':
            existing_cameras.append(camera)

    # Select the correct camera using the same logic as renderer_test.py
    if existing_cameras:
        camera_index = {camera_id} % len(existing_cameras)
        selected_camera = existing_cameras[camera_index]
        bpy.context.scene.camera = selected_camera
        print(f"Using camera: {{selected_camera.name}} (index {{camera_index}} of {{len(existing_cameras)}})")
    else:
        print("No cameras found in scene")
        sys.exit(1)

    # Configure render settings
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.resolution_x = 1024
    bpy.context.scene.render.resolution_y = 1024
    bpy.context.scene.render.resolution_percentage = 100

    # Set output format to PNG with RGBA
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.image_settings.color_depth = '8'

    # Enable CUDA GPU rendering
    scene = bpy.context.scene
    prefs = bpy.context.preferences
    cprefs = prefs.addons['cycles'].preferences
    
    # Enable GPU compute
    cprefs.compute_device_type = 'CUDA'
    
    # Get available CUDA devices and enable them
    cprefs.get_devices()
    cuda_devices = []
    for device in cprefs.devices:
        if device.type == 'CUDA':
            device.use = True
            cuda_devices.append(device.name)
            print(f"Enabled CUDA device: {{device.name}}")
    
    if not cuda_devices:
        print("Warning: No CUDA devices found, falling back to CPU")
        scene.cycles.device = 'CPU'
    else:
        scene.cycles.device = 'GPU'
        print(f"Using GPU rendering with {{len(cuda_devices)}} CUDA device(s)")

    # Set cycles settings - 128 samples as requested
    bpy.context.scene.cycles.samples = 128
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE'

    # Find and update the existing Environment Texture node
    world = bpy.context.scene.world
    if not world or not world.use_nodes:
        print("Error: No world material with nodes found in the blend file")
        sys.exit(1)

    nodes = world.node_tree.nodes
    env_texture_node = None
    
    # Find the Environment Texture node by name or type
    for node in nodes:
        if node.name == "Environment Texture" or node.type == 'TEX_ENVIRONMENT':
            env_texture_node = node
            break
    
    if not env_texture_node:
        print("Error: Environment Texture node not found in world material")
        print("Available nodes:")
        for node in nodes:
            print(f"  - {{node.name}} ({{node.type}})")
        sys.exit(1)

    print(f"Found Environment Texture node: {{env_texture_node.name}}")

    # Load and assign the HDR image to the existing node
    try:
        if "{hdr_path}" in bpy.data.images:
            env_texture_node.image = bpy.data.images["{hdr_path}"]
        else:
            hdr_image = bpy.data.images.load("{hdr_path}")
            env_texture_node.image = hdr_image
        print(f"Successfully loaded HDR image: {hdr_path}")
    except Exception as e:
        print(f"Error: Could not load HDR image {hdr_path}: {{e}}")
        sys.exit(1)

    # Ensure we have a camera
    if not bpy.context.scene.camera:
        print("Error: No camera set in scene")
        sys.exit(1)

    # Create output directory
    output_dir = os.path.dirname("{output_path}")
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Set render output
    bpy.context.scene.render.filepath = "{output_path}"
    
    # Render with error handling
    try:
        bpy.ops.render.render(write_still=True)
        print(f"Successfully rendered to {output_path}")
    except Exception as e:
        print(f"Render failed: {{e}}")
        sys.exit(1)

except Exception as e:
    print(f"Script error: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
    
    return script_content

def process_object(object_path, data_path, blend_file, blender_path, is_gt=False):
    """Process a single object directory"""
    
    object_name = os.path.basename(object_path)

    # Extract camera_id from path structure
    # Path format: .../lighting_XXXX/camera_YYYY/object_ZZZZ
    path_parts = object_path.split(os.sep)
    camera_part = None
    for part in path_parts:
        if part.startswith('camera_'):
            camera_part = part
            break
    
    if camera_part:
        camera_id = int(camera_part.split('_')[1])  # Extract YYYY from camera_YYYY
    else:
        print(f"Warning: Could not extract camera ID from path {object_path}, using camera 0")
        camera_id = 0

    print(f"Processing object: {object_name} with camera_id: {camera_id}")
        
    HDR_dir = os.path.join(object_path, f"hdr")
        
    # Create render output directory - using render_metrics as requested
    render_dir = os.path.join(object_path, f"render_metrics")
    os.makedirs(render_dir, exist_ok=True)
    
    # Find all frame files
    frame_files = glob(os.path.join(HDR_dir, "envmap_*.exr"))
    if is_gt:
        frame_files = glob(os.path.join(object_path, "frame_*"))
    
    if not frame_files:
        print(f"No HDR files found in {HDR_dir}")
        return
    
    #frame_files = frame_files[:1]
    for frame_idx in tqdm(range(1, len(frame_files) + 1), desc=f"Rendering metrics"):
        if is_gt:
            frame_file = os.path.join(object_path, f"frame_{frame_idx:04d}", "hdri", f"image_{frame_idx:04d}.exr")
        else:
            frame_file = os.path.join(HDR_dir, f"envmap_{frame_idx:04d}.exr")
        
        output_file = os.path.join(render_dir, f"render_{frame_idx:04d}.png")
        
        if not os.path.exists(frame_file):
            print(f"Warning: Frame file {frame_file} not found, skipping")
            continue
            
        print(f"Rendering frame: {frame_file}")
        
        # Create Blender script
        script_content = create_blender_script(
            blend_file=blend_file,
            hdr_path=frame_file,
            output_path=output_file,
            frame_number=frame_idx,
            camera_id=camera_id
        )
        
        # Write temporary script
        script_path = f"/tmp/render_metrics_script_{frame_idx}.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Run Blender
        try:
            cmd = [
                blender_path,
                "--background",
                "--python", script_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"    Blender error for {frame_file}:")
                print(f"    Command: {' '.join(cmd)}")
                print(f"    stdout: {result.stdout}")
                print(f"    stderr: {result.stderr}")
            else:
                # Check if output file was created
                if os.path.exists(output_file):
                    print(f"    ✓ Rendered {frame_file}")
                else:
                    print(f"    ✗ Render completed but no output file: {frame_file}")
            
        except Exception as e:
            print(f"    ✗ Error rendering {frame_file}: {e}")
        finally:
            # Clean up script
            if os.path.exists(script_path):
                os.remove(script_path)
    
    # Create video from rendered frames
    if frame_files:
        video_path = os.path.join(render_dir, "video.mp4")
        try:
            subprocess.run([
                'ffmpeg', '-y', 
                '-framerate', '30', 
                '-i', os.path.join(render_dir, 'render_%04d.png'), 
                '-c:v', 'libx264', 
                '-pix_fmt', 'yuv420p',
                '-vf', 'premultiply=inplace=1',  # Handle transparency properly
                video_path
            ], check=True, capture_output=True)
            print(f"    Created video: {video_path}")
        except subprocess.CalledProcessError as e:
            print(f"    Failed to create video: {e}")

def main():
    parser = argparse.ArgumentParser(description="Render with predicted HDR lighting using render_test.blend")
    parser.add_argument("--data_path", required=True, help="Output path from test.py (contains pred_ev* directories)")
    parser.add_argument("--blender_path", help="Path to Blender executable")
    parser.add_argument("--is_gt", action='store_true', help="If set, process ground truth HDRs instead of predictions")
    
    args = parser.parse_args()
    
    # Find Blender
    blender_path = find_blender_executable(args.blender_path)
    print(f"Using Blender: {blender_path}")
    
    # Check if output directory exists
    if not os.path.exists(args.data_path):
        print(f"Error: Output path {args.data_path} does not exist")
        return 1

    # Check if test_scene.blend exists
    blend_file = os.path.join(os.path.dirname(__file__), "test_scene.blend")
    if not os.path.exists(blend_file):
        print(f"Error: test_scene.blend not found at {blend_file}")
        return 1
    
    # Find all object directories
    scene_dirs = glob(os.path.join(args.data_path, "lighting*", "camera*", "object*"))
    
    if not scene_dirs:
        print(f"No objects found in {args.data_path}")
        return 1

    print(f"Found {len(scene_dirs)} objects to process")
    print(f"Using blend file: {blend_file}")

    # Process each object
    for scene_dir in tqdm(scene_dirs, desc="Processing objects"):
        process_object(scene_dir, args.data_path, blend_file, blender_path, is_gt=args.is_gt)
    
    print("Rendering complete!")

if __name__ == "__main__":
    main()