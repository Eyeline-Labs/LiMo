import time
import gradio as gr
import numpy as np
import torch
import cv2
import os
import sys
import tempfile
import shutil
from PIL import Image
import pickle
import subprocess
from typing import List, Tuple
import matplotlib.cm as cm
from torchvision import transforms
import json
import zipfile

sys.path.append("./Video-Depth-Anything")
from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames
from ezexr import imsave
from video.diffsynth import load_state_dict
from video.diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig, WanVideoUnit_ConditionsVideoEmbedder
from video.conditional_utils import add_condition_input
from video.datasets.dataset_spheres_video import treat_data_flux, TARGET_TO_PROMPT
from video.datasets.data_utils import sRGB_to_Lin
from video.WAN_test import WanTrainingModule, forward_preprocess
import copy

# MoGe imports
from moge.model.v2 import MoGeModel
import requests

# HDR optimization imports
from HDRMerge.batch_optimization_pol_manual_gt import (
    optimize_envmap_for_scene,
    render,
    load_reference_images,
    Lin_to_sRGB
)

from image.conditional_pipelines import ConditionalSD3Pipeline, ConditionalFluxPipeline
from diffusers import FluxTransformer2DModel
from image.train_utils_min import encode_prompt
from image.utils import encode_with_vae, prepare_target_and_prompt, prepare_target_and_prompt_ev, add_condition_input, TARGET_TO_PROMPT
from image.test_config import TestConfig, default_test_configs


#Global models
SHARED_VIDEO_DEPTH_MODEL = None
SHARED_INFERENCE_MODEL = None
SHARED_CONDITION_LIST = None
SHARED_FOCAL_LENGTH_MODEL = None
SHARED_target_and_prompt_dict = None
SHARED_prompt_embeds_dict = None
SHARED_pooled_prompt_embeds_dict = None


def load_shared_models():
    global SHARED_VIDEO_DEPTH_MODEL, SHARED_INFERENCE_MODEL, SHARED_CONDITION_LIST, SHARED_FOCAL_LENGTH_MODEL, SHARED_target_and_prompt_dict, SHARED_prompt_embeds_dict, SHARED_pooled_prompt_embeds_dict
    if SHARED_VIDEO_DEPTH_MODEL is None:
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        
        SHARED_VIDEO_DEPTH_MODEL = VideoDepthAnything(**model_configs['vitl'])
        checkpoint_path = './Video-Depth-Anything/checkpoints/metric_video_depth_anything_vitl.pth'
        if os.path.exists(checkpoint_path):
            SHARED_VIDEO_DEPTH_MODEL.load_state_dict(torch.load(checkpoint_path, map_location='cpu'), strict=True)
        SHARED_VIDEO_DEPTH_MODEL = SHARED_VIDEO_DEPTH_MODEL.to(DEVICE).eval()

    if SHARED_INFERENCE_MODEL is None:
        weight_dtype = torch.float32

        condition_list = "image,position,normals,dist_to_sphere,dir_to_sphere",

        weight_dtype = torch.float16

        pretrained_model_name_or_path = "black-forest-labs/FLUX.1-schnell"

        PipelineClass = ConditionalFluxPipeline
        TransformerClass = FluxTransformer2DModel

        model_path = ".checkpoints/Flux/mp_rank_00_model_states.pt"

        num_input_conditions = 5

        if os.path.isdir(model_path):
            pipeline = PipelineClass.from_pretrained(model_path)
        else:
            transformer = TransformerClass.from_pretrained(pretrained_model_name_or_path, subfolder="transformer")
            add_condition_input(transformer, num_input_conditions)
            transformer.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False)['module'], strict=True)
            pipeline = PipelineClass.from_pretrained(pretrained_model_name_or_path, transformer=transformer)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create the prompt embeddings
        tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2]
        text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2]
        if PipelineClass == ConditionalSD3Pipeline:
            tokenizers.append(pipeline.tokenizer_3)
            text_encoders.append(pipeline.text_encoder_3)

        # Put encoders on device
        text_encoders = [encoder.to(device,dtype=weight_dtype) for encoder in text_encoders]

        print("Precomputing targets' prompt embeddings...") 
        cfg = default_test_configs["balls_ev"][1]
        #target_and_prompt_dict, prompt_embeds_dict, pooled_prompt_embeds_dict,text_ids = prepare_target_and_prompt(cfg,encode_prompt,text_encoders,tokenizers,device)
        target_and_prompt_dict, prompt_embeds_dict, pooled_prompt_embeds_dict, text_ids = prepare_target_and_prompt_ev(cfg,encode_prompt,text_encoders,tokenizers,device)

        SHARED_target_and_prompt_dict = target_and_prompt_dict
        SHARED_prompt_embeds_dict = prompt_embeds_dict
        SHARED_pooled_prompt_embeds_dict = pooled_prompt_embeds_dict

        pipeline.text_encoder = None
        pipeline.text_encoder_2 = None
        if PipelineClass == ConditionalSD3Pipeline:
            pipeline.text_encoder_3 = None
        del tokenizers, text_encoders
        torch.cuda.empty_cache()

        print("Uploading pipeline to device...")
        pipeline.to(device,dtype=weight_dtype)
        SHARED_INFERENCE_MODEL = pipeline

    if SHARED_CONDITION_LIST is None:
        SHARED_CONDITION_LIST = "image,position,normals,dist_to_sphere,dir_to_sphere".split(",")

    if SHARED_FOCAL_LENGTH_MODEL is None:
        try:
            # Load MoGe model'
            SHARED_FOCAL_LENGTH_MODEL = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl").to("cuda")
            print("MoGe model loaded successfully!")
        except Exception as e:
            SHARED_FOCAL_LENGTH_MODEL = None
            print(f"Warning: Could not load MoGe model: {e}. Using manual FOV input.")




class VideoDepthDemo:
    def __init__(self):
        #self.setup_models()
        self.temp_dir = None
        self.depth_frames = None
        self.video_frames = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.height = None
        self.width = None
        self.model = SHARED_INFERENCE_MODEL
        self.target_and_prompt_dict = SHARED_target_and_prompt_dict
        self.prompt_embeds_dict = SHARED_prompt_embeds_dict
        print(self.prompt_embeds_dict.keys())
        self.pooled_prompt_embeds_dict = SHARED_pooled_prompt_embeds_dict
        self.condition_list = SHARED_CONDITION_LIST
        self.video_depth_anything = SHARED_VIDEO_DEPTH_MODEL
        self.focal_length_model = SHARED_FOCAL_LENGTH_MODEL
        # Add sphere parameters for first and last frame
        self.first_frame_sphere = None
        self.last_frame_sphere = None
        self.interpolated_spheres = None
        self.sphere_data = None
        self.blend_file_path = None


    def export_to_blender(self):
        """Export depth as animated point cloud and camera setup to Blender file"""
        if self.depth_frames is None:
            return None, "Please predict depth first!"
        
        try:
            # Create Blender script that will be executed
            blender_script = '''
import bpy
import numpy as np
import pickle
import sys

# Load data from pickle file
data_path = sys.argv[-1]
with open(data_path, 'rb') as f:
    data = pickle.load(f)

depth_frames = data['depth_frames']
video_frames = data['video_frames']
fx, fy, cx, cy = data['fx'], data['fy'], data['cx'], data['cy']
height, width = data['height'], data['width']
vfov_rad = data['vfov_rad']
num_frames = len(depth_frames)

# Clear existing scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Create camera with correct FOV
bpy.ops.object.camera_add(location=(0, 0, 0))
camera = bpy.context.object
camera.name = "DepthCamera"

# Set camera parameters
camera.data.lens_unit = 'MILLIMETERS'
# Calculate focal length in mm (using 35mm sensor width as reference)
sensor_width_mm = 36.0  # Standard full-frame sensor width
sensor_height_mm = sensor_width_mm * (height / width)

# Use the average focal length, or use fx since it corresponds to sensor width
focal_length_mm = (fx / width) * sensor_width_mm

camera.data.lens = focal_length_mm
camera.data.sensor_width = sensor_width_mm
camera.data.sensor_height = sensor_height_mm
camera.data.sensor_fit = 'HORIZONTAL'  # Lock to horizontal so sensor_width is respected

# Set as active camera
bpy.context.scene.camera = camera

# Set up scene frame range
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = num_frames

# Create mesh with faces for each frame
for frame_idx, depth in enumerate(depth_frames):
    # Subsample for performance (every 4th pixel)
    subsample = 2
    depth_sub = depth[::subsample, ::subsample]
    
    # Create grid for subsampled resolution
    h_sub, w_sub = depth_sub.shape
    i_sub, j_sub = np.meshgrid(np.arange(0, width, subsample), np.arange(0, height, subsample))
    
    # Calculate 3D positions from subsampled data
    x_sub = (i_sub - cx) / fx * depth_sub
    y_sub = -(j_sub - cy) / fy * depth_sub
    z_sub = -depth_sub  # Negative Z for Blender camera space
    
    # Get colors from video frames if available
    if video_frames is not None:
        frame_rgb = video_frames[frame_idx]
        colors_sub = frame_rgb[::subsample, ::subsample] / 255.0
    else:
        colors_sub = np.ones((x_sub.shape[0], x_sub.shape[1], 3)) * 0.5
    
    # Flatten arrays
    vertices = np.stack([x_sub.flatten(), y_sub.flatten(), z_sub.flatten()], axis=1)
    colors = colors_sub.reshape(-1, 3)
    
    # Create faces (quads connecting adjacent vertices)
    faces = []
    h_sub, w_sub = x_sub.shape
    for row in range(h_sub - 1):
        for col in range(w_sub - 1):
            # Vertex indices for quad
            v0 = row * w_sub + col
            v1 = row * w_sub + (col + 1)
            v2 = (row + 1) * w_sub + (col + 1)
            v3 = (row + 1) * w_sub + col
            faces.append([v0, v1, v2, v3])
    
    # Create mesh for this frame
    mesh_name = f"DepthMesh_Frame_{frame_idx:04d}"
    mesh = bpy.data.meshes.new(mesh_name)
    obj = bpy.data.objects.new(mesh_name, mesh)
    bpy.context.collection.objects.link(obj)
    
    # Add vertices and faces to mesh
    mesh.from_pydata(vertices.tolist(), [], faces)
    mesh.update()
    
    # Add vertex colors to mesh using color attributes (Blender 3.2+)
    if hasattr(mesh, 'color_attributes'):
        color_attr = mesh.color_attributes.new(name='Color', type='BYTE_COLOR', domain='CORNER')
        
        # Set colors for each face corner (loop)
        for face in mesh.polygons:
            for loop_idx in face.loop_indices:
                loop = mesh.loops[loop_idx]
                vertex_idx = loop.vertex_index
                if vertex_idx < len(colors):
                    color_attr.data[loop_idx].color = (*colors[vertex_idx], 1.0)
    
    # Create material with vertex color support
    mat_name = f"PointCloud_Material_{frame_idx:04d}"
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear default nodes
    nodes.clear()
    
    # Add shader nodes
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    color_attr_node = nodes.new(type='ShaderNodeVertexColor')
    color_attr_node.layer_name = 'Color'
    
    # Link nodes
    links.new(color_attr_node.outputs['Color'], bsdf_node.inputs['Base Color'])
    links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])
    
    # Assign material to mesh
    if len(obj.data.materials):
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    
    # Hide all frames except the first one initially
    obj.hide_viewport = (frame_idx != 0)
    obj.hide_render = (frame_idx != 0)
    
    # Keyframe visibility
    obj.keyframe_insert(data_path="hide_viewport", frame=1)
    obj.keyframe_insert(data_path="hide_render", frame=1)
    
    for f in range(1, num_frames + 1):
        obj.hide_viewport = (f != frame_idx + 1)
        obj.hide_render = (f != frame_idx + 1)
        obj.keyframe_insert(data_path="hide_viewport", frame=f)
        obj.keyframe_insert(data_path="hide_render", frame=f)

# Create probe sphere
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=(0, 0, -2))
probe = bpy.context.object
probe.name = "probe"

# Add custom properties to store metadata
probe["sphere_radius"] = 0.5
probe["num_frames"] = num_frames

# Create material for probe
mat = bpy.data.materials.new(name="ProbeMaterial")
mat.use_nodes = True
mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (1.0, 0.5, 0.0, 1.0)  # Orange
mat.node_tree.nodes["Principled BSDF"].inputs[12].default_value = 1.0  # Metallic
probe.data.materials.append(mat)

# Add extraction script as a text block
extraction_script = """import bpy
import json

def extract_sphere_data():
    #Extract sphere center and radius for each frame
    probe = bpy.data.objects.get("probe")
    if not probe:
        print("ERROR: Probe sphere not found!")
        return
    
    num_frames = probe.get("num_frames", bpy.context.scene.frame_end)
    sphere_data = []
    
    for frame in range(1, num_frames + 1):
        bpy.context.scene.frame_set(frame)
        
        # Get world space location
        location = probe.matrix_world.translation
        
        # Get scale (assuming uniform scaling)
        scale = probe.scale[0]
        base_radius = probe.get("sphere_radius", 0.5)
        radius = base_radius * scale
        
        sphere_data.append({
            "frame": frame,
            "center": [location.x, location.y, location.z],
            "radius": radius
        })
    
    # Format output for copying
    output = json.dumps(sphere_data, indent=2)
    print("\\\\n" + "="*50)
    print("SPHERE DATA (Copy the text below):")
    print("="*50)
    print(output)
    print("="*50 + "\\\\n")
    
    return sphere_data

# Run extraction
if __name__ == "__main__":
    extract_sphere_data()
"""

text_block = bpy.data.texts.new("extract_sphere_data.py")
text_block.write(extraction_script)

# Save blend file
blend_path = data['blend_path']
bpy.ops.wm.save_as_mainfile(filepath=blend_path)
'''
            
            # Save data for Blender script
            data_for_blender = {
                'depth_frames': self.depth_frames,
                'video_frames': self.video_frames,
                'fx': self.fx,
                'fy': self.fy,
                'cx': self.cx,
                'cy': self.cy,
                'height': self.height,
                'width': self.width,
                'vfov_rad': 2 * np.arctan(self.height / (2 * self.fy)),
                'blend_path': os.path.join(self.temp_dir, "depth_scene.blend")
            }
            
            data_pickle_path = os.path.join(self.temp_dir, "blender_data.pkl")
            with open(data_pickle_path, 'wb') as f:
                pickle.dump(data_for_blender, f)
            
            # Save Blender script
            script_path = os.path.join(self.temp_dir, "create_scene.py")
            with open(script_path, 'w') as f:
                f.write(blender_script)
            
            # Find Blender executable
            blender_paths = [
                "/root/blender-4.4.0-linux-x64/blender",
                "/root/blender-4.3.2-linux-x64/blender",
                "blender"  # System blender
            ]
            
            blender_exe = None
            for path in blender_paths:
                if os.path.exists(path) or path == "blender":
                    blender_exe = path
                    break
            
            if not blender_exe:
                return None, "Blender executable not found! Please install Blender."
            
            # Run Blender in background mode
            blend_path = data_for_blender['blend_path']
            result = subprocess.run(
                [blender_exe, "--background", "--python", script_path, "--", data_pickle_path],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                return None, f"Blender execution failed:\n{result.stderr}"
            
            if not os.path.exists(blend_path):
                return None, f"Blend file was not created. Blender output:\n{result.stdout}\n{result.stderr}"
            
            self.blend_file_path = blend_path
            
            num_frames = len(self.depth_frames)
            
            # Calculate camera parameters for display
            sensor_width_mm = 36.0
            sensor_height_mm = sensor_width_mm * (self.height / self.width)
            focal_length_mm = (self.fx / self.width) * sensor_width_mm
            vfov_rad = 2 * np.arctan(self.height / (2 * self.fy))
            
            return (
                blend_path,
                f"Blender file created successfully!\n"
                f"Location: {blend_path}\n"
                f"Camera Setup:\n"
                f"  - Focal Length: {focal_length_mm:.2f}mm\n"
                f"  - Sensor Size: {sensor_width_mm:.1f}mm x {sensor_height_mm:.1f}mm\n"
                f"  - FOV: {np.rad2deg(vfov_rad):.2f}° (vertical)\n"
                f"Scene Info:\n"
                f"  - Frames: {num_frames}\n"
                f"  - Point cloud subsampling: 4x\n"
                f"  - Vertex colors: Enabled (from video)\n\n"
                f"Instructions:\n"
                f"1. Open the .blend file in Blender\n"
                f"2. Position and animate the 'probe' sphere\n"
                f"3. Switch to 'Solid' or 'Material Preview' shading to see colors\n"
                f"4. Run the 'extract_sphere_data.py' script in Blender's text editor\n"
                f"5. Copy the JSON output and paste it in the 'Sphere Data' tab"
            )
            
        except subprocess.TimeoutExpired:
            return None, "Blender execution timed out. The scene might be too complex."
        except Exception as e:
            import traceback
            return None, f"Error exporting to Blender: {str(e)}\n{traceback.format_exc()}"

    

    
    def predict_vfov(self, image_pil):
        """Predict vertical field of view for a PIL image using MoGe model"""
        if self.focal_length_model is None:
            return None, "MoGe model not available"
        
        try:
            # Convert PIL image to numpy array (RGB)
            image = np.array(image_pil)
            image = torch.tensor(np.array(image)).permute(2,0,1).float() / 255.0
            
            # # Ensure we have RGB
            # if len(img_rgb.shape) == 3 and img_rgb.shape[2] == 3:
            #     # Convert RGB to BGR for PerspectiveFields
            #     img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            # else:
            #     return None, "Invalid image format"
            
            # Predict using MoGe
            with torch.no_grad():
                output = self.focal_length_model.infer(image)
                intrinsics = output['intrinsics'].cpu().numpy()
                # Extract VFOV from predictions
                self.fy = intrinsics[1,1] * self.height
                self.fx = intrinsics[0,0] * self.width
                self.cx = intrinsics[0,2] * self.width
                self.cy = intrinsics[1,2] * self.height
                pred_vfov = 2 * np.degrees(np.arctan(self.height / (2 * self.fy)))            
            return pred_vfov, f"Predicted VFOV: {pred_vfov:.1f}°"
            
        except Exception as e:
            return None, f"Error predicting VFOV: {str(e)}"
        
    def setup_models(self):
        """Initialize the depth prediction and inference models"""
        # Setup Video Depth Anything
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        
        self.video_depth_anything = VideoDepthAnything(**model_configs['vitl'])
        checkpoint_path = '/root/Project/Video-Depth-Anything/checkpoints/metric_video_depth_anything_vitl.pth'
        if os.path.exists(checkpoint_path):
            self.video_depth_anything.load_state_dict(torch.load(checkpoint_path, map_location='cpu'), strict=True)
        self.video_depth_anything = self.video_depth_anything.to(DEVICE).eval()
        
        
    # def setup_inference_model(self, model_path: str, condition_list: List[str]):
    #     """Setup the inference model with given parameters"""
    #     # try:


    #     condition_list = "image,position,normals,dist_to_sphere,dir_to_sphere"
    #     model = WanTrainingModule(
    #         model_paths=None,
    #         model_id_with_origin_paths="Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth",
    #         trainable_models=None,
    #         lora_base_model=None,
    #         lora_target_modules=None,
    #         lora_rank=None,
    #         lora_checkpoint=None,
    #         use_gradient_checkpointing_offload=False,
    #         extra_inputs=None,
    #         max_timestep_boundary=1.0,
    #         min_timestep_boundary=0.0,
    #         condition_list=condition_list,
    #         targets=None,
    #         output_dir=None
    #     )

    #     state_dict = load_state_dict("/root/Projects/balls/Wan/Wan_26sep_2.2/step-150000.safetensors")
    #     load_result = model.pipe.dit.load_state_dict(state_dict, strict=False)

    #     model.to('cuda')

    #     self.model = model
    #     self.condition_list = condition_list.split(",")
        
    #     return "Model loaded successfully"
    #     # except Exception as e:
    #     #     return f"Error loading model: {str(e)}"
    
    def predict_depth_and_focal(self, video_path: str, fov: float = 90.0, use_ml_focal_length: bool = True):
        """Predict depth for video frames and estimate focal length"""
        num_frames = 21
        try:
            # Check if input is an image or video
            if video_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
                # Load image and duplicate for num_frames frames
                img = Image.open(video_path)
                img_array = np.array(img.convert('RGB'))
                frames = np.stack([img_array] * num_frames, axis=0)
                fps = 30  # Default fps for image sequences
                print(f"Image input detected, duplicated to {frames.shape[0]} frames")
            else:
                # Read video frames (limit to first num_frames frames)
                frames, fps = read_video_frames(video_path, -1, -1, 4000)
                # Limit to first num_frames frames
                frames = frames[:num_frames]
                print(f"Video input: {frames.shape}")

            #save original frames
            orig_frames = copy.deepcopy(frames)

            # Resize to ensure dimensions are divisible by 16 (VAE requirement)
            max_dim = 512
            h, w = frames[0].shape[:2]
            
            # Calculate new dimensions divisible by 16
            if min(h, w) > max_dim:
                scale = max_dim / min(h, w)
                new_h = int(h * scale)
                new_w = int(w * scale)
            else:
                new_h, new_w = h, w
            
            # Round to nearest multiple of mult_factor
            mult_factor = 32
            new_h = (new_h // mult_factor) * mult_factor
            new_w = (new_w // mult_factor) * mult_factor
            
            # Ensure minimum size of mult_factor
            new_h = max(mult_factor, new_h)
            new_w = max(mult_factor, new_w)
            
            if (new_h, new_w) != (h, w):
                resized_frames = []
                for frame in frames:
                    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    resized_frames.append(resized_frame)
                frames = np.array(resized_frames)
                print(f"Resized frames from {h}x{w} to {new_h}x{new_w} (divisible by 16)")
            else:
                frames = np.array(frames)
            
                
            self.video_frames = frames
            
            # Predict depths using original frames
            depths, fps = self.video_depth_anything.infer_video_depth(
                frames, fps, input_size=518, device='cuda', fp32=False
            )
            self.depth_frames = depths

            self.height, self.width = frames[0].shape[:2]  # Changed from depths[0].shape
        
            # Verify depths have same dimensions
            target_h, target_w = frames[0].shape[:2]
            depths_resized = []
            for depth in depths:
                if depth.shape[:2] != (target_h, target_w):
                    depth_resized = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                    depths_resized.append(depth_resized)
                else:
                    depths_resized.append(depth)
            self.depth_frames = np.array(depths_resized)
            

            # Get dimensions from original frames (first frame)
            self.height, self.width = depths[0].shape[:2]
            
            # Predict or use manual focal length
            predicted_vfov = None
            focal_info = ""
            
            if use_ml_focal_length and self.focal_length_model is not None:
                # Use first frame to predict VFOV
                first_frame_pil = Image.fromarray(frames[0])
                predicted_vfov, focal_msg = self.predict_vfov(first_frame_pil)
                
                if predicted_vfov is not None:
                    # Use predicted VFOV directly
                    vfov = predicted_vfov
                    # Calculate corresponding horizontal FOV
                    vfov_rad = np.deg2rad(vfov)
                    hfov_rad = 2 * np.arctan(np.tan(vfov_rad / 2) * self.width / self.height)
                    fov = np.rad2deg(hfov_rad)
                    focal_info = f"{focal_msg}\nConverted to horizontal FOV: {fov:.1f}°"
                else:
                    focal_info = f"{focal_msg}\nUsing manual FOV: {fov:.1f}°"
            else:
                focal_info = f"Using manual FOV: {fov:.1f}°"
            
            # Calculate focal length parameters for camera intrinsics
            # FOV parameter is horizontal FOV
            hfov = np.deg2rad(fov)
            # Calculate focal length from horizontal FOV
            self.fx = self.width / (2 * np.tan(hfov / 2))
            # For a pinhole camera, fy should equal fx (assuming square pixels)
            self.fy = self.fx
            self.cx = self.width / 2
            self.cy = self.height / 2
            
            # # Create temporary directory for outputs
            # if self.temp_dir:
            #     shutil.rmtree(self.temp_dir)
            # self.temp_dir = tempfile.mkdtemp()
            #Temp dir based on current date-time
            self.temp_dir = os.path.join("/tmp", "gradio_results", f"temp_{int(time.time())}")
            os.makedirs(self.temp_dir, exist_ok=True)

            # save original frames as images
            orig_frames_path = os.path.join(self.temp_dir, 'original_frames')
            os.makedirs(orig_frames_path, exist_ok=True)
            for idx, frame in enumerate(orig_frames):
                frame_pil = Image.fromarray(frame)
                frame_pil.save(os.path.join(orig_frames_path, f'frame_{idx:04d}.jpg'))
            
            # Create zip file of original frames
            orig_frames_zip = os.path.join(self.temp_dir, 'original_frames.zip')
            with zipfile.ZipFile(orig_frames_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for idx in range(len(orig_frames)):
                    frame_path = os.path.join(orig_frames_path, f'frame_{idx:04d}.jpg')
                    zipf.write(frame_path, f'frame_{idx:04d}.jpg')

            self.orig_frames_zip = orig_frames_zip

            # Debug depth values
            print(f"Depth shape: {depths[0].shape}")
            print(f"Depth min: {depths[0].min()}, max: {depths[0].max()}")
            print(f"Depth dtype: {depths[0].dtype}")
            
            # Save depth video visualization (following dc_utils.save_video approach)
            depth_vis_path = os.path.join(self.temp_dir, 'depth_visualization.mp4')
            from utils.dc_utils import save_video
            save_video(depths, depth_vis_path, fps=fps, is_depths=True, grayscale=False)
            
            # Store depth range for sphere Z slider
            self.depth_min = float(depths[0].min())
            self.depth_max = float(depths[0].max())
            
            return (
                depth_vis_path,
                self.orig_frames_zip,
                f"Depth predicted successfully!\n{focal_info}\n"
                f"Camera intrinsics: fx={self.fx:.2f}, fy={self.fy:.2f}\n"
                f"Principal point: cx={self.cx:.2f}, cy={self.cy:.2f}\n"
                f"Video dimensions: {self.width}x{self.height}\n"
                f"Number of frames: {len(frames)}\n"
                f"Depth range: {depths[0].min():.3f} to {depths[0].max():.3f}\n"
                f"Depth video saved to: {depth_vis_path}\n"
                f"Original frames zip: {self.orig_frames_zip}"
            )
            
        except Exception as e:
            return None, None, f"Error predicting depth: {str(e)}"
    
    def get_depth_range_info(self):
        """Get depth range information for updating UI"""
        if hasattr(self, 'depth_min') and hasattr(self, 'depth_max'):
            return f"Depth range: [{self.depth_min:.2f}, {self.depth_max:.2f}]"
        else:
            return "Predict depth first to see range"
    
    
    def prepare_maps(self, sphere_data_json: str):
        """Prepare all maps with sphere data from JSON"""
        if self.depth_frames is None:
            return None, None, None, None, None, None, "Please predict depth first!"
        
        try:
            # Parse JSON sphere data
            sphere_data = json.loads(sphere_data_json)
            
            # Validate sphere data
            if not isinstance(sphere_data, list):
                return None, None, None, None, None, None, "Invalid sphere data format. Expected a list of frame data."
            
            if len(sphere_data) != len(self.depth_frames):
                return None, None, None, None, None, None, f"Sphere data has {len(sphere_data)} frames, but video has {len(self.depth_frames)} frames."
            
            # Store sphere data for later use
            self.sphere_data = sphere_data
            
            position_maps = []
            sphere_masks = []
            normal_maps = []
            
            for frame_idx, (depth, frame_sphere_data) in enumerate(zip(self.depth_frames, sphere_data)):
                sphere_center = frame_sphere_data['center']
                sphere_radius = frame_sphere_data['radius']
                
                # Convert Blender coordinates to depth camera coordinates
                # Blender has Z-up, camera space has -Z forward
                sphere_x, sphere_y, sphere_z = sphere_center
                sphere_z = -sphere_z  # Flip Z back to camera space
                
                # Create sphere by ray-sphere intersection in world space
                # This ensures the sphere stays circular regardless of aspect ratio
                sphere_center_array = np.array([sphere_x, sphere_y, sphere_z])
                
                i, j = np.meshgrid(np.arange(self.width), np.arange(self.height))
                
                # Create rays in world space using the camera intrinsics
                # This preserves the correct aspect ratio
                ray_x = (i - self.cx) / self.fx
                ray_y = -(j - self.cy) / self.fy
                ray_z = np.ones_like(ray_x)
                
                # Normalize ray directions to ensure sphere remains circular
                ray_length = np.sqrt(ray_x**2 + ray_y**2 + ray_z**2)
                ray_x_norm = ray_x / ray_length
                ray_y_norm = ray_y / ray_length
                ray_z_norm = ray_z / ray_length
                
                oc_x = -sphere_center_array[0]
                oc_y = -sphere_center_array[1]
                oc_z = -sphere_center_array[2]
                
                # Ray-sphere intersection using normalized rays
                a = 1.0  # ray_x_norm**2 + ray_y_norm**2 + ray_z_norm**2 = 1 (normalized)
                b = 2.0 * (oc_x * ray_x_norm + oc_y * ray_y_norm + oc_z * ray_z_norm)
                c = oc_x**2 + oc_y**2 + oc_z**2 - sphere_radius**2
                
                discriminant = b**2 - 4*a*c
                sphere_mask = (discriminant >= 0).astype(np.float32)
                sphere_depth = np.full_like(depth, np.inf)
                
                valid_pixels = discriminant >= 0
                if np.any(valid_pixels):
                    sqrt_discriminant = np.sqrt(discriminant[valid_pixels])
                    t1 = (-b[valid_pixels] - sqrt_discriminant) / (2.0 * a)
                    t2 = (-b[valid_pixels] + sqrt_discriminant) / (2.0 * a)
                    t_near = np.where(t1 > 0, t1, t2)
                    t_near = np.where(t_near > 0, t_near, np.inf)
                    
                    # Convert parametric distance t along normalized ray to actual depth (z-coordinate)
                    # The intersection point is at: origin + t * ray_direction_normalized
                    # Since ray origin is (0,0,0), the point is just t * ray_direction_normalized
                    # The z-coordinate (depth) is t * ray_z_norm
                    sphere_depth[valid_pixels] = t_near * ray_z_norm[valid_pixels]
                
                # Create sphere position map (actual intersection points on sphere)
                sphere_x = (i - self.cx) / self.fx * sphere_depth
                sphere_y = -(j - self.cy) / self.fy * sphere_depth
                sphere_z = sphere_depth
                sphere_position = np.stack((sphere_x, sphere_y, sphere_z), axis=-1)
                
                # Create sphere normals from actual sphere intersection points
                normal_map = np.zeros((self.height, self.width, 3), dtype=np.float32)
                valid_mask = sphere_mask > 0
                if np.any(valid_mask):
                    directions = sphere_position[valid_mask] - sphere_center_array
                    norms = np.linalg.norm(directions, axis=-1, keepdims=True)
                    normal_map[valid_mask] = directions / (norms + 1e-8)
                
                # Create composite depth and position maps (for background)
                composite_depth = depth.copy()
                sphere_pixels = sphere_mask > 0
                if np.any(sphere_pixels):
                    composite_depth[sphere_pixels] = np.minimum(depth[sphere_pixels], sphere_depth[sphere_pixels])
                
                # Convert composite depth to position map
                x = (i - self.cx) / self.fx * composite_depth
                y = -(j - self.cy) / self.fy * composite_depth
                z = composite_depth
                
                position = np.stack((x, y, z), axis=-1)
                
                position_maps.append(position)
                sphere_masks.append(sphere_mask)
                normal_maps.append(normal_map)
            
            total_sphere_pixels = sum(np.sum(mask > 0) for mask in sphere_masks)
            
            # Process data using treat_data to get the final processed maps
            data_for_processing = {
                "video": [],
                "image": [],
                "position": [],
                "normals": [],
                "sphere_mask": [],
                "sphere_center": [],
                "world2cam": [],
                "target_layer": "sphere_0",
                "ev_augmentation": False,  # Don't apply exposure changes for visualization
                "random_ev": False,
                "mask_img": True
            }
            
            # Convert all frames for processing
            num_process_frames = len(self.video_frames)  # Process all frames
            for i in range(num_process_frames):
                # Convert video frame to tensor
                frame_bgr = np.array(self.video_frames[i])
                frame_tensor = torch.from_numpy(frame_bgr.astype(np.float32) / 255.0).permute(2, 0, 1)
                frame_tensor = sRGB_to_Lin(frame_tensor)  # Convert to linear space
                data_for_processing["video"].append(frame_tensor)
                data_for_processing["image"].append(frame_tensor)
                
                # Convert numpy arrays to tensors
                pos_tensor = torch.from_numpy(position_maps[i].astype(np.float32)).permute(2, 0, 1)
                normal_tensor = torch.from_numpy(normal_maps[i].astype(np.float32)).permute(2, 0, 1) 
                mask_tensor = torch.from_numpy(sphere_masks[i].astype(np.float32)).unsqueeze(0)
                
                pos_tensor[2,:,:] = -pos_tensor[2,:,:]  # Invert Z for correct coordinates
                normal_tensor[2,:,:] = -normal_tensor[2,:,:]  # Invert Z for correct normals
                data_for_processing["position"].append(pos_tensor)
                data_for_processing["normals"].append(normal_tensor)
                data_for_processing["sphere_mask"].append(mask_tensor)
                
                # Use sphere center from JSON data (with Z coordinate inverted to match position tensor)
                frame_sphere = sphere_data[i]
                sphere_center = frame_sphere['center']
                sphere_center_corrected = [sphere_center[0], sphere_center[1], -(-sphere_center[2])]  # Double negative = positive
                data_for_processing["sphere_center"].append(torch.tensor(sphere_center_corrected, dtype=torch.float32))
                data_for_processing["world2cam"].append(torch.eye(4, dtype=torch.float32))
            
            # Stack tensors
            for key in ["video", "image", "position", "normals", "sphere_mask", "sphere_center", "world2cam"]:
                data_for_processing[key] = torch.stack(data_for_processing[key])
            
            # Process data
            position_raw = data_for_processing["position"].clone()
            processed_data = treat_data_flux(data_for_processing, given_ev=0, test=False)

            self.processed_data = processed_data  # Store for later use
            self.processed_data["position_raw"] = position_raw

            # Create visualizations from processed data (first frame)
            first_position_processed = processed_data["position"][0].permute(1, 2, 0).numpy()  # Convert CHW to HWC
            first_image_processed = processed_data["image"][0].permute(1, 2, 0).numpy()  # Convert CHW to HWC
            first_normal_processed = processed_data["normals"][0].permute(1, 2, 0).numpy()
            first_mask_processed = processed_data["sphere_mask"][0].permute(1, 2, 0).numpy()
            first_dir_to_sphere = processed_data["dir_to_sphere"][0].permute(1, 2, 0).numpy()
            first_dist_to_sphere = processed_data["dist_to_sphere"][0].permute(1, 2, 0).numpy()
            
            # Position map visualization (processed data is already normalized [-1,1])
            pos_vis = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            if first_position_processed.max() > first_position_processed.min():
                # Convert from [-1,1] to [0,255]
                pos_norm = (first_position_processed + 1.0) / 2.0  # Convert [-1,1] to [0,1]
                pos_vis = (pos_norm * 255).astype(np.uint8)

            image_vis = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            if first_image_processed.max() > first_image_processed.min():
                image_norm = (first_image_processed + 1.0) / 2.0  # Ensure in [0,1]
                img_vis = (image_norm * 255).astype(np.uint8)
            
            # Normal map visualization (processed normals are already in [-1,1])
            normal_vis = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            if np.any(first_mask_processed > 0):
                # Convert from [-1,1] to [0,255]
                normal_display = (first_normal_processed + 1.0) / 2.0
                normal_vis = (np.clip(normal_display, 0, 1) * 255).astype(np.uint8)
            
            # Mask visualization (processed mask is already normalized [-1,1])
            mask_vis = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            # Handle mask shape - it should be (H, W, 1) after permute, so take first channel
            if first_mask_processed.shape[-1] == 1:
                mask_gray = ((first_mask_processed[:, :, 0] + 1.0) / 2.0 * 255).astype(np.uint8)
            else:
                # If it has 3 channels, take the first one
                mask_gray = ((first_mask_processed[:, :, 0] + 1.0) / 2.0 * 255).astype(np.uint8)
            mask_vis[:, :, 0] = mask_gray
            mask_vis[:, :, 1] = mask_gray
            mask_vis[:, :, 2] = mask_gray
            
            # Direction to sphere visualization
            dir_vis = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            if first_dir_to_sphere.max() > first_dir_to_sphere.min():
                # Convert from [-1,1] to [0,255] (normals/directions are in [-1,1])
                dir_norm = (first_dir_to_sphere + 1.0) / 2.0
                dir_vis = (np.clip(dir_norm, 0, 1) * 255).astype(np.uint8)
            
            # Distance to sphere visualization (single channel, convert to RGB)
            dist_vis = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            if first_dist_to_sphere.max() > first_dist_to_sphere.min():
                # Convert from [-1,1] to [0,255] and use as grayscale
                dist_norm = (first_dist_to_sphere + 1.0) / 2.0
                # Handle distance shape - it should be (H, W, 1) after permute, so take first channel
                if first_dist_to_sphere.shape[-1] == 1:
                    dist_gray = (np.clip(dist_norm[:, :, 0], 0, 1) * 255).astype(np.uint8)
                else:
                    # If it has multiple channels, take the first one
                    dist_gray = (np.clip(dist_norm[:, :, 0], 0, 1) * 255).astype(np.uint8)
                dist_vis[:, :, 0] = dist_gray
                dist_vis[:, :, 1] = dist_gray
                dist_vis[:, :, 2] = dist_gray
            
            # Create videos for all frames instead of just first frame visualizations
            import subprocess
            
            # Create video output directory
            video_output_dir = os.path.join(self.temp_dir, "map_videos")
            os.makedirs(video_output_dir, exist_ok=True)
            
            # Process all frames for each map type
            all_pos_vis = []
            all_img_vis = []
            all_normal_vis = []
            all_mask_vis = []
            all_dir_vis = []
            all_dist_vis = []
            
            for frame_idx in range(len(processed_data["position"])):
                # Get processed data for this frame
                frame_position = processed_data["position"][frame_idx].permute(1, 2, 0).numpy()
                frame_image = processed_data["image"][frame_idx].permute(1, 2, 0).numpy()
                frame_normal = processed_data["normals"][frame_idx].permute(1, 2, 0).numpy()
                frame_mask = processed_data["sphere_mask"][frame_idx].permute(1, 2, 0).numpy()
                frame_dir = processed_data["dir_to_sphere"][frame_idx].permute(1, 2, 0).numpy()
                frame_dist = processed_data["dist_to_sphere"][frame_idx].permute(1, 2, 0).numpy()
                
                # Position map visualization
                pos_vis = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                if frame_position.max() > frame_position.min():
                    pos_norm = (frame_position + 1.0) / 2.0
                    pos_vis = (pos_norm * 255).astype(np.uint8)
                all_pos_vis.append(pos_vis)
                
                # Image visualization
                img_vis = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                if frame_image.max() > frame_image.min():
                    image_norm = (frame_image + 1.0) / 2.0
                    img_vis = (image_norm * 255).astype(np.uint8)
                all_img_vis.append(img_vis)
                
                # Normal map visualization
                normal_vis = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                if np.any(frame_mask > 0):
                    normal_display = (frame_normal + 1.0) / 2.0
                    normal_vis = (np.clip(normal_display, 0, 1) * 255).astype(np.uint8)
                all_normal_vis.append(normal_vis)
                
                # Mask visualization
                mask_vis = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                if frame_mask.shape[-1] == 1:
                    mask_gray = ((frame_mask[:, :, 0] + 1.0) / 2.0 * 255).astype(np.uint8)
                else:
                    mask_gray = ((frame_mask[:, :, 0] + 1.0) / 2.0 * 255).astype(np.uint8)
                mask_vis[:, :, 0] = mask_gray
                mask_vis[:, :, 1] = mask_gray
                mask_vis[:, :, 2] = mask_gray
                all_mask_vis.append(mask_vis)
                
                # Direction visualization
                dir_vis = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                if frame_dir.max() > frame_dir.min():
                    dir_norm = (frame_dir + 1.0) / 2.0
                    dir_vis = (np.clip(dir_norm, 0, 1) * 255).astype(np.uint8)
                all_dir_vis.append(dir_vis)
                
                # Distance visualization
                dist_vis = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                if frame_dist.max() > frame_dist.min():
                    dist_norm = (frame_dist + 1.0) / 2.0
                    if frame_dist.shape[-1] == 1:
                        dist_gray = (np.clip(dist_norm[:, :, 0], 0, 1) * 255).astype(np.uint8)
                    else:
                        dist_gray = (np.clip(dist_norm[:, :, 0], 0, 1) * 255).astype(np.uint8)
                    dist_vis[:, :, 0] = dist_gray
                    dist_vis[:, :, 1] = dist_gray
                    dist_vis[:, :, 2] = dist_gray
                all_dist_vis.append(dist_vis)
            
            # Save videos for each map type
            pos_video_path = os.path.join(video_output_dir, "position_maps.mp4")
            img_video_path = os.path.join(video_output_dir, "image_maps.mp4")
            normal_video_path = os.path.join(video_output_dir, "normal_maps.mp4")
            mask_video_path = os.path.join(video_output_dir, "mask_maps.mp4")
            dir_video_path = os.path.join(video_output_dir, "direction_maps.mp4")
            dist_video_path = os.path.join(video_output_dir, "distance_maps.mp4")
            
            # Convert numpy arrays to videos using ffmpeg
            def save_frames_as_video(frames, output_path, fps=30):
                # Create temporary directory for frames
                temp_frames_dir = os.path.join(video_output_dir, "temp_frames")
                os.makedirs(temp_frames_dir, exist_ok=True)
                
                # Save frames as images
                for i, frame in enumerate(frames):
                    frame_path = os.path.join(temp_frames_dir, f"frame_{i:04d}.png")
                    Image.fromarray(frame).save(frame_path)
                
                # Create video with ffmpeg
                subprocess.run([
                    'ffmpeg', '-y', '-framerate', str(fps),
                    '-i', os.path.join(temp_frames_dir, 'frame_%04d.png'),
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    output_path
                ], capture_output=True, check=True)
                
                # Clean up temporary frames
                shutil.rmtree(temp_frames_dir)
                
                return output_path
            
            # Save all map videos
            pos_video_path = save_frames_as_video(all_pos_vis, pos_video_path)
            img_video_path = save_frames_as_video(all_img_vis, img_video_path)
            normal_video_path = save_frames_as_video(all_normal_vis, normal_video_path)
            mask_video_path = save_frames_as_video(all_mask_vis, mask_video_path)
            dir_video_path = save_frames_as_video(all_dir_vis, dir_video_path)
            dist_video_path = save_frames_as_video(all_dist_vis, dist_video_path)
            
            info_text = f"Maps prepared from JSON sphere data!\n" \
                       f"Total frames processed: {len(sphere_masks)}\n" \
                       f"Total sphere pixels: {total_sphere_pixels}\n" \
                       f"Videos saved to: {video_output_dir}"

            return pos_video_path, img_video_path, normal_video_path, mask_video_path, dir_video_path, dist_video_path, info_text

        except json.JSONDecodeError as e:
            return None, None, None, None, None, None, f"Error parsing JSON: {str(e)}"
        except Exception as e:
            import traceback
            return None, None, None, None, None, None, f"Error preparing maps: {str(e)}\n{traceback.format_exc()}"
            
            position_maps = []
            sphere_masks = []
            normal_maps = []
            
            for frame_idx, (depth, sphere_params) in enumerate(zip(self.depth_frames, interpolated_spheres)):
                sphere_x, sphere_y, sphere_z = sphere_params['center']
                sphere_radius = sphere_params['radius']
                
                # Create sphere by ray-sphere intersection
                sphere_center = np.array([sphere_x, sphere_y, sphere_z])
                
                i, j = np.meshgrid(np.arange(self.width), np.arange(self.height))
                ray_x = (i - self.cx) / self.fx
                ray_y = -(j - self.cy) / self.fy
                ray_z = np.ones_like(ray_x)
                
                ray_length = np.sqrt(ray_x**2 + ray_y**2 + ray_z**2)
                ray_x_norm = ray_x / ray_length
                ray_y_norm = ray_y / ray_length
                ray_z_norm = ray_z / ray_length
                
                oc_x = -sphere_center[0]
                oc_y = -sphere_center[1]
                oc_z = -sphere_center[2]
                
                # Ray-sphere intersection using normalized rays
                a = 1.0  # normalized rays have length 1
                b = 2.0 * (oc_x * ray_x_norm + oc_y * ray_y_norm + oc_z * ray_z_norm)
                c = oc_x**2 + oc_y**2 + oc_z**2 - sphere_radius**2
                
                discriminant = b**2 - 4*a*c
                sphere_mask = (discriminant >= 0).astype(np.float32)
                sphere_depth = np.full_like(depth, np.inf)
                
                valid_pixels = discriminant >= 0
                if np.any(valid_pixels):
                    sqrt_discriminant = np.sqrt(discriminant[valid_pixels])
                    t1 = (-b[valid_pixels] - sqrt_discriminant) / (2.0 * a)
                    t2 = (-b[valid_pixels] + sqrt_discriminant) / (2.0 * a)
                    t_near = np.where(t1 > 0, t1, t2)
                    t_near = np.where(t_near > 0, t_near, np.inf)
                    
                    # Convert parametric distance along normalized ray to depth (z-coordinate)
                    sphere_depth[valid_pixels] = t_near * ray_z_norm[valid_pixels]
                
                # Create composite depth and position maps
                composite_depth = depth.copy()
                sphere_pixels = sphere_mask > 0
                if np.any(sphere_pixels):
                    composite_depth[sphere_pixels] = np.minimum(depth[sphere_pixels], sphere_depth[sphere_pixels])
                
                # Convert composite depth to position map
                x = (i - self.cx) / self.fx * composite_depth
                y = -(j - self.cy) / self.fy * composite_depth
                z = composite_depth
                
                position = np.stack((x, y, z), axis=-1)
                
                # Create sphere normals
                normal_map = np.zeros_like(position)
                valid_mask = sphere_mask > 0
                if np.any(valid_mask):
                    directions = position[valid_mask] - sphere_center
                    norms = np.linalg.norm(directions, axis=-1, keepdims=True)
                    normal_map[valid_mask] = directions / (norms + 1e-8)
                
                position_maps.append(position)
                sphere_masks.append(sphere_mask)
                normal_maps.append(normal_map)
            
            # Store interpolated spheres for later use
            self.interpolated_spheres = interpolated_spheres
            
            # Save data in the format expected by treat_data
            # frame_dir = os.path.join(self.temp_dir, "frame_0001")
            # sphere_dir = os.path.join(frame_dir, sphere_type)
            # os.makedirs(sphere_dir, exist_ok=True)
            
            # for i, (pos, mask, normal) in enumerate(zip(position_maps, sphere_masks, normal_maps)):
            #     frame_num = i + 1
                
            #     # Save position map
            #     imsave(os.path.join(frame_dir, f"position_{frame_num:04d}.exr"), 
            #            pos.astype(np.float32))
            #     imsave(os.path.join(sphere_dir, f"position_{frame_num:04d}.exr"), 
            #            pos.astype(np.float32))
                
            #     # Save sphere mask
            #     imsave(os.path.join(sphere_dir, f"mask_{frame_num:04d}.exr"), 
            #            mask.astype(np.float32))
                
            #     # Save normal map
            #     imsave(os.path.join(sphere_dir, f"normal_{frame_num:04d}.exr"), 
            #            normal.astype(np.float32))
                
            #     # Save original image
            #     frame_bgr = cv2.cvtColor(np.array(self.video_frames[i]), cv2.COLOR_RGB2BGR)
            #     frame_float = frame_bgr.astype(np.float32) / 255.0
            #     imsave(os.path.join(frame_dir, f"image_{frame_num:04d}.exr"), 
            #            frame_float)
                
            #     # Save sphere image (original image for now)
            #     imsave(os.path.join(sphere_dir, f"image_{frame_num:04d}.exr"), 
            #            frame_float)
            
            # Save camera info
            # cam2world = np.eye(4, dtype=np.float32)
            # camera_info = {'cam2world': cam2world}
            # with open(os.path.join(frame_dir, 'camera_info.pkl'), 'wb') as f:
            #     pickle.dump(camera_info, f)
            
            # Save sphere info
            # sphere_info = {
            #     1: {  # sphere_id = 1 (matching the mask value)
            #         'center_3D': [sphere_x, sphere_y, sphere_z],
            #         'radius': sphere_radius
            #     }
            # }
            # with open(os.path.join(sphere_dir, 'spheres_info.pkl'), 'wb') as f:
            #     pickle.dump(sphere_info, f)
            
            total_sphere_pixels = sum(np.sum(mask > 0) for mask in sphere_masks)
            
            # Process data using treat_data to get the final processed maps
            data_for_processing = {
                "video": [],
                "image": [],
                "position": [],
                "normals": [],
                "sphere_mask": [],
                "sphere_center": [],
                "world2cam": [],
                "target_layer": "sphere_0",
                "ev_augmentation": False,  # Don't apply exposure changes for visualization
                "random_ev": False,
                "mask_img": True
            }
            
            # Convert all frames for processing
            num_process_frames = len(self.video_frames)  # Process all frames
            for i in range(num_process_frames):
                # Convert video frame to tensor
                frame_bgr = np.array(self.video_frames[i])
                frame_tensor = torch.from_numpy(frame_bgr.astype(np.float32) / 255.0).permute(2, 0, 1)
                frame_tensor = sRGB_to_Lin(frame_tensor)  # Convert to linear space
                data_for_processing["video"].append(frame_tensor)
                data_for_processing["image"].append(frame_tensor)
                
                # Convert numpy arrays to tensors
                pos_tensor = torch.from_numpy(position_maps[i].astype(np.float32)).permute(2, 0, 1)
                normal_tensor = torch.from_numpy(normal_maps[i].astype(np.float32)).permute(2, 0, 1) 
                mask_tensor = torch.from_numpy(sphere_masks[i].astype(np.float32)).unsqueeze(0)
                
                pos_tensor[2,:,:] = -pos_tensor[2,:,:]  # Invert Z for correct coordinates
                normal_tensor[2,:,:] = -normal_tensor[2,:,:]  # Invert Z for correct normals
                data_for_processing["position"].append(pos_tensor)
                data_for_processing["normals"].append(normal_tensor)
                data_for_processing["sphere_mask"].append(mask_tensor)
                
                # Use interpolated sphere center for this frame (with Z coordinate inverted to match position tensor)
                sphere_center = interpolated_spheres[i]['center']
                # Invert Z coordinate to match the position tensor coordinate system
                sphere_center_corrected = [sphere_center[0], sphere_center[1], -sphere_center[2]]
                data_for_processing["sphere_center"].append(torch.tensor(sphere_center_corrected, dtype=torch.float32))
                data_for_processing["world2cam"].append(torch.eye(4, dtype=torch.float32))
            
            # Stack tensors
            for key in ["video", "image", "position", "normals", "sphere_mask", "sphere_center", "world2cam"]:
                data_for_processing[key] = torch.stack(data_for_processing[key])
            
            # Process data
            processed_data = treat_data(data_for_processing, given_ev=0, test=False)

            self.processed_data = processed_data  # Store for later use
            
            # Create visualizations from processed data (first frame)
            first_position_processed = processed_data["position"][0].permute(1, 2, 0).numpy()  # Convert CHW to HWC
            first_image_processed = processed_data["image"][0].permute(1, 2, 0).numpy()  # Convert CHW to HWC
            first_normal_processed = processed_data["normals"][0].permute(1, 2, 0).numpy()
            first_mask_processed = processed_data["sphere_mask"][0].permute(1, 2, 0).numpy()
            first_dir_to_sphere = processed_data["dir_to_sphere"][0].permute(1, 2, 0).numpy()
            first_dist_to_sphere = processed_data["dist_to_sphere"][0].permute(1, 2, 0).numpy()
            
            # Position map visualization (processed data is already normalized [-1,1])
            pos_vis = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            if first_position_processed.max() > first_position_processed.min():
                # Convert from [-1,1] to [0,255]
                pos_norm = (first_position_processed + 1.0) / 2.0  # Convert [-1,1] to [0,1]
                pos_vis = (pos_norm * 255).astype(np.uint8)

            image_vis = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            if first_image_processed.max() > first_image_processed.min():
                image_norm = (first_image_processed + 1.0) / 2.0  # Ensure in [0,1]
                img_vis = (image_norm * 255).astype(np.uint8)
            
            # Normal map visualization (processed normals are already in [-1,1])
            normal_vis = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            if np.any(first_mask_processed > 0):
                # Convert from [-1,1] to [0,255]
                normal_display = (first_normal_processed + 1.0) / 2.0
                normal_vis = (np.clip(normal_display, 0, 1) * 255).astype(np.uint8)
            
            # Mask visualization (processed mask is already normalized [-1,1])
            mask_vis = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            # Handle mask shape - it should be (H, W, 1) after permute, so take first channel
            if first_mask_processed.shape[-1] == 1:
                mask_gray = ((first_mask_processed[:, :, 0] + 1.0) / 2.0 * 255).astype(np.uint8)
            else:
                # If it has 3 channels, take the first one
                mask_gray = ((first_mask_processed[:, :, 0] + 1.0) / 2.0 * 255).astype(np.uint8)
            mask_vis[:, :, 0] = mask_gray
            mask_vis[:, :, 1] = mask_gray
            mask_vis[:, :, 2] = mask_gray
            
            # Direction to sphere visualization
            dir_vis = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            if first_dir_to_sphere.max() > first_dir_to_sphere.min():
                # Convert from [-1,1] to [0,255] (normals/directions are in [-1,1])
                dir_norm = (first_dir_to_sphere + 1.0) / 2.0
                dir_vis = (np.clip(dir_norm, 0, 1) * 255).astype(np.uint8)
            
            # Distance to sphere visualization (single channel, convert to RGB)
            dist_vis = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            if first_dist_to_sphere.max() > first_dist_to_sphere.min():
                # Convert from [-1,1] to [0,255] and use as grayscale
                dist_norm = (first_dist_to_sphere + 1.0) / 2.0
                # Handle distance shape - it should be (H, W, 1) after permute, so take first channel
                if first_dist_to_sphere.shape[-1] == 1:
                    dist_gray = (np.clip(dist_norm[:, :, 0], 0, 1) * 255).astype(np.uint8)
                else:
                    # If it has multiple channels, take the first one
                    dist_gray = (np.clip(dist_norm[:, :, 0], 0, 1) * 255).astype(np.uint8)
                dist_vis[:, :, 0] = dist_gray
                dist_vis[:, :, 1] = dist_gray
                dist_vis[:, :, 2] = dist_gray
            
            # Create videos for all frames instead of just first frame visualizations
            import subprocess
            
            # Create video output directory
            video_output_dir = os.path.join(self.temp_dir, "map_videos")
            os.makedirs(video_output_dir, exist_ok=True)
            
            # Process all frames for each map type
            all_pos_vis = []
            all_img_vis = []
            all_normal_vis = []
            all_mask_vis = []
            all_dir_vis = []
            all_dist_vis = []
            
            for frame_idx in range(len(processed_data["position"])):
                # Get processed data for this frame
                frame_position = processed_data["position"][frame_idx].permute(1, 2, 0).numpy()
                frame_image = processed_data["image"][frame_idx].permute(1, 2, 0).numpy()
                frame_normal = processed_data["normals"][frame_idx].permute(1, 2, 0).numpy()
                frame_mask = processed_data["sphere_mask"][frame_idx].permute(1, 2, 0).numpy()
                frame_dir = processed_data["dir_to_sphere"][frame_idx].permute(1, 2, 0).numpy()
                frame_dist = processed_data["dist_to_sphere"][frame_idx].permute(1, 2, 0).numpy()
                
                # Position map visualization
                pos_vis = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                if frame_position.max() > frame_position.min():
                    pos_norm = (frame_position + 1.0) / 2.0
                    pos_vis = (pos_norm * 255).astype(np.uint8)
                all_pos_vis.append(pos_vis)
                
                # Image visualization
                img_vis = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                if frame_image.max() > frame_image.min():
                    image_norm = (frame_image + 1.0) / 2.0
                    img_vis = (image_norm * 255).astype(np.uint8)
                all_img_vis.append(img_vis)
                
                # Normal map visualization
                normal_vis = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                if np.any(frame_mask > 0):
                    normal_display = (frame_normal + 1.0) / 2.0
                    normal_vis = (np.clip(normal_display, 0, 1) * 255).astype(np.uint8)
                all_normal_vis.append(normal_vis)
                
                # Mask visualization
                mask_vis = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                if frame_mask.shape[-1] == 1:
                    mask_gray = ((frame_mask[:, :, 0] + 1.0) / 2.0 * 255).astype(np.uint8)
                else:
                    mask_gray = ((frame_mask[:, :, 0] + 1.0) / 2.0 * 255).astype(np.uint8)
                mask_vis[:, :, 0] = mask_gray
                mask_vis[:, :, 1] = mask_gray
                mask_vis[:, :, 2] = mask_gray
                all_mask_vis.append(mask_vis)
                
                # Direction visualization
                dir_vis = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                if frame_dir.max() > frame_dir.min():
                    dir_norm = (frame_dir + 1.0) / 2.0
                    dir_vis = (np.clip(dir_norm, 0, 1) * 255).astype(np.uint8)
                all_dir_vis.append(dir_vis)
                
                # Distance visualization
                dist_vis = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                if frame_dist.max() > frame_dist.min():
                    dist_norm = (frame_dist + 1.0) / 2.0
                    if frame_dist.shape[-1] == 1:
                        dist_gray = (np.clip(dist_norm[:, :, 0], 0, 1) * 255).astype(np.uint8)
                    else:
                        dist_gray = (np.clip(dist_norm[:, :, 0], 0, 1) * 255).astype(np.uint8)
                    dist_vis[:, :, 0] = dist_gray
                    dist_vis[:, :, 1] = dist_gray
                    dist_vis[:, :, 2] = dist_gray
                all_dist_vis.append(dist_vis)
            
            # Save videos for each map type
            pos_video_path = os.path.join(video_output_dir, "position_maps.mp4")
            img_video_path = os.path.join(video_output_dir, "image_maps.mp4")
            normal_video_path = os.path.join(video_output_dir, "normal_maps.mp4")
            mask_video_path = os.path.join(video_output_dir, "mask_maps.mp4")
            dir_video_path = os.path.join(video_output_dir, "direction_maps.mp4")
            dist_video_path = os.path.join(video_output_dir, "distance_maps.mp4")
            
            # Convert numpy arrays to videos using ffmpeg
            def save_frames_as_video(frames, output_path, fps=30):
                # Create temporary directory for frames
                temp_frames_dir = os.path.join(video_output_dir, "temp_frames")
                os.makedirs(temp_frames_dir, exist_ok=True)
                
                # Save frames as images
                for i, frame in enumerate(frames):
                    frame_path = os.path.join(temp_frames_dir, f"frame_{i:04d}.png")
                    Image.fromarray(frame).save(frame_path)
                
                # Create video with ffmpeg
                subprocess.run([
                    'ffmpeg', '-y', '-framerate', str(fps),
                    '-i', os.path.join(temp_frames_dir, 'frame_%04d.png'),
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    output_path
                ], capture_output=True, check=True)
                
                # Clean up temporary frames
                shutil.rmtree(temp_frames_dir)
                
                return output_path
            
            # Save all map videos
            pos_video_path = save_frames_as_video(all_pos_vis, pos_video_path)
            img_video_path = save_frames_as_video(all_img_vis, img_video_path)
            normal_video_path = save_frames_as_video(all_normal_vis, normal_video_path)
            mask_video_path = save_frames_as_video(all_mask_vis, mask_video_path)
            dir_video_path = save_frames_as_video(all_dir_vis, dir_video_path)
            dist_video_path = save_frames_as_video(all_dist_vis, dist_video_path)
            
            info_text = f"Maps prepared with interpolated spheres!\n" \
                       f"First frame: {self.first_frame_sphere['center']} (r={self.first_frame_sphere['radius']:.2f})\n" \
                       f"Last frame: {self.last_frame_sphere['center']} (r={self.last_frame_sphere['radius']:.2f})\n" \
                       f"Total frames processed: {len(sphere_masks)}\n" \
                       f"Total sphere pixels: {total_sphere_pixels}\n" \
                       f"Videos saved to: {video_output_dir}"

            return pos_video_path, img_video_path, normal_video_path, mask_video_path, dir_video_path, dist_video_path, info_text

        except Exception as e:
            return None, None, None, None, None, None, f"Error preparing maps: {str(e)}"
    
    def run_inference(self, prompt, num_steps: int = 50, ev: int = 0, seed: int = 42):
        """Run inference using the loaded model"""
        if self.temp_dir is None:
            return None, "Please add a sphere first!"
        
        if self.model is None:
            return None, "Please load inference model first!"
        
        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        processed_data = self.processed_data
        processed_data["path"] = ""
        
        # Move to device
        for key in processed_data.keys():
            if isinstance(processed_data[key], list):
                for i in range(len(processed_data[key])):
                    if isinstance(processed_data[key][i], torch.Tensor):
                        processed_data[key][i] = processed_data[key][i].to("cuda")

        #prompt = prompt + f" [EV{ev}]"
        #processed_data["prompt"] = prompt
        # Run inference
        with torch.no_grad():
            # inputs = forward_preprocess(processed_data, self.model.pipe, self.condition_list)
            # #models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
            # result_video = self.model.pipe(prompt, num_inference_steps=num_steps, inputs_shared=inputs)
            result_video = []
            for i in range(len(processed_data["image"])):
                conditions = [processed_data[k][i] for k in self.condition_list]
                if conditions[0].dim() == 5:
                    b,l,c,h,w = conditions[0].shape
                    conditions = [condition[:,0] for condition in conditions]
                elif conditions[0].dim() == 4:
                    l=0
                    b,c,h,w = conditions[0].shape
                else:
                    l=0
                    b=1
                    c,h,w = conditions[0].shape
                    conditions = [condition.unsqueeze(0) for condition in conditions]

                latent_conditions = [encode_with_vae(self.model.vae, condition, weight_dtype=self.model.vae.dtype) for condition in conditions]

                generator = torch.Generator(device="cuda").manual_seed(seed)

                pipeline_args = {
                    "prompt_embeds": self.prompt_embeds_dict[prompt][ev],
                    "pooled_prompt_embeds": self.pooled_prompt_embeds_dict[prompt][ev],
                    "latent_conditions": latent_conditions,
                    "width": w,
                    "height": h,
                    "guidance_scale": 1.0,
                    "generator": generator,
                    "num_inference_steps": num_steps,
                    "sigmas": None,
                    "output_type": "pt"
                }
                output_img = self.model(**pipeline_args).images[0]
                result_video.append(Image.fromarray((output_img.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)))
        # Save results
        output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        output_frames = []
        for i, frame in enumerate(result_video):
            frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
            frame.save(frame_path)
            output_frames.append(np.array(frame))
        
        # Create output video
        output_video_path = os.path.join(output_dir, "result.mp4")
        subprocess.run([
            'ffmpeg', '-y', '-framerate', '30', 
            '-i', os.path.join(output_dir, 'frame_%04d.png'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            output_video_path
        ], capture_output=True)
        
        return output_video_path, f"Inference completed successfully!\nOutput saved to: {output_dir}"
            
        # except Exception as e:
        #     return None, f"Error during inference: {str(e)}"
    
    def run_inference_all(self, num_steps: int = 20, seed: int = 42):
        """Run inference for all sphere types and EV values"""
        if self.temp_dir is None:
            return None, "Please add a sphere first!"
        
        if self.model is None:
            return None, "Please load inference model first!"
        
        # Define sphere types and EV values
        sphere_types = ["sphere_0", "sphere_1"]
        ev_values = [0, -3, -6, -9]
        
        # Create output directory for all results
        all_output_dir = os.path.join(self.temp_dir, "output_all")
        os.makedirs(all_output_dir, exist_ok=True)
        
        results_summary = "Inference All completed!\n\n"
        
        # Run inference for each combination
        for sphere_key in sphere_types:
            sphere_prompt = sphere_key
            
            for ev in ev_values:
                # Set random seeds for reproducibility
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                
                processed_data = copy.deepcopy(self.processed_data)
                processed_data["path"] = ""
                
                # Move to device
                for key in processed_data.keys():
                    if isinstance(processed_data[key], list):
                        for i in range(len(processed_data[key])):
                            if isinstance(processed_data[key][i], torch.Tensor):
                                processed_data[key][i] = processed_data[key][i].to("cuda")
                
                #prompt = sphere_prompt + f" [EV{ev}]"
                #processed_data["prompt"] = prompt
                prompt = sphere_prompt
                
                # Run inference
                with torch.no_grad():
                    # inputs = forward_preprocess(processed_data, self.model.pipe, self.condition_list)
                    # result_video = self.model.pipe(prompt, num_inference_steps=num_steps, inputs_shared=inputs)
                    result_video = []
                    for i in range(len(processed_data["image"])):
                        conditions = [processed_data[k][i] for k in self.condition_list]
                        if conditions[0].dim() == 5:
                            b,l,c,h,w = conditions[0].shape
                            conditions = [condition[:,0] for condition in conditions]
                        elif conditions[0].dim() == 4:
                            l=0
                            b,c,h,w = conditions[0].shape
                        else:
                            l=0
                            b=1
                            c,h,w = conditions[0].shape
                            conditions = [condition.unsqueeze(0) for condition in conditions]

                        latent_conditions = [encode_with_vae(self.model.vae, condition, weight_dtype=self.model.vae.dtype) for condition in conditions]

                        generator = torch.Generator(device="cuda").manual_seed(seed)

                        pipeline_args = {
                            "prompt_embeds": self.prompt_embeds_dict[prompt][ev],
                            "pooled_prompt_embeds": self.pooled_prompt_embeds_dict[prompt][ev],
                            "latent_conditions": latent_conditions,
                            "width": w,
                            "height": h,
                            "guidance_scale": 1.0,
                            "generator": generator,
                            "num_inference_steps": num_steps,
                            "sigmas": None,
                            "output_type": "pt"
                        }
                        output_img = self.model(**pipeline_args).images[0]
                        result_video.append(Image.fromarray((output_img.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)))
                
                # Save results for this combination
                combo_name = f"{sphere_key}_EV{ev}"
                combo_dir = os.path.join(all_output_dir, combo_name)
                os.makedirs(combo_dir, exist_ok=True)
                
                for i, frame in enumerate(result_video):
                    frame_path = os.path.join(combo_dir, f"frame_{i:04d}.png")
                    frame.save(frame_path)
                
                # Create video for this combination
                video_path = os.path.join(combo_dir, f"{combo_name}.mp4")
                subprocess.run([
                    'ffmpeg', '-y', '-framerate', '30', 
                    '-i', os.path.join(combo_dir, 'frame_%04d.png'),
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    video_path
                ], capture_output=True)
                
                results_summary += f"✓ {sphere_key} @ EV{ev}: {video_path}\n"
        
        # Create a zip file of all results
        import zipfile
        zip_path = os.path.join(self.temp_dir, "output_all.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(all_output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, self.temp_dir)
                    zipf.write(file_path, arcname)
        
        results_summary += f"\nAll results saved to: {all_output_dir}\n"
        results_summary += f"Zip file created: {zip_path}"
        return zip_path, results_summary
    
    def run_hdri_optimization(self, evs_str: str = "0,-3,-6,-9", num_iterations: int = 1000, 
                             learning_rate: float = 0.005, proj: str = "latlong"):
        """Run HDRi environment map optimization using inferred sphere videos"""
        if self.temp_dir is None:
            return None, None, "Please complete inference first!"
        
        # Check if inference results exist
        all_output_dir = os.path.join(self.temp_dir, "output_all")
        if not os.path.exists(all_output_dir):
            return None, None, "Please run 'Inference All' first to generate sphere videos!"
        
        try:
            import torch.nn.functional as F
            from ezexr import imread, imsave
            from HDRMerge.batch_optimization_pol_manual_gt import (
                render, load_reference_images, Lin_to_sRGB,
                sRGB_to_Lin, multi_importance_sample
            )
            from HDRMerge.torch_PoL.torchpol.torch_pol import PoL
            import random
            import math
            import torchvision
            
            evs = evs_str.split(",")
            
            # Create output directory for HDR results
            hdr_output_dir = os.path.join(self.temp_dir, "hdr_optimization")
            os.makedirs(hdr_output_dir, exist_ok=True)
            
            optimization_log = "HDRi Optimization Started\n\n"
            optimization_log += f"Parameters:\n"
            optimization_log += f"  - EV values: {evs}\n"
            optimization_log += f"  - Iterations: {num_iterations}\n"
            optimization_log += f"  - Learning rate: {learning_rate}\n"
            optimization_log += f"  - Projection: {proj}\n\n"
            
            # Process each frame
            num_frames = len(self.depth_frames)
            init = None
            
            preview_images = []
            
            for frame_idx in range(num_frames):
                frame_start_time = time.time()
                optimization_log += f"Processing frame {frame_idx + 1}/{num_frames}...\n"
                
                # Load camera info
                world2cam = torch.eye(4).cuda()  # Identity for now, assuming camera at origin
                
                # Get geometry from processed data (use frame_idx)
                if frame_idx >= len(self.processed_data["position_raw"]):
                    optimization_log += f"  Warning: Frame {frame_idx} out of range, skipping\n"
                    continue
                
                position_map = self.processed_data["position_raw"][frame_idx].cuda()  # [C, H, W]
                normal_map = self.processed_data["normals"][frame_idx].cuda()
                mask = self.processed_data["sphere_mask"][frame_idx].cuda()  # Take first channel

                #position_map[2,:,:] = -position_map[2,:,:]  # The raw data from the dataloader isn't reversed
                #normal_map[2,:,:] = -normal_map[2,:,:]  # The treatdata function inverts the normal maps
                mask = (mask[0, :, :] > 0).float().unsqueeze(0)  # Binarize mask

                # #resize to 512x512 for optimization
                # position_map = F.interpolate(position_map.unsqueeze(0), size=(256, 256), mode='bilinear').squeeze(0)
                # normal_map = F.interpolate(normal_map.unsqueeze(0), size=(256, 256), mode='bilinear').squeeze(0)
                # mask = F.interpolate(mask.unsqueeze(0), size=(256, 256), mode='nearest').squeeze(0)
                
                # Load reference images for both spheres and all EVs
                ref_img_0 = {}
                ref_img_1 = {}
                
                for ev in evs:
                    # Load sphere_0 reference
                    sphere0_path = os.path.join(all_output_dir, f"sphere_0_EV{ev}", f"frame_{frame_idx:04d}.png")
                    if os.path.exists(sphere0_path):
                        ref_img = Image.open(sphere0_path).convert("RGB")
                        #ref_img = ref_img.resize((256, 256), Image.LANCZOS)
                        ref_img = np.array(ref_img, dtype=np.float32) / 255.0
                        ref_img = torch.from_numpy(ref_img).permute(2, 0, 1).cuda()
                        ref_img_0[ev] = ref_img
                    
                    # Load sphere_1 reference
                    sphere1_path = os.path.join(all_output_dir, f"sphere_1_EV{ev}", f"frame_{frame_idx:04d}.png")
                    if os.path.exists(sphere1_path):
                        ref_img = Image.open(sphere1_path).convert("RGB")
                        #ref_img = ref_img.resize((256, 256), Image.LANCZOS)
                        ref_img = np.array(ref_img, dtype=np.float32) / 255.0
                        ref_img = torch.from_numpy(ref_img).permute(2, 0, 1).cuda()
                        ref_img_1[ev] = ref_img
                
                if not ref_img_0 and not ref_img_1:
                    optimization_log += f"  Warning: No reference images found for frame {frame_idx}\n"
                    continue
                
                # Crop around the sphere
                indices = (mask[0, :, :] == 1).nonzero(as_tuple=False)
                if len(indices) == 0:
                    optimization_log += f"  Warning: No valid mask pixels for frame {frame_idx}\n"
                    continue
                
                min_indices = torch.amin(indices, dim=0)
                max_indices = torch.amax(indices, dim=0)
                
                mask = mask[:, min_indices[0]:max_indices[0]+1, min_indices[1]:max_indices[1]+1]
                normal_map = normal_map[:, min_indices[0]:max_indices[0]+1, min_indices[1]:max_indices[1]+1]
                position_map = position_map[:, min_indices[0]:max_indices[0]+1, min_indices[1]:max_indices[1]+1]
                
                # Crop reference images
                for sphere_refs in [ref_img_0, ref_img_1]:
                    for ev in sphere_refs.keys():
                        sphere_refs[ev] = sphere_refs[ev][:, min_indices[0]:max_indices[0]+1, min_indices[1]:max_indices[1]+1]
                
                print(position_map.shape, normal_map.shape, mask.shape)
                print(torch.min(mask), torch.max(mask))
                # # save temp
                # torchvision.utils.save_image(ref_img_0[evs[0]], f"img{frame_idx:04d}.png")
                # torchvision.utils.save_image(normal_map/2 + 0.5, f"normal_map_{frame_idx:04d}.png")
                # torchvision.utils.save_image(position_map/2 + 0.5, f"position_map_{frame_idx:04d}.png")
                # torchvision.utils.save_image(mask, f"mask_{frame_idx:04d}.png")
                # raise('ok')
    
                # Resize if too large
                # max_size = 256
                # _, H, W = mask.shape
                # if H > max_size or W > max_size:
                #     scale_factor = max_size / max(H, W)
                #     new_H, new_W = int(H * scale_factor), int(W * scale_factor)
                    
                #     mask = F.interpolate(mask.unsqueeze(0), size=(new_H, new_W), mode='nearest').squeeze(0)
                #     normal_map = F.interpolate(normal_map.unsqueeze(0), size=(new_H, new_W), mode='bilinear').squeeze(0)
                #     position_map = F.interpolate(position_map.unsqueeze(0), size=(new_H, new_W), mode='bilinear').squeeze(0)
                    
                #     for sphere_refs in [ref_img_0, ref_img_1]:
                #         for ev in sphere_refs.keys():
                #             sphere_refs[ev] = F.interpolate(sphere_refs[ev].unsqueeze(0), size=(new_H, new_W), mode='bilinear').squeeze(0)
                
                # Initialize environment map
                _, H, W = position_map.shape
                size = max(H, W)
                init = torch.ones(1, 3, 256, 512).cuda() * 0.5
                
                # Setup PoL optimization
                pol = PoL(init, learning_rate=learning_rate, weight_decay=0.05, padding_mode="circular").cuda()
                optimizer = torch.optim.AdamW(pol.parameters(), betas=(0.9, 0.99), fused=True)
                
                # Optimization setup
                list_spheres = []
                modes = {}
                references = {}
                weights = {}
                
                if ref_img_0:
                    list_spheres.append("sphere_0")
                    modes["sphere_0"] = "mirror"
                    references["sphere_0"] = ref_img_0
                    weights["sphere_0"] = 1.0
                
                if ref_img_1:
                    list_spheres.append("sphere_1")
                    modes["sphere_1"] = "diffuse"
                    references["sphere_1"] = ref_img_1
                    weights["sphere_1"] = 1.0
                
                # Optimization loop
                for i in range(num_iterations):
                    optimizer.zero_grad()
                    recon = pol()
                    recon = recon.squeeze(0)
                    recon = torch.pow(2, recon)
                    recon = torch.nan_to_num(recon, nan=0.0)
                    
                    random_sphere = random.choice(list_spheres)
                    ev = random.choice(evs)
                    
                    ref_img = references[random_sphere][ev]
                    mode = modes[random_sphere]
                    
                    img = render(position_map, world2cam, normal_map, recon, mode=mode, num_samples=64, proj=proj)
                    img = img * (2**int(ev))
                    img = Lin_to_sRGB(img).clamp(0, 1)
                    
                    mask_sat = torch.ones_like(ref_img)
                    threshold = 0.90
                    mask_sat = torch.where((ref_img > threshold) & (img > threshold), 0.0, 1.0)
                    
                    loss = weights[random_sphere] * torch.mean(torch.square((img - ref_img) * mask * mask_sat))
                    loss.backward()
                    optimizer.step()
                    pol.end_iter_callback(i)
                
                # Save HDR result
                envmap_hdr = recon.permute(1, 2, 0).cpu().detach().numpy()
                hdr_path = os.path.join(hdr_output_dir, f"envmap_{frame_idx:04d}.exr")
                imsave(hdr_path, envmap_hdr, compression='ZIP')
                
                # Save LDR preview
                envmap_ldr = Lin_to_sRGB(recon).clamp(0, 1)
                ldr_path = os.path.join(hdr_output_dir, f"envmap_ldr_{frame_idx:04d}.png")
                torchvision.utils.save_image(envmap_ldr, ldr_path)
                preview_images.append(ldr_path)

                # #final render diffuse and mirror for ev -3
                recon = pol()
                recon = recon.squeeze(0)

                recon = torch.pow(2, recon)
                recon = torch.nan_to_num(recon, nan=0.0)
                ev = "0"
                mode = "mirror"

                img = render(position_map, world2cam, normal_map, recon, mode=mode, num_samples=64, proj=proj)
                img = img * (2**int(ev))  # Scale the image according to the exposure value
                
                img = Lin_to_sRGB(img).clamp(0,1)

                torchvision.utils.save_image(img, "mirror_pred.png")


                frame_time = time.time() - frame_start_time
                optimization_log += f"  ✓ Frame {frame_idx + 1} completed in {frame_time:.2f}s\n"
                optimization_log += f"    HDR saved: {hdr_path}\n"
                
                # Reset init for next frame (or keep for temporal consistency)
                init = None
            
            # Create zip of all results
            zip_path = os.path.join(self.temp_dir, "hdr_envmaps.zip")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(hdr_output_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, self.temp_dir)
                        zipf.write(file_path, arcname)
            
            optimization_log += f"\n✓ Optimization complete!\n"
            optimization_log += f"HDR results saved to: {hdr_output_dir}\n"
            optimization_log += f"Zip file created: {zip_path}\n"
            
            # Return first preview image and zip
            first_preview = preview_images[0] if preview_images else None
            
            return first_preview, zip_path, optimization_log
            
        except Exception as e:
            import traceback
            return None, None, f"Error during optimization: {str(e)}\n\n{traceback.format_exc()}"
    
    def create_blender_scene_with_hdri(self):
        """Create Blender scene with depth mesh as shadow catcher and optimized HDRi as lighting"""
        if self.temp_dir is None:
            return None, "Please complete depth prediction first!"
        
        # Check if HDR optimization results exist
        hdr_output_dir = os.path.join(self.temp_dir, "hdr_optimization")
        if not os.path.exists(hdr_output_dir):
            return None, "Please run HDRi optimization first!"
        
        try:
            # Create Blender script
            blender_script = '''
import bpy
import numpy as np
import pickle
import sys
import os

# Load data from pickle file
data_path = sys.argv[-1]
with open(data_path, 'rb') as f:
    data = pickle.load(f)

depth_frames = data['depth_frames']
video_frames = data['video_frames']
fx, fy, cx, cy = data['fx'], data['fy'], data['cx'], data['cy']
height, width = data['height'], data['width']
vfov_rad = data['vfov_rad']
num_frames = len(depth_frames)
hdr_paths = data['hdr_paths']

# Clear existing scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Set up render engine to Cycles
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.device = 'GPU'

# Create camera with correct FOV
bpy.ops.object.camera_add(location=(0, 0, 0))
camera = bpy.context.object
camera.name = "DepthCamera"

# Set camera parameters
camera.data.lens_unit = 'MILLIMETERS'
sensor_width_mm = 36.0
sensor_height_mm = sensor_width_mm * (height / width)
focal_length_mm = (fx / width) * sensor_width_mm

camera.data.lens = focal_length_mm
camera.data.sensor_width = sensor_width_mm
camera.data.sensor_height = sensor_height_mm
camera.data.sensor_fit = 'HORIZONTAL'

# Set as active camera
bpy.context.scene.camera = camera

# Set up scene frame range
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = num_frames

# Create world environment with HDRi
world = bpy.data.worlds.new("HDRi_World")
bpy.context.scene.world = world
world.use_nodes = True

# Clear default nodes
nodes = world.node_tree.nodes
links = world.node_tree.links
nodes.clear()

# Create nodes for world shader
output_node = nodes.new(type='ShaderNodeOutputWorld')
output_node.location = (300, 0)

background_node = nodes.new(type='ShaderNodeBackground')
background_node.location = (0, 0)

env_tex_node = nodes.new(type='ShaderNodeTexEnvironment')
env_tex_node.location = (-300, 0)

# Add mapping node to rotate environment texture
mapping_node = nodes.new(type='ShaderNodeMapping')
mapping_node.location = (-500, 0)

tex_coord_node = nodes.new(type='ShaderNodeTexCoord')
tex_coord_node.location = (-700, 0)

# Rotate environment: X=90, Z=-90 degrees (camera facing -Z)
mapping_node.inputs['Rotation'].default_value = (1.5708, 0.0, -1.5708)  # radians

# Link texture coordinate to mapping to environment texture
links.new(tex_coord_node.outputs['Generated'], mapping_node.inputs['Vector'])
links.new(mapping_node.outputs['Vector'], env_tex_node.inputs['Vector'])

# Link nodes
links.new(env_tex_node.outputs['Color'], background_node.inputs['Color'])
links.new(background_node.outputs['Background'], output_node.inputs['Surface'])

# Load HDR image sequence
if len(hdr_paths) > 0 and os.path.exists(hdr_paths[0]):
    # Load the first image as a sequence
    env_tex_node.image = bpy.data.images.load(hdr_paths[0])
    env_tex_node.image.colorspace_settings.name = 'Linear Rec.709'
    
    # Enable image sequence if we have multiple frames
    if len(hdr_paths) > 1:
        env_tex_node.image.source = 'SEQUENCE'
        env_tex_node.image_user.frame_duration = len(hdr_paths)
        env_tex_node.image_user.frame_start = 1
        env_tex_node.image_user.frame_offset = -1
        env_tex_node.image_user.use_auto_refresh = True
        env_tex_node.image_user.use_cyclic = False

# Create mesh with shadow catcher material for each frame
for frame_idx, depth in enumerate(depth_frames):
    # Subsample for performance
    subsample = 1
    depth_sub = depth[::subsample, ::subsample]
    
    # Create grid for subsampled resolution
    h_sub, w_sub = depth_sub.shape
    i_sub, j_sub = np.meshgrid(np.arange(0, width, subsample), np.arange(0, height, subsample))
    
    # Calculate 3D positions
    x_sub = (i_sub - cx) / fx * depth_sub
    y_sub = -(j_sub - cy) / fy * depth_sub
    z_sub = -depth_sub
    
    # Flatten arrays
    vertices = np.stack([x_sub.flatten(), y_sub.flatten(), z_sub.flatten()], axis=1)
    
    # Create faces
    faces = []
    for row in range(h_sub - 1):
        for col in range(w_sub - 1):
            v0 = row * w_sub + col
            v1 = row * w_sub + (col + 1)
            v2 = (row + 1) * w_sub + (col + 1)
            v3 = (row + 1) * w_sub + col
            faces.append([v0, v1, v2, v3])
    
    # Create mesh
    mesh_name = f"ShadowCatcher_Frame_{frame_idx:04d}"
    mesh = bpy.data.meshes.new(mesh_name)
    obj = bpy.data.objects.new(mesh_name, mesh)
    bpy.context.collection.objects.link(obj)
    
    # Add vertices and faces
    mesh.from_pydata(vertices.tolist(), [], faces)
    mesh.update()
    
    # Create shadow catcher material
    mat_name = f"ShadowCatcher_Material_{frame_idx:04d}"
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    
    # Basic diffuse shader for shadow catcher
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    diffuse_node = nodes.new(type='ShaderNodeBsdfDiffuse')
    diffuse_node.inputs['Color'].default_value = (0.8, 0.8, 0.8, 1.0)
    
    mat.node_tree.links.new(diffuse_node.outputs['BSDF'], output_node.inputs['Surface'])
    
    # Assign material
    if len(obj.data.materials):
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    
    # Enable shadow catcher on the object (not material)
    obj.is_shadow_catcher = True
    
    # Make mesh invisible to all rays except camera
    obj.visible_diffuse = False
    obj.visible_glossy = False
    obj.visible_transmission = False
    obj.visible_volume_scatter = False
    obj.visible_shadow = False
    
    # Hide all except first frame
    obj.hide_viewport = (frame_idx != 0)
    obj.hide_render = (frame_idx != 0)
    
    # Keyframe visibility
    for f in range(1, num_frames + 1):
        obj.hide_viewport = (f != frame_idx + 1)
        obj.hide_render = (f != frame_idx + 1)
        obj.keyframe_insert(data_path="hide_viewport", frame=f)
        obj.keyframe_insert(data_path="hide_render", frame=f)

# Create single animated sphere from sphere data
sphere_data = data.get('sphere_data', [])
if len(sphere_data) > 0:
    # Get first frame data to create the sphere
    first_frame = sphere_data[0]
    sphere_center = first_frame['center']
    sphere_radius = first_frame['radius']
    
    # Convert Blender coordinates to camera space (reverse Z)
    sphere_x, sphere_y, sphere_z = sphere_center
    sphere_z = sphere_z  # Don't flip Z - keep it as is from Blender
    
    # Create sphere mesh
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=sphere_radius,
        location=(sphere_x, sphere_y, sphere_z),
        segments=64,
        ring_count=32
    )
    sphere_obj = bpy.context.object
    sphere_obj.name = "AnimatedSphere"
    
    # Create material for sphere
    sphere_mat = bpy.data.materials.new(name="Sphere_Material")
    sphere_mat.use_nodes = True
    
    sphere_nodes = sphere_mat.node_tree.nodes
    sphere_nodes.clear()
    
    sphere_output = sphere_nodes.new(type='ShaderNodeOutputMaterial')
    sphere_bsdf = sphere_nodes.new(type='ShaderNodeBsdfPrincipled')
    
    # Set sphere material properties (make it slightly reflective)
    sphere_bsdf.inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1.0)
    sphere_bsdf.inputs['Metallic'].default_value = 0.5
    sphere_bsdf.inputs['Roughness'].default_value = 0.2
    
    sphere_mat.node_tree.links.new(sphere_bsdf.outputs['BSDF'], sphere_output.inputs['Surface'])
    
    # Assign material
    if len(sphere_obj.data.materials):
        sphere_obj.data.materials[0] = sphere_mat
    else:
        sphere_obj.data.materials.append(sphere_mat)
    
    # Animate sphere position and scale for each frame
    for frame_idx, frame_sphere in enumerate(sphere_data):
        frame_num = frame_idx + 1
        sphere_center = frame_sphere['center']
        sphere_radius = frame_sphere['radius']
        
        # Convert coordinates (keep Z as is)
        sphere_x, sphere_y, sphere_z = sphere_center
        
        # Set location for this frame
        sphere_obj.location = (sphere_x, sphere_y, sphere_z)
        sphere_obj.keyframe_insert(data_path="location", frame=frame_num)
        
        # Set scale for this frame (uniform scaling from radius)
        scale = sphere_radius / first_frame['radius']
        sphere_obj.scale = (scale, scale, scale)
        sphere_obj.keyframe_insert(data_path="scale", frame=frame_num)

# Add video frames as camera background (animated)
if video_frames is not None and len(video_frames) > 0:
    # Create image sequence in Blender
    bg_images = []
    for i, frame in enumerate(video_frames):
        # Create new image in Blender
        img_name = f"bg_frame_{i:04d}"
        h, w = frame.shape[:2]
        bg_img = bpy.data.images.new(img_name, width=w, height=h, alpha=False)
        
        # Flatten frame data and normalize to [0, 1]
        pixels = frame.astype(np.float32) / 255.0
        
        # Blender images are stored as flat array in RGBA format
        # Convert RGB to RGBA and flatten
        rgba = np.zeros((h, w, 4), dtype=np.float32)
        rgba[:, :, :3] = pixels  # RGB channels
        rgba[:, :, 3] = 1.0  # Alpha channel
        
        # Blender expects bottom-to-top, so flip vertically
        rgba = np.flipud(rgba)
        
        # Flatten to 1D array
        bg_img.pixels = rgba.flatten()
        bg_img.pack()  # Pack image into .blend file
        bg_images.append(bg_img)
    
    # Setup animated background by creating multiple background image slots
    # Each slot will be visible for one frame
    if len(bg_images) > 0:
        camera.data.show_background_images = True
        
        # Create a background slot for each frame
        for i, bg_img in enumerate(bg_images):
            bg = camera.data.background_images.new()
            bg.image = bg_img
            bg.alpha = 1.0
            bg.display_depth = 'BACK'
            bg.frame_method = 'STRETCH'
            
            # Animate show/hide for each background slot
            # Show only on its corresponding frame
            for f in range(1, num_frames + 1):
                bg.show_background_image = (f == i + 1)
                bg.keyframe_insert(data_path="show_background_image", frame=f)

# Set up render settings
bpy.context.scene.render.resolution_x = width
bpy.context.scene.render.resolution_y = height
bpy.context.scene.render.resolution_percentage = 100
bpy.context.scene.render.film_transparent = True  # Enable transparent background

# Save blend file
blend_path = data['blend_path']
bpy.ops.wm.save_as_mainfile(filepath=blend_path)
'''
            
            # Find all HDR files
            hdr_files = sorted([
                os.path.join(hdr_output_dir, f) 
                for f in os.listdir(hdr_output_dir) 
                if f.endswith('.exr') and f.startswith('envmap_')
            ])
            
            if not hdr_files:
                return None, "No HDR environment maps found! Please run optimization first."
            
            # Prepare data for Blender
            data_for_blender = {
                'depth_frames': self.depth_frames,
                'video_frames': self.video_frames,
                'fx': self.fx,
                'fy': self.fy,
                'cx': self.cx,
                'cy': self.cy,
                'height': self.height,
                'width': self.width,
                'vfov_rad': 2 * np.arctan(self.height / (2 * self.fy)),
                'hdr_paths': hdr_files,
                'sphere_data': self.sphere_data if hasattr(self, 'sphere_data') and self.sphere_data else [],
                'blend_path': os.path.join(self.temp_dir, "hdri_scene.blend")
            }
            
            data_pickle_path = os.path.join(self.temp_dir, "blender_hdri_data.pkl")
            with open(data_pickle_path, 'wb') as f:
                pickle.dump(data_for_blender, f)
            
            # Save Blender script
            script_path = os.path.join(self.temp_dir, "create_hdri_scene.py")
            with open(script_path, 'w') as f:
                f.write(blender_script)
            
            # Find Blender executable
            blender_paths = [
                "/root/blender-4.4.0-linux-x64/blender",
                "/root/blender-4.3.2-linux-x64/blender",
                "blender"
            ]
            
            blender_exe = None
            for path in blender_paths:
                if os.path.exists(path) or path == "blender":
                    blender_exe = path
                    break
            
            if not blender_exe:
                return None, "Blender executable not found!"
            
            # Run Blender in background
            blend_path = data_for_blender['blend_path']
            result = subprocess.run(
                [blender_exe, "--background", "--python", script_path, "--", data_pickle_path],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                return None, f"Blender error (return code {result.returncode}):\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
            
            if not os.path.exists(blend_path):
                return None, f"Failed to create Blender file at {blend_path}\n\nBlender output:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
            
            num_frames = len(self.depth_frames)
            num_hdris = len(hdr_files)
            
            return (
                blend_path,
                f"Blender HDRi scene created successfully!\n"
                f"Location: {blend_path}\n\n"
                f"Scene Setup:\n"
                f"  - Render Engine: Cycles\n"
                f"  - Shadow Catcher: Enabled on depth mesh\n"
                f"  - HDR Environment Maps: {num_hdris} frames\n"
                f"  - Resolution: {self.width}x{self.height}\n"
                f"  - Frames: {num_frames}\n\n"
                f"Usage:\n"
                f"1. Open the .blend file in Blender\n"
                f"2. Switch to 'Rendered' viewport shading to see lighting\n"
                f"3. The depth mesh acts as a shadow catcher\n"
                f"4. HDRi lighting changes per frame (if animated)\n"
                f"5. Press F12 to render or Space to play animation\n"
                f"6. Adjust background strength in World Properties if needed"
            )
            
        except subprocess.TimeoutExpired:
            return None, "Blender execution timed out"
        except Exception as e:
            return None, f"Error creating Blender scene: {str(e)}\n{traceback.format_exc()}"

# Initialize the demo
load_shared_models()
demo_instance = VideoDepthDemo()


# Create Gradio interface
with gr.Blocks(title="Video Depth + Sphere Inference Demo") as demo:
    gr.Markdown("# Sphere Inference Demo")
    
    with gr.Tab("1. Depth Prediction"):
        with gr.Row():
            with gr.Column():
                video_input = gr.File(label="Input Video or Image", file_types=["video", "image"])
                with gr.Row():
                    use_ml_focal = gr.Checkbox(value=True, label="Use ML Focal Length Prediction")
                    fov_slider = gr.Slider(30, 120, value=90, label="Manual Field of View (degrees)")
                gr.Markdown("**Input**: Upload a video or image. Images will be duplicated to 21 frames.")
                gr.Markdown("**ML Prediction**: Automatically infer field of view from the first frame using MoGe")
                gr.Markdown("**Manual FOV**: Use the slider value if ML prediction is disabled or fails")
                predict_btn = gr.Button("Predict Depth & Focal Length", variant="primary")
            
            with gr.Column():
                depth_video_output = gr.Video(label="Depth Video")
                orig_frames_zip_output = gr.File(label="Original Frames (ZIP)")
                depth_info = gr.Textbox(label="Depth Info", lines=8)
        
        predict_btn.click(
            demo_instance.predict_depth_and_focal,
            inputs=[video_input, fov_slider, use_ml_focal],
            outputs=[depth_video_output, orig_frames_zip_output, depth_info]
        )
    
    with gr.Tab("2. Export to Blender"):
        gr.Markdown("### Export Depth as Point Cloud to Blender")
        gr.Markdown("Creates a Blender file with animated depth point cloud, camera setup, and a probe sphere.")
        
        with gr.Row():
            with gr.Column():
                export_blender_btn = gr.Button("Export to Blender", variant="primary")
                blender_info = gr.Textbox(label="Export Info", lines=12)
            
            with gr.Column():
                gr.Markdown("### Download Blender File")
                blender_file_output = gr.File(label="Blender File (.blend)")
        
        export_blender_btn.click(
            demo_instance.export_to_blender,
            inputs=[],
            outputs=[blender_file_output, blender_info]
        )
    
    with gr.Tab("3. Sphere Data Input"):
        gr.Markdown("### Paste Sphere Data from Blender")
        gr.Markdown("""
        **Instructions:**
        1. Open the .blend file in Blender
        2. Position and animate the 'probe' sphere as desired
        3. Open the 'extract_sphere_data.py' script in Blender's Text Editor
        4. Run the script (Alt+P or click 'Run Script')
        5. Copy the JSON output from the Console
        6. Paste it in the text box below
        """)
        
        with gr.Row():
            with gr.Column():
                sphere_data_input = gr.Textbox(
                    label="Sphere Data (JSON)",
                    placeholder='[{"frame": 1, "center": [0.0, 0.0, -2.0], "radius": 0.5}, ...]',
                    lines=10
                )
            
            with gr.Column():
                gr.Markdown("### Expected JSON Format")
                gr.Code(
                    value='''[
  {
    "frame": 1,
    "center": [x, y, z],
    "radius": r
  },
  ...
]''',
                    language="json",
                    label="Example"
                )
    
    with gr.Tab("4. Prepare Maps"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Generate Dataset Maps from Sphere Data")
                gr.Markdown("Creates position maps, normal maps, sphere masks for all frames using sphere data from Blender")
                prepare_maps_btn = gr.Button("Prepare Maps", variant="primary")
                maps_info = gr.Textbox(label="Maps Info", lines=6)
            
            with gr.Column():
                gr.Markdown("### Processed Maps (All Frames)")
                gr.Markdown("**After treat_data processing - ready for inference**")
                with gr.Row():
                    position_map_vis = gr.Video(label="Position Map Video")
                    normal_map_vis = gr.Video(label="Normal Map Video")
                with gr.Row():
                    mask_map_vis = gr.Video(label="Sphere Mask Video")
                    dir_map_vis = gr.Video(label="Direction to Sphere Video")
                with gr.Row():
                    dist_map_vis = gr.Video(label="Distance to Sphere Video")
                    image_map_vis = gr.Video(label="Image Map Video")
        
        prepare_maps_btn.click(
            demo_instance.prepare_maps,
            inputs=[sphere_data_input],
            outputs=[position_map_vis, image_map_vis, normal_map_vis, mask_map_vis, dir_map_vis, dist_map_vis, maps_info]
        )
    
    # with gr.Tab("4. Model Setup"):
    #     with gr.Row():
    #         with gr.Column():
    #             model_path = gr.Textbox(
    #                 label="Model Path", 
    #                 placeholder="/root/Projects/balls/Wan/Wan_26sep_2.2/step-150000.safetensors",
    #                 value="/root/Projects/balls/Wan/Wan_26sep_2.2/step-150000.safetensors",
    #                 interactive=False
    #             )
    #             conditions = gr.Textbox(
    #                 placeholder="position,normals,dir_to_sphere,dist_to_sphere,sphere_mask",
    #                 label="Condition List (comma-separated)",
    #                 value="position,normals,dir_to_sphere,dist_to_sphere,sphere_mask",
    #                 interactive=False
    #             )
    #             setup_model_btn = gr.Button("Load Model", variant="primary")
            
    #         with gr.Column():
    #             model_status = gr.Textbox(label="Model Status", lines=3)
        
    #     setup_model_btn.click(
    #         demo_instance.setup_inference_model,
    #         inputs=[model_path, conditions],
    #         outputs=[model_status]
    #     )
    
    with gr.Tab("5. Inference"):
        with gr.Row():
            with gr.Column():
                sphere = gr.Dropdown(
                    choices=[key for key in "sphere_0,sphere_1".split(",")],
                    value="sphere_0",
                    label="Sphere"
                )
                num_steps = gr.Slider(10, 100, value=20, label="Inference Steps")
                ev_value = gr.Slider(-12, 0, value=0, step=3, label="EV Value")
                seed_value = gr.Number(value=42, label="Random Seed", precision=0)
                inference_btn = gr.Button("Run Inference", variant="primary")
            
            with gr.Column():
                output_video = gr.Video(label="Generated Video")
                inference_info = gr.Textbox(label="Inference Info", lines=4)
        
        inference_btn.click(
            demo_instance.run_inference,
            inputs=[sphere, num_steps, ev_value, seed_value],
            outputs=[output_video, inference_info]
        )
    
    with gr.Tab("6. Inference All"):
        gr.Markdown("### Batch Inference")
        gr.Markdown("Run inference for all combinations of sphere types and EV values")
        gr.Markdown("**Sphere types:** sphere_0, sphere_1")
        gr.Markdown("**EV values:** 0, -3, -6, -9")
        
        with gr.Row():
            with gr.Column():
                num_steps_all = gr.Slider(10, 100, value=20, label="Inference Steps")
                seed_value_all = gr.Number(value=42, label="Random Seed", precision=0)
                inference_all_btn = gr.Button("Run Inference All", variant="primary")
            
            with gr.Column():
                all_output_folder = gr.File(label="Output Folder")
                all_inference_info = gr.Textbox(label="Batch Inference Info", lines=12)
        
        inference_all_btn.click(
            demo_instance.run_inference_all,
            inputs=[num_steps_all, seed_value_all],
            outputs=[all_output_folder, all_inference_info]
        )
    
    with gr.Tab("7. HDRi Optimization"):
        gr.Markdown("### HDR Environment Map Optimization")
        gr.Markdown("Optimize HDR environment maps from inferred sphere videos")
        gr.Markdown("""
        **Requirements:**
        - Must run 'Inference All' first to generate sphere videos
        - Uses both sphere_0 (mirror) and sphere_1 (diffuse) for optimization
        - Produces HDR environment maps (.exr) and LDR previews (.png)
        """)
        
        with gr.Row():
            with gr.Column():
                evs_input = gr.Textbox(
                    value="0,-3,-6,-9",
                    label="EV Values (comma-separated)",
                    info="Exposure values to use for optimization"
                )
                num_iterations = gr.Slider(
                    100, 2000, value=1000, step=100,
                    label="Optimization Iterations",
                    info="More iterations = better quality but slower"
                )
                learning_rate = gr.Slider(
                    0.001, 0.01, value=0.005, step=0.001,
                    label="Learning Rate",
                    info="Optimization learning rate"
                )
                optimize_btn = gr.Button("Run HDRi Optimization", variant="primary")
            
            with gr.Column():
                preview_image = gr.Image(label="First Frame Preview (LDR)")
                hdr_output_zip = gr.File(label="HDR Results (ZIP)")
                optimization_log = gr.Textbox(label="Optimization Log", lines=20)
        
        optimize_btn.click(
            demo_instance.run_hdri_optimization,
            inputs=[evs_input, num_iterations, learning_rate],
            outputs=[preview_image, hdr_output_zip, optimization_log]
        )
    
    with gr.Tab("8. Blender Scene"):
        gr.Markdown("### Create Blender Scene with HDRi Lighting")
        gr.Markdown("""
        **Create a complete Blender scene:**
        - Depth mesh as shadow catcher
        - Optimized HDRi as environment lighting
        - Animated lighting per frame
        - Cycles render engine setup
        
        **Requirements:**
        - Must complete depth prediction first
        - Must run HDRi optimization first
        """)
        
        with gr.Row():
            with gr.Column():
                create_scene_btn = gr.Button("Create Blender Scene with HDRi", variant="primary")
            
            with gr.Column():
                scene_file_output = gr.File(label="Blender Scene File")
                scene_info = gr.Textbox(label="Scene Info", lines=15)
        
        create_scene_btn.click(
            demo_instance.create_blender_scene_with_hdri,
            inputs=[],
            outputs=[scene_file_output, scene_info]
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=True)