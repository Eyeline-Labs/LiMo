import torch
import torch.nn.functional as F
import numpy as np
import cv2
import math
import time
import os
import pickle
import argparse
from PIL import Image
from ezexr import imread, imsave
from torch_PoL.torchpol.torch_pol import PoL, inv_laplacian_pyramid, fwd_laplacian_pyramid
import torchvision
from tqdm import tqdm
import glob

import random

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# Copy all utility functions from optimization_pol_manual.py
def sample_envmap(envmap, uv):
    uv = uv * 2 - 1  # Convert to [-1, 1] for grid_sample
    grid = uv.view(1, -1, 1, 2)
    color = F.grid_sample(envmap, grid, mode='bilinear', align_corners=True)
    return color.squeeze(-1).squeeze(0)  # [3, N]

def sRGB_to_Lin(im):
    """Convert sRGB to linear RGB using PyTorch."""
    im = torch.clamp(im, 0.0, 1.0)
    linear_part = im / 12.92
    gamma_part = ((im + 0.055) / 1.055).pow(2.4)
    return torch.where(im <= 0.04045, linear_part, gamma_part)

def Lin_to_sRGB(im):
    """Convert linear RGB to sRGB using PyTorch."""
    linear_part = im * 12.92
    gamma_part = 1.055 * im.pow(1 / 2.4) - 0.055
    return torch.where(im <= 0.0031308, linear_part, gamma_part)

def vectors_to_latlong(vectors):
    """
    vectors: Tensor of shape (..., 3), assumed to be normalized
    Returns: Tensor of shape (..., 2) with (u, v) coordinates in [0, 1]
    """
    x, y, z = vectors[..., 0], vectors[..., 1], vectors[..., 2]
    theta = torch.acos(torch.clamp(y, -1.0, 1.0))  # [0, pi]
    phi = torch.atan2(x, z)                        # [-pi, pi]
    phi = (phi + 2 * torch.pi) % (2 * torch.pi)    # [0, 2pi]
    u = 1.0 - phi / (2 * torch.pi)  # longitude
    v = theta / torch.pi      # latitude
    return torch.stack([u, v], dim=-1)

def latlong_to_vectors(latlong):
    """
    latlong: Tensor of shape (..., 2) with (u, v) ∈ [0, 1]
    Returns: Tensor of shape (..., 3) normalized direction vectors
    """
    u, v = latlong[..., 0], latlong[..., 1]
    theta = v * torch.pi                 # polar angle from +Y (latitude)
    phi = (1.0 - u) * 2 * torch.pi       # azimuth from +X around Y axis
    sin_theta = torch.sin(theta)
    x = sin_theta * torch.sin(phi)
    y = torch.cos(theta)
    z = sin_theta * torch.cos(phi)
    return torch.stack([x, y, z], dim=-1)

def vectors_to_equisolid_uv(vectors, f=0.5):
    """Convert direction vectors to equisolid projection UV coordinates."""
    x, y, z = vectors[..., 0], vectors[..., 1], vectors[..., 2]
    theta = torch.acos(torch.clamp(z, -1.0, 1.0))
    r = 0.5 * torch.sin(theta / 2)
    phi = torch.atan2(y, x)
    u = 0.5 + r * torch.cos(phi)
    v = 1.0 - (0.5 + r * torch.sin(phi))
    uv = torch.stack([u, v], dim=-1)
    return uv

def cosine_weighted_sample_hemisphere(normals, num_samples):
    """Sample directions in hemisphere with cosine weighting."""
    X = normals.shape[0]
    u1 = torch.rand(X, num_samples, device=normals.device)
    u2 = torch.rand(X, num_samples, device=normals.device)
    
    r = torch.sqrt(u1)
    theta = 2 * torch.pi * u2
    
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    z = torch.sqrt(1 - u1)
    
    local_dirs = torch.stack([x, y, z], dim=-1)
    
    n = normals / torch.norm(normals, dim=-1, keepdim=True)
    arbitrary = torch.tensor([0.0, 0.0, 1.0], device=normals.device).expand_as(n)
    mask = (torch.abs(n[:, 2]) > 0.999).unsqueeze(-1)
    arbitrary = torch.where(mask, torch.tensor([0.0, 1.0, 0.0], device=n.device), arbitrary)
    
    tangent = torch.nn.functional.normalize(torch.cross(arbitrary, n), dim=-1)
    bitangent = torch.cross(n, tangent)
    TBN = torch.stack([tangent, bitangent, n], dim=-1)
    
    world_dirs = torch.matmul(local_dirs, TBN.transpose(1, 2))
    return world_dirs

def compute_luminance(envmap):
    return 0.2126 * envmap[0] + 0.7152 * envmap[1] + 0.0722 * envmap[2]

def multi_importance_sample_bad(normals, envmap, num_samples=64):
    """Multi-importance sampling for environment map."""
    envmap = F.interpolate(envmap.unsqueeze(0), size=(64, 128), mode='area').squeeze(0)
    
    with torch.no_grad():
        luminance = compute_luminance(envmap)
        H, W = luminance.shape
        
        normals = normals / torch.norm(normals, dim=-1, keepdim=True)
        normals = normals.unsqueeze(1)
        normals = torch.nan_to_num(normals, nan=1.0)
        
        grid = torch.meshgrid(torch.arange(W, device=luminance.device), 
                            torch.arange(H, device=luminance.device), indexing='xy')
        uv = torch.stack(grid, dim=-1).float() + 0.5
        uv = uv / torch.tensor([W, H], device=envmap.device)
        sin_theta = torch.sin(uv[:, :, 1] * np.pi)
        sin_theta = sin_theta.view(-1, 1)
        uv = uv.view(-1, 2)
        envmap_rays = latlong_to_vectors(uv)
        envmap_rays = envmap_rays.unsqueeze(0)
        envmap_rays = envmap_rays / torch.norm(envmap_rays, dim=-1, keepdim=True)
        summed = normals * envmap_rays
        cos_weights = (summed).sum(dim=-1)
        cos_weights = torch.clamp(cos_weights, min=0.0)
        
        luminance_flat = luminance.reshape(-1)
        importance_map = cos_weights * luminance_flat * sin_theta.permute(1, 0)
        importance_map = importance_map / importance_map.sum(dim=1, keepdim=True)
        
        samples = importance_map.multinomial(num_samples, replacement=True)
    
    envmap = envmap.permute(1,2,0).reshape(-1, 3)
    envmap_samples = envmap[samples]
    envmap_samples = envmap_samples.permute(2,0,1)
    
    pdfs_selected = torch.gather(importance_map, dim=1, index=samples)
    cosines = torch.gather(cos_weights, dim=1, index=samples)
    envmap_samples = envmap_samples * cosines[None]
    
    return envmap_samples, pdfs_selected

def multi_importance_sample(normals, envmap, num_samples=64):
    """
    normals: [N, 3]
    envmap: [3, H, W]
    
    returns: [X, N, 3] sampled uv
    """
    
    
    # Compute luminance
    # Downscale envmap to reduce computation (AREA)
    # For each normal: 
    #    - Compute cosine with normal
    #    - multiply by luminance
    #    - Normalize to a sum of 1
    #    - Sample from the luminance map using probabilities (Torch function?)
    #    - PDF is the sampled importance map
    
    # Downscale envmap map to reduce computation
    envmap = F.interpolate(envmap.unsqueeze(0), size=(64, 128), mode='area').squeeze(0)  # [3, 64, 128]
    
    with torch.no_grad():
        luminance = compute_luminance(envmap)  # [H, W]

        
        H, W = luminance.shape
        
        # Compute cosine weights
        normals = normals / torch.norm(normals, dim=-1, keepdim=True)
        normals = normals.unsqueeze(1)  # [N, 1, 3]
        normals = torch.nan_to_num(normals, nan=1.0)  # Handle NaNs
        #Get ray for each pixel in the envmap
        grid = torch.meshgrid(torch.arange(W, device=luminance.device), torch.arange(H, device=luminance.device), indexing='xy')
        uv = torch.stack(grid, dim=-1).float() + 0.5  # Shape: (W, H, 2)
        uv = uv / torch.tensor([W, H], device=envmap.device)  # Normalize to [0, 1]
        sin_theta = torch.sin(uv[:, :, 1] * np.pi)  # [W, H]
        sin_theta = sin_theta.view(-1, 1)  # Flatten to [H*W, 1]
        uv = uv.view(-1, 2)  # Flatten to (H*W, 2)
        envmap_rays = latlong_to_vectors(uv)  # Convert to direction vectors [H*W, 3]
        envmap_rays = envmap_rays.unsqueeze(0)  # Add batch dimension [1, H*W, 3]
        envmap_rays = envmap_rays / torch.norm(envmap_rays, dim=-1, keepdim=True)  # Normalize the rays [1, H*W, 3]
        summed = normals * envmap_rays  # [N, H*W, 3]
        cos_weights = (summed).sum(dim=-1)  # [N, H*W]
        cos_weights = torch.clamp(cos_weights, min=0.0)  # Clamp to [0, 1]
        
        # cos_weights = cos_weights  / cos_weights.sum(dim=1, keepdim=True)  # Normalize to sum to 1 [N, H*W]
        
        
        mine = True

        luminance = luminance # .clamp(max=1000.0)+0.01  # Clamp luminance to [0, 1]
        
        if mine:
            importance_map = cos_weights * luminance.reshape(1,-1) * sin_theta.permute(1, 0)  # [H*W, 1]
            importance_map = importance_map / importance_map.sum(dim=1, keepdim=True)  # Normalize to sum to 1 [N, H*W]
            delta_omega = (2 * np.pi / W) * (np.pi / H) * sin_theta.permute(1, 0) / 2  # Solid angle for each pixel
            pdfs = importance_map / delta_omega  # [N, H*W]
        else:
            lum_cosine = luminance.reshape(1, -1) * (cos_weights>0)  # [N, H*W]
            lum_cosine /= lum_cosine.sum(dim=1, keepdim=True)
            lum_cosine *= 4096# Normalize to sum to 1 [N, H*W]
            importance_map_no_cos = luminance.reshape(1,-1) * sin_theta.permute(1, 0)
            importance_map = cos_weights * importance_map_no_cos
            summed_importance_map = importance_map.sum(dim=1, keepdim=True)
            importance_map = importance_map / summed_importance_map  # Normalize to sum to 1
            
            importance_map_cos = cos_weights * torch.ones_like(luminance.reshape(1,-1)) * sin_theta.permute(1, 0)
            #importance_map_cos = importance_map_cos / importance_map_cos.sum(dim=1, keepdim=True)  # Normalize to sum to 1
            importance_map = importance_map / importance_map_cos
            importance_map = importance_map * (cos_weights>0)
            #importance_map = importance_map / importance_map.sum(dim=1, keepdim=True)  # Normalize to sum to 1
            #importance_map = torch.nan_to_num(importance_map, nan=1.0)  # Handle NaNs
            #lum_cosine = importance_map * (2 * np.pi / W) * (np.pi / H) * sin_theta.permute(1, 0) * 0.5

        
        #cos_weights = cos_weights / summed_importance_map  # Normalize to sum to 1
        
        #importance_map_no_cos = importance_map_no_cos/ importance_map_no_cos.sum(dim=1, keepdim=True)  # [H*W, 1]
        
        
        # delta_omega = (2 * np.pi / W) * (np.pi / H) * sin_theta.permute(1, 0)  # Solid angle for each pixel
        # pdfs_mine = importance_map / delta_omega  # [N, H*W]
        # importance_map_mine = cos_weights * luminance.reshape(1,-1) * sin_theta.permute(1, 0)
        # importance_map_mine = importance_map_mine / importance_map_mine.sum(dim=1, keepdim=True)  # Normalize to sum to 1 [N, H*W]
        
        
            
        samples = importance_map.multinomial(num_samples, replacement=True)  # Sample from the multinomial distribution
    
    envmap = envmap.permute(1,2,0).reshape(-1, 3)  # Flatten env map to [H*W, 1]
    # get luminance for samples
    envmap_samples = envmap[samples]  # [N, num_samples, 3]
    envmap_samples = envmap_samples.permute(2,0,1)  # [3, N, num_samples]
    
    if mine:
        pdfs_selected = torch.gather(pdfs, dim=1, index=samples)
        cosines = torch.gather(cos_weights, dim=1, index=samples)
        envmap_samples = envmap_samples * cosines[None]  # Scale samples by cosine weights
    else:
        pdfs_selected = torch.gather(lum_cosine, dim=1, index=samples)
    
    #delta_select = torch.gather(delta_omega.expand(cos_weights.shape[0],-1), dim=1, index=samples)  # [N, num_samples]
    
    envmap_samples = envmap_samples #* cosines[None] #* delta_select
    
    return envmap_samples, pdfs_selected

def render(position_map, world2cam, normal, envmap, mode='mirror', num_samples=16, proj='latlong'):
    """Render using environment map with given geometry."""
    normal = torch.where(normal.norm(dim=0, keepdim=True) < 1e-6, torch.ones_like(normal), normal)
    position_map = torch.where(position_map.norm(dim=0, keepdim=True) < 1e-6, torch.ones_like(position_map), position_map)
    position_map_flat = position_map.reshape(3, -1).permute(1, 0)
    normal_flat = normal.reshape(3, -1).permute(1, 0)
    
    env_H, env_W = envmap.shape[1], envmap.shape[2]
    map_H, map_W = position_map.shape[1], position_map.shape[2]
    
    grid = torch.meshgrid(torch.arange(map_H, device=envmap.device), 
                         torch.arange(map_W, device=envmap.device), indexing='ij')
    grid = torch.stack(grid, dim=-1).float()
    grid = grid + torch.rand_like(grid)
    grid = grid / torch.tensor([map_H, map_W], device=envmap.device)
    grid = torch.clamp(grid, 0, 1)
    
    normal_flat = sample_envmap(normal.unsqueeze(0), grid.view(-1, 2)[:,[1,0]]).permute(1,0)
    position_map_flat = sample_envmap(position_map.unsqueeze(0), grid.view(-1, 2)[:,[1,0]]).permute(1,0)
    
    position_map_homogeneous = torch.cat([position_map_flat, torch.ones(position_map_flat.shape[0], 1).cuda()], dim=1)
    I = world2cam @ position_map_homogeneous.T
    I = I.T[:,:3]
    
    normal_homogeneous = torch.cat([normal_flat, torch.zeros(normal_flat.shape[0], 1).cuda()], dim=1)
    normal_homogeneous = world2cam @ normal_homogeneous.T
    normal_flat = normal_homogeneous.T[:, :3]
    normal_flat = normal_flat / normal_flat.norm(dim=1, keepdim=True)
    
    if mode == 'mirror':
        with torch.no_grad():
            I_normalized = I / I.norm(dim=1, keepdim=True)
            R = I_normalized - 2 * (I_normalized * normal_flat).sum(dim=1, keepdim=True) * normal_flat
            
            if proj == 'latlong':
                uv = vectors_to_latlong(R)
            else:
                uv = vectors_to_equisolid_uv(R, f=0.5)
        
        img = sample_envmap(envmap.unsqueeze(0), uv)
    elif mode == 'diffuse':
        samples, pdfs = multi_importance_sample(normal_flat, envmap, num_samples=num_samples)
        samples = samples / (pdfs.unsqueeze(0) + 1e-8)  # Normalize by PDF
        img = samples.sum(dim=2) / num_samples
    
    img = img.reshape(3, position_map.shape[1], position_map.shape[2])
    return img

def load_reference_images(sphere_out_path, evs, frame):
    """Load reference images for different exposure values."""
    ref_imgs = {}
    
    for ev in evs:
        # Try to load predicted image from test.py output
        pred_path = os.path.join(sphere_out_path, f"pred_ev{ev}", f"frame{frame}.png")
        if os.path.exists(pred_path):
            ref_img = Image.open(pred_path).convert("RGB")
            ref_img = ref_img.resize((512, 512), Image.LANCZOS)
            ref_img = np.array(ref_img, dtype=np.float32) / 255.0
            ref_img = torch.from_numpy(ref_img).permute(2, 0, 1)
            ref_imgs[ev] = ref_img
        else:
            print(f"Warning: Reference image not found at {pred_path}")
    
    return ref_imgs

def optimize_envmap_for_scene(scene_path, data_path, out_path, evs, proj='latlong', mask_id=255, temporal_loss_weight=0.1):
    """Optimize environment map for a single scene with temporal consistency."""
    print(f"Processing scene: {scene_path}")
    
    t_start = time.time()
    
    num_frames = len(glob.glob(os.path.join(scene_path, "frame_*")))
    print(f"Found {num_frames} frames to optimize simultaneously")
    
    # Store all frame data
    frame_data = []
    
    # Load all frame data first
    for frame in range(1, num_frames + 1):
        print(f"Loading frame {frame} data...")
        
        # Load camera info
        camera_info_path = os.path.join(scene_path, f"frame_{frame:04d}", "camera_info.pkl")
        if not os.path.exists(camera_info_path):
            print(f"Camera info not found at {camera_info_path}")
            continue
        
        with open(camera_info_path, 'rb') as f:
            camera_data = pickle.load(f)
        
        cam2world = np.array(camera_data["cam2world"])
        world2cam = torch.from_numpy(cam2world).float().cuda().inverse()
        
        # Load geometry data
        mask_path = os.path.join(scene_path, f"frame_{frame:04d}", "sphere_0", f"mask_{frame:04d}.exr")
        normal_path = os.path.join(scene_path, f"frame_{frame:04d}", "sphere_0", f"normal_{frame:04d}.exr")
        position_path = os.path.join(scene_path, f"frame_{frame:04d}", "sphere_0", f"position_{frame:04d}.exr")

        if not all(os.path.exists(p) for p in [mask_path, normal_path, position_path]):
            print(f"Missing geometry files for frame {frame}")
            continue
        
        # Load mask, normal, and position maps
        mask = imread(mask_path)[:,:,:3]
        mask = np.where(mask == mask_id, 1, 0)
        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask.repeat(3, axis=2)
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        mask = torch.from_numpy(mask).float().permute(2, 0, 1)
        
        normal_map = imread(normal_path).astype(np.float32)[:,:,:3]
        normal_map = cv2.resize(normal_map, (512, 512), interpolation=cv2.INTER_LINEAR)
        normal_map = torch.from_numpy(normal_map).permute(2, 0, 1)
        
        position_map = imread(position_path).astype(np.float32)[:,:,:3]
        position_map = cv2.resize(position_map, (512, 512), interpolation=cv2.INTER_LINEAR)
        position_map = torch.from_numpy(position_map).permute(2, 0, 1)
        
        # Load inferred images for different spheres and EVs
        ref_img_0 = load_reference_images(os.path.join(scene_path.replace(data_path, out_path), "sphere_0"), evs, frame-1)  # frame-1 because test.py uses 0-based indexing
        ref_img_1 = load_reference_images(os.path.join(scene_path.replace(data_path, out_path), "sphere_1"), evs, frame-1)
        
        if not ref_img_0 and not ref_img_1:
            print(f"No reference images found for frame {frame}")
            continue
        
        # Crop around the sphere
        indices = (mask[0,:,:] == 1).nonzero(as_tuple=False)
        if len(indices) == 0:
            print(f"No valid mask pixels for frame {frame}")
            continue
        
        min_indices = torch.amin(indices, dim=0)
        max_indices = torch.amax(indices, dim=0)
        
        mask = mask[:, min_indices[0]:max_indices[0]+1, min_indices[1]:max_indices[1]+1].cuda()
        normal_map = normal_map[:, min_indices[0]:max_indices[0]+1, min_indices[1]:max_indices[1]+1].cuda()
        position_map = position_map[:, min_indices[0]:max_indices[0]+1, min_indices[1]:max_indices[1]+1].cuda()
        
        # Crop reference images
        for sphere_refs in [ref_img_0, ref_img_1]:
            for ev in sphere_refs.keys():
                sphere_refs[ev] = sphere_refs[ev][:, min_indices[0]:max_indices[0]+1, min_indices[1]:max_indices[1]+1].cuda()
        
        # Resize if too large
        max_size = 256
        _, H, W = mask.shape
        if H > max_size or W > max_size:
            scale_factor = max_size / max(H, W)
            new_H, new_W = int(H * scale_factor), int(W * scale_factor)
            
            mask = F.interpolate(mask.unsqueeze(0), size=(new_H, new_W), mode='nearest').squeeze(0)
            normal_map = F.interpolate(normal_map.unsqueeze(0), size=(new_H, new_W), mode='bilinear').squeeze(0)
            position_map = F.interpolate(position_map.unsqueeze(0), size=(new_H, new_W), mode='bilinear').squeeze(0)
            
            for sphere_refs in [ref_img_0, ref_img_1]:
                for ev in sphere_refs.keys():
                    sphere_refs[ev] = F.interpolate(sphere_refs[ev].unsqueeze(0), size=(new_H, new_W), mode='bilinear').squeeze(0)
        
        # Setup sphere modes and references
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
        
        frame_data.append({
            'frame': frame,
            'world2cam': world2cam,
            'mask': mask,
            'normal_map': normal_map,
            'position_map': position_map,
            'list_spheres': list_spheres,
            'modes': modes,
            'references': references,
            'weights': weights
        })
    
    if len(frame_data) == 0:
        print("No valid frames to process")
        return
    
    print(f"Loaded {len(frame_data)} frames successfully")
    
    # Initialize environment maps for all frames
    _, H, W = frame_data[0]['position_map'].shape
    size = max(H, W)
    size = 2 ** math.ceil(math.log2(size))
    init_size = min(512, size * 4)
    
    # Create PoL optimizers for each frame
    pols = []
    all_params = []
    for i in range(len(frame_data)):
        init = torch.ones(1, 3, init_size // 2, init_size).cuda() * 0.5
        pol = PoL(init, learning_rate=0.005, weight_decay=0.05, padding_mode="circular").cuda()
        pols.append(pol)
        all_params.extend(pol.parameters())
    
    print(f"Initialized {len(pols)} environment maps with shape [3, {init_size // 2}, {init_size}]")
    
    # Single optimizer for all frames
    optimizer = torch.optim.AdamW(all_params, betas=(0.9, 0.99), fused=True)
    
    # Optimization loop
    num_iterations = 1000 * len(frame_data)
    print(f"Starting optimization for {num_iterations} iterations...")

    frames_stack = []
    
    for i in range(num_iterations):
        optimizer.zero_grad()
        
        # Randomly select a frame to optimize in this iteration
        if len(frames_stack) == 0:
            #make a list of all frame indices
            frames_stack = list(range(len(frame_data)))
        frame_idx = frames_stack.pop(random.randint(0, len(frames_stack) - 1))
        fdata = frame_data[frame_idx]
        
        # Get current frame's environment map
        recon = pols[frame_idx]()
        recon = recon.squeeze(0)
        recon = torch.pow(2, recon)
        recon = torch.nan_to_num(recon, nan=0.0)
        
        # Randomly select sphere and exposure value
        random_sphere = random.choice(fdata['list_spheres'])
        ev = random.choice(evs)
        
        ref_img = fdata['references'][random_sphere][ev]
        mode = fdata['modes'][random_sphere]
        
        # Render with current environment map
        img = render(fdata['position_map'], fdata['world2cam'], fdata['normal_map'], 
                    recon, mode=mode, num_samples=64, proj=proj)
        img = img * (2**int(ev))
        img = Lin_to_sRGB(img).clamp(0, 1)
        
        # Saturation mask
        mask_sat = torch.ones_like(ref_img)
        threshold = 0.90
        mask_sat = torch.where((ref_img > threshold) & (img > threshold), 0.0, 1.0)
        
        # Reconstruction loss
        recon_loss = fdata['weights'][random_sphere] * torch.mean(
            torch.square((img - ref_img.to(recon.device)) * fdata['mask'] * mask_sat)
        )
        
        # Temporal consistency loss (L1 between current frame and its neighbors)
        temporal_loss = 0.0
        if temporal_loss_weight > 0 and len(pols) > 1:
            # Compare with previous frame
            if frame_idx > 0:
                envmap_prev = pols[frame_idx - 1]().squeeze(0).detach()
                envmap_curr = pols[frame_idx]().squeeze(0)

                envmap_prev = torch.pow(2, envmap_prev)
                envmap_curr = torch.pow(2, envmap_curr)
                temporal_loss += F.l1_loss(envmap_curr, envmap_prev)
            
            if frame_idx < len(pols) - 1:
                envmap_next = pols[frame_idx + 1]().squeeze(0).detach()
                envmap_curr = pols[frame_idx]().squeeze(0)

                envmap_next = torch.pow(2, envmap_next)
                envmap_curr = torch.pow(2, envmap_curr)
                temporal_loss += F.l1_loss(envmap_curr, envmap_next)
            
            # Average the temporal loss (divide by number of neighbors)
            num_neighbors = int(frame_idx > 0) + int(frame_idx < len(pols) - 1)
            if num_neighbors > 0:
                temporal_loss = temporal_loss / num_neighbors
        
        # Total loss
        loss = recon_loss + temporal_loss_weight * temporal_loss
        
        loss.backward()
        optimizer.step()
        
        # Update all PoL schedulers
        for pol in pols:
            pol.end_iter_callback(i//len(pols))
        
        if i % 100 == 0:
            print(f"Iteration {i}/{num_iterations}, Recon Loss: {recon_loss.item():.6f}, "
                  f"Temporal Loss: {temporal_loss if isinstance(temporal_loss, float) else temporal_loss.item():.6f}, "
                  f"Total Loss: {loss.item():.6f}")
    
    # Save results for all frames
    frame_output_dir = os.path.join(scene_path.replace(data_path, out_path), "hdr")
    os.makedirs(frame_output_dir, exist_ok=True)
    
    for idx, fdata in enumerate(frame_data):
        with torch.no_grad():
            recon = pols[idx]()
            recon = recon.squeeze(0)
            recon = torch.pow(2, recon)
            recon = torch.nan_to_num(recon, nan=0.0)
            
            envmap_hdr = recon.permute(1, 2, 0).cpu().detach().numpy()
            output_path = os.path.join(frame_output_dir, f"envmap_{fdata['frame']:04d}.exr")
            imsave(output_path, envmap_hdr, compression='ZIP')
            print(f"Saved: {output_path}")
    
    print(f"Scene optimization complete in {time.time() - t_start:.2f} seconds")

def main():
    parser = argparse.ArgumentParser(description="Batch HDR environment map optimization")
    parser.add_argument("--data_path", required=True, help="Root path containing scene data")
    parser.add_argument("--out_path", required=True, help="Root path containing inferred spheres")
    parser.add_argument("--evs", default="0,-3,-6,-9,-12", help="Comma-separated exposure values")
    parser.add_argument("--proj", default="latlong", choices=["latlong", "equisolid"], help="Projection type")
    parser.add_argument("--mask_id", type=int, default=255, help="Mask ID for sphere detection")
    parser.add_argument("--temporal_loss_weight", type=float, default=0.1, help="Weight for temporal consistency loss between frames")
    
    args = parser.parse_args()
    
    evs = args.evs.split(",")
    
    # Find all scene directories that match the expected structure
    scene_pattern = os.path.join(args.data_path, "lighting_*", "*", "object_*")
    scene_dirs = glob.glob(scene_pattern)
    
    print(f"Found {len(scene_dirs)} scenes to process")
    print(f"Temporal loss weight: {args.temporal_loss_weight}")
    
    for scene_dir in tqdm(scene_dirs, desc="Processing scenes"):
        optimize_envmap_for_scene(
            scene_dir, 
            args.data_path,
            args.out_path,
            evs, 
            proj=args.proj,
            mask_id=args.mask_id,
            temporal_loss_weight=args.temporal_loss_weight,
        )
    
    print("Batch processing complete!")

if __name__ == "__main__":
    main()