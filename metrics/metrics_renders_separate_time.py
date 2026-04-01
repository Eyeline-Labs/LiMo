import torch
import torch.nn.functional as F
import numpy as np
import os
import json
import argparse
from PIL import Image
import glob
from tqdm import tqdm
import lpips

import sys
sys.path.append('core')

# Try to import RAFT for optical flow
try:
    from raft import RAFT
    from utils.utils import InputPadder
    RAFT_AVAILABLE = True
except ImportError:
    RAFT_AVAILABLE = False
    print("Warning: RAFT module not found. Flow-based warping metric will be disabled.")




def load_image_with_alpha(path, target_size=None):
    """Load image with alpha channel and convert to tensor."""
    img = Image.open(path).convert("RGBA")
    img_array = np.array(img).astype(np.float32) / 255.0
    
    if target_size:
        # Resize using PIL to preserve alpha
        img = img.resize(target_size, Image.LANCZOS)
        img_array = np.array(img).astype(np.float32) / 255.0
    
    # Split RGB and alpha
    rgb = torch.from_numpy(img_array[:, :, :3]).permute(2, 0, 1)  # (3, H, W)
    alpha = torch.from_numpy(img_array[:, :, 3]).unsqueeze(0)     # (1, H, W)
    
    return rgb, alpha


def extract_quadrants(rgb, alpha):
    """Extract the four quadrants from the image."""
    _, h, w = rgb.shape
    
    # Calculate quadrant boundaries
    h_mid = h // 2
    w_mid = w // 2
    
    quadrants = {
        'Mirror': {  # Top Left
            'rgb': rgb[:, :h_mid, :w_mid],
            'alpha': alpha[:, :h_mid, :w_mid]
        },
        'Diffuse': {  # Top Right
            'rgb': rgb[:, :h_mid, w_mid:],
            'alpha': alpha[:, :h_mid, w_mid:]
        },
        'Matte': {  # Bottom Left
            'rgb': rgb[:, h_mid:, :w_mid],
            'alpha': alpha[:, h_mid:, :w_mid]
        },
        'Glossy': {  # Bottom Right
            'rgb': rgb[:, h_mid:, w_mid:],
            'alpha': alpha[:, h_mid:, w_mid:]
        }
    }
    
    return quadrants


def warp_image_with_flow(image, flow):
    """
    Warp an image using optical flow.
    
    Args:
        image: Tensor of shape (C, H, W) in range [0, 1]
        flow: Tensor of shape (2, H, W) representing (dx, dy) at each pixel
    
    Returns:
        Warped image of shape (C, H, W)
    """
    _, h, w = image.shape
    
    # Create normalized grid [-1, 1]
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, h, device=image.device),
        torch.linspace(-1, 1, w, device=image.device),
        indexing='ij'
    )
    
    # Normalize flow to [-1, 1] range
    flow_x_normalized = 2.0 * flow[0] / (w - 1)
    flow_y_normalized = 2.0 * flow[1] / (h - 1)
    
    # Apply flow to grid
    warped_grid_x = grid_x + flow_x_normalized
    warped_grid_y = grid_y + flow_y_normalized
    
    # Stack to (H, W, 2) format expected by grid_sample
    warped_grid = torch.stack([warped_grid_x, warped_grid_y], dim=-1)
    
    # Add batch dimension for grid_sample
    image_batch = image.unsqueeze(0)  # (1, C, H, W)
    warped_grid_batch = warped_grid.unsqueeze(0)  # (1, H, W, 2)
    
    # Warp image
    warped_image = F.grid_sample(
        image_batch, 
        warped_grid_batch, 
        mode='bilinear', 
        padding_mode='border',
        align_corners=True
    )
    
    return warped_image.squeeze(0)  # (C, H, W)


def compute_optical_flow(model, image1, image2):
    """
    Compute optical flow from image1 to image2 using RAFT.
    
    Args:
        model: RAFT model
        image1: Tensor (C, H, W) in range [0, 1]
        image2: Tensor (C, H, W) in range [0, 1]
    
    Returns:
        flow: Tensor (2, H, W) representing (dx, dy)
    """
    device = next(model.parameters()).device
    
    # Convert to format expected by RAFT (B, C, H, W) in range [0, 255]
    # Make sure to move to the correct device
    img1 = (image1 * 255.0).unsqueeze(0).to(device)
    img2 = (image2 * 255.0).unsqueeze(0).to(device)
    
    # Pad images
    padder = InputPadder(img1.shape)
    img1_padded, img2_padded = padder.pad(img1, img2)
    
    # Compute flow
    with torch.no_grad():
        _, flow = model(img1_padded, img2_padded, iters=20, test_mode=True)
    
    # Unpad flow
    flow = padder.unpad(flow)
    
    return flow.squeeze(0)  # (2, H, W)


def compute_flow_warping_error(pred_renders_list, gt_renders_list, pred_mask_list=None, gt_mask_list=None, raft_model=None):
    """
    Compute flow-based warping error.
    
    This computes optical flow from GT(t) -> GT(t+1), then warps Pred(t) using this flow
    and compares with Pred(t+1). This measures if predictions follow the same motion as GT.
    
    Args:
        pred_renders_list: List of prediction tensors (T frames)
        gt_renders_list: List of ground truth tensors (T frames)
        pred_mask_list: Optional list of prediction masks
        gt_mask_list: Optional list of ground truth masks
        raft_model: Pre-initialized RAFT model (if None or RAFT unavailable, returns NaN)
    
    Returns:
        Dictionary with flow warping error metrics
    """
    if not RAFT_AVAILABLE or raft_model is None:
        return {
            'flow_warp_error': float('nan')
        }
    
    if len(pred_renders_list) < 2 or len(gt_renders_list) < 2:
        return {
            'flow_warp_error': float('nan')
        }
    
    warp_errors = []
    
    for i in range(min(len(pred_renders_list), len(gt_renders_list)) - 1):
        gt_t = gt_renders_list[i]
        gt_t1 = gt_renders_list[i+1]
        pred_t = pred_renders_list[i]
        pred_t1 = pred_renders_list[i+1]
        
        try:
            # Compute optical flow from GT(t) to GT(t+1)
            flow = compute_optical_flow(raft_model, gt_t, gt_t1)
            
            # Move pred_t to the same device as flow for warping
            device = flow.device
            pred_t_device = pred_t.to(device)
            pred_t1_device = pred_t1.to(device)
            
            # Warp Pred(t) using this flow
            pred_t_warped = warp_image_with_flow(pred_t_device, flow)
            
            # Compute error between warped Pred(t) and Pred(t+1)
            error = torch.pow(pred_t_warped - pred_t1_device, 2)
            
            # Apply mask if available
            if pred_mask_list is not None and i+1 < len(pred_mask_list):
                mask = pred_mask_list[i+1].to(device)
                valid_pixels = mask.sum()
                if valid_pixels > 0:
                    masked_error = error * mask
                    mean_error = masked_error.sum() / valid_pixels
                else:
                    continue
            else:
                mean_error = torch.mean(error)
            
            mean_error = torch.sqrt(mean_error)  # RMSE
            warp_errors.append(mean_error.item())
            
        except Exception as e:
            print(f"Warning: Flow computation failed for frame pair {i}: {e}")
            continue
    
    if not warp_errors:
        return {
            'flow_warp_error': float('nan')
        }
    
    return {
        'flow_warp_error': np.mean(warp_errors)
    }


def compute_temporal_gradient_error(pred_renders_list, gt_renders_list, pred_mask_list=None, gt_mask_list=None):
    """
    Compute temporal gradient error: E_temp = ||∂_t I - ∂_t I_GT||_1
    
    This measures the L1 norm of the difference between prediction and GT temporal derivatives.
    Lower values indicate better temporal consistency with ground truth.
    
    Args:
        pred_renders_list: List of prediction tensors (T frames)
        gt_renders_list: List of ground truth tensors (T frames)
        pred_mask_list: Optional list of prediction masks
        gt_mask_list: Optional list of ground truth masks
    
    Returns:
        Dictionary with temporal error metrics
    """
    if len(pred_renders_list) < 2 or len(gt_renders_list) < 2:
        return {
            'temporal_error': float('nan')
        }
    
    temporal_errors = []
    
    for i in range(min(len(pred_renders_list), len(gt_renders_list)) - 1):
        # Compute temporal derivatives (∂_t I and ∂_t I_GT)
        pred_temporal_derivative = pred_renders_list[i+1] - pred_renders_list[i]
        gt_temporal_derivative = gt_renders_list[i+1] - gt_renders_list[i]
        
        # Compute difference of temporal derivatives
        derivative_diff = torch.pow((pred_temporal_derivative) - (gt_temporal_derivative), 2)
        
        # Compute L1 norm
        l1_error = torch.abs(derivative_diff)
        
        # Apply mask if available (use GT mask as reference)
        if gt_mask_list is not None and i < len(gt_mask_list) - 1:
            mask = gt_mask_list[i+1]
            valid_pixels = mask.sum()
            if valid_pixels > 0:
                masked_error = l1_error * mask
                mean_error = masked_error.sum() / valid_pixels
            else:
                continue
        else:
            mean_error = torch.mean(l1_error)
        
        mean_error = torch.sqrt(mean_error)  # RMSE of the differences
        
        temporal_errors.append(mean_error.item())
    
    if not temporal_errors:
        return {
            'temporal_error': float('nan')
        }
    
    # Return mean temporal error across all frame pairs
    return {
        'temporal_error': np.mean(temporal_errors)
    }


def compute_temporal_lpips(pred_renders_list, gt_renders_list, pred_mask_list=None, gt_mask_list=None, lpips_model=None):
    """
    Compute temporal LPIPS error: measures perceptual difference between consecutive frames.
    
    This computes LPIPS(I(t), I(t+1)) for predictions and ground truth separately,
    then compares them to measure if predictions have similar temporal perceptual changes as GT.
    
    Args:
        pred_renders_list: List of prediction tensors (T frames)
        gt_renders_list: List of ground truth tensors (T frames)
        pred_mask_list: Optional list of prediction masks (not used for LPIPS)
        gt_mask_list: Optional list of ground truth masks (not used for LPIPS)
        lpips_model: Pre-initialized LPIPS model (if None, will create one)
    
    Returns:
        Dictionary with temporal LPIPS metrics
    """
    if len(pred_renders_list) < 2 or len(gt_renders_list) < 2:
        return {
            'temporal_lpips_pred': float('nan'),
            'temporal_lpips_gt': float('nan'),
            'temporal_lpips_diff': float('nan')
        }
    
    # Initialize LPIPS model if not provided
    if lpips_model is None:
        lpips_model = lpips.LPIPS(net='alex').eval()
        if torch.cuda.is_available():
            lpips_model = lpips_model.cuda()
    
    pred_lpips_values = []
    gt_lpips_values = []
    
    for i in range(min(len(pred_renders_list), len(gt_renders_list)) - 1):
        # Get consecutive frames
        pred_t = pred_renders_list[i]
        pred_t1 = pred_renders_list[i+1]
        gt_t = gt_renders_list[i]
        gt_t1 = gt_renders_list[i+1]
        
        # LPIPS expects (B, C, H, W) with values in [-1, 1]
        # Our images are in [0, 1], so convert to [-1, 1]
        pred_t_norm = pred_t * 2.0 - 1.0
        pred_t1_norm = pred_t1 * 2.0 - 1.0
        gt_t_norm = gt_t * 2.0 - 1.0
        gt_t1_norm = gt_t1 * 2.0 - 1.0
        
        # Add batch dimension
        pred_t_norm = pred_t_norm.unsqueeze(0)
        pred_t1_norm = pred_t1_norm.unsqueeze(0)
        gt_t_norm = gt_t_norm.unsqueeze(0)
        gt_t1_norm = gt_t1_norm.unsqueeze(0)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            pred_t_norm = pred_t_norm.cuda()
            pred_t1_norm = pred_t1_norm.cuda()
            gt_t_norm = gt_t_norm.cuda()
            gt_t1_norm = gt_t1_norm.cuda()
        
        # Compute LPIPS between consecutive frames
        with torch.no_grad():
            pred_lpips = lpips_model(pred_t_norm, pred_t1_norm).item()
            gt_lpips = lpips_model(gt_t_norm, gt_t1_norm).item()
        
        pred_lpips_values.append(pred_lpips)
        gt_lpips_values.append(gt_lpips)
    
    if not pred_lpips_values or not gt_lpips_values:
        return {
            'temporal_lpips_pred': float('nan'),
            'temporal_lpips_gt': float('nan'),
            'temporal_lpips_diff': float('nan')
        }
    
    # Compute mean temporal LPIPS for predictions and GT
    mean_pred_lpips = np.mean(pred_lpips_values)
    mean_gt_lpips = np.mean(gt_lpips_values)
    
    # Compute absolute difference to measure if pred changes at similar rate as GT
    lpips_diff = abs(mean_pred_lpips - mean_gt_lpips)
    
    return {
        'temporal_lpips_pred': mean_pred_lpips,
        'temporal_lpips_gt': mean_gt_lpips,
        'temporal_lpips_diff': lpips_diff
    }


def process_object(gt_object_path, pred_object_path, scene_name, lpips_model=None, raft_model=None):
    """Process a single scene and compute temporal metrics for all renders."""
    object_metrics = {
        'scene_name': scene_name,
        'temporal_metrics': {}
    }
    
    # Find all ground truth renders
    gt_pattern = os.path.join(gt_object_path, "render_metrics", "render_*.png")
    gt_files = sorted(glob.glob(gt_pattern))
    
    if not gt_files:
        print(f"No ground truth renders found for {scene_name}")
        return object_metrics
    
    if len(gt_files) < 2:
        print(f"Not enough renders for temporal analysis in {scene_name} (need at least 2)")
        return object_metrics
    
    # Collect all renders for temporal analysis
    gt_renders_full = []
    pred_renders_full = []
    gt_masks_full = []
    pred_masks_full = []
    
    quadrant_renders = {
        'Mirror': {'gt': [], 'pred': [], 'gt_masks': [], 'pred_masks': []},
        'Diffuse': {'gt': [], 'pred': [], 'gt_masks': [], 'pred_masks': []},
        'Matte': {'gt': [], 'pred': [], 'gt_masks': [], 'pred_masks': []},
        'Glossy': {'gt': [], 'pred': [], 'gt_masks': [], 'pred_masks': []}
    }
    
    for gt_file in gt_files:
        # Extract render number
        render_name = os.path.splitext(os.path.basename(gt_file))[0]
        render_num = render_name.split('_')[1]
        
        # Find corresponding prediction file
        pred_file = os.path.join(pred_object_path, "render_metrics", f"render_{render_num}.png")
        
        if not os.path.exists(pred_file):
            print(f"Prediction file not found: {pred_file}")
            continue
        
        try:
            # Load images with alpha
            gt_rgb, gt_alpha = load_image_with_alpha(gt_file)
            pred_rgb, pred_alpha = load_image_with_alpha(pred_file)
            
            # Store full images
            gt_renders_full.append(gt_rgb)
            pred_renders_full.append(pred_rgb)
            gt_masks_full.append(torch.where(gt_alpha > 0.5, 1.0, 0.0))
            pred_masks_full.append(torch.where(pred_alpha > 0.5, 1.0, 0.0))
            
            # Extract quadrants
            gt_quadrants = extract_quadrants(gt_rgb, gt_alpha)
            pred_quadrants = extract_quadrants(pred_rgb, pred_alpha)
            
            # Store quadrant data
            for quadrant_name in ['Mirror', 'Diffuse', 'Matte', 'Glossy']:
                gt_quad_rgb = gt_quadrants[quadrant_name]['rgb']
                gt_quad_alpha = gt_quadrants[quadrant_name]['alpha']
                pred_quad_rgb = pred_quadrants[quadrant_name]['rgb']
                pred_quad_alpha = pred_quadrants[quadrant_name]['alpha']
                
                quadrant_renders[quadrant_name]['gt'].append(gt_quad_rgb)
                quadrant_renders[quadrant_name]['pred'].append(pred_quad_rgb)
                quadrant_renders[quadrant_name]['gt_masks'].append(
                    torch.where(gt_quad_alpha > 0.5, 1.0, 0.0)
                )
                quadrant_renders[quadrant_name]['pred_masks'].append(
                    torch.where(pred_quad_alpha > 0.5, 1.0, 0.0)
                )
            
        except Exception as e:
            print(f"Error processing {render_name} for {scene_name}: {e}")
            continue
    
    # Compute temporal metrics for full image
    if len(gt_renders_full) > 1:
        temporal_gradient = compute_temporal_gradient_error(
            pred_renders_full, gt_renders_full, 
            pred_masks_full, gt_masks_full
        )
        
        temporal_lpips_metrics = compute_temporal_lpips(
            pred_renders_full, gt_renders_full,
            pred_masks_full, gt_masks_full,
            lpips_model
        )
        
        flow_warp_metrics = compute_flow_warping_error(
            pred_renders_full, gt_renders_full,
            pred_masks_full, gt_masks_full,
            raft_model
        )
        
        object_metrics['temporal_metrics']['Overall'] = {
            **temporal_gradient,
            **temporal_lpips_metrics,
            **flow_warp_metrics
        }
    
    # Compute temporal metrics per quadrant
    for quadrant_name in ['Mirror', 'Diffuse', 'Matte', 'Glossy']:
        if len(quadrant_renders[quadrant_name]['gt']) > 1:
            gt_list = quadrant_renders[quadrant_name]['gt']
            pred_list = quadrant_renders[quadrant_name]['pred']
            gt_mask_list = quadrant_renders[quadrant_name]['gt_masks']
            pred_mask_list = quadrant_renders[quadrant_name]['pred_masks']
            
            temporal_gradient = compute_temporal_gradient_error(
                pred_list, gt_list,
                pred_mask_list, gt_mask_list
            )
            
            temporal_lpips_metrics = compute_temporal_lpips(
                pred_list, gt_list,
                pred_mask_list, gt_mask_list,
                lpips_model
            )
            
            flow_warp_metrics = compute_flow_warping_error(
                pred_list, gt_list,
                pred_mask_list, gt_mask_list,
                raft_model
            )
            
            object_metrics['temporal_metrics'][quadrant_name] = {
                **temporal_gradient,
                **temporal_lpips_metrics,
                **flow_warp_metrics
            }
    
    return object_metrics


def main():
    parser = argparse.ArgumentParser(description="Compute metrics for render predictions")
    parser.add_argument("--gt", required=True, help="Ground truth data path")
    parser.add_argument("--pred", required=True, help="Predictions path")
    parser.add_argument("--raft_model", default=None, help="Path to RAFT model checkpoint for flow-based warping metric")
    
    args = parser.parse_args()
    
    # Initialize LPIPS model once for all computations
    print("Initializing LPIPS model...")
    lpips_model = lpips.LPIPS(net='alex').eval()
    if torch.cuda.is_available():
        lpips_model = lpips_model.cuda()
        print("LPIPS model loaded on GPU")
    else:
        print("LPIPS model loaded on CPU")
    
    # Initialize RAFT model if available and checkpoint provided
    raft_model = None
    if RAFT_AVAILABLE and args.raft_model is not None:
        print(f"Initializing RAFT model from {args.raft_model}...")
        try:
            # Create args object for RAFT that supports attribute checking
            class RAFTArgs:
                def __init__(self):
                    self.small = False
                    self.mixed_precision = False
                    self.alternate_corr = False
                    self.dropout = 0
                    self.corr_levels = 4
                    self.corr_radius = 4
                
                def __contains__(self, key):
                    """Make the object support 'in' operator for attribute checking"""
                    return hasattr(self, key)
            
            raft_args = RAFTArgs()
            raft_model = torch.nn.DataParallel(RAFT(raft_args))
            raft_model.load_state_dict(torch.load(args.raft_model))
            raft_model = raft_model.module
            
            if torch.cuda.is_available():
                raft_model = raft_model.cuda()
                print("RAFT model loaded on GPU")
            else:
                print("RAFT model loaded on CPU")
            raft_model.eval()
        except Exception as e:
            print(f"Failed to load RAFT model: {e}")
            print("Flow-based warping metric will be disabled.")
            raft_model = None
    elif not RAFT_AVAILABLE:
        print("RAFT not available. Flow-based warping metric will be disabled.")
    else:
        print("No RAFT model checkpoint provided (--raft_model). Flow-based warping metric will be disabled.")
    
    # Find all lighting directories
    lighting_pattern = os.path.join(args.gt, "lighting_*")
    lighting_dirs = sorted(glob.glob(lighting_pattern))
    
    print(f"Found {len(lighting_dirs)} lighting directories to process")
    
    all_results = {
        'config': {
            'gt_path': args.gt,
            'pred_path': args.pred
        },
        'scenes': {},
        'global_temporal_averages': {}
    }
    
    # Process each lighting/camera/object combination
    scene_count = 0
    for lighting_dir in lighting_dirs:
        lighting_name = os.path.basename(lighting_dir)
        
        # Find all camera directories within this lighting
        camera_pattern = os.path.join(lighting_dir, "camera_*")
        camera_dirs = sorted(glob.glob(camera_pattern))
        
        for camera_dir in camera_dirs:
            camera_name = os.path.basename(camera_dir)
            
            # Find all object directories within this camera
            object_pattern = os.path.join(camera_dir, "object_*")
            object_dirs = sorted(glob.glob(object_pattern))
            
            for object_dir in tqdm(object_dirs, desc=f"Processing {lighting_name}/{camera_name}", leave=False):
                object_name = os.path.basename(object_dir)
                scene_name = f"{lighting_name}/{camera_name}/{object_name}"
                scene_count += 1
                
                gt_scene_path = object_dir
                pred_scene_path = os.path.join(args.pred, lighting_name, camera_name, object_name)
                
                if not os.path.exists(pred_scene_path):
                    print(f"Prediction directory not found: {pred_scene_path}")
                    continue
                
                scene_results = process_object(gt_scene_path, pred_scene_path, scene_name, lpips_model, raft_model)
                all_results['scenes'][scene_name] = scene_results
    
    print(f"Processed {scene_count} scenes total")
    
    # Compute global averages across all scenes for temporal metrics
    temporal_metrics_global = {
        'Overall': {
            'temporal_error': [],
            'temporal_lpips_pred': [],
            'temporal_lpips_gt': [],
            'temporal_lpips_diff': [],
            'flow_warp_error': []
        },
        'Mirror': {
            'temporal_error': [],
            'temporal_lpips_pred': [],
            'temporal_lpips_gt': [],
            'temporal_lpips_diff': [],
            'flow_warp_error': []
        },
        'Diffuse': {
            'temporal_error': [],
            'temporal_lpips_pred': [],
            'temporal_lpips_gt': [],
            'temporal_lpips_diff': [],
            'flow_warp_error': []
        },
        'Matte': {
            'temporal_error': [],
            'temporal_lpips_pred': [],
            'temporal_lpips_gt': [],
            'temporal_lpips_diff': [],
            'flow_warp_error': []
        },
        'Glossy': {
            'temporal_error': [],
            'temporal_lpips_pred': [],
            'temporal_lpips_gt': [],
            'temporal_lpips_diff': [],
            'flow_warp_error': []
        }
    }
    
    # Collect all temporal metrics across scenes
    for scene_name, scene_data in all_results['scenes'].items():
        if 'temporal_metrics' in scene_data:
            for key in ['Overall', 'Mirror', 'Diffuse', 'Matte', 'Glossy']:
                if key in scene_data['temporal_metrics']:
                    tm = scene_data['temporal_metrics'][key]
                    for metric, value in tm.items():
                        if not np.isnan(value) and not np.isinf(value):
                            temporal_metrics_global[key][metric].append(value)
    
    # Compute global averages for temporal metrics
    for key in temporal_metrics_global.keys():
        avg_temporal = {}
        for metric, values in temporal_metrics_global[key].items():
            if values:
                avg_temporal[metric] = np.mean(values)
            else:
                avg_temporal[metric] = float('nan')
        all_results['global_temporal_averages'][key] = avg_temporal
    
    # Save results to prediction path root
    results_path = os.path.join(args.pred, "temporal_metrics_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    print(f"\nResults saved to {results_path}")
    
    # Print summary
    print("\n=== TEMPORAL METRICS SUMMARY ===")
    print("E_temp = ||∂_t I - ∂_t I_GT||_1")
    print("Temporal LPIPS = LPIPS(I(t), I(t+1))")
    print("Flow Warp Error = RMSE(Warp(Pred(t), Flow_GT(t->t+1)), Pred(t+1))")
    print("(Lower values indicate better temporal consistency with GT)")
    
    print("\nGlobal Overall Temporal Metrics:")
    if 'Overall' in all_results['global_temporal_averages']:
        tm = all_results['global_temporal_averages']['Overall']
        print(f"  Temporal L1 Error: {tm['temporal_error']:.6f}")
        print(f"  Temporal LPIPS (Pred): {tm['temporal_lpips_pred']:.6f}")
        print(f"  Temporal LPIPS (GT): {tm['temporal_lpips_gt']:.6f}")
        print(f"  Temporal LPIPS Difference: {tm['temporal_lpips_diff']:.6f}")
        print(f"  Flow Warp Error: {tm['flow_warp_error']:.6f}")
    
    print("\nGlobal Quadrant Temporal Metrics:")
    for quadrant in ['Mirror', 'Diffuse', 'Matte', 'Glossy']:
        if quadrant in all_results['global_temporal_averages']:
            tm = all_results['global_temporal_averages'][quadrant]
            print(f"\n  {quadrant}:")
            print(f"    Temporal L1 Error: {tm['temporal_error']:.6f}")
            print(f"    Temporal LPIPS (Pred): {tm['temporal_lpips_pred']:.6f}")
            print(f"    Temporal LPIPS (GT): {tm['temporal_lpips_gt']:.6f}")
            print(f"    Temporal LPIPS Difference: {tm['temporal_lpips_diff']:.6f}")
            print(f"    Flow Warp Error: {tm['flow_warp_error']:.6f}")
    
    print("\n" + "="*50)
    print("INTERPRETATION:")
    print("  Temporal L1 Error:")
    print("    - Lower: Prediction temporal changes match GT better (pixel-level)")
    print("  Temporal LPIPS:")
    print("    - Pred: Perceptual difference between consecutive pred frames")
    print("    - GT: Perceptual difference between consecutive GT frames")
    print("    - Difference: How much pred temporal changes differ from GT (perceptual)")
    print("  Lower LPIPS Difference = predictions change at similar perceptual rate as GT")
    print("  Flow Warp Error:")
    print("    - Lower: Prediction motion matches GT motion better")
    print("    - Measures error when warping Pred(t) with GT flow and comparing to Pred(t+1)")
    print("="*50)
if __name__ == "__main__":
    main()