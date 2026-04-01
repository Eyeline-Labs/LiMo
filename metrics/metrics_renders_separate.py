import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import json
import argparse
from PIL import Image
import glob
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import math


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


def compute_rmse(pred, gt, mask=None):
    """Compute Root Mean Square Error."""
    if mask is not None:
        pred = pred * mask
        gt = gt * mask
        valid_pixels = mask.sum()
        if valid_pixels == 0:
            return float('nan')
        mse = torch.sum((pred - gt) ** 2) / valid_pixels
    else:
        mse = torch.mean((pred - gt) ** 2)
    return torch.sqrt(mse).item()


def compute_psnr(pred, gt, mask=None):
    """Compute Peak Signal-to-Noise Ratio."""
    if mask is not None:
        pred = pred * mask
        gt = gt * mask
        valid_pixels = mask.sum()
        if valid_pixels == 0:
            return float('nan')
        mse = torch.sum((pred - gt) ** 2) / valid_pixels
    else:
        mse = torch.mean((pred - gt) ** 2)
    
    if mse == 0:
        return float('inf')
    
    # Assuming pixel values are in range [0, 1]
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr.item()


def compute_si_rmse(pred, gt, mask=None):
    """Compute Scale-Invariant Root Mean Square Error."""
    if mask is not None:
        pred_flat = pred[mask.expand_as(pred) > 0.5]
        gt_flat = gt[mask.expand_as(gt) > 0.5]
    else:
        pred_flat = pred.flatten()
        gt_flat = gt.flatten()
    
    if len(pred_flat) == 0:
        return float('nan')
    
    # Convert to log space
    pred_log = torch.log(torch.clamp(pred_flat, min=1e-6))
    gt_log = torch.log(torch.clamp(gt_flat, min=1e-6))
    
    # Compute scale-invariant error
    diff = pred_log - gt_log
    si_mse = torch.mean(diff ** 2) - (torch.mean(diff) ** 2)
    return torch.sqrt(si_mse).item()


def compute_ssim_metric(pred, gt, mask=None):
    """Compute Structural Similarity Index."""
    pred_np = pred.permute(1, 2, 0).cpu().numpy()
    gt_np = gt.permute(1, 2, 0).cpu().numpy()
    
    if mask is not None:
        mask_np = mask[0].cpu().numpy().astype(bool)
        if not mask_np.any():
            return float('nan')
        # Apply mask by setting non-masked areas to 0
        pred_np = pred_np * mask_np[..., None]
        gt_np = gt_np * mask_np[..., None]
    
    try:
        return ssim(gt_np, pred_np, data_range=1.0, channel_axis=2)
    except:
        return float('nan')


def compute_angular_error(pred, gt, mask=None):
    """Compute Angular Error between normal vectors."""
    if mask is not None:
        # Apply mask to both images
        pred_masked = pred * mask
        gt_masked = gt * mask
        valid_mask = mask[0] > 0.5
    else:
        pred_masked = pred
        gt_masked = gt
        valid_mask = torch.ones(pred.shape[1:], dtype=torch.bool)
    
    if not valid_mask.any():
        return float('nan')
    
    # Flatten spatial dimensions for valid pixels only
    pred_flat = pred_masked.permute(1, 2, 0)[valid_mask]  # (N_valid, 3)
    gt_flat = gt_masked.permute(1, 2, 0)[valid_mask]      # (N_valid, 3)
    
    # Normalize vectors
    pred_norm = F.normalize(pred_flat, dim=1, eps=1e-8)
    gt_norm = F.normalize(gt_flat, dim=1, eps=1e-8)
    
    # Compute dot product and clamp to avoid numerical issues
    dot_product = torch.sum(pred_norm * gt_norm, dim=1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    
    # Compute angular error in degrees
    angular_error = torch.acos(torch.abs(dot_product)) * 180.0 / math.pi
    
    return torch.mean(angular_error).item()


def process_object(gt_object_path, pred_object_path, scene_name):
    """Process a single scene and compute metrics for all renders."""
    object_metrics = {
        'scene_name': scene_name,
        'renders': {},
        'quadrant_averages': {},
        'overall_averages': {}
    }
    
    # Find all ground truth renders
    gt_pattern = os.path.join(gt_object_path, "render_metrics", "render_*.png")
    gt_files = sorted(glob.glob(gt_pattern))
    
    if not gt_files:
        print(f"No ground truth renders found for {scene_name}")
        return object_metrics
    
    render_metrics = []
    quadrant_metrics = {'Mirror': [], 'Diffuse': [], 'Matte': [], 'Glossy': []}
    
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
            
            # Extract quadrants
            gt_quadrants = extract_quadrants(gt_rgb, gt_alpha)
            pred_quadrants = extract_quadrants(pred_rgb, pred_alpha)
            
            render_result = {}
            
            # Compute metrics for each quadrant
            for quadrant_name in ['Mirror', 'Diffuse', 'Matte', 'Glossy']:
                gt_quad_rgb = gt_quadrants[quadrant_name]['rgb']
                gt_quad_alpha = gt_quadrants[quadrant_name]['alpha']
                pred_quad_rgb = pred_quadrants[quadrant_name]['rgb']
                pred_quad_alpha = pred_quadrants[quadrant_name]['alpha']
                
                # Use ground truth alpha as mask (binary)
                quad_mask = torch.where(gt_quad_alpha > 0.5, 1.0, 0.0)
                
                # Compute metrics for this quadrant
                quad_rmse = compute_rmse(pred_quad_rgb, gt_quad_rgb, quad_mask)
                quad_si_rmse = compute_si_rmse(pred_quad_rgb, gt_quad_rgb, quad_mask)
                quad_ssim = compute_ssim_metric(pred_quad_rgb, gt_quad_rgb, quad_mask)
                quad_angular_error = compute_angular_error(pred_quad_rgb, gt_quad_rgb, quad_mask)
                quad_psnr = compute_psnr(pred_quad_rgb, gt_quad_rgb, quad_mask)
                
                quadrant_result = {
                    'rmse': quad_rmse,
                    'si_rmse': quad_si_rmse,
                    'ssim': quad_ssim,
                    'angular_error': quad_angular_error,
                    'psnr': quad_psnr
                }
                
                render_result[quadrant_name] = quadrant_result
                quadrant_metrics[quadrant_name].append(quadrant_result)
            
            # Also compute overall metrics for the full image
            mask = torch.where(gt_alpha > 0.5, 1.0, 0.0)
            overall_result = {
                'rmse': compute_rmse(pred_rgb, gt_rgb, mask),
                'si_rmse': compute_si_rmse(pred_rgb, gt_rgb, mask),
                'ssim': compute_ssim_metric(pred_rgb, gt_rgb, mask),
                'angular_error': compute_angular_error(pred_rgb, gt_rgb, mask),
                'psnr': compute_psnr(pred_rgb, gt_rgb, mask)
            }
            render_result['Overall'] = overall_result
            
            object_metrics['renders'][render_name] = render_result
            render_metrics.append(overall_result)
            
        except Exception as e:
            print(f"Error processing {render_name} for {scene_name}: {e}")
            continue
    
    # Compute averages for each quadrant
    for quadrant_name in ['Mirror', 'Diffuse', 'Matte', 'Glossy']:
        if quadrant_metrics[quadrant_name]:
            avg_metrics = {}
            for metric in ['rmse', 'si_rmse', 'ssim', 'angular_error', 'psnr']:
                values = [qm[metric] for qm in quadrant_metrics[quadrant_name] if not np.isnan(qm[metric]) and not np.isinf(qm[metric])]
                if values:
                    avg_metrics[metric] = np.mean(values)
                else:
                    avg_metrics[metric] = float('nan')
            
            object_metrics['quadrant_averages'][quadrant_name] = avg_metrics
    
    # Compute overall averages
    if render_metrics:
        avg_metrics = {}
        for metric in ['rmse', 'si_rmse', 'ssim', 'angular_error', 'psnr']:
            values = [rm[metric] for rm in render_metrics if not np.isnan(rm[metric]) and not np.isinf(rm[metric])]
            if values:
                avg_metrics[metric] = np.mean(values)
            else:
                avg_metrics[metric] = float('nan')
        
        object_metrics['overall_averages'] = avg_metrics
    
    return object_metrics


def main():
    parser = argparse.ArgumentParser(description="Compute metrics for render predictions")
    parser.add_argument("--gt", required=True, help="Ground truth data path")
    parser.add_argument("--pred", required=True, help="Predictions path")
    
    args = parser.parse_args()
    
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
        'global_quadrant_averages': {},
        'global_overall_averages': {}
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
                
                scene_results = process_object(gt_scene_path, pred_scene_path, scene_name)
                all_results['scenes'][scene_name] = scene_results
    
    print(f"Processed {scene_count} scenes total")
    
    # Compute global averages across all scenes for each quadrant
    quadrant_global_metrics = {
        'Mirror': {'rmse': [], 'si_rmse': [], 'ssim': [], 'angular_error': [], 'psnr': []},
        'Diffuse': {'rmse': [], 'si_rmse': [], 'ssim': [], 'angular_error': [], 'psnr': []},
        'Matte': {'rmse': [], 'si_rmse': [], 'ssim': [], 'angular_error': [], 'psnr': []},
        'Glossy': {'rmse': [], 'si_rmse': [], 'ssim': [], 'angular_error': [], 'psnr': []}
    }
    
    overall_global_metrics = {
        'rmse': [],
        'si_rmse': [],
        'ssim': [],
        'angular_error': [],
        'psnr': []
    }
    
    # Collect all render metrics across scenes
    for scene_name, scene_data in all_results['scenes'].items():
        if 'renders' in scene_data:
            for render_data in scene_data['renders'].values():
                # Collect quadrant metrics
                for quadrant in ['Mirror', 'Diffuse', 'Matte', 'Glossy']:
                    if quadrant in render_data:
                        for metric in quadrant_global_metrics[quadrant].keys():
                            if not np.isnan(render_data[quadrant][metric]) and not np.isinf(render_data[quadrant][metric]):
                                quadrant_global_metrics[quadrant][metric].append(render_data[quadrant][metric])
                
                # Collect overall metrics
                if 'Overall' in render_data:
                    for metric in overall_global_metrics.keys():
                        if not np.isnan(render_data['Overall'][metric]) and not np.isinf(render_data['Overall'][metric]):
                            overall_global_metrics[metric].append(render_data['Overall'][metric])
    
    # Compute global averages for each quadrant
    for quadrant in ['Mirror', 'Diffuse', 'Matte', 'Glossy']:
        quadrant_avg = {}
        for metric, values in quadrant_global_metrics[quadrant].items():
            if values:
                quadrant_avg[metric] = np.mean(values)
            else:
                quadrant_avg[metric] = float('nan')
        all_results['global_quadrant_averages'][quadrant] = quadrant_avg
    
    # Compute global overall averages
    global_overall_avg = {}
    for metric, values in overall_global_metrics.items():
        if values:
            global_overall_avg[metric] = np.mean(values)
        else:
            global_overall_avg[metric] = float('nan')
    
    all_results['global_overall_averages'] = global_overall_avg
    
    # Save results to prediction path root
    results_path = os.path.join(args.pred, "render_metrics_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    print(f"Results saved to {results_path}")
    
    # Print summary
    
    
    print("\nPer-Scene Quadrant Averages:")
    for scene_name, scene_data in all_results['scenes'].items():
        if 'quadrant_averages' in scene_data and scene_data['quadrant_averages']:
            print(f"  {scene_name}:")
            for quadrant in ['Mirror', 'Diffuse', 'Matte', 'Glossy']:
                if quadrant in scene_data['quadrant_averages']:
                    avg_data = scene_data['quadrant_averages'][quadrant]
                    print(f"    {quadrant}:")
                    print(f"      RMSE: {avg_data['rmse']:.4f}")
                    print(f"      SI-RMSE: {avg_data['si_rmse']:.4f}")
                    print(f"      SSIM: {avg_data['ssim']:.4f}")
                    print(f"      Angular Error: {avg_data['angular_error']:.2f}°")
                    print(f"      PSNR: {avg_data['psnr']:.2f} dB")

    print("\n=== SUMMARY ===")
    print("Global Overall Averages:")
    print(f"  RMSE: {global_overall_avg['rmse']:.4f}")
    print(f"  SI-RMSE: {global_overall_avg['si_rmse']:.4f}")
    print(f"  SSIM: {global_overall_avg['ssim']:.4f}")
    print(f"  Angular Error: {global_overall_avg['angular_error']:.2f}°")
    print(f"  PSNR: {global_overall_avg['psnr']:.2f} dB")
    
    print("\nGlobal Quadrant Averages:")
    for quadrant in ['Mirror', 'Diffuse', 'Matte', 'Glossy']:
        if quadrant in all_results['global_quadrant_averages']:
            quad_avg = all_results['global_quadrant_averages'][quadrant]
            print(f"  {quadrant}:")
            print(f"    RMSE: {quad_avg['rmse']:.4f}")
            print(f"    SI-RMSE: {quad_avg['si_rmse']:.4f}")
            print(f"    SSIM: {quad_avg['ssim']:.4f}")
            print(f"    Angular Error: {quad_avg['angular_error']:.2f}°")
            print(f"    PSNR: {quad_avg['psnr']:.2f} dB")
if __name__ == "__main__":
    main()