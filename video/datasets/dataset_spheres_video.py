from dataclasses import dataclass, field
import math
import random
import shutil
import time
import csv
from typing import List, Optional
import torch
from scipy.special import sph_harm
from typing import List, Optional, Union

from torch.utils.data import Dataset, Sampler
import OpenEXR
from pyexr.exr import InputFile as PyexrInputFile
import pyexr
import os
from tqdm import tqdm
import pickle
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import DataLoader
import numpy as np
from .data_utils import list_s3_files, get_auto_exposure, white_balance, Lin_to_sRGB, Lin_to_Log_, depth_to_scaled_depth, position_to_scaled_position, Lin_to_Reinhard, depth_to_scaled_depth_vectorized
#from data_utils import download_s3_file, list_s3_files, get_auto_exposure, white_balance, Lin_to_sRGB, Lin_to_Log_, depth_to_scaled_depth, position_to_scaled_position



from metaflow import S3
import uuid

import boto3
import botocore

from glob import glob
from natsort import natsorted
import fnmatch

from ezexr import imsave


from torch.profiler import record_function

#### Dataset ####
@dataclass
class SpheresAugConfig:
    horizontal_flip: bool = False
    vignetting: bool = False
    auto_exposure: bool = False
    noise: bool = False
    haze: bool = False
    white_balance: bool = False
    reverse: bool = False  # Randomly reverse the order of the frames

@dataclass
class SpheresDatasetConfig:
    data_dir: Union[str, List[str]]
    val_data_dir: Optional[Union[str, List[str]]] = None
    augmentation_cfg: Optional[SpheresAugConfig] = None
    sampling_weight: float = 1.0
    random_resolution: bool = False
    random_ev: bool = False
    HDR: bool = False
    mask_img: bool = True

ALL_LAYERS = ['image', 'alpha', 'composed_image', 'depth', 'albedo', 'world_normals', 'env', 'camera_normals', 'lighting_r_1', 'lighting_r_0.5', 'lighting_r_0.25', 'lighting_r_0', 'intrinsics']
IMAGE_LAYERS = ['image', 'composed_image', 'env']
LIGHTING_LAYERS = ['lighting_r_1', 'lighting_r_0.5', 'lighting_r_0.25', 'lighting_r_0']

TARGET_TO_PROMPT = {
    "sphere_0":"Metallic sphere",
    #"sphere_1":"half diffuse, metallic sphere",
    "sphere_1":"Diffuse sphere"
}


def clean_nan(data):
    for key in data.keys():
        #If is a tensor, replace NaN values with 0
        if isinstance(data[key], list):
            if isinstance(data[key][0], torch.Tensor):
                for i in range(len(data[key])):
                    data[key][i] = torch.nan_to_num_(data[key][i])

def manage_colors(data):
    for key in data.keys():
        if key in ['image', 'sphere_0', 'sphere_1', 'sphere_2', 'sphere', 'video', 'image_full']:
            for i in range(len(data[key])):
                torch.clamp_(data[key][i], 1e-4, 1.0)
                data[key][i] = Lin_to_sRGB(data[key][i])
                data[key][i].mul_(2).sub_(1) 
        elif 'sphere_mask' in key:
            for i in range(len(data[key])):
                data[key][i] = torch.clamp_(data[key][i], 1e-4, 1.0).repeat(3, 1, 1)
                data[key][i].mul_(2).sub_(1) 
        elif 'position' in key:
            for i in range(len(data[key])):
                data[key][i] = position_to_scaled_position(data[key][i])
                data[key][i].mul_(2).sub_(1) 
                #data[key][i] = torch.zeros_like(data[key][i])
        elif 'dist_to_sphere' in key:
            for i in range(len(data[key])):
                data[key][i] = depth_to_scaled_depth(data[key][i].repeat(3, 1, 1), torch.ones_like(data[key][i]))  # Repeat the depth channel to match RGB
                data[key][i] = torch.nan_to_num(data[key][i])
                data[key][i].mul_(2).sub_(1)
        elif key in ['normals', 'dir_to_sphere']:
            for i in range(len(data[key])):
                data[key][i] = data[key][i].clamp(-1, 1)


def clean_nan_vectorized(data):
    """Vectorized version of clean_nan"""
    for key in data.keys():
        if isinstance(data[key], torch.Tensor):
            torch.nan_to_num_(data[key])

def manage_colors_vectorized(data):
    """Vectorized version of manage_colors"""
    for key in data.keys():
        if key in ['image', 'sphere_0', 'sphere_1', 'sphere_2', 'sphere', 'video', 'image_full']:
            torch.clamp_(data[key], 1e-4, 1.0)
            data[key] = Lin_to_sRGB(data[key])
            data[key].mul_(2).sub_(1)
        elif 'sphere_mask' in key:
            torch.clamp_(data[key], 1e-4, 1.0)
            if data[key].shape[1] == 1:  # If single channel, repeat to 3 channels
                data[key] = data[key].repeat(1, 3, 1, 1)
            data[key].mul_(2).sub_(1)
        elif 'position' in key:
            data[key] = position_to_scaled_position(data[key])
            data[key].mul_(2).sub_(1)
        elif 'dist_to_sphere' in key:
            data[key] = depth_to_scaled_depth_vectorized(data[key].repeat(1, 3, 1, 1), torch.ones_like(data[key].repeat(1, 3, 1, 1)))
            data[key] = torch.nan_to_num(data[key])
            data[key].mul_(2).sub_(1)
        elif key in ['normals', 'dir_to_sphere']:
            data[key] = data[key].clamp(-1, 1)

def treat_data(data, legacy=False, given_ev=None, test=False, bad_incident=False):
    """
    Does the world2camera transformations
    Computes the dist and dir to sphere
    Normalizes the data
    """
    output_dict = data

    batch_size = output_dict['position'].shape[0]
    device = output_dict['position'].device

    output_dict['dir_to_sphere'] = torch.zeros_like(output_dict['position'])
    output_dict['dist_to_sphere'] = torch.zeros_like(output_dict['position'][:, 0:1, :, :])  # Shape: (B, 1, H, W)


    with record_function("find_ev"):
        ev = 0
        if output_dict['random_ev']:
            #Find out what the minimum EV is for the sphere images
            sphere_image = output_dict['video'][0]  # Assuming the first sphere image is representative
            maximum_value = sphere_image.max()
            if maximum_value <= 0:
                minimum_ev = 0
                print(maximum_value)
                print("Maximum value in sphere images is negative, setting minimum EV to 0")
            elif maximum_value == 0:
                minimum_ev = 0
                print("Maximum value in sphere images is 0, setting minimum EV to 0")
            else:
                minimum_ev = - math.log2(2 * maximum_value)
                try:
                    minimum_ev = math.floor(minimum_ev / 3) * 3  # Round to the nearest multiple of 3
                except:
                    print("Error in computing minimum_ev, setting to 0")
                    minimum_ev = 0
            minimum_ev = max(minimum_ev, -12)  # Ensure minimum EV is not too low
            minimum_ev = min(minimum_ev, 0)  # Ensure minimum EV is not positive
            #Sample a random EV in the range [minimum_ev, 0]
            start = minimum_ev + (3 - minimum_ev % 3) if minimum_ev % 3 != 0 else minimum_ev
            # Create list of valid multiples of 3 from start to 0 (inclusive)
            valid_evs = list(range(start, 1, 3))  # step is 3, end at 1 to include 0

            # Sample one
            ev = random.choice(valid_evs)

        if given_ev is not None:
            ev = given_ev

    if output_dict['ev_augmentation']:
        # Apply auto exposure to all frames at once
        exposure_factors = torch.stack([get_auto_exposure(output_dict['image'][i]) for i in range(batch_size)])
        # if test:
        #     exposure_factors = torch.ones_like(exposure_factors)
        
        # Reshape for broadcasting: (B, 1, 1, 1)
        exposure_factors = exposure_factors.view(batch_size, 1, 1, 1)
        
        output_dict['image'] *= exposure_factors
        output_dict['video'] *= exposure_factors
    
    # Apply the EV to all sphere images at once
    output_dict['video'] *= (2 ** ev) #At this point it's just the sphere

    output_dict['video'] = (output_dict['video'] * output_dict['sphere_mask'] + 
                           output_dict['image'] * (1 - output_dict['sphere_mask']))

    if output_dict['mask_img']:
        output_dict['image'] = output_dict['image'] * (1 - output_dict['sphere_mask'])

    #Cam view
    with record_function("cam_view"):
        world2cam = output_dict['world2cam']  # Shape: (B, 4, 4)
        
        # Reshape position for batch matrix multiplication
        B, C, H, W = output_dict['position'].shape
        position_map = output_dict['position'].view(B, C, -1)  # (B, 3, H*W)
        
        # Add homogeneous coordinates
        ones = torch.ones(B, 1, H*W, dtype=position_map.dtype, device=position_map.device)
        position_map = torch.cat((position_map, ones), dim=1)  # (B, 4, H*W)
        
        # Batch matrix multiplication
        position_map = torch.bmm(world2cam, position_map)  # (B, 4, H*W)
        position_map = position_map[:, :3, :]  # Remove homogeneous coordinate
        position_map = position_map.reshape(B, 3, H, W)  # Reshape back
        
        # Invert Z coordinate for all frames
        position_map[:, 2, :, :] = -position_map[:, 2, :, :]
        output_dict['position'] = position_map
        
        # Transform normal maps to camera coordinates
        normal_map = output_dict['normals'].view(B, 3, -1)  # (B, 3, H*W)
        # Use only rotation part of transformation matrix
        normal_map = torch.bmm(world2cam[:, :3, :3], normal_map)  # (B, 3, H*W)
        normal_map = normal_map.reshape(B, 3, H, W)
        normal_map[:, 2, :, :] = -normal_map[:, 2, :, :]
        output_dict['normals'] = normal_map
    
    with record_function("sphere"):
        sphere_center = output_dict['sphere_center']  # Shape: (B, 3)
        
        # Add homogeneous coordinates
        ones = torch.ones(B, 1, dtype=sphere_center.dtype, device=sphere_center.device)
        sphere_center = torch.cat((sphere_center, ones), dim=1)  # (B, 4)
        
        # Transform sphere centers
        sphere_center = torch.bmm(world2cam, sphere_center.unsqueeze(-1)).squeeze(-1)  # (B, 4)
        sphere_center = sphere_center[:, :3]  # (B, 3)
        sphere_center[:, 2] = -sphere_center[:, 2]  # Invert Z coordinate
        
        
        # Calculate vector to sphere for all frames
        vector_to_sphere = sphere_center.unsqueeze(-1).unsqueeze(-1) - position_map  # (B, 3, H, W)
        
        # Distance and direction calculations
        dist_to_sphere = vector_to_sphere.norm(dim=1, keepdim=True)  # (B, 1, H, W)
        dir_to_sphere = vector_to_sphere / (dist_to_sphere + 1e-8)  # (B, 3, H, W)
        
        # Handle zero distances
        min_vals = torch.where(dist_to_sphere != 0, dist_to_sphere, torch.inf).min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        dist_to_sphere = torch.where(dist_to_sphere == 0, min_vals, dist_to_sphere)
        
        # Compute incident ray direction and reflection
        cam_to_point = position_map
        cam_to_point_norm = cam_to_point.norm(dim=1, keepdim=True)
        cam_to_point_dir = cam_to_point / (cam_to_point_norm + 1e-8)
        incident_ray = -cam_to_point_dir
        
        if bad_incident:
            incident_ray = -dir_to_sphere
        
        # Dot product and reflection calculation
        dot_product = torch.sum(incident_ray * normal_map, dim=1, keepdim=True).clamp(-1, 1)
        reflection = incident_ray - 2 * dot_product * normal_map
        reflection = reflection / (reflection.norm(dim=1, keepdim=True) + 1e-8)
        
        # Apply reflection where sphere mask is 1
        sphere_mask = output_dict['sphere_mask']  # (B, 1, H, W)
        dir_to_sphere = torch.where(sphere_mask == 1, reflection, dir_to_sphere)
        
        output_dict['dir_to_sphere'] = dir_to_sphere
        output_dict['dist_to_sphere'] = dist_to_sphere

    # Clean NaN values
    with record_function("clean nan"):
        clean_nan_vectorized(output_dict)
    
    # Manage colors
    with record_function("manage colors"):
        manage_colors_vectorized(output_dict)
    
    # for i in range(len(output_dict['video'])):
        
    #     with record_function("auto_expose"):
    #      #Auto exposure 
    #         if output_dict['ev_augmentation']:
    #             exposure_factor = get_auto_exposure(output_dict['image_full'][i])
    #             if test:
    #                 exposure_factor = 1
                
    #             output_dict['image'][i] *= exposure_factor
    #             output_dict['video'][i] *= exposure_factor
            
    #         #Apply the EV to the sphere image
    #         output_dict['video'][i] *= 2 ** ev
        

    #     with record_function("cam_view"):
    #         world2cam = output_dict['world2cam'][i]
            
    #         position_map = output_dict['position'][i].view(3, -1)  # Flatten the spatial dimensions

    #         #Add extra channel for homegeneous coordinates
    #         position_map = torch.cat((position_map, torch.ones(1, position_map.shape[1], dtype=position_map.dtype, device=position_map.device)), dim=0)  # Add a row of ones for homogeneous coordinates
    #         position_map = torch.matmul(world2cam, position_map)  # Transform to world coordinates
    #         position_map = position_map[:3, :]  # Remove the homogeneous coordinate row
    #         position_map = position_map.reshape(3, *output_dict['position'][i].shape[-2:])  # Reshape back to (3, H, W)

    #         position_map[2, :, :] = -position_map[2, :, :]  # Invert the Z coordinate to have positive Z pointing outwards
    #         output_dict['position'][i] = position_map
            
            
    #         #Bring normal map to camera coordinates
    #         normal_map = output_dict['normals'][i].view(3, -1)  # Flatten the spatial dimensions
    #         #Add extra channel for homegeneous coordinates
    #         #normal_map = torch.cat((normal_map, torch.ones(1, normal_map.shape[1], dtype=normal_map.dtype, device=normal_map.device)), dim=0)
    #         #normal_map = torch.matmul(world2cam, normal_map)  # Transform to world coordinates
    #         normal_map = torch.matmul(world2cam[:3,:3], normal_map)
    #         #normal_map = normal_map[:3, :]  # Remove the homogeneous coordinate row
    #         normal_map = normal_map.reshape(3, *output_dict['normals'][i].shape[-2:])  # Reshape back to (3, H, W)
    #         normal_map[2, :, :] = -normal_map[2, :, :]  # Invert the Z coordinate to have positive Z pointing outwards
    #         output_dict['normals'][i] = normal_map
        
    #     with record_function("sphere"):
    #         # #Direction to sphere
    #         sphere_center = output_dict['sphere_center'][i]
    #         #sphere_center appended with 1 for homogeneous coordinates
    #         sphere_center = torch.cat((sphere_center, torch.ones(1, dtype=sphere_center.dtype, device=sphere_center.device)), dim=0)
    #         sphere_center = torch.matmul(world2cam, sphere_center.T).T 
    #         sphere_center = sphere_center[:3]  # Remove the homogeneous coordinate row
    #         sphere_center[2] = -sphere_center[2]  # Invert the Z coordinate to have positive Z pointing outwards
                
    #         # Get the vector from the camera to the sphere center
    #         vector_to_sphere = sphere_center[:,None,None] - position_map
        
    #         dist_to_sphere = vector_to_sphere.norm(dim=0, keepdim=True)  # Get the distance to the sphere
    #         dir_to_sphere = vector_to_sphere / (dist_to_sphere + 1e-8)  # Normalize the vector to get the direction to the sphere
            

    #         #Set zeros to the non-zero minimum value
                
    #         #dist_to_sphere[dist_to_sphere == 0] = dist_to_sphere[dist_to_sphere != 0].min().item()
    #         min_val = torch.where(dist_to_sphere != 0, dist_to_sphere, torch.inf).min()
    #         dist_to_sphere = torch.where(dist_to_sphere == 0, min_val, dist_to_sphere)

    #         #Compute the incident ray direction on the sphere
    #         cam_to_point = position_map
    #         cam_to_point_dir = cam_to_point / (cam_to_point.norm(dim=0, keepdim=True) + 1e-8)
    #         incident_ray = -cam_to_point_dir
    #         if bad_incident:
    #             incident_ray = -dir_to_sphere  # Temp for backward compatibility.
    #         dot_product = (torch.sum(incident_ray * normal_map, dim=0, keepdim=True)).clamp(-1, 1)  # Dot product between incident ray and normal
    #         reflection = incident_ray - 2 * dot_product * normal_map  # Reflection direction
    #         reflection = reflection / (reflection.norm(dim=0, keepdim=True) + 1e-8)  # Normalize the reflection vector

    #         dir_to_sphere = torch.where(output_dict['sphere_mask'][i]==1, reflection, dir_to_sphere)
                        
    #         output_dict['dir_to_sphere'][i] = dir_to_sphere
    #         output_dict['dist_to_sphere'][i] = dist_to_sphere
                
                        
    # with record_function("clean nan"):
    #     clean_nan(output_dict)
    # with record_function("manage colors"):
    #     manage_colors(output_dict)


    for key in output_dict.keys():
        if isinstance(output_dict[key], torch.Tensor):
            output_dict[key] = [output_dict[key][i] for i in range(output_dict[key].shape[0])]

    output_dict['ev'] = ev
    # output_dict["prompt"] = TARGET_TO_PROMPT[random_req_layer]+ f" [EV{ev}]"
    output_dict["prompt"] = TARGET_TO_PROMPT[output_dict['target_layer']]+ f" [EV{ev}]"

    return output_dict


def treat_data_flux(data, legacy=False, given_ev=None, test=False, bad_incident=False):
    """
    Does the world2camera transformations
    Computes the dist and dir to sphere
    Normalizes the data
    """
    output_dict = data

    batch_size = output_dict['position'].shape[0]
    device = output_dict['position'].device

    output_dict['dir_to_sphere'] = torch.zeros_like(output_dict['position'])
    output_dict['dist_to_sphere'] = torch.zeros_like(output_dict['position'][:, 0:1, :, :])  # Shape: (B, 1, H, W)


    with record_function("find_ev"):
        ev = 0
        if output_dict['random_ev']:
            #Find out what the minimum EV is for the sphere images
            sphere_image = output_dict['video'][0]  # Assuming the first sphere image is representative
            maximum_value = sphere_image.max()
            if maximum_value <= 0:
                minimum_ev = 0
                print(maximum_value)
                print("Maximum value in sphere images is negative, setting minimum EV to 0")
            elif maximum_value == 0:
                minimum_ev = 0
                print("Maximum value in sphere images is 0, setting minimum EV to 0")
            else:
                minimum_ev = - math.log2(2 * maximum_value)
                try:
                    minimum_ev = math.floor(minimum_ev / 3) * 3  # Round to the nearest multiple of 3
                except:
                    print("Error in computing minimum_ev, setting to 0")
                    minimum_ev = 0
            minimum_ev = max(minimum_ev, -12)  # Ensure minimum EV is not too low
            minimum_ev = min(minimum_ev, 0)  # Ensure minimum EV is not positive
            #Sample a random EV in the range [minimum_ev, 0]
            start = minimum_ev + (3 - minimum_ev % 3) if minimum_ev % 3 != 0 else minimum_ev
            # Create list of valid multiples of 3 from start to 0 (inclusive)
            valid_evs = list(range(start, 1, 3))  # step is 3, end at 1 to include 0

            # Sample one
            ev = random.choice(valid_evs)

        if given_ev is not None:
            ev = given_ev

    if output_dict['ev_augmentation']:
        # Apply auto exposure to all frames at once
        exposure_factors = torch.stack([get_auto_exposure(output_dict['image'][i]) for i in range(batch_size)])
        # if test:
        #     exposure_factors = torch.ones_like(exposure_factors)
        
        # Reshape for broadcasting: (B, 1, 1, 1)
        exposure_factors = exposure_factors.view(batch_size, 1, 1, 1)
        
        output_dict['image'] *= exposure_factors
        output_dict['video'] *= exposure_factors
    
    # Apply the EV to all sphere images at once
    output_dict['video'] *= (2 ** ev) #At this point it's just the sphere

    output_dict['video'] = (output_dict['video'] * output_dict['sphere_mask'] + 
                           output_dict['image'] * (1 - output_dict['sphere_mask']))

    if output_dict['mask_img']:
        output_dict['image'] = output_dict['image'] * (1 - output_dict['sphere_mask'])

    #Cam view
    with record_function("cam_view"):
        world2cam = output_dict['world2cam']  # Shape: (B, 4, 4)
        
        # Reshape position for batch matrix multiplication
        B, C, H, W = output_dict['position'].shape
        position_map = output_dict['position'].view(B, C, -1)  # (B, 3, H*W)
        
        # Add homogeneous coordinates
        ones = torch.ones(B, 1, H*W, dtype=position_map.dtype, device=position_map.device)
        position_map = torch.cat((position_map, ones), dim=1)  # (B, 4, H*W)
        
        # Batch matrix multiplication
        position_map = torch.bmm(world2cam, position_map)  # (B, 4, H*W)
        position_map = position_map[:, :3, :]  # Remove homogeneous coordinate
        position_map = position_map.reshape(B, 3, H, W)  # Reshape back
        
        # Invert Z coordinate for all frames
        position_map[:, 2, :, :] = -position_map[:, 2, :, :]
        output_dict['position'] = position_map
        
        # Transform normal maps to camera coordinates
        normal_map = output_dict['normals'].view(B, 3, -1)  # (B, 3, H*W)
        # Use only rotation part of transformation matrix
        normal_map = torch.bmm(world2cam[:, :3, :3], normal_map)  # (B, 3, H*W)
        normal_map = normal_map.reshape(B, 3, H, W)
        normal_map[:, 2, :, :] = normal_map[:, 2, :, :]
        output_dict['normals'] = normal_map
    
    with record_function("sphere"):
        sphere_center = output_dict['sphere_center']  # Shape: (B, 3)
        
        # Add homogeneous coordinates
        ones = torch.ones(B, 1, dtype=sphere_center.dtype, device=sphere_center.device)
        sphere_center = torch.cat((sphere_center, ones), dim=1)  # (B, 4)
        
        # Transform sphere centers
        sphere_center = torch.bmm(world2cam, sphere_center.unsqueeze(-1)).squeeze(-1)  # (B, 4)
        sphere_center = sphere_center[:, :3]  # (B, 3)
        sphere_center[:, 2] = -sphere_center[:, 2]  # Invert Z coordinate
        
        
        # Calculate vector to sphere for all frames
        vector_to_sphere = sphere_center.unsqueeze(-1).unsqueeze(-1) - position_map  # (B, 3, H, W)
        
        # Distance and direction calculations
        dist_to_sphere = vector_to_sphere.norm(dim=1, keepdim=True)  # (B, 1, H, W)
        dir_to_sphere = vector_to_sphere / (dist_to_sphere + 1e-8)  # (B, 3, H, W)
        
        # Handle zero distances
        min_vals = torch.where(dist_to_sphere != 0, dist_to_sphere, torch.inf).min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        dist_to_sphere = torch.where(dist_to_sphere == 0, min_vals, dist_to_sphere)
        
        output_dict['dir_to_sphere'] = dir_to_sphere
        output_dict['dist_to_sphere'] = dist_to_sphere

    # Clean NaN values
    with record_function("clean nan"):
        clean_nan_vectorized(output_dict)
    
    # Manage colors
    with record_function("manage colors"):
        manage_colors_vectorized(output_dict)
    
    # for i in range(len(output_dict['video'])):
        
    #     with record_function("auto_expose"):
    #      #Auto exposure 
    #         if output_dict['ev_augmentation']:
    #             exposure_factor = get_auto_exposure(output_dict['image_full'][i])
    #             if test:
    #                 exposure_factor = 1
                
    #             output_dict['image'][i] *= exposure_factor
    #             output_dict['video'][i] *= exposure_factor
            
    #         #Apply the EV to the sphere image
    #         output_dict['video'][i] *= 2 ** ev
        

    #     with record_function("cam_view"):
    #         world2cam = output_dict['world2cam'][i]
            
    #         position_map = output_dict['position'][i].view(3, -1)  # Flatten the spatial dimensions

    #         #Add extra channel for homegeneous coordinates
    #         position_map = torch.cat((position_map, torch.ones(1, position_map.shape[1], dtype=position_map.dtype, device=position_map.device)), dim=0)  # Add a row of ones for homogeneous coordinates
    #         position_map = torch.matmul(world2cam, position_map)  # Transform to world coordinates
    #         position_map = position_map[:3, :]  # Remove the homogeneous coordinate row
    #         position_map = position_map.reshape(3, *output_dict['position'][i].shape[-2:])  # Reshape back to (3, H, W)

    #         position_map[2, :, :] = -position_map[2, :, :]  # Invert the Z coordinate to have positive Z pointing outwards
    #         output_dict['position'][i] = position_map
            
            
    #         #Bring normal map to camera coordinates
    #         normal_map = output_dict['normals'][i].view(3, -1)  # Flatten the spatial dimensions
    #         #Add extra channel for homegeneous coordinates
    #         #normal_map = torch.cat((normal_map, torch.ones(1, normal_map.shape[1], dtype=normal_map.dtype, device=normal_map.device)), dim=0)
    #         #normal_map = torch.matmul(world2cam, normal_map)  # Transform to world coordinates
    #         normal_map = torch.matmul(world2cam[:3,:3], normal_map)
    #         #normal_map = normal_map[:3, :]  # Remove the homogeneous coordinate row
    #         normal_map = normal_map.reshape(3, *output_dict['normals'][i].shape[-2:])  # Reshape back to (3, H, W)
    #         normal_map[2, :, :] = -normal_map[2, :, :]  # Invert the Z coordinate to have positive Z pointing outwards
    #         output_dict['normals'][i] = normal_map
        
    #     with record_function("sphere"):
    #         # #Direction to sphere
    #         sphere_center = output_dict['sphere_center'][i]
    #         #sphere_center appended with 1 for homogeneous coordinates
    #         sphere_center = torch.cat((sphere_center, torch.ones(1, dtype=sphere_center.dtype, device=sphere_center.device)), dim=0)
    #         sphere_center = torch.matmul(world2cam, sphere_center.T).T 
    #         sphere_center = sphere_center[:3]  # Remove the homogeneous coordinate row
    #         sphere_center[2] = -sphere_center[2]  # Invert the Z coordinate to have positive Z pointing outwards
                
    #         # Get the vector from the camera to the sphere center
    #         vector_to_sphere = sphere_center[:,None,None] - position_map
        
    #         dist_to_sphere = vector_to_sphere.norm(dim=0, keepdim=True)  # Get the distance to the sphere
    #         dir_to_sphere = vector_to_sphere / (dist_to_sphere + 1e-8)  # Normalize the vector to get the direction to the sphere
            

    #         #Set zeros to the non-zero minimum value
                
    #         #dist_to_sphere[dist_to_sphere == 0] = dist_to_sphere[dist_to_sphere != 0].min().item()
    #         min_val = torch.where(dist_to_sphere != 0, dist_to_sphere, torch.inf).min()
    #         dist_to_sphere = torch.where(dist_to_sphere == 0, min_val, dist_to_sphere)

    #         #Compute the incident ray direction on the sphere
    #         cam_to_point = position_map
    #         cam_to_point_dir = cam_to_point / (cam_to_point.norm(dim=0, keepdim=True) + 1e-8)
    #         incident_ray = -cam_to_point_dir
    #         if bad_incident:
    #             incident_ray = -dir_to_sphere  # Temp for backward compatibility.
    #         dot_product = (torch.sum(incident_ray * normal_map, dim=0, keepdim=True)).clamp(-1, 1)  # Dot product between incident ray and normal
    #         reflection = incident_ray - 2 * dot_product * normal_map  # Reflection direction
    #         reflection = reflection / (reflection.norm(dim=0, keepdim=True) + 1e-8)  # Normalize the reflection vector

    #         dir_to_sphere = torch.where(output_dict['sphere_mask'][i]==1, reflection, dir_to_sphere)
                        
    #         output_dict['dir_to_sphere'][i] = dir_to_sphere
    #         output_dict['dist_to_sphere'][i] = dist_to_sphere
                
                        
    # with record_function("clean nan"):
    #     clean_nan(output_dict)
    # with record_function("manage colors"):
    #     manage_colors(output_dict)


    for key in output_dict.keys():
        if isinstance(output_dict[key], torch.Tensor):
            output_dict[key] = [output_dict[key][i] for i in range(output_dict[key].shape[0])]

    output_dict['ev'] = ev
    # output_dict["prompt"] = TARGET_TO_PROMPT[random_req_layer]+ f" [EV{ev}]"
    output_dict["prompt"] = TARGET_TO_PROMPT[output_dict['target_layer']]+ f" [EV{ev}]"

    return output_dict


def treat_data_old(data, legacy=False, given_ev=None, test=False, bad_incident=False, factor=1.0):
    """
    Does the world2camera transformations
    Computes the dist and dir to sphere
    Normalizes the data
    """
    output_dict = data

    for key in output_dict.keys():
        if isinstance(output_dict[key], torch.Tensor):
            output_dict[key] = [output_dict[key][i] for i in range(output_dict[key].shape[0])]


    output_dict['dir_to_sphere'] = [torch.zeros_like(output_dict['position'][i], device=output_dict['position'][i].device) for i in range(len(output_dict['position']))]
    output_dict['dist_to_sphere'] = [torch.zeros_like(output_dict['position'][i][0:1, :, :], device=output_dict['position'][i].device) for i in range(len(output_dict['position']))]  # Keep the shape (1, H, W) for consistency


    with record_function("find_ev"):
        ev = 0
        if output_dict['random_ev']:
            #Find out what the minimum EV is for the sphere images
            sphere_image = output_dict['video'][0]  # Assuming the first sphere image is representative
            maximum_value = sphere_image.max()
            if maximum_value <= 0:
                minimum_ev = 0
                print(maximum_value)
                print("Maximum value in sphere images is negative, setting minimum EV to 0")
            elif maximum_value == 0:
                minimum_ev = 0
                print("Maximum value in sphere images is 0, setting minimum EV to 0")
            else:
                minimum_ev = - math.log2(2 * maximum_value)
                try:
                    minimum_ev = math.floor(minimum_ev / 3) * 3  # Round to the nearest multiple of 3
                except:
                    print("Error in computing minimum_ev, setting to 0")
                    minimum_ev = 0
            minimum_ev = max(minimum_ev, -12)  # Ensure minimum EV is not too low
            minimum_ev = min(minimum_ev, 0)  # Ensure minimum EV is not positive
            #Sample a random EV in the range [minimum_ev, 0]
            start = minimum_ev + (3 - minimum_ev % 3) if minimum_ev % 3 != 0 else minimum_ev
            # Create list of valid multiples of 3 from start to 0 (inclusive)
            valid_evs = list(range(start, 1, 3))  # step is 3, end at 1 to include 0

            # Sample one
            ev = random.choice(valid_evs)

        if given_ev is not None:
            ev = given_ev
            
    output_dict['image_full'] = []
    for i in range(len(output_dict['video'])):

        if output_dict['mask_img']:
            output_dict['image_full'].append(output_dict['image'][i].clone())
            output_dict['image'][i] = output_dict['image'][i] * (1 - output_dict['sphere_mask'][i])
        
        with record_function("auto_expose"):
         #Auto exposure 
            if output_dict['ev_augmentation']:
                exposure_factor = get_auto_exposure(output_dict['image_full'][i])
                # if test:
                #     exposure_factor = 1
                
                output_dict['image'][i] *= exposure_factor
                output_dict['video'][i] *= exposure_factor
            
            #Apply manual exposure correction
            output_dict['image'][i] *= factor
            output_dict['video'][i] *= factor
            output_dict['image_full'][i] = output_dict['image_full'][i] * factor

            #Apply the EV to the sphere image
            output_dict['video'][i] *= 2 ** ev
        

        

        with record_function("cam_view"):
            world2cam = output_dict['world2cam'][i]
            
            position_map = output_dict['position'][i].view(3, -1)  # Flatten the spatial dimensions

            #Add extra channel for homegeneous coordinates
            position_map = torch.cat((position_map, torch.ones(1, position_map.shape[1], dtype=position_map.dtype, device=position_map.device)), dim=0)  # Add a row of ones for homogeneous coordinates
            position_map = torch.matmul(world2cam, position_map)  # Transform to world coordinates
            position_map = position_map[:3, :]  # Remove the homogeneous coordinate row
            position_map = position_map.reshape(3, *output_dict['position'][i].shape[-2:])  # Reshape back to (3, H, W)

            position_map[2, :, :] = -position_map[2, :, :]  # Invert the Z coordinate to have positive Z pointing outwards
            output_dict['position'][i] = position_map
            
            
            #Bring normal map to camera coordinates
            normal_map = output_dict['normals'][i].view(3, -1)  # Flatten the spatial dimensions
            #Add extra channel for homegeneous coordinates
            #normal_map = torch.cat((normal_map, torch.ones(1, normal_map.shape[1], dtype=normal_map.dtype, device=normal_map.device)), dim=0)
            #normal_map = torch.matmul(world2cam, normal_map)  # Transform to world coordinates
            normal_map = torch.matmul(world2cam[:3,:3], normal_map)
            #normal_map = normal_map[:3, :]  # Remove the homogeneous coordinate row
            normal_map = normal_map.reshape(3, *output_dict['normals'][i].shape[-2:])  # Reshape back to (3, H, W)
            normal_map[2, :, :] = -normal_map[2, :, :]  # Invert the Z coordinate to have positive Z pointing outwards
            output_dict['normals'][i] = normal_map
        
        with record_function("sphere"):
            # #Direction to sphere
            if legacy:
                if output_dict['sphere_mask'][i].nonzero(as_tuple=False).size(0) > 0:
                    sphere_center_pixel = output_dict['sphere_mask'][i].nonzero(as_tuple=False).float().mean(dim=0)
                    sphere_center_pixel = sphere_center_pixel[1:]  # Remove the batch dimension
                    sphere_center_pixel = sphere_center_pixel.long()  # Convert to integer pixel coordinates
                    sphere_center = position_map[:, sphere_center_pixel[0], sphere_center_pixel[1]]  # Get the position of the sphere center in world coordinates
                    position_map_filled_neg = position_map.clone().masked_fill(~(output_dict["sphere_mask"][i]==1), float('-inf'))  # Fill the non-sphere pixels with -inf
                    sphere_max = position_map_filled_neg.amax(dim=(1,2))  # Get the maximum position value for the sphere
                    position_map_filled_pos = position_map.clone().masked_fill(~(output_dict["sphere_mask"][i]==1), float('inf'))  # Fill the non-sphere pixels with -inf
                    sphere_min = position_map_filled_pos.amin(dim=(1,2))
                    sphere_radius = (sphere_max[0]-sphere_min[0]) / 2  # Calculate the radius of the sphere
                    sphere_center[2] += sphere_radius
                    vector_to_sphere = sphere_center[:,None,None] - position_map
            else:

                sphere_center = output_dict['sphere_center'][i]
                #sphere_center appended with 1 for homogeneous coordinates
                sphere_center = torch.cat((sphere_center, torch.ones(1, dtype=sphere_center.dtype, device=sphere_center.device)), dim=0)
                sphere_center = torch.matmul(world2cam, sphere_center.T).T 
                sphere_center = sphere_center[:3]  # Remove the homogeneous coordinate row
                sphere_center[2] = -sphere_center[2]  # Invert the Z coordinate to have positive Z pointing outwards
                
                # Get the vector from the camera to the sphere center
                vector_to_sphere = sphere_center[:,None,None] - position_map
            
            dist_to_sphere = vector_to_sphere.norm(dim=0, keepdim=True)  # Get the distance to the sphere
            dir_to_sphere = vector_to_sphere / (dist_to_sphere + 1e-8)  # Normalize the vector to get the direction to the sphere
            

            #Set zeros to the non-zero minimum value
                
            #dist_to_sphere[dist_to_sphere == 0] = dist_to_sphere[dist_to_sphere != 0].min().item()
            min_val = torch.where(dist_to_sphere != 0, dist_to_sphere, torch.inf).min()
            dist_to_sphere = torch.where(dist_to_sphere == 0, min_val, dist_to_sphere)

            #Compute the incident ray direction on the sphere
            cam_to_point = position_map
            cam_to_point_dir = cam_to_point / (cam_to_point.norm(dim=0, keepdim=True) + 1e-8)
            incident_ray = -cam_to_point_dir
            if bad_incident:
                incident_ray = -dir_to_sphere  # Temp for backward compatibility.
            dot_product = (torch.sum(incident_ray * normal_map, dim=0, keepdim=True)).clamp(-1, 1)  # Dot product between incident ray and normal
            reflection = incident_ray - 2 * dot_product * normal_map  # Reflection direction
            reflection = reflection / (reflection.norm(dim=0, keepdim=True) + 1e-8)  # Normalize the reflection vector

            dir_to_sphere = torch.where(output_dict['sphere_mask'][i]==1, reflection, dir_to_sphere)
                        
            output_dict['dir_to_sphere'][i] = dir_to_sphere
            output_dict['dist_to_sphere'][i] = dist_to_sphere
                
                        
    with record_function("clean nan"):
        clean_nan(output_dict)
    with record_function("manage colors"):
        manage_colors(output_dict)

    output_dict['ev'] = ev
    # output_dict["prompt"] = TARGET_TO_PROMPT[random_req_layer]+ f" [EV{ev}]"
    output_dict["prompt"] = TARGET_TO_PROMPT[output_dict['target_layer']]+ f" [EV{ev}]"

    
    return output_dict
    

class SpheresDataset(Dataset):
    def __init__(self, cfg : SpheresDatasetConfig, target_layers=["sphere_0","sphere_1","sphere_2"], required_layers=None, split='train', num_frames=81, key=255):
        
        self.local_folder = '/dev/shm/s3_cache' # Use /dev/shm to store the cache in memory
        #self.local_folder = '/root/s3_cache'
        self.temp_dir = os.path.join(self.local_folder, "tmp_" + str(uuid.uuid4().hex) + "_" + str(time.time()))
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.local_folder, exist_ok=True)

        self.is_s3 = cfg.data_dir.startswith('s3://') if isinstance(cfg.data_dir, str) else cfg.data_dir[0].startswith('s3://')

        #s3_client = boto3.client('s3', config=botocore.client.Config(max_pool_connections=8))
        self.s3_client = S3(tmproot=self.temp_dir)
        self.s3_bucket = None
    
        #Build the list of scenes
        print("Building list of scenes...")
        self.scenes = []
        scenes_list = []
        if split == 'train' or split == 'val' or split == 'val_test':
            if isinstance(cfg.data_dir, str):
                if self.is_s3:
                    self.s3_bucket = cfg.data_dir.split('/')[2]
                    data_dir = cfg.data_dir.replace('s3://nflx-studio-algo-research-sl-awsprod-us-east-1/nflx-scl-ml', '/fsx_scanline')
                    scene_list = natsorted(glob(os.path.join(data_dir, '*')))
                else:
                    scene_list = natsorted(glob(os.path.join(cfg.data_dir, '*')))
            else: 
                if self.is_s3:
                    self.s3_bucket = cfg.data_dir[0].split('/')[2]
                    for folder in cfg.data_dir:
                        folder = folder.replace('s3://nflx-studio-algo-research-sl-awsprod-us-east-1/nflx-scl-ml', '/fsx_scanline')
                        list_dir = natsorted(glob(os.path.join(folder, '*')))
                        scenes_list.extend(list_dir)
                else:
                    for folder in cfg.data_dir:
                        list_dir = natsorted(glob(os.path.join(folder, '*')))
                        scenes_list.extend(list_dir)
            for scene in tqdm(scenes_list):
                if os.path.exists(os.path.join(scene, 'render_complete')) and "failed" not in os.path.basename(scene[:-1]):
                    lightings = natsorted(glob(os.path.join(scene, 'lighting_*')))
                    for lighting in lightings:
                        cameras = natsorted(glob(os.path.join(lighting, 'camera_*')))
                        for camera in cameras:
                            if os.path.exists(os.path.join(camera, 'frame_0001')):
                                if self.is_s3:
                                    self.scenes.append(camera.replace('/fsx_scanline', 'nflx-scl-ml'))
                                else:
                                    self.scenes.append(camera)
        
        elif split == 'test':
            if self.is_s3:
                self.s3_bucket = cfg.data_dir.split('/')[2]
                self.scenes = [cfg.data_dir.replace('s3://nflx-studio-algo-research-sl-awsprod-us-east-1/nflx-scl-ml', 'nflx-scl-ml')]
            else:
                self.scenes = [cfg.data_dir]
            
        elif split == 'objects':
            print(cfg.data_dir)
            if isinstance(cfg.data_dir, str):
                if self.is_s3:
                    self.s3_bucket = cfg.data_dir.split('/')[2]
                    data_dir = cfg.data_dir.replace('s3://nflx-studio-algo-research-sl-awsprod-us-east-1/nflx-scl-ml', '/fsx_scanline')
                    #scene_list = natsorted(glob(os.path.join(data_dir, '*')))
                    scenes_list = [data_dir]
                else:
                    scenes_list = [cfg.data_dir]
            else: 
                if self.is_s3:
                    self.s3_bucket = cfg.data_dir[0].split('/')[2]
                    for folder in cfg.data_dir:
                        folder = folder.replace('s3://nflx-studio-algo-research-sl-awsprod-us-east-1/nflx-scl-ml', '/fsx_scanline')
                        list_dir = natsorted(glob(os.path.join(folder, '*')))
                        scenes_list.extend(list_dir)
                else:
                    for folder in cfg.data_dir:
                        list_dir = natsorted(glob(os.path.join(folder, '*')))
                        scenes_list.extend(list_dir)
            for scene in tqdm(scenes_list):
                #if os.path.exists(os.path.join(scene, 'render_complete')) and "failed" not in os.path.basename(scene[:-1]):
                lightings = natsorted(glob(os.path.join(scene, 'lighting_*')))
                for lighting in lightings:
                    cameras = natsorted(glob(os.path.join(lighting, 'camera_*')))
                    for camera in cameras:
                        objects = natsorted(glob(os.path.join(camera, 'object_*')))
                        for obj in objects:
                            if self.is_s3:
                                self.scenes.append(obj.replace('/fsx_scanline', 'nflx-scl-ml'))
                            else:
                                self.scenes.append(obj)

        
        print(f"Found {len(self.scenes)} valid scenes.")
        
        #Save the list to file
        if split == 'train':
            with open('train_scenes.txt', 'w') as f:
                for scene in self.scenes:
                    f.write(scene + '\n')

        
        self.augmentation_cfg = cfg.augmentation_cfg
        self.max_resolution = 512*512
        self.random_resolution = cfg.random_resolution
        self.target_layers = target_layers
        
        self.split = split
        self.random_ev = cfg.random_ev
        self.known_ev = False
        self.HDR = cfg.HDR
        self.mask_img = cfg.mask_img
        
        self.bad_indices = []
        self.num_frames = num_frames
        
        #augment current state
        self.h_flip = None
        self.white_balance_temperature = None
        self.do_haze = None
        self.haze_amount = None
        self.noise_a = None
        self.noise_b = None
        self.noise_tensor = None

        self.key = key
        
        self.s3_client.close()
        self.s3_client = None
        shutil.rmtree(self.temp_dir)
        
        self.layer = 'sphere_0'
        
    def s3_listdir_natsorted_boto(self, s3, bucket, scene_prefix, pattern='*', return_full_prefix=True):
        """
        scene_prefix: e.g. 'projects/foo/scene123/'  (NO 's3://bucket', just the key prefix)
        Returns subfolder prefixes matching pattern, naturally sorted.
        """
        if not scene_prefix.endswith('/'):
            scene_prefix += '/'

        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=scene_prefix, Delimiter='/')

        matches = []
        for page in pages:
            for cp in page.get('CommonPrefixes', []):
                subprefix = cp['Prefix']                       # e.g. 'projects/foo/scene123/lighting_0010/'
                name = subprefix[len(scene_prefix):].rstrip('/')  # 'lighting_0010'
                if fnmatch.fnmatch(name, pattern):
                    matches.append(subprefix if return_full_prefix else name)

        return natsorted(matches)
    
    def s3_listdir_natsorted(self, s3, bucket, scene_prefix, pattern='*', return_full_prefix=True):
        """
        List *subfolder prefixes* directly under `scene_prefix` (no 's3://bucket', just the key prefix),
        filter with a shell-style pattern, and return them in natural order.
        """
        if not scene_prefix.endswith('/'):
            scene_prefix += '/'

        # S3 root is the bucket; we list immediate children under `scene_prefix`.
        matches = []
        # list_paths returns the *next level* of prefixes/objects under a prefix
        for obj in s3.list_paths(["s3://"+bucket+"/"+scene_prefix]):
            # A "prefix" (i.e., subfolder) has exists == False
            if not obj.exists:
                name = obj.key.rstrip('/')             # e.g., 'lighting_0010'
                if fnmatch.fnmatch(name, pattern):
                    matches.append(scene_prefix + name + '/' if return_full_prefix else name)

        return natsorted(matches)
        
    def is_valid_scene_boto(self, element, s3_client=None):
        render_complete_key = element
        try:
            if s3_client is None:
                s3_client = self.s3_client
            s3_client.head_object(Bucket=self.s3_bucket, Key=render_complete_key)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                # The key does not exist.
                return False
        return True
    
    def is_valid_scene(self, element):
        """
        Return True if `s3://{bucket}/{key}` exists (object present), else False.
        """
        #return_missing=True avoids raising if the key is absent
        obj = self.s3_client.get("s3://"+self.s3_bucket+"/"+element, return_missing=True)
        return bool(obj.exists)
        # key = element.rstrip("/")
        # results = list(self.s3_client.list_paths(["s3://"+self.s3_bucket+"/"+key]))
        # return any(obj.exists for obj in results if obj.key == key)
    
    def download_s3_file(self, s3_client, bucket, s3_path, local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        s3obj = s3_client.get("s3://"+bucket+"/"+s3_path)

        if s3obj is None:
            return False
        
        with open(local_path, 'wb') as f:
            f.write(s3obj.blob)
        
        return True

    def download_s3_files(self, s3_client, bucket, s3_paths, local_paths):

        urls = ["s3://"+bucket+"/"+s3_path for s3_path in s3_paths]

        s3objs = s3_client.get_many(urls)

        if s3objs is None:
            return False
        
        for local_path, s3obj in zip(local_paths, s3objs):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'wb') as f:
                f.write(s3obj.blob)
        
        return True

    
    def _get_pyexr_from_s3_download(self, s3_path):
        tmp_path = os.path.join(self.temp_dir,s3_path)
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        if self.verbose:
            print("ID",self.idx," - ","Starting: download_s3_file at ", time.time()-self.start_time)
        self.download_s3_file(self.s3_client, self.s3_bucket,s3_path, tmp_path)
        if self.verbose:
            print("ID",self.idx," - ","Done: download_s3_file at ", time.time()-self.start_time)
        return PyexrInputFile(OpenEXR.InputFile(tmp_path))

    def _get_pyexr_from_s3(self, s3_path, max_retries=8):
        attempt = 0
        while attempt < max_retries:
            try:
                return self._get_pyexr_from_s3_download(s3_path)
            except botocore.exceptions.ClientError as e:
                if e.response['Error']['Code'] == "404":
                    raise FileNotFoundError(f"The object {s3_path} does not exist.")
                else:
                    self._handle_s3_error(e, attempt, s3_path)
                    attempt += 1
            except Exception as e:
                self._handle_s3_error(e, attempt, s3_path)
                attempt += 1
        raise Exception(f"Failed to download {s3_path} after {max_retries} attempts.")


    
    def _get_tensor_from_pyexrfile(self, pyexr_file, get3channels=False):
        if self.verbose:
            print("ID",self.idx," - ","Starting: get_tensor_from_pyexrfile at ", time.time()-self.start_time)
        tensor = torch.from_numpy(pyexr_file.get(precision=pyexr_file.precisions[0])).permute(2, 0, 1)
        if self.verbose:
            print("ID",self.idx," - ","Done: get_tensor_from_pyexrfile at ", time.time()-self.start_time)
        if get3channels:
            if 'R' in pyexr_file.channels:
                tensor = tensor[0:3]
            elif 'X' in pyexr_file.channels:
                tensor = tensor[-3:]
            else:
                raise Exception("No 3 channel data found")
        return tensor

    def get_tensor(self, camera_path_s3, prefix, get3channels=False):
        exr_file_path_s3 = camera_path_s3 + f'{prefix}_0001.exr'
        pyexr_file = self._get_pyexr_from_s3(exr_file_path_s3)
        return self._get_tensor_from_pyexrfile(pyexr_file, get3channels)
        
    
    def __len__(self):
        return len(self.scenes)

    def _clean_nan(self, data):
        for key in data.keys():
            torch.nan_to_num_(data[key])

    def _get_random_resolution(self,batch_random_int):
        max_resolution = self.max_resolution
        if batch_random_int == 0:
            ratio_h_over_w = 1
        else:
            rng = np.random.default_rng(seed=batch_random_int)
            ratio_h_over_w = 2**rng.uniform(-1,1)
        w_float = math.sqrt(max_resolution/ratio_h_over_w)
        h_float = w_float*ratio_h_over_w
        #Round to the nearest multiple of 16
        w = int(round(w_float/16)*16)
        h = int(round(h_float/16)*16)
        #Check that the resolution does not exceed the maximum resolution
        while w*h > max_resolution:
            if h/w > ratio_h_over_w:
                h -= 16
            else:
                w -= 16
        return [h,w]

    def _resize_data(self, data, batch_random_int):
        #Pick resolution randomly
        if self.random_resolution:
            resolution_hw = self._get_random_resolution(batch_random_int)
        else:
            #resolution_hw = [1024, 1024]  # Default resolution if not random
            resolution_hw = [512, 512]  # Default resolution if not random
            #resolution_hw = [1536, 1536]
            max_resolution = max(data['image'].shape[-2], data['image'].shape[-1])
            factor = 512 / min(data['image'].shape[-2], data['image'].shape[-1])
            if factor != 1:
                resolution_hw = [int(data['image'].shape[-2]*factor), int(data['image'].shape[-1]*factor)]
        h,w=resolution_hw
        max_resolution = max(resolution_hw)

        for key in data.keys(): 
            max_dim = data[key].shape[-2] if max_resolution == h else data[key].shape[-1]
            factor = max_resolution / max_dim
            #if factor != 1:
            #    data[key] = torch.nn.functional.interpolate(data[key], scale_factor=factor, mode='area')
            # Center crop to match the resolution
            #data[key] = data[key][..., (data[key].shape[-2]-h)//2:(data[key].shape[-2]+h)//2, (data[key].shape[-1]-w)//2:(data[key].shape[-1]+w)//2]
            data[key] = torch.nn.functional.interpolate(data[key].unsqueeze(0), size=resolution_hw, mode='area').squeeze(0)  # Resize to the target resolution

    def _resize_single_data(self, data, batch_random_int, nearest=False):
        #Pick resolution randomly
        if self.random_resolution:
            resolution_hw = self._get_random_resolution(batch_random_int)
        else:
            #resolution_hw = [1024, 1024]  # Default resolution if not random
            resolution_hw = [512, 512]  # Default resolution if not random
            #resolution_hw = [1536, 1536]
            max_resolution = max(data.shape[-2], data.shape[-1])
            factor = 512 / min(data.shape[-2], data.shape[-1])
            if factor != 1:
                resolution_hw = [int(data.shape[-2]*factor), int(data.shape[-1]*factor)]

        h,w=resolution_hw
        max_resolution = max(resolution_hw)

        max_dim = data.shape[-2] if max_resolution == h else data.shape[-1]
        factor = max_resolution / max_dim
        #if factor != 1:
        #    data[key] = torch.nn.functional.interpolate(data[key], scale_factor=factor, mode='area')
        # Center crop to match the resolution
        #data[key] = data[key][..., (data[key].shape[-2]-h)//2:(data[key].shape[-2]+h)//2, (data[key].shape[-1]-w)//2:(data[key].shape[-1]+w)//2]
        if nearest:
            data_resized = torch.nn.functional.interpolate(data.unsqueeze(0), size=resolution_hw, mode='nearest').squeeze(0)  # Resize to the target resolution
        else:
            data_resized = torch.nn.functional.interpolate(data.unsqueeze(0), size=resolution_hw, mode='area').squeeze(0)  # Resize to the target resolution
        return data_resized

    def _augment_data(self, data, cfg: SpheresAugConfig):
        if cfg is None:
            return
        
        if self.h_flip is None:
            h_flip = cfg.horizontal_flip and random.random() > 0.5
            self.h_flip = h_flip

        if self.white_balance_temperature is None:
            white_balance_temperature = random.gauss(6600, 800)
            self.white_balance_temperature = white_balance_temperature
            
        if self.do_haze is None:
            do_haze = random.random() > 0.5
            self.do_haze = do_haze
        
        if self.haze_amount is None:
            self.haze_amount = np.abs(random.gauss(0.0, 0.004))
            
        if self.noise_a is None:
            self.noise_a = 10**random.uniform(3,5)
            
        if self.noise_b is None:
            self.noise_b = 10**random.uniform(-4,-2.3)
            
        if self.noise_tensor is None:
            self.noise_tensor = torch.randn_like(data["image"])
            
        for key in data.keys():
            if self.h_flip:
                data[key] = torch.flip(data[key], [-1])

            if cfg.haze:
                if key == 'image':
                    #Generate a haze map and add it to the image
                    if self.do_haze: #Sample haze from normal distribution
                        haze = min(0.1,self.haze_amount) 
                        data[key] += haze        
            if cfg.noise:
                if key == 'image':
                    #Noise is derived from https://arxiv.org/pdf/2210.04866
                    # a is picked in [100,10000] uniformly in log space and controls the poisson noise
                    # b is picked in [0.0001,0.005] uniformly in log space and controls the gaussian noise. b is the standard deviation
                    
                    original_dtype = data[key].dtype #Converting to float32 to avoid overflow when multiplying by a
                    data[key] = (torch.poisson(data[key].clamp(1e-3).to(dtype=torch.float32)*self.noise_a)/self.noise_a + self.noise_tensor*self.noise_b).to(dtype=original_dtype)
            
            if cfg.white_balance:
                if key in ['image', 'sphere_0', 'sphere_1', 'sphere_2']:
                    data[key] = white_balance(data[key], self.white_balance_temperature)
        
    def _manage_colors(self, data):
        for key in data.keys():
            if key in ['image', 'sphere_0', 'sphere_1', 'sphere_2', 'image_full']:
                if self.HDR:
                    data[key] = Lin_to_Reinhard(data[key], power=1/2)  # Convert to Reinhard scale
                    data[key].mul_(2).sub_(1) 
                else:
                    torch.clamp_(data[key], 1e-4, 1.0)
                    data[key] = Lin_to_sRGB(data[key])
                    data[key].mul_(2).sub_(1) 
            elif 'sphere_mask' in key:
                data[key] = torch.clamp_(data[key], 1e-4, 1.0).repeat(3, 1, 1)
                data[key].mul_(2).sub_(1) 
            elif 'position' in key:
                data[key] = position_to_scaled_position(data[key])
                data[key].mul_(2).sub_(1) 
                #data[key] = torch.zeros_like(data[key])
            elif 'dist_to_sphere' in key:
                data[key] = depth_to_scaled_depth(data[key], torch.ones_like(data[key])).repeat(3, 1, 1)  # Repeat the depth channel to match RGB
                data[key] = torch.nan_to_num(data[key])
                data[key].mul_(2).sub_(1)
            elif key in ['normals', 'dir_to_sphere']:
                data[key] = data[key].clamp(-1, 1)
                

    @staticmethod
    def get_gpu_postprocessing_args(**kwargs):
        return {}

    @staticmethod
    def gpu_batch_postprocess(batch,**kwargs):
        pass
    
    def load_frame(self, idx, path, path_local, sphere_id, req_layer, batch_random_int, frame, s3_client_frame):
        
        output_dict = {}
        
        def download_and_load(s3_client, s3_bucket, s3_path, local_path):
            # t0 = time.time()
            # with record_function("loading_exr"):
            if self.is_s3:
                self.download_s3_file(s3_client, s3_bucket, s3_path, local_path)
            else:
                local_path = s3_path
            img = pyexr.read(local_path)
            # t1 = time.time()
            # print("load time: ", t1-t0)
            img = torch.from_numpy(img).float().permute(2, 0, 1)
            img = self._resize_single_data(img, batch_random_int,nearest=('mask' in s3_path))
            return img
        
        def download_and_load_cam(s3_client, s3_bucket, s3_path, local_path):
            if self.is_s3:
                self.download_s3_file(s3_client, s3_bucket, s3_path, local_path)
            else:
                local_path = s3_path
            with open(local_path, 'rb') as f:
                camera_info = pickle.load(f)
            return camera_info

        # with record_function("downloading_exr"):
        #thread the downloads of the files
        with ThreadPoolExecutor(max_workers=7) as download_executor:
            
            sphere_path = os.path.join(path, req_layer)
            sphere_path_local = os.path.join(path_local, req_layer)
            
            mask_future = download_executor.submit(
                download_and_load, s3_client_frame, self.s3_bucket,
                os.path.join(sphere_path, f'mask_{frame:04d}.exr'),
                os.path.join(sphere_path_local, f'mask_{frame:04d}.exr')
            )
            
            image_future = download_executor.submit(
                download_and_load, s3_client_frame, self.s3_bucket,
                os.path.join(path, f'image_{frame:04d}.exr'),
                os.path.join(path_local, f'image_{frame:04d}.exr')
            )
            
            sphere_image_future = download_executor.submit(
                download_and_load, s3_client_frame, self.s3_bucket,
                os.path.join(sphere_path, f'image_{frame:04d}.exr'),
                os.path.join(sphere_path_local, f'image_{frame:04d}.exr')
            )
            
            sphere_position_future = download_executor.submit(
                download_and_load, s3_client_frame, self.s3_bucket,
                os.path.join(sphere_path, f'position_{frame:04d}.exr'),
                os.path.join(sphere_path_local, f'position_{frame:04d}.exr')
            )
            
            image_position_future = download_executor.submit(
                download_and_load, s3_client_frame, self.s3_bucket,
                os.path.join(path, f'position_{frame:04d}.exr'),
                os.path.join(path_local, f'position_{frame:04d}.exr')
            )
            
            sphere_normal_future = download_executor.submit(
                download_and_load, s3_client_frame, self.s3_bucket,
                os.path.join(sphere_path, f'normal_{frame:04d}.exr'),
                os.path.join(sphere_path_local, f'normal_{frame:04d}.exr')
            )

            camera_info_path = os.path.join(path, 'camera_info.pkl')
            camera_info_path_local = os.path.join(path_local, 'camera_info.pkl')
            camera_info_future = download_executor.submit(
                download_and_load_cam, s3_client_frame, self.s3_bucket,
                camera_info_path, camera_info_path_local
            )

            # Wait for all downloads to complete and get results
            mask = mask_future.result()
            image = image_future.result()
            sphere_image = sphere_image_future.result()
            sphere_position_map = sphere_position_future.result()
            image_position_map = image_position_future.result()
            sphere_normal_map = sphere_normal_future.result()
            camera_info = camera_info_future.result()

        sphere_path = os.path.join(path, req_layer)
        sphere_path_local = os.path.join(path_local, req_layer)
        
        mask = mask[0,:,:]
    
        output_dict['sphere_mask'] = (mask == sphere_id).float().unsqueeze(0)  # Shape: (1, H, W)
        
        output_dict['image'] = image[:3, :, :]  # Keep only RGB channels
        #output_dict['image_full'] = image[:3, :, :].clone()
        
            
        sphere_image = sphere_image[:3, :, :]

        # sphere_image = sphere_image * output_dict['sphere_mask'] + output_dict['image'] * (torch.ones_like(output_dict['sphere_mask']) - output_dict['sphere_mask'])
        output_dict['sphere'] = sphere_image
        

        sphere_position_map = sphere_position_map[:3,:,:]
        image_position_map = image_position_map[:3,:,:]
        position_map = sphere_position_map * output_dict['sphere_mask'] + image_position_map * (1 - output_dict['sphere_mask'])
        output_dict['position'] = position_map[:3, :, :]
        
        sphere_normal_map = sphere_normal_map[:3, :, :] * output_dict['sphere_mask']
        output_dict['normals'] = sphere_normal_map
        
        
        cam2world = torch.from_numpy(camera_info['cam2world']).float()
        world2cam = torch.inverse(cam2world)
            
        self._clean_nan(output_dict)

        if self.split == 'train':
            self._augment_data(output_dict, self.augmentation_cfg)

        
        output_dict['world2cam'] = world2cam
        output_dict['target_layer'] = req_layer 
        
        output_dict["scene"] = self.scenes[idx]
        
        
        return output_dict

    def __getitem__(self, item):
        with record_function("getitem total"):
            #augment current state
            self.h_flip = None
            self.white_balance_temperature = None
            self.do_haze = None
            self.haze_amount = None
            self.noise_a = None
            self.noise_b = None
            self.noise_tensor = None
            self.known_ev = False
            
            #if item is a tuple
            if isinstance(item, tuple):
                idx, batch_random_int, self.max_resolution = item
            else:
                idx = item
                batch_random_int = random.randint(0, 2**32-1)
                
            scene_path = self.scenes[idx]
            
            
            self.temp_dir = os.path.join(self.local_folder, "tmp_" + str(hash(scene_path)) + "_" + str(time.time()))
            os.makedirs(self.temp_dir, exist_ok=True)
            
            if self.s3_client is None and self.is_s3:
                self.s3_client = S3(tmproot=self.temp_dir)
                
            # with record_function("find sphere"):
            #Image without spheres logic
            while True:
                if idx in self.bad_indices:
                    idx = random.randint(0, len(self.scenes)-1)
                    continue
                camera_path = self.scenes[idx]
                
                self.idx = idx
                
                #Check if static or multi frames logic
                frames = False
                # with record_function("is valid"):
                if os.path.exists(os.path.join(self.scenes[idx].replace('nflx-scl-ml', '/fsx_scanline'), 'frame_0001', 'camera_info.pkl')):
                    frames = True
                # if self.is_valid_scene(os.path.join(self.scenes[idx], 'frame_0001', 'camera_info.pkl')):
                #     frames = True
                    
                
                spheres_path = camera_path
                if frames:
                    camera_info_path = os.path.join(camera_path, 'frame_0001', 'camera_info.pkl')
                else:
                    camera_info_path = os.path.join(camera_path, 'camera_info.pkl')
                    if not os.path.exists(camera_info_path):
                        print(camera_info_path, " does not exist, trying to find it in parent directory")
                        camera_path = os.path.dirname(camera_path)
                        print(camera_path)
                        camera_info_path = os.path.join(camera_path, 'camera_info.pkl')
                
                camera_path_local = os.path.join(self.temp_dir,"camera_path")
                camera_info_path_local = os.path.join(camera_path_local, 'camera_info.pkl')
                            
                #Load mask and select random sphere
                if frames:
                    sphere_0_path = os.path.join(spheres_path, 'frame_0001', 'sphere_0')
                    sphere_0_path_local = os.path.join(camera_path_local, 'frame_0001', 'sphere_0')
                else:
                    sphere_0_path = os.path.join(spheres_path, 'sphere_0')
                    sphere_0_path_local = os.path.join(camera_path_local, 'sphere_0')
                
                # with record_function("s3 download"):
                if self.is_s3:
                    self.download_s3_file(self.s3_client, self.s3_bucket,os.path.join(sphere_0_path, 'mask_0001.exr'), os.path.join(sphere_0_path_local, 'mask_0001.exr'))
                else:
                    sphere_0_path_local = sphere_0_path
                # with record_function("pyexr load"):
                mask = pyexr.read(os.path.join(sphere_0_path_local, 'mask_0001.exr'))
                mask = np.round(mask).astype(np.int32)[:,:,0]
                id_spheres = np.unique(mask)[1:] # Exclude the background   
                if len(id_spheres) > 0:
                    break
                else:
                    print("No spheres found in scene ", camera_path)
                    self.bad_indices.append(idx)
                    idx = random.randint(0, len(self.scenes)-1)
            
            random_req_layer = random.choice(self.target_layers)
            sphere_id = random.choice(id_spheres)
            if self.split == 'test' or self.split == 'objects' or self.split == 'val_test':
                #For test set, we use a fixed sphere id
                sphere_id = self.key
                random_req_layer = self.layer

            
            #List number of frames
            if frames:
                num_frames = len(natsorted(glob(os.path.join(camera_path.replace('nflx-scl-ml', '/fsx_scanline'), 'frame_*'))))
            else:
                num_frames = 1
                
            final_output = {"video": [], "prompt": None, "target_layer": random_req_layer, "ev": None}
            #output_dicts = [{} for _ in range(num_frames)]

            # s3_clients = []
            # with ThreadPoolExecutor(max_workers=6) as executor:
            #     for i in range(6):
            #         s3_clients.append(S3(tmproot=self.temp_dir))
            #         #Dummy download to initialize the client
            #         executor.submit(self.download_s3_file, s3_clients[i], self.s3_bucket,os.path.join(sphere_0_path, 'mask_0001.exr'), os.path.join(sphere_0_path_local, 'mask_0001.exr'))

            threads_data = {}
            with ThreadPoolExecutor(max_workers=6) as executor:
                for frame in range(1, num_frames+1):
                    if frames:
                        frame_path = os.path.join(camera_path, f'frame_{frame:04d}')
                        frame_path_local = os.path.join(camera_path_local, f'frame_{frame:04d}')
                    else:
                        frame_path = camera_path
                        frame_path_local = camera_path_local
                    
                    if frame > self.num_frames:
                        break
                    
                    #Load the frame
                    threads_data[frame] = executor.submit(self.load_frame, idx, frame_path, frame_path_local, sphere_id, random_req_layer, batch_random_int, frame, self.s3_client)
                    #output_dict = self.load_frame(idx, frame_path, frame_path_local, sphere_id, random_req_layer, batch_random_int, frame)
                    #output_dicts[frame-1] = output_dict
                
            used_frames = 0
            for frame in range(1, num_frames+1):
                if frame > self.num_frames:
                    break
                output_dict = threads_data[frame].result()

                # ###### TEMP
                # if frames:
                #     frame_path = os.path.join(camera_path, f'frame_{frame:04d}')
                #     frame_path_local = os.path.join(camera_path_local, f'frame_{frame:04d}')
                # else:
                #     frame_path = camera_path
                #     frame_path_local = camera_path_local

                # output_dict = self.load_frame(idx, frame_path, frame_path_local, sphere_id, random_req_layer, batch_random_int, frame)
                ########


                # with record_function("append stuff"):
                final_output["video"].append(output_dict["sphere"])
                #final_output["prompt"] = TARGET_TO_PROMPT[random_req_layer]+ f" [EV{self.known_ev}]"
                #final_output["ev"] = output_dicts[i-1]['ev']
                for key in output_dict.keys():
                    if key not in ["video", "prompt", "target_layer", "ev", "sphere"]:
                        if key not in final_output:
                            final_output[key] = []
                        final_output[key].append(output_dict[key])
                    
                used_frames += 1
                #Check if the mask is empty to deal with spheres getting out of the frame
                sphere_mask_rectified = output_dict['sphere_mask']# * 0.5 + 0.5
                if sphere_mask_rectified.sum() < 10:
                    break
                    
            # with record_function("load sphere centers"):
            sphere_infos = []
            def load_sphere_info(camera_path, frame):
                if frames:
                    frame_path = os.path.join(camera_path, f'frame_{frame:04d}')
                    frame_path_local = os.path.join(camera_path_local, f'frame_{frame:04d}')
                else:
                    frame_path = camera_path
                    frame_path_local = camera_path_local
                if "sphere_center" not in final_output:
                    final_output["sphere_center"] = []
                sphere_path = os.path.join(frame_path, random_req_layer)
                sphere_path_local = os.path.join(frame_path_local, random_req_layer)
                spheres_info_path = os.path.join(sphere_path, f'spheres_info.pkl')
                spheres_info_path_local = os.path.join(sphere_path_local, f'spheres_info.pkl')
                if self.is_s3:
                    valid = self.download_s3_file(self.s3_client, self.s3_bucket,spheres_info_path, spheres_info_path_local)
                else:
                    spheres_info_path_local = spheres_info_path
                with open(spheres_info_path_local, 'rb') as f:
                    sphere_info = pickle.load(f)
                return sphere_info
                
            # threads_data = {}
            with ThreadPoolExecutor(max_workers=6) as executor:
                for frame in range(1, num_frames+1):
                    threads_data[frame] = executor.submit(load_sphere_info, camera_path, frame)
                for frame in range(1, num_frames+1):
                    if frame > self.num_frames or frame > used_frames:
                        break
                    sphere_info = threads_data[frame].result()
                    try:
                        final_output['sphere_center'].append(torch.tensor(sphere_info[sphere_id]['center_3D']).float())
                    except:
                        print(sphere_id)
                        print(sphere_info)
                        print(id_spheres)
                        print(os.path.join(camera_path, f'frame_{frame:04d}'))

            #repeat if not enough frames
            if len(final_output["video"]) < self.num_frames:
                for key in final_output.keys():
                    if key not in ["target_layer", "ev", "prompt"]:
                        mirror_seq = final_output[key] + final_output[key][-2:0:-1]  # forward + reversed (without duplicating ends)
                        final_output[key] = (mirror_seq * ((self.num_frames // len(mirror_seq)) + 1))[:self.num_frames]
                        #final_output[key] = (final_output[key] * ((self.num_frames // len(final_output[key])) + 1))[:self.num_frames]
            
            if self.augmentation_cfg.reverse and random.random() > 0.5:
                #Reverse the video sequence
                final_output["video"].reverse()
                for key in final_output.keys():
                    if key not in ["video", "prompt", "target_layer", "ev"]:
                        final_output[key].reverse()
            
            for key in final_output.keys():
                if isinstance(final_output[key], list):
                    if isinstance(final_output[key][0], torch.Tensor):
                        final_output[key] = torch.stack(final_output[key])

            final_output["path"] = camera_path
            
            #Store info on EV augmentation
            final_output["ev_augmentation"] = self.augmentation_cfg.auto_exposure
            final_output["random_ev"] = self.random_ev
            final_output["mask_img"] = self.mask_img
        
            if self.is_s3:
                self.s3_client.close()
                self.s3_client = None

            # for client in s3_clients:
            #     client.close()
            
            shutil.rmtree(self.temp_dir)

        return final_output

def collate_fn(batch):
    return batch[0]

if __name__ == '__main__':
    # targets = "sphere_0,sphere_1,sphere_2"
    # cfg = SpheresDatasetConfig(
    #     data_dir = "/root/Data/test_sets/synthetic/classroom",
    #     augmentation_cfg=SpheresAugConfig(
    #         horizontal_flip = False,
    #         noise=False,
    #         vignetting=False,
    #         auto_exposure=False,
    #         haze=False,
    #         white_balance=False
    #     ),
    # )
    
    # dataset = SpheresDataset(cfg=cfg, split='objects', target_layers=targets.split(','))
    
    # import sys
    # sys.exit(0)
    
    config = SpheresDatasetConfig(
                    #data_dir = "s3://nflx-studio-algo-research-sl-awsprod-us-east-1/nflx-scl-ml/from_eyeline/users/jphilip/synthetic_data/outdoor_10k_20241211/",
                    data_dir = ["s3://nflx-studio-algo-research-sl-awsprod-us-east-1/nflx-scl-ml/from_eyeline/users/cbolduc/synthetic_spheres/spheres_bk_indoor_anim_lighting_16sep"],
                                # "s3://nflx-studio-algo-research-sl-awsprod-us-east-1/nflx-scl-ml/from_eyeline/users/cbolduc/synthetic_spheres/spheres_bk_indoor_anim_camera_16sep",
                                # "s3://nflx-studio-algo-research-sl-awsprod-us-east-1/nflx-scl-ml/from_eyeline/users/cbolduc/synthetic_spheres/spheres_bk_indoor_anim_spheres_16sep"],
                    augmentation_cfg=SpheresAugConfig(
                        horizontal_flip = True,
                        noise=True,
                        vignetting=True,
                        auto_exposure=True,
                        haze=True,
                        white_balance=True
                    ),
                    random_ev = True,
                )
    dataset = SpheresDataset(config, split='train', target_layers=['sphere_0', 'sphere_1'], num_frames=21)

    # Use DataLoader to iterate over the dataset



    dataloader = DataLoader(dataset, batch_size=1, num_workers=16, shuffle=False, collate_fn=collate_fn, pin_memory=True)
    name_report = "dataset_trace_"+str(time.time())
    output_dir = "/root/Project/DiffSynth-Studio"
    os.makedirs(os.path.join(output_dir,"logs"), exist_ok=True)

    profile = True
    wait_iter=0


    if profile:
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
            ],
            schedule=torch.profiler.schedule(wait=wait_iter, warmup=2, active=5, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(output_dir,"logs",name_report)),
            record_shapes=True,
            profile_memory=True,
            with_modules=True,
            #record_shapes=True,
            #with_stack=True
        )
        prof.start()

    for it in range(3):
        count = 0
        for batch in tqdm(dataset):
            with record_function("treat_data"):
                #To cuda
                for key in batch.keys():
                    #Check if type list
                    if type(batch[key]) == list:
                        for i, element in enumerate(batch[key]):
                            if type(element) == torch.Tensor:
                                batch[key][i] = element.cuda()
                batch = treat_data(batch)
            #save conditions
            # from ezexr import imsave
            # dir_to_sphere = batch['dir_to_sphere'][0].permute(1,2,0).numpy() * 0.5 + 0.5
            # imsave("dir_to_sphere.exr", dir_to_sphere)
            # normals = batch['normals'][0].permute(1,2,0).numpy() * 0.5 + 0.5
            # imsave("normals.exr", normals)
            # raise('ok')

            if profile:
                prof.step()
            count += 1
            if count > 50:
                break
        break
    if profile:
        prof.stop()

