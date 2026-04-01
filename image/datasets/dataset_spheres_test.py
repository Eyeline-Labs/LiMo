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
from datasets.data_utils import download_s3_file, list_s3_files, get_auto_exposure, white_balance, Lin_to_sRGB, Lin_to_Log_, depth_to_scaled_depth, position_to_scaled_position, Lin_to_Reinhard
#from data_utils import download_s3_file, list_s3_files, get_auto_exposure, white_balance, Lin_to_sRGB, Lin_to_Log_, depth_to_scaled_depth, position_to_scaled_position

import boto3
import botocore

from glob import glob
from natsort import natsorted

#### Dataset ####
@dataclass
class SpheresAugConfig:
    horizontal_flip: bool = False
    vignetting: bool = False
    auto_exposure: bool = False
    noise: bool = False
    haze: bool = False
    white_balance: bool = False

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

# TARGET_TO_PROMPT = {
#     "sphere_0":"Metallic sphere",
#     "sphere_1":"half diffuse, metallic sphere",
#     "sphere_2":"Diffuse sphere"
# }
TARGET_TO_PROMPT = {
    "sphere_0":"Metallic sphere",
    #"sphere_1":"matte sphere",
    "sphere_1":"Diffuse sphere"
}

    
class SpheresDataset(Dataset):
    def __init__(self, cfg : SpheresDatasetConfig, target_layers=["sphere_0","sphere_1","sphere_2"], required_layers=None, split='train'):
        
        #Build the list of scenes
        print("Building list of scenes...")
        self.scenes = []
        self.frames = False
        scenes_list = []
        if split == 'train':
            if isinstance(cfg.data_dir, str):
                scenes_list = natsorted(glob(os.path.join(cfg.data_dir, '*')))
            else: 
                for folder in cfg.data_dir:
                    scenes_list.extend(natsorted(glob(os.path.join(folder, '*'))))
            for scene in tqdm(scenes_list):
                if not os.path.isdir(scene):
                    continue
                if os.path.exists(os.path.join(scene, 'finished')) and "failed" not in os.path.basename(scene):
                    lightings = natsorted(glob(os.path.join(scene, 'lighting_*')))
                    for lighting in lightings:
                        cameras = natsorted(glob(os.path.join(lighting, 'camera_*')))
                        for camera in cameras:
                            self.scenes.append(camera)
                            
        elif split == 'test':
            self.scenes = [cfg.data_dir]
            
        elif split == 'objects':
            if isinstance(cfg.data_dir, str):
                scenes_list = natsorted(glob(os.path.join(cfg.data_dir, 'lighting*')))
            else: 
                for folder in cfg.data_dir:
                    scenes_list.extend(natsorted(glob(os.path.join(folder, 'lighting*'))))
            for scene in tqdm(scenes_list):
                print(scene)
                if not os.path.isdir(scene):
                    continue
                objects = natsorted(glob(os.path.join(scene, '**/object_*'), recursive=True))
                for obj in objects:
                    print(obj)
                    frames = natsorted(glob(os.path.join(obj, "frame_*")))
                    if len(frames) == 0:
                        self.scenes.append(obj)
                    else:
                        self.frames = True
                        for frame in frames:
                            print(frame)
                            self.scenes.append(frame)
                    
        elif split == 'val':
            if cfg.val_data_dir is None:
                self.scenes = []
            else:
                if isinstance(cfg.val_data_dir, str):
                    scenes_list = natsorted(glob(os.path.join(cfg.val_data_dir, '*')))
                else: 
                    for folder in cfg.val_data_dir:
                        scenes_list.extend(natsorted(glob(os.path.join(folder, '*'))))
                for scene in tqdm(scenes_list):
                    if not os.path.isdir(scene):
                        continue
                    if os.path.exists(os.path.join(scene, 'finished')) and "failed" not in os.path.basename(scene):
                        lightings = natsorted(glob(os.path.join(scene, 'lighting_*')))
                        for lighting in lightings:
                            cameras = natsorted(glob(os.path.join(lighting, 'camera_*')))
                            for camera in cameras:
                                self.scenes.append(camera)

        
        print(f"Found {len(self.scenes)} valid scenes.")
        
        #Save the list to file
        if split == 'train':
            with open('train_scenes.txt', 'w') as f:
                for scene in self.scenes:
                    f.write(scene + '\n')

        
        self.augmentation_cfg = cfg.augmentation_cfg
        self.max_resolution = 1024*1024
        self.random_resolution = cfg.random_resolution
        self.target_layers = target_layers
        
        self.split = split
        self.random_ev = cfg.random_ev
        self.known_ev = False
        self.HDR = cfg.HDR
        self.mask_img = cfg.mask_img
        
        self.bad_indices = []
        
    
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
            #resize smallest size to 512
            # max_resolution = max(data['image'].shape[-2], data['image'].shape[-1])
            # factor = 512 / min(data['image'].shape[-2], data['image'].shape[-1])
            # if factor != 1:
            #     resolution_hw = [int(data['image'].shape[-2]*factor), int(data['image'].shape[-1]*factor)]
            # print(resolution_hw)
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

    def _augment_data(self, data, cfg: SpheresAugConfig):
        if cfg is None:
            return
        
        h_flip = cfg.horizontal_flip and random.random() > 0.5

        white_balance_temperature = random.gauss(6600, 800)
        for key in data.keys():
            if h_flip:
                data[key] = torch.flip(data[key], [-1])

            if cfg.haze:
                if key == 'image':
                    #Generate a haze map and add it to the image
                    if random.random() > 0.5: #Sample haze from normal distribution
                        haze = min(0.1,np.abs(random.gauss(0.0, 0.004))) 
                        data[key] += haze        
            if cfg.noise:
                if key == 'image':
                    #Noise is derived from https://arxiv.org/pdf/2210.04866
                    # a is picked in [100,10000] uniformly in log space and controls the poisson noise
                    a = 10**random.uniform(3,5)
                    # b is picked in [0.0001,0.005] uniformly in log space and controls the gaussian noise. b is the standard deviation
                    b = 10**random.uniform(-4,-2.3)
                    original_dtype = data[key].dtype #Converting to float32 to avoid overflow when multiplying by a
                    data[key] = (torch.poisson(data[key].clamp(1e-3).to(dtype=torch.float32)*a)/a + torch.randn_like(data[key])*b).to(dtype=original_dtype)
            
            if cfg.white_balance:
                if key in ['image', 'sphere_0', 'sphere_1', 'sphere_2']:
                    data[key] = white_balance(data[key], white_balance_temperature)
        
    def _manage_colors(self, data):
        for key in data.keys():
            if key in ['image', 'sphere_0', 'sphere_1', 'sphere_2']:
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
            elif 'position' in key or 'image_position' in key:
                data[key] = position_to_scaled_position(data[key])
                data[key].mul_(2).sub_(1) 
                #data[key] = torch.zeros_like(data[key])
            elif 'dist_to_sphere' in key:
                data[key] = depth_to_scaled_depth(data[key], torch.ones_like(data[key])).repeat(3, 1, 1)  # Repeat the depth channel to match RGB
                data[key].mul_(2).sub_(1)
            elif key in ['normals', 'dir_to_sphere']:
                data[key] = data[key].clamp(-1, 1)
                

    @staticmethod
    def get_gpu_postprocessing_args(**kwargs):
        return {}

    @staticmethod
    def gpu_batch_postprocess(batch,**kwargs):
        pass

    def __getitem__(self, item):
        self.start_time = time.time()
        #if item is a tuple
        if isinstance(item, tuple):
            idx, batch_random_int, self.max_resolution = item
        else:
            idx = item
            batch_random_int = random.randint(0, 2**32-1)
            
        #Image without spheres logic
        while True:
            if idx in self.bad_indices:
                idx = random.randint(0, len(self.scenes)-1)
                continue
            camera_path = self.scenes[idx]
            
            self.idx = idx

            number = "0001"
            if self.frames:
                number = os.path.basename(camera_path).split('_')[-1]
            
            spheres_path = camera_path
            camera_info_path = os.path.join(camera_path, 'camera_info.pkl')
            if not os.path.exists(camera_info_path):
                print(camera_info_path, " does not exist, trying to find it in parent directory")
                camera_path = os.path.dirname(camera_path)
                print(camera_path)
                camera_info_path = os.path.join(camera_path, 'camera_info.pkl')
            with open(camera_info_path, 'rb') as f:
                camera_info = pickle.load(f)
            
            output_dict = {}
            
            #Load main image
            image_path = os.path.join(camera_path, f'image_{number}.exr')
            image = pyexr.read(image_path)
            image = torch.from_numpy(image).float().permute(2, 0, 1)
            output_dict['image'] = image[:3, :, :]  # Keep only RGB channels
            
            
            random_req_layer = random.choice(self.target_layers)
            
            
            #Load mask and select random sphere
            sphere_0_path = os.path.join(spheres_path, 'sphere_0')
            sphere_1_path = os.path.join(spheres_path, 'sphere_1')
            #sphere_2_path = os.path.join(spheres_path, 'sphere_2')
            mask = pyexr.read(os.path.join(sphere_0_path, f"mask_{number}.exr"))
            mask = mask.astype(np.int32)[:,:,0]
            id_spheres = np.unique(mask)[1:] # Exclude the background   
            if self.split == 'test' or self.split == 'objects':
                if 255 not in id_spheres:
                    print("Sphere with id 255 not found in scene")
                    self.bad_indices.append(idx)
                    if self.split == 'test' or self.split == 'objects':
                        idx = (idx + 1) % len(self.scenes)
                    continue
                
            if len(id_spheres) > 0:
                break
            else:
                print("No spheres found in scene")
                self.bad_indices.append(idx)
                if self.split == 'test' or self.split == 'objects':
                    idx = (idx + 1) % len(self.scenes)
                else:
                    idx = random.randint(0, len(self.scenes)-1)

        sphere_id = random.choice(id_spheres)
        if self.split == 'test' or self.split == 'objects':
            #For test set, we use a fixed sphere id
            sphere_id = 255
        
        #output_dict['sphere_id'] = sphere_id
        output_dict['sphere_mask'] = torch.from_numpy((mask == sphere_id).astype(np.float32)).unsqueeze(0)  # Shape: (1, H, W)
        
        #load sphere images
        sphere_0_image_path = os.path.join(sphere_0_path, f'image_{number}.exr')
        sphere_1_image_path = os.path.join(sphere_1_path, f'image_{number}.exr')
        #sphere_2_image_path = os.path.join(sphere_2_path, f'image_{number}.exr')

        sphere_0_image = pyexr.read(sphere_0_image_path)
        sphere_0_image = torch.from_numpy(sphere_0_image).float().permute(2, 0, 1)[:3, :, :]
        
        sphere_1_image = pyexr.read(sphere_1_image_path)
        sphere_1_image = torch.from_numpy(sphere_1_image).float().permute(2, 0, 1)[:3, :, :]
        
        #sphere_2_image = pyexr.read(sphere_2_image_path)
        #sphere_2_image = torch.from_numpy(sphere_2_image).float().permute(2, 0, 1)[:3, :, :]
        
        ev = 0
        if self.random_ev or self.known_ev:
            #Find out what the minimum EV is for the sphere images
            if self.known_ev:
                ev = self.known_ev
            else:
                maximum_value = max(sphere_0_image.max(), sphere_1_image.max())#, sphere_2_image.max())
                if maximum_value <= 0:
                    minimum_ev = 0
                    print(maximum_value)
                    print("Maximum value in sphere images is negative, setting minimum EV to 0")
                else:
                    minimum_ev = - math.log2(2 * maximum_value)
                    minimum_ev = math.floor(minimum_ev / 3) * 3  # Round to the nearest multiple of 3
                minimum_ev = max(minimum_ev, -12)  # Ensure minimum EV is not too low
                minimum_ev = min(minimum_ev, 0)  # Ensure minimum EV is not positive
                #Sample a random EV in the range [minimum_ev, 0]
                start = minimum_ev + (3 - minimum_ev % 3) if minimum_ev % 3 != 0 else minimum_ev
                # Create list of valid multiples of 3 from start to 0 (inclusive)
                valid_evs = list(range(start, 1, 3))  # step is 3, end at 1 to include 0

                # Sample one
                ev = random.choice(valid_evs)
                # ev = random.uniform(minimum_ev, 0)
                # ev = int(ev)
                # ev = min(ev, 0) # Clamp EV to be non-positive
            #Apply the EV to the sphere images
            sphere_0_image *= 2 ** ev
            sphere_1_image *= 2 ** ev
            #sphere_2_image *= 2 ** ev

        
        sphere_0_image = sphere_0_image * output_dict['sphere_mask'] + output_dict['image'] * (1 - output_dict['sphere_mask'])
        output_dict['sphere_0'] = sphere_0_image

        
        sphere_1_image = sphere_1_image * output_dict['sphere_mask'] + output_dict['image'] * (1 - output_dict['sphere_mask'])
        output_dict['sphere_1'] = sphere_1_image

        
        # sphere_2_image = sphere_2_image * output_dict['sphere_mask'] + output_dict['image'] * (1 - output_dict['sphere_mask'])
        # output_dict['sphere_2'] = sphere_2_image

        
        # for key in self.target_layers:
        #     if key not in output_dict.keys():
        #         output_dict[key] = torch.zeros_like(output_dict[random_req_layer])
                    
        #load position map
        sphere_position_map_path = os.path.join(sphere_0_path, f'position_{number}.exr')
        image_position_map_path = os.path.join(camera_path, f'position_{number}.exr')
        sphere_position_map = pyexr.read(sphere_position_map_path)
        image_position_map = pyexr.read(image_position_map_path)
        sphere_position_map = torch.from_numpy(sphere_position_map).float().permute(2, 0, 1)
        image_position_map = torch.from_numpy(image_position_map).float().permute(2, 0, 1)
        sphere_position_map = sphere_position_map[:3,:,:]
        image_position_map = image_position_map[:3,:,:]
        position_map = sphere_position_map * output_dict['sphere_mask'] + image_position_map * (1 - output_dict['sphere_mask'])
        position_map = position_map[:3, :, :]
        
        #Bring position map to camera coordinates
        cam2world = torch.from_numpy(camera_info['cam2world']).float()
        world2cam = torch.inverse(cam2world)
        position_map = position_map.view(3, -1)  # Flatten the spatial dimensions
        #Add extra channel for homegeneous coordinates
        position_map = torch.cat((position_map, torch.ones(1, position_map.shape[1], dtype=position_map.dtype, device=position_map.device)), dim=0)  # Add a row of ones for homogeneous coordinates
        position_map = torch.matmul(world2cam, position_map)  # Transform to world coordinates
        position_map = position_map[:3, :]  # Remove the homogeneous coordinate row
        position_map = position_map.reshape(3, *image.shape[-2:])  # Reshape back to (3, H, W)

        position_map[2, :, :] = -position_map[2, :, :]  # Invert the Z coordinate to have positive Z pointing outwards
        output_dict['position'] = position_map

        # Image position map
        image_position_map = image_position_map.view(3, -1)  # Flatten the spatial dimensions
        #Add extra channel for homegeneous coordinates
        image_position_map = torch.cat((image_position_map, torch.ones(1, image_position_map.shape[1], dtype=image_position_map.dtype, device=image_position_map.device)), dim=0)  # Add a row of ones for homogeneous coordinates
        image_position_map = torch.matmul(world2cam, image_position_map)  # Transform to world coordinates
        image_position_map = image_position_map[:3, :]  # Remove the homogeneous coordinate row
        image_position_map = image_position_map.reshape(3, *image.shape[-2:])

        image_position_map[2, :, :] = -image_position_map[2, :, :]  # Invert the Z coordinate to have positive Z pointing outwards
        output_dict['image_position'] = image_position_map
        
        
        
        #Normal map
        sphere_normal_map_path = os.path.join(sphere_0_path, f'normal_{number}.exr')
        sphere_normal_map = pyexr.read(sphere_normal_map_path)
        sphere_normal_map = torch.from_numpy(sphere_normal_map).float().permute(2, 0, 1)
        sphere_normal_map = sphere_normal_map[:3, :, :] * output_dict['sphere_mask']
        
        #Bring normal map to camera coordinates
        normal_map = sphere_normal_map.view(3, -1)  # Flatten the spatial dimensions
        #Add extra channel for homegeneous coordinates
        #normal_map = torch.cat((normal_map, torch.ones(1, normal_map.shape[1], dtype=normal_map.dtype, device=normal_map.device)), dim=0)
        #normal_map = torch.matmul(world2cam, normal_map)  # Transform to world coordinates
        normal_map = torch.matmul(world2cam[:3,:3], normal_map)
        #normal_map = normal_map[:3, :]  # Remove the homogeneous coordinate row
        normal_map = normal_map.reshape(3, *image.shape[-2:])  # Reshape back to (3, H, W)
        output_dict['normals'] = normal_map
        
        
        #Direction to sphere
        
        sphere_center_pixel = output_dict['sphere_mask'].nonzero(as_tuple=False).float().mean(dim=0)
        sphere_center_pixel = sphere_center_pixel[1:]  # Remove the batch dimension
        sphere_center_pixel = sphere_center_pixel.long()  # Convert to integer pixel coordinates
        sphere_center = position_map[:, sphere_center_pixel[0], sphere_center_pixel[1]]  # Get the position of the sphere center in world coordinates
        position_map_filled_neg = position_map.clone().masked_fill(~(output_dict["sphere_mask"]==1), float('-inf'))  # Fill the non-sphere pixels with -inf
        sphere_max = position_map_filled_neg.amax(dim=(1,2))  # Get the maximum position value for the sphere
        position_map_filled_pos = position_map.clone().masked_fill(~(output_dict["sphere_mask"]==1), float('inf'))  # Fill the non-sphere pixels with -inf
        sphere_min = position_map_filled_pos.amin(dim=(1,2))
        sphere_radius = (sphere_max[0]-sphere_min[0]) / 2  # Calculate the radius of the sphere
        sphere_center[2] += sphere_radius
        
        vector_to_sphere = sphere_center[:,None,None] - position_map  # Get the vector from the camera to the sphere center
        
        dir_to_sphere = vector_to_sphere / (vector_to_sphere.norm(dim=0, keepdim=True) + 1e-8)  # Normalize the vector to get the direction to the sphere
        dist_to_sphere = vector_to_sphere.norm(dim=0, keepdim=True)  # Get the distance to the sphere
        
        output_dict['dir_to_sphere'] = dir_to_sphere
        output_dict['dist_to_sphere'] = dist_to_sphere
        
        #Auto exposure 
        if self.augmentation_cfg.auto_exposure:
            exposure_factor = get_auto_exposure(output_dict['image'])
            
            output_dict['image'] *= exposure_factor
            output_dict['sphere_0'] *= exposure_factor
            output_dict['sphere_1'] *= exposure_factor
            #output_dict['sphere_2'] *= exposure_factor
            
        if self.mask_img:
            #Mask the input image with the sphere mask
            output_dict['image'] = output_dict['image'] * (1 - output_dict['sphere_mask'])
        
        self._clean_nan(output_dict)
        self._resize_data(output_dict,batch_random_int)
        if self.split == 'train':
            self._augment_data(output_dict, self.augmentation_cfg)

        self._manage_colors(output_dict)
        
        output_dict['target_layer'] = random_req_layer # self.target_layers
        output_dict['ev'] = ev
        
        output_dict["scene"] = self.scenes[idx]
        
        return output_dict


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
    
    from data_utils import seed_worker
    config = SpheresDatasetConfig(
                    #data_dir = "s3://nflx-studio-algo-research-sl-awsprod-us-east-1/nflx-scl-ml/from_eyeline/users/jphilip/synthetic_data/outdoor_10k_20241211/",
                    data_dir = ["/root/Data/synthetic_data/spheres_bk_indoor", 
                                #"/root/Data/synthetic_data/spheres_bk_outdoor",
                                "/root/Data/synthetic_data/spheres_rand_indoor",
                                "/root/Data/synthetic_data/spheres_rand_outdoor"],
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
    dataset = SpheresDataset(config)

    # Use DataLoader to iterate over the dataset

    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False, worker_init_fn=seed_worker)
    output_dir = "/root/outputs/spheres_dataset_test"
    os.makedirs(os.path.join(output_dir,"logs"), exist_ok=True)
    name_report = "dataset_trace_"+str(time.time())

    profile = False
    wait_iter=16

    if profile:
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=wait_iter, warmup=8, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(output_dir,"logs",name_report)),
            #record_shapes=True,
            #with_stack=True
        )
        prof.start()

    for it in range(3):
        for id,batch in tqdm(enumerate(dataloader)):
            print("image: ", batch["image"].shape)
            print("sphere_0:", batch["sphere_0"].shape)
            print("sphere_1:", batch["sphere_1"].shape)
            print("sphere_2:", batch["sphere_2"].shape)
            print("position:", batch["position"].shape)
            print("sphere_mask:", batch["sphere_mask"].shape)
            print("ev:", batch["ev"])
            #Save temp results
            from ezexr import imsave
            os.makedirs(os.path.join(output_dir, f"{id}"), exist_ok=True)
            imsave(os.path.join(output_dir, f"{id}", "image.exr"), batch["image"].cpu().squeeze().permute(1,2,0).numpy())
            imsave(os.path.join(output_dir, f"{id}", "sphere_0.exr"), batch["sphere_0"].cpu().squeeze().permute(1,2,0).numpy())
            imsave(os.path.join(output_dir, f"{id}", "sphere_1.exr"), batch["sphere_1"].cpu().squeeze().permute(1,2,0).numpy())
            imsave(os.path.join(output_dir, f"{id}", "sphere_2.exr"), batch["sphere_2"].cpu().squeeze().permute(1,2,0).numpy())
            imsave(os.path.join(output_dir, f"{id}", "position.exr"), batch["position"].cpu().squeeze().permute(1,2,0).numpy())
            imsave(os.path.join(output_dir, f"{id}", "sphere_mask.exr"), batch["sphere_mask"].cpu().squeeze().permute(1,2,0).numpy())

            if profile:
                prof.step()
                if id>wait_iter+12:
                    break
            if id == 2:
                break
        break

    if profile:
        prof.stop()

