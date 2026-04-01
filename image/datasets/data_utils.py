import math
import os
import random
import time
import numpy as np
import bisect
import torch
from torch.utils.data import ConcatDataset, Sampler
import boto3
from boto3.s3.transfer import TransferConfig
import botocore

#### Color management ####
@torch.jit.script
def sRGB_to_Lin(im):
    """Convert sRGB to linear."""
    return torch.where(im <= 0.04045, im / 12.92, ((im + 0.055) / 1.055) ** 2.4)

@torch.jit.script
def Lin_to_sRGB(im):
    """Convert linear to sRGB."""
    linear_part = 12.92 * im
    gamma_part = im.pow(1.0 / 2.4).mul_(1.055).sub_(0.055)
    return torch.where(im <= 0.0031308, linear_part, gamma_part)

#http://filmicworlds.com/blog/filmic-tonemapping-operators/
def Lin_to_Hejl(im):
    im = torch.clamp(im-0.004,0)
    im = (im*(6.2*im+.5))/(im*(6.2*im+1.7)+0.06)
    return im

def Lin_to_Log(im,max_val=128):
    im = torch.clamp(im,max=max_val)
    im = torch.log(2.2*im + 1.0)/math.log(2.2*max_val + 1.0)
    return torch.pow(im,1.0/2.2)

def Log_to_Lin(im,max_val=128):
    im = torch.pow(im,2.2)
    im = torch.exp(im*math.log(2.2*max_val + 1.0))-1.0
    return im/2.2

def Lin_to_Log_(im,max_val=128):
    torch.clamp_(im,max=max_val)
    im.mul_(2.2).add_(1.0).log_().div_(math.log(2.2*max_val + 1.0)).pow_(1.0/2.2)
    
def Lin_to_Reinhard(im, power=1/2):
    im_power = im**power
    return im_power / (im_power + 1)

def Reinhard_to_Lin(im, power=1/2):
    return (im / (1 - im)) ** (1/power)

def depth_to_scaled_depth(depth,alpha):
    log_masked_depth = torch.log10(depth[0:3,alpha[0]>0.99])
    mean_log_depth = log_masked_depth.mean()
    #Shift the depth to have a log-mean of 1
    depth[0:3] = depth[0:3]/(10**(mean_log_depth-1))
    filled_depth = depth[0:3]+(1-alpha)*(10**3)
    log_depth = (torch.log10(filled_depth).clamp(-1,3)+1)/4
    return log_depth

def position_to_scaled_position(position):
    # log_position = torch.log10(position)
    # mean_log_position = log_position.mean()
    # position[0:3] = position[0:3]/(10**(mean_log_position-1))
    # log_position = (torch.log10(position).clamp(-1,3)+1)/4
    # return log_position
    
    # position_max = position.max()
    # position_min = position.min()
    # position = (position - position_min) / (position_max - position_min)
    # return position
    
    #position = position - position.min()
    linear_depth = position[2:, ...]
    #Repeat 3 times on channel dimension
    linear_depth = linear_depth.repeat(3, 1, 1)
    
    #linear_depth = linear_depth - linear_depth.min()
    log_depth = depth_to_scaled_depth(linear_depth, torch.ones_like(linear_depth))
    #scaled_position = position * log_depth/linear_depth
    scaled_position = log_depth
    scaled_position = torch.nan_to_num(scaled_position)
    return scaled_position 


def _tone_map(rgb, exposure, wb_rgb):
    rgb = rgb * torch.pow(2.0, wb_rgb)
    rgb = rgb * torch.pow(2.0, exposure)
    rgb = torch.clamp(rgb, 1e-4, 1.0)
    rgb = Lin_to_sRGB(rgb)
    return rgb

def get_auto_exposure(image):
    is_3dim = image.dim() == 3
    if is_3dim:
        image = image.unsqueeze(0)
    exposure = torch.ones([image.shape[0],1,1,1], device=image.device)
    #Compute luminance using the standard coefficient
    luminance = image.mul(torch.tensor([0.2126, 0.7152, 0.0722], device=image.device).view(1,3, 1, 1)).sum(dim=1, keepdim=True)
    torch.clamp_(luminance, 1e-3, 128) #Clamp to avoid extreme values steering the exposure
    torch.log_(luminance)
    #Compute the 90th percentile of the log luminance
    quant90_log_luminance = torch.quantile(luminance.view(image.shape[0], -1), 0.95, dim=-1).view(image.shape[0], 1, 1, 1)
    target_quant90_log_luminance = math.log(0.95)
    #Compute the exposure to have a log average luminance of 0.16
    exposure *= torch.exp(target_quant90_log_luminance - quant90_log_luminance)
    if is_3dim:
        exposure = exposure.squeeze(0)
    return exposure

def white_balance(image, temperature):
    """
    Adjust the white balance of an image based on a given color temperature in Kelvin.

    Args:
    - image (torch.Tensor): Input image tensor in linear space with shape (C, H, W) where C=3 for RGB.
    - temperature (float): Color temperature in Kelvin.

    Returns:
    - torch.Tensor: White balanced image tensor.
    """
    # Define the reference white point (D65 standard illuminant)
    d65_white_point = torch.tensor([0.95047, 1.00000, 1.08883], dtype=image.dtype, device=image.device)

    # Convert the temperature to a white point using a simple approximation
    def kelvin_to_linear_rgb(temp):
        temp = temp / 100
        if temp <= 66:
            red = 255
            green = temp
            green = 99.4708025861 * math.log(green) - 161.1195681661
            if temp <= 19:
                blue = 0
            else:
                blue = temp - 10
                blue = 138.5177312231 * math.log(blue) - 305.0447927307
        else:
            red = temp - 60
            red = 329.698727446 * math.pow(red, -0.1332047592)
            green = temp - 60
            green = 288.1221695283 * math.pow(green, -0.0755148492)
            blue = 255

        # Convert to linear RGB by applying inverse gamma correction
        red_linear = math.pow(red / 255.0, 2.2)
        green_linear = math.pow(green / 255.0, 2.2)
        blue_linear = math.pow(blue / 255.0, 2.2)

        return torch.tensor([red_linear, green_linear, blue_linear], dtype=image.dtype, device=image.device)

    # Get the target white point for the given temperature
    target_white_point = kelvin_to_linear_rgb(temperature)
    # Calculate the scaling factors for each channel
    scale_factors = d65_white_point / target_white_point
    # Apply the scaling factors to the image
    balanced_image = image * scale_factors.view(3, 1, 1)
    return balanced_image

#Batch sampler to allow feeding a shared random number to all elements in the batch
class RandomSizeBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, shared_dict):
        self.sampler = sampler
        self.batch_size = batch_size
        self.shared_dict = shared_dict
        
    def __iter__(self):
        # Yield indices in batches
        batch = []
        max_resolution = self.shared_dict['max_resolution']
        batch_random_int = 0 #So the first batch is highest resolution. 
        batch_count = 0
        for idx in self.sampler:
            batch.append((idx, batch_random_int, max_resolution))
            if len(batch) == self.batch_size:
                yield batch
                batch_count += 1
                batch = []
                max_resolution = self.shared_dict['max_resolution']
                if batch_count>16: #So the first batch is at high resolution accross all GPUs, highest resolution avoid some OOM.
                    batch_random_int = random.randint(0, 2**32-1)

    def __len__(self):
        return len(self.sampler) // self.batch_size

class CustomConcatDataset(ConcatDataset):
    def __getitem__(self, item):
        #if item is a tuple
        if isinstance(item, tuple):
            idx, batch_random_int, max_resolution = item
        else:
            idx = item
            batch_random_int = 0
            max_resolution = 1024**2
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][(sample_idx, batch_random_int, max_resolution)]

# Function to set the seed for each worker
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def reproducible_seeding(worker_id):
    seed = worker_id
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


#### S3 Related functions ####
def handle_s3_error(client, error, attempt, s3_path):
    sleep_time = 0.25 * (2 ** attempt)
    print(f"Attempt {attempt + 1} to access {s3_path} failed with {type(error).__name__}. Retrying in {sleep_time} seconds...")
    client = boto3.client('s3',config=botocore.client.Config(max_pool_connections=8))
    time.sleep(sleep_time)

def download_s3_file(client, bucket, s3_path, local_path, max_retries=8):
    attempt = 0
    # Make sure the directory exists
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    while attempt < max_retries:
        try:
            config = TransferConfig(use_threads=False)
            client.download_file(bucket, s3_path, local_path, Config=config)
            return
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                raise FileNotFoundError(f"The object {s3_path} does not exist.")
            else:
                handle_s3_error(client, e, attempt, s3_path)
                attempt += 1
        #Handle any error
        except Exception as e:
            handle_s3_error(client, e, attempt, s3_path)
            attempt += 1
    raise Exception(f"Failed to download {s3_path} after {max_retries} attempts.")

def list_s3_folders(client, bucket, s3_path, max_retries=8):
    attempt = 0
    while attempt < max_retries:
        try:
            paginator = client.get_paginator('list_objects_v2')
            folder_pages = paginator.paginate(Bucket=bucket, Prefix=s3_path, Delimiter='/')
            list_dir = []
            for page in folder_pages:
                temp_list = page.get('CommonPrefixes', [])
                temp_list = [obj['Prefix'] for obj in temp_list]
                list_dir.extend(temp_list)
            return sorted(list_dir)
        except (botocore.exceptions.ClientError, botocore.exceptions.CredentialRetrievalError, botocore.exceptions.HTTPClientError) as e:
            handle_s3_error(client, e, attempt, s3_path)
            attempt += 1
    raise Exception(f"Failed to list {s3_path} after {max_retries} attempts.")

def list_s3_files(client, bucket, s3_path, max_retries=8, extension=None):
    attempt = 0
    while attempt < max_retries:
        try:
            paginator = client.get_paginator('list_objects_v2')
            folder_pages = paginator.paginate(Bucket=bucket, Prefix=s3_path)
            list_dir = []
            for page in folder_pages:
                temp_list = page.get('Contents', [])
                temp_list = [obj['Key'] for obj in temp_list]
                if extension is not None:
                    temp_list = [obj for obj in temp_list if obj.endswith(extension)]
                list_dir.extend(temp_list)
            return sorted(list_dir)
        except (botocore.exceptions.ClientError, botocore.exceptions.CredentialRetrievalError, botocore.exceptions.HTTPClientError) as e:
            handle_s3_error(client, e, attempt, s3_path)
            attempt += 1
    raise Exception(f"Failed to list {s3_path} after {max_retries} attempts.")

