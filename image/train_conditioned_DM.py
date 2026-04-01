#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import copy
import logging
import math
import os
import shutil
from contextlib import nullcontext
from pathlib import Path
import tyro
from datetime import datetime
from multiprocessing import Manager
import numpy as np
import torch
import torchvision
import transformers
from accelerate import Accelerator, DeepSpeedPlugin, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, PretrainedConfig, T5EncoderModel, T5TokenizerFast
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps

import diffusers
from diffusers import (
    AutoencoderKL,
    SD3Transformer2DModel,
    FluxTransformer2DModel,
    SD3ControlNetModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, free_memory
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
)
from diffusers.utils.torch_utils import is_compiled_module

from train_config import TrainConfig, default_train_configs
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler
from datasets.data_utils import (
    RandomSizeBatchSampler,
    CustomConcatDataset,
    seed_worker
)
from utils import cfg_to_dict,encode_with_vae,prepare_target_and_prompt,prepare_target_and_prompt_ev,add_condition_input, decode_with_vae,TARGET_TO_PROMPT
import train_utils as tu
from conditional_pipelines import ConditionalSD3Pipeline
import lpips
from pytorch_msssim import ssim

if is_wandb_available():
    import wandb
import time
import random


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.32.0.dev0")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = get_logger(__name__)

@torch.no_grad()
def log_first_train(
    pipeline,
    cfg,
    accelerator,
    prompt_embeds_dict,
    pooled_prompt_embeds_dict,
    validation_dataset, 
    global_step
):
    pipeline.text_encoder = None
    pipeline.text_encoder_2 = None
    if hasattr(pipeline, "text_encoder_3"):
        pipeline.text_encoder_3 = None

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    
    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(1)
    
    autocast_ctx = nullcontext()
        
    validation_dataset.load_all = True
    batch = validation_dataset[0]
    
    for key in batch.keys():
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(pipeline.vae.device)
            
    conditions = [batch[k] for k in cfg.condition_list.split(',')]
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

    # Convert conditions to latent space
    latent_conditions = [encode_with_vae(pipeline.vae, condition, weight_dtype=pipeline.vae.dtype) for condition in conditions]

    target_key = batch['target_layer']

    latent_conditions = [l_c.to(dtype=pipeline.transformer.dtype) for l_c in latent_conditions]
    pipeline_args_0 = {
        "prompt_embeds": prompt_embeds_dict['sphere_0'][batch['ev']].to(dtype=pipeline.transformer.dtype),
        "pooled_prompt_embeds": pooled_prompt_embeds_dict['sphere_0'][batch['ev']].to(dtype=pipeline.transformer.dtype),
        "latent_conditions": latent_conditions,
        "width": w,
        "height": h,
        "guidance_scale": 1.0,
        "lotus_formulation": cfg.lotus_formulation,
        "lotus_num_steps": cfg.lotus_num_steps,
        "num_inference_steps": cfg.val_denoising_steps,
        #"sigmas": np.array([1.0,0.3,0.1,0.03]),
    }
    pipeline_args_1 = {
        "prompt_embeds": prompt_embeds_dict['sphere_1'][batch['ev']].to(dtype=pipeline.transformer.dtype),
        "pooled_prompt_embeds": pooled_prompt_embeds_dict['sphere_1'][batch['ev']].to(dtype=pipeline.transformer.dtype),
        "latent_conditions": latent_conditions,
        "width": w,
        "height": h,
        "guidance_scale": 1.0,
        "lotus_formulation": cfg.lotus_formulation,
        "lotus_num_steps": cfg.lotus_num_steps,
        "num_inference_steps": cfg.val_denoising_steps,
        #"sigmas": np.array([1.0,0.3,0.1,0.03]),
    }
    pipeline_args_2 = {
        "prompt_embeds": prompt_embeds_dict['sphere_2'][batch['ev']].to(dtype=pipeline.transformer.dtype),
        "pooled_prompt_embeds": pooled_prompt_embeds_dict['sphere_2'][batch['ev']].to(dtype=pipeline.transformer.dtype),
        "latent_conditions": latent_conditions,
        "width": w,
        "height": h,
        "guidance_scale": 1.0,
        "lotus_formulation": cfg.lotus_formulation,
        "lotus_num_steps": cfg.lotus_num_steps,
        "num_inference_steps": cfg.val_denoising_steps,
        #"sigmas": np.array([1.0,0.3,0.1,0.03]),
    }

    with autocast_ctx:
        image_0 = pipeline(**pipeline_args_0, generator=generator).images[0]
        image_1 = pipeline(**pipeline_args_1, generator=generator).images[0]
        image_2 = pipeline(**pipeline_args_2, generator=generator).images[0]
    
    gt_0 = batch['sphere_0']
    gt_1 = batch['sphere_1']
    gt_2 = batch['sphere_2']
    condition = batch[cfg.condition_list.split(',')[0]]
    # gen_img = gen_img[0,:,:,:]
    # condition = condition[0,:,:,:]
    # gt = gt[0,:,:,:]
    image_0 = torchvision.transforms.functional.to_tensor(image_0).to(accelerator.device)
    image_1 = torchvision.transforms.functional.to_tensor(image_1).to(accelerator.device)
    image_2 = torchvision.transforms.functional.to_tensor(image_2).to(accelerator.device)
    
    #condition = torchvision.transforms.functional.to_tensor(condition).to(accelerator.device)
    image_0 = image_0.clamp(0, 1)
    image_1 = image_1.clamp(0, 1)
    image_2 = image_2.clamp(0, 1)
    condition = (condition * 0.5 + 0.5).clamp(0, 1)
    gt_0 = (gt_0 * 0.5 + 0.5).clamp(0, 1)
    gt_1 = (gt_1 * 0.5 + 0.5).clamp(0, 1)
    gt_2 = (gt_2 * 0.5 + 0.5).clamp(0, 1)
    grid = torch.stack([condition, gt_0, image_0, gt_1, image_1, gt_2, image_2], dim=0)
    grid = torchvision.utils.make_grid(grid, nrow=3)
    wandb.log({"train_images": wandb.Image(grid, caption=f"Step {global_step}")}, step=global_step)
    validation_dataset.load_all = False
        

@torch.no_grad()
def log_spatial_test(
    pipeline,
    cfg,
    accelerator,
    prompt_embeds_dict,
    pooled_prompt_embeds_dict,
    spatial_dataset, 
    global_step
):
    pipeline.text_encoder = None
    pipeline.text_encoder_2 = None
    if hasattr(pipeline, "text_encoder_3"):
        pipeline.text_encoder_3 = None

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    
    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(1)
    
    autocast_ctx = nullcontext()
    
    obj = []
    gts = []
    for batch_id, batch in enumerate(spatial_dataset):
    
        for key in batch.keys():
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(pipeline.vae.device)
                
        conditions = [batch[k] for k in cfg.condition_list.split(',')]
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

        # Convert conditions to latent space
        latent_conditions = [encode_with_vae(pipeline.vae, condition, weight_dtype=pipeline.vae.dtype) for condition in conditions]

        target_key = batch['target_layer']

        latent_conditions = [l_c.to(dtype=pipeline.transformer.dtype) for l_c in latent_conditions]
        pipeline_args_0 = {
            "prompt_embeds": prompt_embeds_dict['sphere_0'][batch['ev']].to(dtype=pipeline.transformer.dtype),
            "pooled_prompt_embeds": pooled_prompt_embeds_dict['sphere_0'][batch['ev']].to(dtype=pipeline.transformer.dtype),
            "latent_conditions": latent_conditions,
            "width": w,
            "height": h,
            "guidance_scale": 1.0,
            "lotus_formulation": cfg.lotus_formulation,
            "lotus_num_steps": cfg.lotus_num_steps,
            "num_inference_steps": cfg.val_denoising_steps,
            #"sigmas": np.array([1.0,0.3,0.1,0.03]),
        }

        with autocast_ctx:
            image_0 = pipeline(**pipeline_args_0, generator=generator).images[0]
        
        gt_0 = batch['sphere_0']
        condition = batch[cfg.condition_list.split(',')[0]]
        # gen_img = gen_img[0,:,:,:]
        # condition = condition[0,:,:,:]
        # gt = gt[0,:,:,:]
        image_0 = torchvision.transforms.functional.to_tensor(image_0).to(accelerator.device)
        
        #condition = torchvision.transforms.functional.to_tensor(condition).to(accelerator.device)
        image_0 = image_0.clamp(0, 1)
        condition = (condition * 0.5 + 0.5).clamp(0, 1)
        gt_0 = (gt_0 * 0.5 + 0.5).clamp(0, 1)
        
        obj.append(image_0)
        gts.append(gt_0)
        
    grid = torch.stack([gts[0], gts[1], gts[2],obj[0], obj[1], obj[2]], dim=0)
    grid = torchvision.utils.make_grid(grid, nrow=3)
    wandb.log({"spatial_images": wandb.Image(grid, caption=f"Step {global_step}")}, step=global_step)
    

@torch.no_grad()
def log_validation(
    pipeline,
    cfg,
    accelerator,
    target_and_prompt_dict,
    prompt_embeds_dict,
    pooled_prompt_embeds_dict,
    epoch,
    validation_dataset, 
    is_final_validation=False,
    batch_postprocess_args=None,
    DatasetClass=None,
):
    logger.info(
        f"Running validation... \n Generating {cfg.num_validation_images} images"
    )
    pipeline.text_encoder = None
    pipeline.text_encoder_2 = None
    if hasattr(pipeline, "text_encoder_3"):
        pipeline.text_encoder_3 = None

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(cfg.seed) if cfg.seed else None
    # autocast_ctx = torch.autocast(accelerator.device.type) if not is_final_validation else nullcontext()
    autocast_ctx = nullcontext()

    val_dataloader = DataLoader(
        validation_dataset,
        sampler=RandomSampler(validation_dataset, num_samples=cfg.num_validation_images),
        batch_size=1,
        num_workers=4,
        worker_init_fn=seed_worker,
    )

    images=[]
    #Initialize with empty list for each condition
    condition_images = {condition: [] for condition in cfg.condition_list.split(',')}
    target_images = []
    list_target_keys = list(target_and_prompt_dict.keys())
    if cfg.lotus_formulation:
        #Remove the potential condition_list from the list of target keys
        #Lotus trains also on the input image to better preserve details
        list_target_keys = [key for key in list_target_keys if key not in cfg.condition_list.split(',')]

    num_tgt_keys = len(list_target_keys)
    for batch_id, batch in tqdm(enumerate(val_dataloader)):
        # Convert images to latent space
        for key in batch.keys():
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(pipeline.vae.device)
        # post process the batch on device
        if batch_postprocess_args is not None and DatasetClass is not None:
            DatasetClass.gpu_batch_postprocess(batch=batch, **batch_postprocess_args)

        conditions = [batch[k] for k in cfg.condition_list.split(',')]
        if conditions[0].dim() == 5:
            b,l,c,h,w = conditions[0].shape
            conditions = [condition[:,0] for condition in conditions]
        else:
            l=0
            b,c,h,w = conditions[0].shape

        # Convert conditions to latent space
        latent_conditions = [encode_with_vae(pipeline.vae, condition, weight_dtype=pipeline.vae.dtype) for condition in conditions]

        #target_key = list_target_keys[batch_id%num_tgt_keys]
        target_key = batch['target_layer'][0]        

        latent_conditions = [l_c.to(dtype=pipeline.transformer.dtype) for l_c in latent_conditions]
        pipeline_args = {
            "prompt_embeds": prompt_embeds_dict[target_key][batch['ev'].item()].to(dtype=pipeline.transformer.dtype),
            "pooled_prompt_embeds": pooled_prompt_embeds_dict[target_key][batch['ev'].item()].to(dtype=pipeline.transformer.dtype),
            "latent_conditions": latent_conditions,
            "width": w,
            "height": h,
            "guidance_scale": 1.0,
            "lotus_formulation": cfg.lotus_formulation,
            "lotus_num_steps": cfg.lotus_num_steps,
            "num_inference_steps": cfg.val_denoising_steps,
            #"sigmas": np.array([1.0,0.3,0.1,0.03]),
        }

        with autocast_ctx:
            images.append(pipeline(**pipeline_args, generator=generator).images[0])
        for cond_id,cond in enumerate(cfg.condition_list.split(',')):
            condition_images[cond].append((0.5*conditions[cond_id][0].float()+0.5).detach().cpu())
        target_tensor = batch[target_key]
        if target_tensor.dim() == 5:
            target_tensor = target_tensor[:,0]
        target_images.append((0.5*batch[target_key][0].float()+0.5).detach().cpu())

    #Compute LPIPS, PSNR, SSIM between the generated images and the target images
    import torchvision.transforms.functional as F

    lpips_fn = lpips.LPIPS(net='alex').to(accelerator.device)

    lpips_scores = []
    psnr_scores = []
    ssim_scores = []

    lpips_scores_per_key = {key: [] for key in list_target_keys}
    psnr_scores_per_key = {key: [] for key in list_target_keys}
    ssim_scores_per_key = {key: [] for key in list_target_keys}

    for batch_id, (gen_img, tgt_img) in enumerate(zip(images, target_images)):

        target_key = list_target_keys[batch_id%num_tgt_keys]

        gen_img = F.to_tensor(gen_img).unsqueeze(0).to(accelerator.device)
        tgt_img = tgt_img.unsqueeze(0).to(accelerator.device)

        lpips_score = lpips_fn(2*gen_img.clamp(0,1)-1, 2*tgt_img.clamp(0,1)-1).item()
        psnr_score =  - 10 * torch.log10(torch.mean((gen_img - tgt_img) ** 2)).item()
        ssim_score = ssim(gen_img.clamp(0,1), tgt_img.clamp(0,1), data_range=1.0).item()

        lpips_scores.append(lpips_score)
        psnr_scores.append(psnr_score)
        ssim_scores.append(ssim_score)
        lpips_scores_per_key[target_key].append(lpips_score)
        psnr_scores_per_key[target_key].append(psnr_score)
        ssim_scores_per_key[target_key].append(ssim_score)

    avg_lpips = np.mean(lpips_scores)
    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)

    avg_lpips_per_key = {key: np.mean(lpips_scores_per_key[key]) for key in list_target_keys}
    avg_psnr_per_key = {key: np.mean(psnr_scores_per_key[key]) for key in list_target_keys}
    avg_ssim_per_key = {key: np.mean(ssim_scores_per_key[key]) for key in list_target_keys}

    logger.info(f"Validation metrics: LPIPS: {avg_lpips}, PSNR: {avg_psnr}, SSIM: {avg_ssim}")

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            tracker.writer.add_scalar("val_lpips", avg_lpips, epoch)
            tracker.writer.add_scalar("val_psnr", avg_psnr, epoch)
            tracker.writer.add_scalar("val_ssim", avg_ssim, epoch)
            for key in list_target_keys:
                tracker.writer.add_scalar(f"val_lpips_{key}", avg_lpips_per_key[key], epoch)
                tracker.writer.add_scalar(f"val_psnr_{key}", avg_psnr_per_key[key], epoch)
                tracker.writer.add_scalar(f"val_ssim_{key}", avg_ssim_per_key[key], epoch)
        if tracker.name == "wandb":
            tracker.log({
                "val_lpips": avg_lpips,
                "val_psnr": avg_psnr,
                "val_ssim": avg_ssim,
            })
            for key in list_target_keys:
                tracker.log({
                    f"val_lpips_{key}": avg_lpips_per_key[key],
                    f"val_psnr_{key}": avg_psnr_per_key[key],
                    f"val_ssim_{key}": avg_ssim_per_key[key],
                })

    num_display_images = min(cfg.num_validation_images, cfg.num_validation_images_display)
    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images[:num_display_images]])
            tracker.writer.add_images(f"{phase_name}_output", np_images, epoch, dataformats="NHWC")
            for cond in cfg.condition_list.split(','):
                np_images = np.stack([np.asarray(img) for img in condition_images[cond][:num_display_images]])
                tracker.writer.add_images(f"{phase_name}_{cond}_cond", np_images, epoch, dataformats="NCHW")
            np_images = np.stack([np.asarray(img) for img in target_images[:num_display_images]])
            tracker.writer.add_images(f"{phase_name}_target", np_images, epoch, dataformats="NCHW")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}") for i, image in enumerate(images[:num_display_images])
                    ]
                }
            )

    del pipeline
    del lpips_fn
    free_memory()

    return images

def update_cfg(cfg: TrainConfig):

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != cfg.local_rank:
        cfg.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if cfg.non_ema_revision is None:
        cfg.non_ema_revision = cfg.revision

    return cfg

def train(cfg: TrainConfig):
    # if cfg.report_to == "wandb" and cfg.hub_token is not None:
    #     raise ValueError(
    #         "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
    #         " Please use `huggingface-cli login` to authenticate with the Hub."
    #     )

    if torch.backends.mps.is_available() and cfg.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(cfg.output_dir, cfg.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=cfg.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(gradient_as_bucket_view=True)
    hf_ds_config={
            "zero_optimization": {
                "stage": 2,
                "overlap_comm": True,
                # 'offload_optimizer': {
                #     'device': 'cpu',
                #     "pin_memory": True,
                # },
                }, 
            "train_micro_batch_size_per_gpu": cfg.train_batch_size,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        }
    if cfg.trainable_precision in ["fp16", "bf16"]:
        hf_ds_config[cfg.trainable_precision] = {"enabled": True}  # Enable mixed precision training
    ds_plugin = DeepSpeedPlugin(
       hf_ds_config=hf_ds_config, 
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.trainable_precision,
        log_with=cfg.report_to,
        project_config=accelerator_project_config,
        deepspeed_plugin=ds_plugin,
        kwargs_handlers=[kwargs],
    )

    print(f"Number of GPUs: {accelerator.num_processes}")
    
    if cfg.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
    
    if accelerator.is_main_process:
        wandb.init(
            project=os.path.basename(cfg.output_dir),         # name of the project (can be created automatically)
            name=time.strftime("%Y-%m-%d_%H-%M-%S"),             # optional: run name
        )
    #accelerator.wait_for_everyone()
    
    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if cfg.output_dir is not None:
            os.makedirs(cfg.output_dir, exist_ok=True)
    
    # Load scheduler and models
    noise_scheduler = tu.get_noise_scheduler(cfg)
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    tokenizers, text_encoders = tu.get_tokenizers_and_text_encoders(cfg)

    vae = AutoencoderKL.from_pretrained(
        cfg.pretrained_model_name_or_path,
        subfolder="vae",
        revision=cfg.revision,
        variant=cfg.variant,
    )

    transformer = tu.get_transformer(cfg)
    #Modify the transformer to allow additional input channels for conditions
    add_condition_input(transformer,len(cfg.condition_list.split(',')))
    controlnet = None #TODO PROPERLY IMPLEMENT CONTROLNET

    vae.requires_grad_(False)

    for text_encoder in text_encoders:
        text_encoder.requires_grad_(False)

    # We allow different precision for trainable and frozen parameters
    # Even if mixed precision training is enabled, we still need to input float32 to the model
    frozen_weight_dtype = torch.float32
    if cfg.frozen_precision == "fp16":
        frozen_weight_dtype = torch.float16
    elif cfg.frozen_precision == "bf16":
        frozen_weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=frozen_weight_dtype)
    for text_encoder in text_encoders:
        text_encoder.to(accelerator.device, dtype=frozen_weight_dtype)
    if cfg.controlnet:
        if cfg.trainable_precision == "fp16":
            transformer.to(accelerator.device, dtype=torch.float16)
        elif cfg.trainable_precision == "bf16":
            transformer.to(accelerator.device, dtype=torch.bfloat16)
        else:
            transformer.to(accelerator.device, dtype=torch.float32)
        

    if cfg.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for i, model in enumerate(models):
                if isinstance(unwrap_model(model), SD3Transformer2DModel) or isinstance(unwrap_model(model), FluxTransformer2DModel):
                    unwrap_model(model).save_pretrained(os.path.join(output_dir, "transformer"))
                elif isinstance(unwrap_model(model), SD3ControlNetModel):
                    unwrap_model(model).save_pretrained(os.path.join(output_dir, "controlnet"))
                elif isinstance(unwrap_model(model), (CLIPTextModelWithProjection, T5EncoderModel)):
                    if isinstance(unwrap_model(model), CLIPTextModelWithProjection):
                        hidden_size = unwrap_model(model).config.hidden_size
                        if hidden_size == 768:
                            unwrap_model(model).save_pretrained(os.path.join(output_dir, "text_encoder"))
                        elif hidden_size == 1280:
                            unwrap_model(model).save_pretrained(os.path.join(output_dir, "text_encoder_2"))
                    else:
                        unwrap_model(model).save_pretrained(os.path.join(output_dir, "text_encoder_3"))
                else:
                    raise ValueError(f"Wrong model supplied: {type(model)=}.")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

    def load_model_hook(models, input_dir):
        for _ in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            if isinstance(unwrap_model(model), SD3Transformer2DModel):
                load_model = SD3Transformer2DModel.from_pretrained(input_dir, subfolder="transformer")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
            elif isinstance(unwrap_model(model), SD3ControlNetModel):
                load_model = SD3ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
            elif isinstance(unwrap_model(model), FluxTransformer2DModel):
                load_model = FluxTransformer2DModel.from_pretrained(input_dir, subfolder="transformer")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
            elif isinstance(unwrap_model(model), (CLIPTextModelWithProjection, T5EncoderModel)):
                try:
                    load_model = CLIPTextModelWithProjection.from_pretrained(input_dir, subfolder="text_encoder")
                    model(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                except Exception:
                    try:
                        load_model = CLIPTextModelWithProjection.from_pretrained(input_dir, subfolder="text_encoder_2")
                        model(**load_model.config)
                        model.load_state_dict(load_model.state_dict())
                    except Exception:
                        try:
                            load_model = T5EncoderModel.from_pretrained(input_dir, subfolder="text_encoder_3")
                            model(**load_model.config)
                            model.load_state_dict(load_model.state_dict())
                        except Exception:
                            raise ValueError(f"Couldn't load the model of type: ({type(model)}).")
            else:
                raise ValueError(f"Unsupported model found: {type(model)=}")

            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if cfg.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    opt_cfg = cfg.optimizer_cfg
    if opt_cfg.scale_lr:
        opt_cfg.learning_rate = (
            opt_cfg.learning_rate * cfg.gradient_accumulation_steps * cfg.train_batch_size * accelerator.num_processes
        )

    # Optimization parameters
    if cfg.controlnet:
        controlnet_parameters_with_lr = {"params": controlnet.parameters(), "lr": opt_cfg.learning_rate}
        params_to_optimize = [controlnet_parameters_with_lr]
    else:
        transformer_parameters_with_lr = {"params": transformer.parameters(), "lr": opt_cfg.learning_rate}
        params_to_optimize = [transformer_parameters_with_lr]

    # Optimizer creation
    optimizer = tu.get_optimizer(opt_cfg, params_to_optimize,logger)

    if cfg.lotus_formulation and 'composed_image' not in cfg.targets:
        print(f"Warning: Lotus formulation is enabled but composed_image is not in the targets. Adding it to the targets.")
        cfg.targets = cfg.targets + ',composed_image'

    required_layers = cfg.condition_list.split(',')
    target_layers = cfg.targets.split(',')

    # Prepare the target and prompt embeddings which are static in our case
    #target_and_prompt_dict, prompt_embeds_dict, pooled_prompt_embeds_dict, text_ids = prepare_target_and_prompt(cfg,tu.encode_prompt,text_encoders,tokenizers,accelerator.device)
    target_and_prompt_dict, prompt_embeds_dict, pooled_prompt_embeds_dict, text_ids = prepare_target_and_prompt_ev(cfg,tu.encode_prompt,text_encoders,tokenizers,accelerator.device)

    # Clear the memory here
    for tokenizer in tokenizers:
        del tokenizer
    for text_encoder in text_encoders:
        text_encoder = text_encoder.cpu()
        del text_encoder
    del tokenizers, text_encoders
    free_memory()

    # Dataset and DataLoaders creation:
    train_datasets = []
    val_datasets = []
    sampling_weights = []
    manager = Manager()
    shared_dataset_dict = manager.dict()
    shared_dataset_dict['max_resolution'] = cfg.fine_resolution**2 
    for dataset_cfg in cfg.dataset_cfg:
        DatasetClass = tu.get_dataset_class(dataset_cfg)
        dataset = DatasetClass(dataset_cfg,required_layers=required_layers,target_layers=target_layers)
        train_datasets.append(dataset)
        sampling_weights.extend([dataset_cfg.sampling_weight] * len(dataset))
        #Validation counterpart, we require all layers for validation
        dataset = DatasetClass(dataset_cfg, split="val", required_layers=list(set(required_layers+target_layers)))
        val_datasets.append(dataset)
        
    #Spatial dataset:
    spatial_test = False
    if spatial_test:
        from datasets.dataset_spheres import SpheresAugConfig, SpheresDatasetConfig, SpheresDataset
        cfg_spatial = SpheresDatasetConfig(
            data_dir = "/root/Data/test_sets/synthetic/temple_multiple_fixed",
            augmentation_cfg=SpheresAugConfig(
                horizontal_flip = False,
                noise=False,
                vignetting=False,
                auto_exposure=False,
                haze=False,
                white_balance=False
            ),
        )
        
        spatial_dataset = SpheresDataset(cfg=cfg_spatial, split='objects', target_layers=target_layers)
        spatial_dataset.known_ev = 0

    train_dataset = CustomConcatDataset(train_datasets)
    val_dataset = CustomConcatDataset(val_datasets)

    train_sampler = WeightedRandomSampler(sampling_weights, num_samples=cfg.train_iter_per_epoch, replacement=True)
    train_batch_sampler = RandomSizeBatchSampler(
        train_sampler,
        batch_size=cfg.train_batch_size,
        shared_dict=shared_dataset_dict
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=cfg.dataloader_num_workers,
        prefetch_factor=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        #persistent_workers=True, #Can't use persistent worker otherwise dataset is not reinitialized and won't change size
    )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)

    lr_scheduler = get_scheduler(
        opt_cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=opt_cfg.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=cfg.max_train_steps * accelerator.num_processes,
        num_cycles=opt_cfg.lr_num_cycles,
        power=opt_cfg.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if cfg.controlnet:
        controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            controlnet, optimizer, train_dataloader, lr_scheduler
        )
    else:
        transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_dataloader, lr_scheduler
        )
    if cfg.compile_network:
        print("Compiling network, this will slow down drastically the early steps as the network is compiled for different input sizes.")
        transformer.compile(compile_kwargs={})

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = cfg_to_dict(cfg)
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        tracker_project_name = f"{cfg.tracker_project_name}_{current_time}"
        accelerator.init_trackers(tracker_project_name, tracker_config)

    # Train!
    total_batch_size = cfg.train_batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Instantaneous batch size per device = {cfg.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfg.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            path = os.path.basename(cfg.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(cfg.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{cfg.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            cfg.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(cfg.output_dir, path))
            #If the base learning rate changed comapred to previous run we ensure it is set correctly
            #This is usefull when extending a finished training
            lr_scheduler.scheduler.base_lrs[0]=cfg.optimizer_cfg.learning_rate
            optimizer.param_groups[0]['initial_lr']=cfg.optimizer_cfg.learning_rate
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, cfg.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    if accelerator.is_main_process and cfg.profile:
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=64, warmup=8, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(cfg.output_dir,"logs",tracker_project_name)),
            #record_shapes=True,
            #with_stack=True
        )
        prof.start()

    ema_loss = None
    ema_rate = 0.999
    epoch = first_epoch
    batch_postprocess_args = DatasetClass.get_gpu_postprocessing_args(device=accelerator.device)
    while global_step < cfg.max_train_steps: 
        if cfg.controlnet:
            controlnet.train()
        else:
            transformer.train()
            
        # ###### TEMP
        # if accelerator.is_main_process:
        #     if  epoch % cfg.validation_epochs == cfg.validation_epochs - 1:
        #         # create pipeline

        #         pipeline = tu.get_val_pipeline(cfg,vae,transformer,controlnet,accelerator,frozen_weight_dtype) 

        #         images = log_validation(
        #             pipeline=pipeline,
        #             cfg=cfg,
        #             accelerator=accelerator,
        #             target_and_prompt_dict=target_and_prompt_dict,
        #             prompt_embeds_dict=prompt_embeds_dict,
        #             pooled_prompt_embeds_dict=pooled_prompt_embeds_dict,
        #             epoch=epoch,
        #             validation_dataset=val_dataset,
        #             batch_postprocess_args=batch_postprocess_args,
        #             DatasetClass=DatasetClass,
        #         )
                
        #         free_memory()
                
        # ########

        for step, batch in enumerate(train_dataloader):
            if batch is None:
                continue
            models_to_accumulate = [controlnet] if cfg.controlnet else [transformer]
            #We allow some processing of the batch to happen on the GPU
            with torch.no_grad():
                DatasetClass.gpu_batch_postprocess(batch=batch, **batch_postprocess_args)

            with accelerator.accumulate(models_to_accumulate):
                
                input_dtype = controlnet.dtype if cfg.controlnet else transformer.dtype
                # Convert images to latent space
                #Get the conditions, most likely the input images and alpha masks
                conditions = [batch[k] for k in cfg.condition_list.split(',')]
                # if accelerator.is_main_process:
                #     print("depth size:", conditions[1].shape)
                #     from ezexr import imsave
                #     depth_np = ((conditions[1][0]+1)/2).cpu().permute(1,2,0).numpy()
                #     imsave("train_depth.exr", depth_np)
                #     exit()
                if conditions[0].dim() == 5:
                    b,l,c,h,w = conditions[0].shape
                else:
                    b,c,h,w = conditions[0].shape
                    l = 1
                conditions = [condition.view(b*l,c,h,w) for condition in conditions]
            
                
                # Get target key for each batch element. i.e the type of output we want to generate
                target_keys = batch['target_layer'] #random.choices(list(target_and_prompt_dict.keys()), k=cfg.train_batch_size) 
                
                # Get the pixel value for each target key. Concat them to get the pixel values for the batch forming the target tensor
                pixel_values = torch.cat([batch[key][batch_sample_id:batch_sample_id+1].view(l,c,h,w) for batch_sample_id,key in enumerate(target_keys)], dim=0).to(dtype=vae.dtype)
                
                # Similarly get the prompt for each target key and concatenate them to get the prompts embedings for the batch
                
                if batch['ev'] is not None:
                    prompt_embeds = torch.cat([prompt_embeds_dict[key][batch['ev'].item()] for key in target_keys], dim=0).to(dtype=input_dtype)
                    pooled_prompt_embeds = torch.cat([pooled_prompt_embeds_dict[key][batch['ev'].item()] for key in target_keys], dim=0).to(dtype=input_dtype)
                else:
                    prompt_embeds = torch.cat([prompt_embeds_dict[key][0] for key in target_keys], dim=0).to(dtype=input_dtype)
                    pooled_prompt_embeds = torch.cat([pooled_prompt_embeds_dict[key][0] for key in target_keys], dim=0).to(dtype=input_dtype)
                
                #repeat for each 'l', making sure we interleave correctly
                prompt_embeds = prompt_embeds[:,None].repeat(1,l,1,1).reshape(b*l,prompt_embeds.shape[-2],prompt_embeds.shape[-1])
                pooled_prompt_embeds = pooled_prompt_embeds[:,None].repeat(1,l,1).reshape(b*l,pooled_prompt_embeds.shape[-1])
                if 'FLUX' in cfg.pretrained_model_name_or_path:
                    text_ids = text_ids.to(dtype=input_dtype)
                else:
                    text_ids = None

                # Convert conditions to latent space
                latent_conditions = [encode_with_vae(vae, condition, weight_dtype=input_dtype) for condition in conditions]
                
                #Randomly drop the image condition
                if random.random() < 0.1:
                    latent_conditions[0] = torch.zeros_like(latent_conditions[0], dtype=latent_conditions[0].dtype, device=latent_conditions[0].device)

                # Convert images to latent space
                model_input = encode_with_vae(vae,pixel_values, weight_dtype=input_dtype)
                vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input.view((b,l)+model_input.shape[-3:])[:,0:1]).repeat(1,l,1,1,1).reshape(model_input.shape)
                bsz = b 

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                if cfg.lotus_formulation:
                    potential_timesteps,_ = retrieve_timesteps(noise_scheduler_copy, cfg.lotus_num_steps)
                    indices = torch.randint(0, len(potential_timesteps), (bsz,))
                    timesteps = potential_timesteps[indices].to(device=model_input.device)
                else:
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=cfg.weighting_scheme,
                        batch_size=bsz,
                        logit_mean=cfg.logit_mean,
                        logit_std=cfg.logit_std,
                        mode_scale=cfg.mode_scale,
                    )
                    #u = torch.zeros_like(u) #Sample only the first timestep
                    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                    indices = torch.clamp(indices,0,len(noise_scheduler_copy.timesteps)-1) # Very occasionally the index is out of bounds
                    timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)
                    timesteps = timesteps.view(b,1).repeat(1,l).view(b*l)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                model_pred = tu.get_model_pred(
                    cfg,
                    transformer,
                    controlnet,
                    latent_conditions,
                    noisy_model_input,
                    prompt_embeds,
                    pooled_prompt_embeds,
                    text_ids,
                    timesteps,
                    input_dtype,
                    vae_scale_factor)

                # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                # Preconditioning of the model outputs.
                if cfg.precondition_outputs:
                    model_pred = model_pred * (-sigmas) + noisy_model_input

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=cfg.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                if cfg.precondition_outputs:
                    target = model_input
                else:
                    target = noise - model_input

                model_pred = model_pred.float().view((b,l)+model_pred.shape[-3:])
                target = target.float().view((b,l)+target.shape[-3:])
                weighting = weighting.float().view(b,l,1,1,1)
                
                if cfg.sphere_weighting:
                    sphere_mask = batch["sphere_mask"].to(device=model_input.device, dtype=model_input.dtype)
                    sphere_mask = torch.nn.functional.interpolate(
                        sphere_mask, size=model_pred.shape[-2:], mode="area"
                    )
                    sphere_mask = sphere_mask[:,:,0,:,:] if sphere_mask.ndim == 5 else sphere_mask[:,0,:,:]
                    sphere_mask = sphere_mask.view((b,l,1)+model_pred.shape[-2:])
                    weighting = torch.where(sphere_mask > 0.5, weighting*10, weighting) # We multiply the weighting by 10 for the sphere mask

                # Compute regular loss.
                l2_diff_loss = torch.mean(
                    (weighting * (model_pred - target) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                l2_diff_loss = l2_diff_loss.mean()
                loss = l2_diff_loss
                
                #Consistency loss for nomals and intrinsic images
                if cfg.consistency_loss and l>1:
                    consistency_loss = torch.tensor(0.0, device=accelerator.device)
                    for batch_sample_id,tgt_key in enumerate(target_keys):
                        if 'intrinsic' in tgt_key or 'normal' in tgt_key:
                            consistency_loss += 0.2*torch.nn.functional.mse_loss(model_pred[batch_sample_id][0],model_pred[batch_sample_id][1])
                    loss += consistency_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = controlnet.parameters() if cfg.controlnet else transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, opt_cfg.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                    if global_step % cfg.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if cfg.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(cfg.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= cfg.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - cfg.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(cfg.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            loss_value = loss.item()
            ema_loss = ema_rate*ema_loss + (1-ema_rate)*loss_value if ema_loss is not None else loss_value
            logs = {
                "loss": loss_value,
                "ema_loss": ema_loss,
                "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            logs["l2_diff_loss"] = l2_diff_loss.item()
            if cfg.consistency_loss and l>1:
                logs["consistency_loss"] = consistency_loss.item()
            # if lpips_loss is not None:
            #     logs["lpips_loss"] = lpips_loss.item()
            
            if accelerator.is_main_process:
                if step % cfg.save_img_every_n_steps == 0:
                #if step % 10 == 0:
                    # gt = decode_with_vae(vae, model_input, weight_dtype=torch.half)[0]
                    # gen_img = decode_with_vae(vae, model_pred[0,:,:,:,:], weight_dtype=torch.half)[0]
                    # condition = decode_with_vae(vae, latent_conditions[0], weight_dtype=torch.half)[0]
                    # # gen_img = gen_img[0,:,:,:]
                    # # condition = condition[0,:,:,:]
                    # # gt = gt[0,:,:,:]
                    
                    # print(gt.min(), gt.max())
                    # grid = torch.stack([condition, gt, gen_img], dim=0)
                    # grid = torchvision.utils.make_grid(grid, nrow=3, normalize=True, value_range=(-1, 1))
                    # wandb.log({"train_images": wandb.Image(grid, caption=f"Step {global_step}")}, step=global_step)
                    pipeline = tu.get_val_pipeline(cfg,vae,transformer,controlnet,accelerator,frozen_weight_dtype) 
                    log_first_train(
                        pipeline=pipeline,
                        cfg=cfg,
                        accelerator=accelerator,
                        prompt_embeds_dict=prompt_embeds_dict,
                        pooled_prompt_embeds_dict=pooled_prompt_embeds_dict,
                        validation_dataset=val_dataset,
                        global_step=global_step,
                    )
                    if spatial_test:
                        log_spatial_test(
                            pipeline=pipeline,
                            cfg=cfg,
                            accelerator=accelerator,
                            spatial_dataset=spatial_dataset,
                            prompt_embeds_dict=prompt_embeds_dict,
                            pooled_prompt_embeds_dict=pooled_prompt_embeds_dict,
                            global_step=global_step,
                        )
                                
                        

            if cfg.coarse_to_fine and global_step >0:
                #We keep the last 20% of the training steps at the fine resolution
                r_t = (global_step/(0.8*cfg.max_train_steps))**2 # We square the progression to grow slowly at the beginning and faster at the end
                f_c_ratio = cfg.fine_resolution/cfg.coarse_resolution
                step_resolution = int(min(cfg.fine_resolution,(f_c_ratio**r_t)*cfg.coarse_resolution))
                if step_resolution**2 < 0.5*cfg.fine_resolution**2 and train_dataloader.batch_sampler.batch_size != 2*cfg.train_batch_size:
                    #We can use a bigger batch size
                    train_dataloader.batch_sampler.batch_sampler.batch_size = 2*cfg.train_batch_size
                    train_dataloader.batch_sampler.batch_size = 2*cfg.train_batch_size
                elif step_resolution**2 >= 0.5*cfg.fine_resolution**2 and train_dataloader.batch_sampler.batch_size != cfg.train_batch_size:
                    train_dataloader.batch_sampler.batch_sampler.batch_size = cfg.train_batch_size
                    train_dataloader.batch_sampler.batch_size = cfg.train_batch_size
                previous_resolution = shared_dataset_dict['max_resolution']
                shared_dataset_dict['max_resolution'] = step_resolution**2
                if previous_resolution != step_resolution**2:
                    free_memory()

            logs["resolution"] = shared_dataset_dict['max_resolution']**0.5

            accelerator.log(logs, step=global_step)

            if cfg.profile and accelerator.is_main_process:
                prof.step()
                if step >= 64+8+5:
                    prof.stop()
                    quit()
            
            if global_step >= cfg.max_train_steps:
                break

        if accelerator.is_main_process:
            if  epoch % cfg.validation_epochs == cfg.validation_epochs - 1:
                # create pipeline

                pipeline = tu.get_val_pipeline(cfg,vae,transformer,controlnet,accelerator,frozen_weight_dtype) 

                images = log_validation(
                    pipeline=pipeline,
                    cfg=cfg,
                    accelerator=accelerator,
                    target_and_prompt_dict=target_and_prompt_dict,
                    prompt_embeds_dict=prompt_embeds_dict,
                    pooled_prompt_embeds_dict=pooled_prompt_embeds_dict,
                    epoch=epoch,
                    validation_dataset=val_dataset,
                    batch_postprocess_args=batch_postprocess_args,
                    DatasetClass=DatasetClass,
                )
                
                free_memory()
        epoch += 1

    # Save the layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)

        pipeline = ConditionalSD3Pipeline.from_pretrained(
            cfg.pretrained_model_name_or_path, transformer=transformer
        )

        # save the pipeline
        pipeline.save_pretrained(cfg.output_dir)

        # Final inference
        # Load previous pipeline
        pipeline = ConditionalSD3Pipeline.from_pretrained(
            cfg.output_dir,
            revision=cfg.revision,
            variant=cfg.variant,
            torch_dtype=frozen_weight_dtype,
        )

        # run inference
        # images = []
        # if cfg.num_validation_images > 0:
        #     pipeline_args = {"prompt": ""}
        #     images = log_validation(
        #         pipeline=pipeline,
        #         cfg=cfg,
        #         accelerator=accelerator,
        #         pipeline_args=pipeline_args,
        #         epoch=epoch,
        #         is_final_validation=True,
        #         validation_dataset=val_dataset,
        #     )

    accelerator.end_training()


if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(default_train_configs)
    train(config)
