import math
import random
import numpy as np
import torch
from .datasets.dataset_relighting import TARGET_TO_PROMPT as TARGET_TO_PROMPT_relighting
from .datasets.dataset_light_estimation import TARGET_TO_PROMPT as TARGET_TO_PROMPT_light_estimation
from .datasets.dataset_supres import TARGET_TO_PROMPT as TARGET_TO_PROMPT_supres
from .datasets.dataset_spheres import TARGET_TO_PROMPT as TARGET_TO_PROMPT_spheres
from diffusers import (
    SD3Transformer2DModel,
    FluxTransformer2DModel,
    SD3ControlNetModel,
)

#Merge the dictionaries
TARGET_TO_PROMPT = {**TARGET_TO_PROMPT_relighting, **TARGET_TO_PROMPT_supres, **TARGET_TO_PROMPT_light_estimation, **TARGET_TO_PROMPT_spheres}

def cfg_to_dict(cfg):
    dict_cfg = dict(vars(cfg))
    # Recursively convert dataclasses to dictionaries
    for key, value in dict_cfg.items():
        if hasattr(value, "__dataclass_fields__"):
            sub_dict_cfg = cfg_to_dict(value)
            #Merge the dictionary with the current one adding the key as a prefix
            dict_cfg = {**dict_cfg, **{f"{key}.{sub_key}": sub_value for sub_key, sub_value in sub_dict_cfg.items()}}
            del dict_cfg[key]
        #value is a list
        elif isinstance(value, list):
            #Convert list to single string
            if not hasattr(value[0], "__dataclass_fields__"):
                dict_cfg[key] = ', '.join(value)
            else:
                for i, sub_value in enumerate(value):
                    sub_dict_cfg = cfg_to_dict(sub_value)
                    #Merge the dictionary with the current one adding the key as a prefix
                    dict_cfg = {**dict_cfg, **{f"{key}[{i}].{sub_key}": sub_value for sub_key, sub_value in sub_dict_cfg.items()}}
                del dict_cfg[key]
    
    return dict_cfg

def encode_with_vae(vae,_inp,vae_nondeterministic=False,weight_dtype=None):

    cast_inp = _inp.to(vae.dtype)

    if vae_nondeterministic:
        _latents = vae.encode(cast_inp).latent_dist.sample()
    else:
        h = vae.encoder(cast_inp)
        mean, logvar = torch.chunk(h, 2, dim=1)
        _latents = mean

    _latents = (_latents - vae.config.shift_factor) * vae.config.scaling_factor
    _latents = _latents.to(weight_dtype)
    return _latents

def decode_with_vae(vae,_latents,weight_dtype=None):

    _latents = _latents.to(vae.dtype)
    _latents = _latents / vae.config.scaling_factor + vae.config.shift_factor

    _out = vae.decode(_latents,return_dict=False)[0]
    _out = _out.to(weight_dtype)
    return _out

def decode_with_cond_decoder(decoder,_latents,condition,scaling_factor,shift_factor,weight_dtype=None):
    
    decoder_dtype = next(iter(decoder.parameters())).dtype
    _latents = _latents.to(decoder_dtype)
    condition = condition.to(decoder_dtype)
    _latents = _latents / scaling_factor + shift_factor

    _out = decoder(_latents,condition)
    _out = _out.to(weight_dtype)
    return _out

def add_condition_input(transformer, ntimes):
    
    if isinstance(transformer,SD3Transformer2DModel):
        original_conv = transformer.pos_embed.proj
        ori_ch = original_conv.in_channels
        new_conv = torch.nn.Conv2d(ori_ch*(1+ntimes), original_conv.out_channels, original_conv.kernel_size, original_conv.stride, original_conv.padding, original_conv.dilation, original_conv.groups)
        new_conv.weight.data[:,-ori_ch:] = 1.0*original_conv.weight.data
        new_conv.weight.data[:,:-ori_ch] = (0.05/ntimes)*torch.cat([original_conv.weight.data for _ in range(ntimes)], dim=1)
        new_conv.bias.data = original_conv.bias.data
        transformer.pos_embed.proj = new_conv
        transformer.config["in_channels"] = ori_ch*(ntimes+1)
        transformer.pos_embed.proj.requires_grad_(True)
    elif isinstance(transformer,FluxTransformer2DModel):
        old_linear = transformer.x_embedder
        ori_ch = old_linear.in_features
        new_linear = torch.nn.Linear(ori_ch*(1+ntimes), old_linear.out_features)
        new_linear.weight.data[:,-ori_ch:] = 1.0*old_linear.weight.data
        new_linear.weight.data[:,:-ori_ch] = 0.02*torch.cat([old_linear.weight.data for _ in range(ntimes)], dim=1)
        new_linear.bias.data = old_linear.bias.data
        transformer.x_embedder = new_linear
        transformer.config["in_channels"] = ori_ch*(ntimes+1)
        transformer.x_embedder.requires_grad_(True)
    elif isinstance(transformer,SD3ControlNetModel):
        #Custom modifications of controlnet init
        new_conv_weights = controlnet.pos_embed.proj.weight.data.repeat(1,2,1,1)*(0.05/ntimes)
        controlnet.pos_embed_input.proj.weight.data = new_conv_weights
        for block in controlnet.controlnet_blocks:
            size_block = block.weight.data.shape[0]
            block.weight.data.copy_(0.05*torch.eye(size_block))
        transformer.requires_grad_(False)
        controlnet.requires_grad_(True)
    else:
        raise NotImplementedError("Only SD3Transformer2DModel, FluxTransformer2DModel and SD3ControlNetModel are supported")
    # if cfg.controlnet:
    #     logger.info("Initializing controlnet weights from transformer")
    #     ntimes = len(cfg.condition_list.split(','))
    #     controlnet = SD3ControlNetModel.from_transformer(
    #         transformer, num_extra_conditioning_channels=16*(ntimes-1)
    #     )

def prepare_target_and_prompt(cfg,encode_prompt,text_encoders,tokenizers,device):
    
    cfg_targets = cfg.targets
    if hasattr(cfg, "max_sequence_length"):
        cfg_max_sequence_length = cfg.max_sequence_length
    else:
        cfg_max_sequence_length = 77

    target_and_prompt_dict = {}
    for tgt in cfg_targets.split(','):
        target_and_prompt_dict[tgt] = TARGET_TO_PROMPT[tgt] 

    def compute_text_embeddings(prompt, text_encoders, tokenizers):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                text_encoders, tokenizers, prompt, cfg_max_sequence_length, cfg
            )
            prompt_embeds = prompt_embeds.to(device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(device)
            if text_ids is not None:
                text_ids = text_ids.to(device)
        return prompt_embeds, pooled_prompt_embeds, text_ids

    prompt_embeds_dict = {}
    pooled_prompt_embeds_dict = {}
    for tgt_key, prompt in target_and_prompt_dict.items():
        pmpt_embeds, pooled_pmpt_embeds, text_ids = compute_text_embeddings(prompt, text_encoders, tokenizers)
        prompt_embeds_dict[tgt_key] = pmpt_embeds
        pooled_prompt_embeds_dict[tgt_key] = pooled_pmpt_embeds
    
    if text_ids is not None:
        text_ids = text_ids[0]

    return target_and_prompt_dict,prompt_embeds_dict, pooled_prompt_embeds_dict, text_ids

def prepare_target_and_prompt_ev(cfg,encode_prompt,text_encoders,tokenizers,device, min_ev=-12):
    
    cfg_targets = cfg.targets
    if hasattr(cfg, "max_sequence_length"):
        cfg_max_sequence_length = cfg.max_sequence_length
    else:
        cfg_max_sequence_length = 77

    target_and_prompt_dict = {}
    for tgt in cfg_targets.split(','):
        target_and_prompt_dict[tgt] = TARGET_TO_PROMPT[tgt] 

    def compute_text_embeddings(prompt, text_encoders, tokenizers):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                text_encoders, tokenizers, prompt, cfg_max_sequence_length, cfg
            )
            prompt_embeds = prompt_embeds.to(device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(device)
            if text_ids is not None:
                text_ids = text_ids.to(device)
        return prompt_embeds, pooled_prompt_embeds, text_ids

    prompt_embeds_dict = {}
    pooled_prompt_embeds_dict = {}
    for tgt_key, prompt in target_and_prompt_dict.items():
        prompt_embeds_dict[tgt_key] = {}
        pooled_prompt_embeds_dict[tgt_key] = {}
        for ev in range(min_ev, 1):
            prompt = f"{prompt} [EV{ev}]"
            pmpt_embeds, pooled_pmpt_embeds, text_ids = compute_text_embeddings(prompt, text_encoders, tokenizers)
            prompt_embeds_dict[tgt_key][ev] = pmpt_embeds
            pooled_prompt_embeds_dict[tgt_key][ev] = pooled_pmpt_embeds
    
    if text_ids is not None:
        text_ids = text_ids[0]

    return target_and_prompt_dict,prompt_embeds_dict, pooled_prompt_embeds_dict, text_ids

def compute_lotus_loss(decoded_pred,pixel_values,loss_fn=None):
    if loss_fn is None:
        #MSE loss
        return torch.nn.functional.mse_loss(decoded_pred.float(), pixel_values.float())
    else:
        return loss_fn(decoded_pred.float(), pixel_values.float())
