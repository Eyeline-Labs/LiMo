from contextlib import contextmanager
import torch
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, PretrainedConfig, T5EncoderModel, T5TokenizerFast
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    FluxTransformer2DModel,
    SD3ControlNetModel,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
)
from .conditional_pipelines import ConditionalSD3Pipeline, ConditionalFluxPipeline


def load_text_encoders_SD3(class_one, class_two, class_three, cfg):
    text_encoder_one = class_one.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="text_encoder", revision=cfg.revision, variant=cfg.variant
    )
    text_encoder_two = class_two.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=cfg.revision, variant=cfg.variant
    )
    text_encoder_three = class_three.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="text_encoder_3", revision=cfg.revision, variant=cfg.variant
    )
    return text_encoder_one, text_encoder_two, text_encoder_three

def load_text_encoders_FLUX(class_one, class_two, cfg):
    text_encoder_one = class_one.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="text_encoder", revision=cfg.revision, variant=cfg.variant
    )
    text_encoder_two = class_two.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=cfg.revision, variant=cfg.variant
    )
    return text_encoder_one, text_encoder_two

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    else:
        raise ValueError(f"{model_class} is not supported.")
    
def get_tokenizers_and_text_encoders_SD3(cfg):
    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        cfg.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=cfg.revision,
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        cfg.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=cfg.revision,
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        cfg.pretrained_model_name_or_path,
        subfolder="tokenizer_3",
        revision=cfg.revision,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        cfg.pretrained_model_name_or_path, cfg.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        cfg.pretrained_model_name_or_path, cfg.revision, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        cfg.pretrained_model_name_or_path, cfg.revision, subfolder="text_encoder_3"
    )

    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders_SD3(
        text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three, cfg
    )
    return [tokenizer_one, tokenizer_two, tokenizer_three], [text_encoder_one, text_encoder_two, text_encoder_three]

def get_tokenizers_and_text_encoders_FLUX(cfg):
        # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        cfg.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=cfg.revision,
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        cfg.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=cfg.revision,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        cfg.pretrained_model_name_or_path, cfg.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        cfg.pretrained_model_name_or_path, cfg.revision, subfolder="text_encoder_2"
    )

    text_encoder_one, text_encoder_two = load_text_encoders_FLUX(
        text_encoder_cls_one, text_encoder_cls_two, cfg
    )

    return [tokenizer_one, tokenizer_two], [text_encoder_one, text_encoder_two]

def get_tokenizers_and_text_encoders(cfg):
    if 'stable-diffusion-3' in cfg.pretrained_model_name_or_path:
        return get_tokenizers_and_text_encoders_SD3(cfg)
    elif 'FLUX' in cfg.pretrained_model_name_or_path:
        return get_tokenizers_and_text_encoders_FLUX(cfg)
    else:
        raise ValueError(f"Model {cfg.pretrained_model_name_or_path} not supported")

def get_noise_scheduler(cfg):
    if 'stable-diffusion-3' in cfg.pretrained_model_name_or_path:
        return FlowMatchEulerDiscreteScheduler(shift=3.0)
    elif 'FLUX' in cfg.pretrained_model_name_or_path:
        return FlowMatchEulerDiscreteScheduler.from_pretrained(
            cfg.pretrained_model_name_or_path, subfolder="scheduler"
        )
    elif 'Wan' in cfg.pretrained_model_name_or_path:
        return None
    else:
        raise ValueError(f"Model {cfg.pretrained_model_name_or_path} not supported")

#Encoding and tokenization functions

def tokenize_prompt(tokenizer, prompt, max_sequence_length):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds

def _encode_prompt_with_clip_SD3(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds

def _encode_prompt_with_clip_FLUX(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds

def encode_prompt_SD3(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for i, (tokenizer, text_encoder) in enumerate(zip(clip_tokenizers, clip_text_encoders)):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip_SD3(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
            text_input_ids=text_input_ids_list[i] if text_input_ids_list else None,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[-1].device,
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    return prompt_embeds, pooled_prompt_embeds, None

def encode_prompt_FLUX(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    dtype = text_encoders[0].dtype
    device = device if device is not None else text_encoders[1].device
    pooled_prompt_embeds = _encode_prompt_with_clip_FLUX(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )

    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )

    text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
    text_ids = text_ids.repeat(num_images_per_prompt, 1, 1)

    return prompt_embeds, pooled_prompt_embeds, text_ids

def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    cfg,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None):
    if 'stable-diffusion-3' in cfg.pretrained_model_name_or_path:
        return encode_prompt_SD3(text_encoders, tokenizers, prompt, max_sequence_length, device, num_images_per_prompt, text_input_ids_list)
    elif 'FLUX' in cfg.pretrained_model_name_or_path:
        return encode_prompt_FLUX(text_encoders, tokenizers, prompt, max_sequence_length, device, num_images_per_prompt, text_input_ids_list)
    else:
        raise ValueError(f"Model {text_encoders[0].config.architectures[0]} not supported")

def get_transformer(cfg):
    if 'stable-diffusion-3' in cfg.pretrained_model_name_or_path:
        return SD3Transformer2DModel.from_pretrained(
            cfg.pretrained_model_name_or_path, subfolder="transformer", revision=cfg.revision, variant=cfg.variant
        )
    elif 'FLUX' in cfg.pretrained_model_name_or_path:
        return FluxTransformer2DModel.from_pretrained(
            cfg.pretrained_model_name_or_path, subfolder="transformer", revision=cfg.revision, variant=cfg.variant
        )
    else:
        raise ValueError(f"Model {cfg.pretrained_model_name_or_path} not supported")

def get_optimizer(opt_cfg,params_to_optimize,logger):
    if not (opt_cfg.type.lower() == "prodigy" or opt_cfg.type.lower() == "adamw" or opt_cfg.type.lower() == "deepspeedadam"):
        logger.warning(
            f"Unsupported choice of optimizer: {opt_cfg.type}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        opt_cfg.type = "adamw"

    if opt_cfg.use_8bit_adam and not opt_cfg.type.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {opt_cfg.type.lower()}"
        )

    if opt_cfg.type.lower() == "adamw":
        if opt_cfg.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(opt_cfg.adam_beta1, opt_cfg.adam_beta2),
            weight_decay=opt_cfg.adam_weight_decay,
            eps=opt_cfg.adam_epsilon,
        )

    if opt_cfg.type.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if opt_cfg.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(opt_cfg.adam_beta1, opt_cfg.adam_beta2),
            beta3=opt_cfg.prodigy_beta3,
            weight_decay=opt_cfg.adam_weight_decay,
            eps=opt_cfg.adam_epsilon,
            decouple=opt_cfg.prodigy_decouple,
            use_bias_correction=opt_cfg.prodigy_use_bias_correction,
            safeguard_warmup=opt_cfg.prodigy_safeguard_warmup,
        )
        
    if opt_cfg.type.lower() == "deepspeedadam":
        try:
            import deepspeed
        except ImportError:
            raise ImportError("To use DeepSpeed Adam, please install the deepspeed library: `pip install deepspeed`")

        optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(
            params_to_optimize,
            lr=opt_cfg.learning_rate,
            betas=(opt_cfg.adam_beta1, opt_cfg.adam_beta2),
            weight_decay=opt_cfg.adam_weight_decay,
            eps=opt_cfg.adam_epsilon,
        )
        
    return optimizer

def get_model_pred(
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
        vae_scale_factor
):
    if cfg.controlnet and 'stable-diffusion-3' in cfg.pretrained_model_name_or_path:
        cond_model_input = torch.cat([*latent_conditions], dim=1).to(dtype=input_dtype)
        noisy_model_input = noisy_model_input.to(dtype=input_dtype)
        control_block_res_samples = controlnet(
            hidden_states=noisy_model_input,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            controlnet_cond=cond_model_input,
            return_dict=False,
        )[0]
        control_block_res_samples = [sample.to(dtype=transformer.dtype) for sample in control_block_res_samples]
        noisy_model_input = noisy_model_input.to(dtype=transformer.dtype)
        # Predict the noise residual
        model_pred = transformer(
            hidden_states=noisy_model_input,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            block_controlnet_hidden_states=control_block_res_samples,
            return_dict=False,
        )[0]
    elif 'stable-diffusion-3' in cfg.pretrained_model_name_or_path:
        #timestep_maps = (timesteps/1000).view(-1,1,1,1).expand((-1,1)+noisy_model_input.shape[2:])
        #cond_and_noisy_model_input = torch.cat([*latent_conditions, timestep_maps, noisy_model_input], dim=1).to(dtype=input_dtype)
        cond_and_noisy_model_input = torch.cat([*latent_conditions, noisy_model_input], dim=1).to(dtype=input_dtype)
        # Predict the noise residual
        model_pred = transformer(
            hidden_states=cond_and_noisy_model_input,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]
    elif 'FLUX' in cfg.pretrained_model_name_or_path:

        latent_image_ids = FluxPipeline._prepare_latent_image_ids(
            noisy_model_input.shape[0],
            noisy_model_input.shape[2] // 2,
            noisy_model_input.shape[3] // 2,
            text_ids.device,
            transformer.dtype,
        )

        cond_and_noisy_model_input = torch.cat([*latent_conditions, noisy_model_input], dim=1).to(dtype=input_dtype)

        packed_noisy_model_input = FluxPipeline._pack_latents(
            cond_and_noisy_model_input,
            batch_size=noisy_model_input.shape[0],
            num_channels_latents=cond_and_noisy_model_input.shape[1],
            height=noisy_model_input.shape[2],
            width=noisy_model_input.shape[3],
        )

        # handle guidance
        # if transformer.config.guidance_embeds:
        #     guidance = torch.tensor([cfg.guidance_scale], device=accelerator.device)
        #     guidance = guidance.expand(model_input.shape[0])
        # else:
        #     guidance = None
        guidance = None

        # Predict the noise residual
        model_pred = transformer(
            hidden_states=packed_noisy_model_input,
            # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
            timestep=timesteps / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]
        # upscaling height & width as discussed in https://github.com/huggingface/diffusers/pull/9257#discussion_r1731108042
        model_pred = FluxPipeline._unpack_latents(
            model_pred,
            height=noisy_model_input.shape[2] * vae_scale_factor,
            width=noisy_model_input.shape[3] * vae_scale_factor,
            vae_scale_factor=vae_scale_factor,
        )

    return model_pred


## Pipeline related stuff

def get_val_pipeline(cfg,vae,transformer,controlnet,accelerator,frozen_weight_dtype):
    if 'stable-diffusion-3' in cfg.pretrained_model_name_or_path:
        return ConditionalSD3Pipeline.from_pretrained(
                    cfg.pretrained_model_name_or_path,
                    vae=vae,
                    #Since tokenizer and text_encoder are not used, we pass None to avoid a slow loading
                    tokenizer=None,tokenizer_2=None,tokenizer_3=None,
                    text_encoder=None,text_encoder_2=None,text_encoder_3=None,
                    transformer=transformer if cfg.controlnet else accelerator.unwrap_model(transformer),
                    revision=cfg.revision,
                    variant=cfg.variant,
                    torch_dtype=frozen_weight_dtype,
                    controlnet=accelerator.unwrap_model(controlnet) if cfg.controlnet else None,
                )
    elif 'FLUX' in cfg.pretrained_model_name_or_path:
        return  ConditionalFluxPipeline.from_pretrained(
                    cfg.pretrained_model_name_or_path,
                    vae=vae,
                    tokenizer=None,tokenizer_2=None,
                    text_encoder=None,text_encoder_2=None,
                    transformer=accelerator.unwrap_model(transformer, keep_fp32_wrapper=False),
                    revision=cfg.revision,
                    variant=cfg.variant,
                    torch_dtype=frozen_weight_dtype,
                )
    else:
        raise ValueError(f"Model {cfg.pretrained_model_name_or_path} not supported")

#### DeepSpeed Zero-3
#from https://github.com/huggingface/diffusers/issues/10743
