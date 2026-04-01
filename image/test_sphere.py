import tyro
from test_config import TestConfig, default_test_configs
from conditional_pipelines import ConditionalSD3Pipeline, ConditionalFluxPipeline
from datasets.data_utils import sRGB_to_Lin,Lin_to_sRGB,Log_to_Lin
from PIL import Image
import pyexr
import os
import torch
from utils import encode_with_vae, prepare_target_and_prompt, prepare_target_and_prompt_ev, add_condition_input, TARGET_TO_PROMPT
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
from diffusers import SD3Transformer2DModel,FluxTransformer2DModel
from train_utils_min import encode_prompt,get_tokenizers_and_text_encoders 
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

import torchvision

import pickle
from datasets.dataset_spheres_test import SpheresAugConfig, SpheresDatasetConfig, SpheresDataset

def center_crop_input(img,max_resolution):
    #Center crop so both width and height are divisible by 16
    crop_h = max(img.shape[-2]%16,img.shape[-2]-max_resolution)
    crop_w = max(img.shape[-1]%16,img.shape[-1]-max_resolution)
    img = transforms.CenterCrop((img.shape[-2]-crop_h, img.shape[-1]-crop_w))(img)
    return img

def make_input_complient(img):
    #Center crop so both width and height are divisible by 16
    crop_h = img.shape[-2] % 16
    crop_w = img.shape[-1] % 16
    img = transforms.CenterCrop((img.shape[-2]-crop_h, img.shape[-1]-crop_w))(img)
    return img

def deal_with_input(path, targets=None):
    cfg = SpheresDatasetConfig(
        data_dir = path,
        augmentation_cfg=SpheresAugConfig(
            horizontal_flip = False,
            noise=False,
            vignetting=False,
            auto_exposure=False,
            haze=False,
            white_balance=False
        ),
    )
    
    dataset = SpheresDataset(cfg=cfg, split='objects', target_layers=targets.split(','))
    
    return dataset
    

@torch.no_grad()
def test(cfg: TestConfig):

    weight_dtype = torch.float32
    if cfg.test_model_dtype == "fp16":
        weight_dtype = torch.float16

    PipelineClass = ConditionalSD3Pipeline if "stable-diffusion-3" in cfg.pretrained_model_name_or_path else ConditionalFluxPipeline
    TransformerClass = SD3Transformer2DModel if "stable-diffusion-3" in cfg.pretrained_model_name_or_path else FluxTransformer2DModel

    if os.path.isdir(cfg.model_path):
        pipeline = PipelineClass.from_pretrained(cfg.model_path)
    else:
        transformer = TransformerClass.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="transformer")
        add_condition_input(transformer, cfg.num_input_conditions)
        transformer.load_state_dict(torch.load(cfg.model_path, map_location="cpu", weights_only=False)['module'], strict=True)
        pipeline = PipelineClass.from_pretrained(cfg.pretrained_model_name_or_path, transformer=transformer)

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
    #target_and_prompt_dict, prompt_embeds_dict, pooled_prompt_embeds_dict,text_ids = prepare_target_and_prompt(cfg,encode_prompt,text_encoders,tokenizers,device)
    target_and_prompt_dict, prompt_embeds_dict, pooled_prompt_embeds_dict, text_ids = prepare_target_and_prompt_ev(cfg,encode_prompt,text_encoders,tokenizers,device)

    pipeline.text_encoder = None
    pipeline.text_encoder_2 = None
    if PipelineClass == ConditionalSD3Pipeline:
        pipeline.text_encoder_3 = None
    del tokenizers, text_encoders
    torch.cuda.empty_cache()

    print("Uploading pipeline to device...")
    pipeline.to(device,dtype=weight_dtype)

    input_dir = cfg.input_path if os.path.isdir(cfg.input_path) else os.path.dirname(cfg.input_path)
    print(f"Input directory: {input_dir}")
    print(f"{cfg.input_path}")

    dataset = deal_with_input(cfg.input_path, cfg.targets)

    if cfg.pred_ev:
        #evs = [-8, -4, 0]
        evs = [-9, -6, -3, 0]
    else:
        evs = [0]

    for ev in evs[::-1]:
        print(f"Processing exposure value: {ev}")
        dataset.known_ev = ev
        #Iterate over all .png|.jpg files in the input_path
        print(len(dataset))
        p_bar=tqdm(dataset, desc="Processing images", unit="img")
        for idx, batch in enumerate(p_bar):
            
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
            
            generator = torch.Generator(device=device).manual_seed(0)
                
            output_dir = os.path.join(batch["scene"], "preds")
            output_dir = batch["scene"].replace(input_dir, cfg.output_path)
            os.makedirs(output_dir, exist_ok=True)

            #Save depth
            depth = batch['image_position'].cpu().squeeze(0).permute(1,2,0).numpy() * 0.5 + 0.5
            depth = (depth*255).astype('uint8')
            im = Image.fromarray(depth)
            im.save(os.path.join(output_dir, f"depth.png"))

            #save sphere_mask
            sphere_mask = batch['sphere_mask'].cpu().squeeze(0).permute(1,2,0).numpy()
            sphere_mask = (sphere_mask*255).astype('uint8')
            im = Image.fromarray(sphere_mask)
            im.save(os.path.join(output_dir, f"sphere_mask.png"))

            # #Temp save input conditions
            # for i, condition in enumerate(conditions):
            #     condition = condition.cpu().squeeze(0).permute(1,2,0).numpy() * 0.5 + 0.5
            #     condition = (condition*255).astype('uint8')
            #     im = Image.fromarray(condition)
            #     im.save(os.path.join(output_dir, f"input_{cfg.condition_list.split(',')[i]}.png"))

            p_bar.set_description(str(idx)+f" computing", refresh=True)

            depth = batch["position"]
            depth = depth / 2 + 0.5  #Normalize to 0-1
            filename_depth = os.path.join(output_dir, f"depth_{ev}.png")
            torchvision.utils.save_image(depth, filename_depth)

            for target in target_and_prompt_dict.keys():
                    
                    pipeline_args = {
                        "prompt_embeds": prompt_embeds_dict[target][ev],
                        "pooled_prompt_embeds": pooled_prompt_embeds_dict[target][ev],
                        "latent_conditions": latent_conditions,
                        "width": w,
                        "height": h,
                        "guidance_scale": 1.0,
                        "generator": generator,
                        "num_inference_steps": cfg.num_inference_steps,
                        "sigmas": np.array(cfg.sigmas) if cfg.sigmas is not None else None,
                        "output_type": "pt"
                    }
                    output_img = pipeline(**pipeline_args).images[0]

                    filename = os.path.join(output_dir, f"{target}_{ev}.png")
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    Image.fromarray((output_img.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)).save(filename)

            print("Done with image ", idx)
        print("Done with ev ", ev)
    print("All done.")


if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(default_test_configs)
    with torch.no_grad():
        test(config)