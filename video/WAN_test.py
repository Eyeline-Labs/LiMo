import torch, os, json
import sys
sys.path.append("/root/Project/DiffSynth-Studio")
from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig, WanVideoUnit_ConditionsVideoEmbedder
from diffsynth.trainers.utils import DiffusionTrainingModule, VideoDataset, ModelLogger, launch_training_task, wan_parser
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from conditional_utils import add_condition_input
from datasets.dataset_spheres_video import SpheresDatasetConfig, SpheresAugConfig, SpheresDataset, treat_data, treat_data_old

import subprocess
from tqdm import tqdm
from PIL import Image

import copy
class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        condition_list=None,
        targets=None,
        output_dir=None,
        validation_steps=50
    ):
        super().__init__()
        # Load models
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            for i in model_id_with_origin_paths:
                if ":" in i:
                    model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1])]
                else:
                    model_configs += [ModelConfig(path=i)]
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs)
        
        self.condition_list = condition_list.split(",") if condition_list is not None else []
         
        # Add linear layer for added inputs
        add_condition_input(self.pipe.dit, len(condition_list.split(',')))
        # Add condition preprocessing pipeline
        self.pipe.units.append(WanVideoUnit_ConditionsVideoEmbedder(condition_list.split(",")))
        
        # Reset training scheduler
        self.pipe.scheduler_train.set_timesteps(1000, training=True)
        
        # Freeze untrainable models
        self.pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        
        # Add LoRA to the base models
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(self.pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank
            )
            if lora_checkpoint is not None:
                state_dict = load_state_dict(lora_checkpoint)
                state_dict = self.mapping_lora_state_dict(state_dict)
                load_result = model.load_state_dict(state_dict, strict=False)
                if len(load_result[1]) > 0:
                    print(f"Warning, LoRA key mismatch! Unexpected keys in LoRA checkpoint: {load_result[1]}")
            setattr(self.pipe, lora_base_model, model)
            
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.validation_steps = validation_steps
        
        
        
    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        
        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].shape[1],
            "width": data["video"][0].shape[2],
            "num_frames": len(data["video"]),
            #"num_conditions": len(self.condition_list),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            "path": data["path"]
        }
        for condition in self.condition_list:
            inputs_shared[condition] = data[condition]
        
        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    def data_to_device(self, data, device):
        for key in data.keys():
            if isinstance(data[key], list):
                if isinstance(data[key][0], torch.Tensor):
                    for i in range(len(data[key])):
                        data[key][i] = data[key][i].to(device)
        return data
    
    def validation(self, data, inputs=None):
        data = treat_data(data)
        #data = self.data_to_device(data, self.pipe.device)
        if inputs is None: inputs = self.forward_preprocess(data)
        #models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        video = self.pipe(data["prompt"], num_inference_steps=self.validation_steps, inputs_shared=inputs)
        return video
    
    def forward(self, data, inputs=None):
        data = treat_data(data)
        #data = self.data_to_device(data, self.pipe.device)
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss

def data_to_device(data, device):
    for key in data.keys():
        if isinstance(data[key], list):
            if isinstance(data[key][0], torch.Tensor):
                for i in range(len(data[key])):
                    data[key][i] = data[key][i].to(device)
    return data
    
def forward_preprocess(data, pipe, condition_list):
    # CFG-sensitive parameters
    inputs_posi = {"prompt": data["prompt"]}
    inputs_nega = {}
    
    # CFG-unsensitive parameters
    inputs_shared = {
        # Assume you are using this pipeline for inference,
        # please fill in the input parameters.
        "input_video": data["video"],
        "height": data["video"][0].shape[1],
        "width": data["video"][0].shape[2],
        "num_frames": len(data["video"]),
        #"num_conditions": len(self.condition_list),
        # Please do not modify the following parameters
        # unless you clearly know what this will cause.
        "cfg_scale": 1,
        "tiled": False,
        "rand_device": pipe.device,
        "use_gradient_checkpointing": True,
        "use_gradient_checkpointing_offload": False,
        "cfg_merge": False,
        "vace_scale": 1,
        "max_timestep_boundary": 1.0,
        "min_timestep_boundary": 0.0,
        "path": data["path"]
    }
    for condition in condition_list:
        inputs_shared[condition] = data[condition]
    
    
    # Pipeline units will automatically process the input parameters.
    for unit in pipe.units:
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(unit, pipe, inputs_shared, inputs_posi, inputs_nega)
    return {**inputs_shared, **inputs_posi}

if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()

    data_dir = args.data_path
    out_path = args.out_path
    
    # Spheres dataset
    config = SpheresDatasetConfig(
        # data_dir = [#"/root/Data/synthetic_data/spheres_bk_indoor",
        #             # "/root/Data/synthetic_data/spheres_bk_indoor_anim_lighting",
        #             # "/root/Data/synthetic_data/spheres_bk_indoor_anim_camera",
        #             # "/root/Data/synthetic_data/spheres_bk_indoor_anim_camera_jump",
        #             # "/root/Data/synthetic_data/spheres_bk_indoor_anim_spheres",
        #             # "/root/Data/synthetic_data/spheres_bk_indoor_anim_spheres_jump"],
        #             #"s3://nflx-studio-algo-research-sl-awsprod-us-east-1/nflx-scl-ml/from_eyeline/users/cbolduc/synthetic_spheres/spheres_bk_indoor_anim_spheres_test"],
        #             "s3://nflx-studio-algo-research-sl-awsprod-us-east-1/nflx-scl-ml/from_eyeline/users/cbolduc/test/synthetic/temple"],
                    #"s3://nflx-studio-algo-research-sl-awsprod-us-east-1/nflx-scl-ml/from_eyeline/users/cbolduc/synthetic_spheres/spheres_bk_indoor_anim_camera"],

        #data_dir = "s3://nflx-studio-algo-research-sl-awsprod-us-east-1/nflx-scl-ml/from_eyeline/users/cbolduc/test/synthetic/temple_anim_screen_center",
        data_dir = data_dir,
        #val_data_dir = ["/root/Data/synthetic_data/spheres_val"],
        augmentation_cfg=SpheresAugConfig(
            horizontal_flip = False,
            noise=False,
            vignetting=False,
            auto_exposure=False,
            haze=False,
            white_balance=False,
            reverse=False
        ),
        random_ev = True, #For multiple EV
        HDR = False, #For reinhard
        mask_img = True, #Mask the input image
    )
    dataset = SpheresDataset(config, target_layers=args.targets.split(","), split='objects', num_frames=args.num_frames)
    
    
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        condition_list=args.condition_list,
        targets=args.targets,
        output_dir=args.output_path
    )
    
    # model_configs = []
    # model_id_with_origin_paths = args.model_id_with_origin_paths
    # if model_id_with_origin_paths is not None:
    #     model_id_with_origin_paths = model_id_with_origin_paths.split(",")
    #     for i in model_id_with_origin_paths:
    #         if ":" in i:
    #             model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1])]
    #         else:
    #             model_configs += [ModelConfig(path=i)]
        
    # pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs)

    # condition_list = args.condition_list.split(",") if args.condition_list is not None else []
    
    # add_condition_input(pipe.dit, len(condition_list))
    
    # pipe.units.append(WanVideoUnit_ConditionsVideoEmbedder(condition_list))
    
    # Load trained weights
    print("Loading trained weights from:", args.trained_weights_path)
    state_dict = load_state_dict(args.trained_weights_path)
    load_result = model.pipe.dit.load_state_dict(state_dict, strict=False)
    
        
    evs = args.evs.split(",") if args.evs is not None else [0]

    condition_list = args.condition_list.split(",")
    
    model.to("cuda")

    for sphere in args.targets.split(","):
        dataset.layer = sphere
        for data_raw in tqdm(dataset):
            #first_ev = True
            sample_seed = hash(data_raw["path"]) % (2**32)  # Use path hash as consistent seed

            #frame = data_raw["position"][0].cpu().permute(1,2,0).numpy()

            for ev in evs:
                # Set the same seed for each EV to get identical initial noise
                torch.manual_seed(sample_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(sample_seed)
                    torch.cuda.manual_seed_all(sample_seed)

                data = copy.deepcopy(data_raw)
                factor = 2 ** (-2.5)
                data = treat_data_old(data, legacy=False, given_ev=int(ev), test=False, bad_incident=False, factor=factor)
                
                data = data_to_device(data, model.pipe.device)
                inputs = forward_preprocess(data, model.pipe, condition_list)
                #models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
                video = model.pipe(data["prompt"], num_inference_steps=50, inputs_shared=inputs)#, shift=2)

                # Extract relative path from data path to maintain hierarchy
                relative_path = os.path.relpath(data["path"], data_dir)
                sample_out_dir = os.path.join(out_path, relative_path, sphere)


                #Save video
                # path_dir = os.path.join(sample_out_dir, "dir")
                # path_dist = os.path.join(sample_out_dir, "dist")
                # path_normal = os.path.join(sample_out_dir, "normal")
                # path_image = os.path.join(sample_out_dir, "image")
                # path_mask = os.path.join(sample_out_dir, "mask")
                # os.makedirs(path_dir, exist_ok=True)
                # os.makedirs(path_dist, exist_ok=True)
                # os.makedirs(path_normal, exist_ok=True)
                # os.makedirs(path_image, exist_ok=True)
                # os.makedirs(path_mask, exist_ok=True)
                path_gt = os.path.join(sample_out_dir, f"gt_ev{ev}")
                path_pred = os.path.join(sample_out_dir, f"pred_ev{ev}")
                os.makedirs(path_gt, exist_ok=True)
                os.makedirs(path_pred, exist_ok=True)
                for i in range(len(data["video"])):
                    video[i].save(os.path.join(path_pred, f"frame{i}.png"))

                    gt = data["video"][i].cpu().permute(1,2,0).numpy() * 0.5 + 0.5
                    gt = (gt*255).astype('uint8')
                    im = Image.fromarray(gt)
                    im.save(os.path.join(path_gt, f"frame{i}.png"))
                    
                    # if first_ev:
                    #     frame = data["video"][i].cpu().permute(1,2,0).numpy() * 0.5 + 0.5
                    #     frame = (frame*255).astype('uint8')
                        
                    #     im = Image.fromarray(frame)
                    #     im.save(os.path.join(path_gt, f"frame{i}.png"))
                    #     #dir
                    #     frame = data["dir_to_sphere"][i].cpu().permute(1,2,0).numpy() * 0.5 + 0.5
                    #     frame = (frame*255).astype('uint8')
                    #     im = Image.fromarray(frame)
                    #     im.save(os.path.join(path_dir, f"frame{i}.png"))
                    #     #dist
                    #     frame = data["dist_to_sphere"][i].cpu().permute(1,2,0).numpy() * 0.5 + 0.5
                    #     frame = (frame*255).astype('uint8')
                    #     im = Image.fromarray(frame)
                    #     im.save(os.path.join(path_dist, f"frame{i}.png"))
                    #     #normal
                    #     frame = data["normals"][i].cpu().permute(1,2,0).numpy() * 0.5 + 0.5
                    #     frame = (frame*255).astype('uint8')
                    #     im = Image.fromarray(frame)
                    #     im.save(os.path.join(path_normal, f"frame{i}.png"))
                    #     #image
                    #     frame = data["image"][i].cpu().permute(1,2,0).numpy() * 0.5 + 0.5
                    #     frame = (frame*255).astype('uint8')
                    #     im = Image.fromarray(frame)
                    #     im.save(os.path.join(path_image, f"frame{i}.png"))
                    #     #Sphere mask
                    #     frame = data["sphere_mask"][i].cpu().permute(1,2,0).numpy() * 0.5 + 0.5
                    #     frame = (frame*255).astype('uint8')
                    #     im = Image.fromarray(frame)
                    #     im.save(os.path.join(path_mask, f"frame{i}.png"))
                    
                
                subprocess.run(['ffmpeg', '-y', '-framerate', '30', '-i', os.path.join(path_pred, 'frame%d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', os.path.join(path_pred, 'video.mp4')])
                # if first_ev:
                #     subprocess.run(['ffmpeg', '-y', '-framerate', '30', '-i', os.path.join(path_gt, 'frame%d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', os.path.join(path_gt, 'video.mp4')])

                #     subprocess.run(['ffmpeg', '-y', '-framerate', '30', '-i', os.path.join(path_dir, 'frame%d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', os.path.join(path_dir, 'video.mp4')])
                #     subprocess.run(['ffmpeg', '-y', '-framerate', '30', '-i', os.path.join(path_dist, 'frame%d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', os.path.join(path_dist, 'video.mp4')])
                #     subprocess.run(['ffmpeg', '-y', '-framerate', '30', '-i', os.path.join(path_normal, 'frame%d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', os.path.join(path_normal, 'video.mp4')])
                #     subprocess.run(['ffmpeg', '-y', '-framerate', '30', '-i', os.path.join(path_image, 'frame%d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', os.path.join(path_image, 'video.mp4')])
                #     subprocess.run(['ffmpeg', '-y', '-framerate', '30', '-i', os.path.join(path_mask, 'frame%d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', os.path.join(path_mask, 'video.mp4')])
                #     first_ev = False
                        
            