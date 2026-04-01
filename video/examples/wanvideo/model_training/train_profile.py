import torch, os, json
import sys
sys.path.append("/root/Project/DiffSynth-Studio")
from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig, WanVideoUnit_ConditionsVideoEmbedder
from diffsynth.trainers.utils import DiffusionTrainingModule, VideoDataset, ModelLogger, launch_training_task, wan_parser
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from conditional_utils import add_condition_input
from datasets.dataset_spheres_video import SpheresDatasetConfig, SpheresAugConfig, SpheresDataset, treat_data

import time

from torch.profiler import record_function

def attach_dit_token_count_hook(pipe):
    info = {}
    handles = []

    # Helper to normalize shapes and record token counts
    def _record_input(name, x):
        # Linear gets (B, N, D) for transformers; sometimes (..., D)
        if isinstance(x, tuple):
            x = x[0]
        if hasattr(x, "shape"):
            shape = tuple(x.shape)
            info[f"{name}_in_shape"] = shape
            if len(shape) == 3:
                _, N, _ = shape
                info["tokens"] = N  # number of tokens as seen by attention/projections
            elif len(shape) == 5:
                # (B, D, T, H, W) or (B, C, T, H, W) depending on layout
                info["raw_5d_in_shape"] = shape

    def _record_output(name, y):
        if hasattr(y, "shape"):
            info[f"{name}_out_shape"] = tuple(y.shape)

    # 1) Hook the self-attention module itself to get its input token count
    def attn_io_hook(module, inp, out):
        _record_input("self_attn", inp)
        if hasattr(out, "shape"):
            info["self_attn_out_shape"] = tuple(out.shape)

    blk0 = pipe.dit.blocks[0]
    handles.append(blk0.self_attn.register_forward_hook(attn_io_hook))

    # 2) Hook q/k/v Linear layers to see what goes in/out of each projection
    def make_qkv_hook(name):
        def qkv_hook(module, inp, out):
            _record_input(name, inp)
            _record_output(name, out)
        return qkv_hook

    for name in ("q", "k", "v"):
        layer = getattr(blk0.self_attn, name)
        handles.append(layer.register_forward_hook(make_qkv_hook(name)))

    return info, handles

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
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1]) for i in model_id_with_origin_paths]
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
        with record_function("unit_runner"):
            for unit in self.pipe.units:
                with record_function(f"unit_{unit.__class__.__name__}"):
                    inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)

        return {**inputs_shared, **inputs_posi}
    
    def data_to_dtype(self, data, dtype):
        for key in data.keys():
            if isinstance(data[key], list):
                if isinstance(data[key][0], torch.Tensor):
                    for i in range(len(data[key])):
                        data[key][i] = data[key][i].to(dtype=dtype)
        return data
    
    def validation(self, data, inputs=None):
        
        data = treat_data(data)
        #data = self.data_to_dtype(data, torch.bfloat16)
        if inputs is None: inputs = self.forward_preprocess(data)
        #models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        with record_function("inference"):
            video = self.pipe(data["prompt"], num_inference_steps=self.validation_steps, inputs_shared=inputs)
        return video
    
    def forward(self, data, inputs=None):
        
        data = treat_data(data)
        #data = self.data_to_dtype(data, torch.bfloat16)
        if inputs is None: inputs = self.forward_preprocess(data)

        token_info = {}
        handle = None
        if False:
            token_info, handle = attach_dit_token_count_hook(self.pipe)

        with record_function("loss"):
            models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
            loss = self.pipe.training_loss(**models, **inputs)
        
        if handle is not None:
            print("DiT token debug:", token_info)

        return loss


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()

    data_dir = args.data_path.split(",")
    num_frames = 21
    
    # Spheres dataset
    config = SpheresDatasetConfig(
        data_dir = data_dir,
        augmentation_cfg=SpheresAugConfig(
            horizontal_flip = True,
            noise=True,
            vignetting=True,
            auto_exposure=False,
            haze=True,
            white_balance=True,
            reverse=True
        ),
        random_ev = args.train_ev, #For multiple EV
        HDR = False, #For reinhard
        mask_img = True, #Mask the input image
    )
    dataset = SpheresDataset(config, target_layers=args.targets.split(","), num_frames=num_frames)
    
    # Spheres dataset
    config_val = SpheresDatasetConfig(
        #data_dir = ["s3://nflx-studio-algo-research-sl-awsprod-us-east-1/nflx-scl-ml/from_eyeline/users/cbolduc/synthetic_spheres/spheres_bk_indoor_anim_lighting_16sep"],
        data_dir = "s3://nflx-studio-algo-research-sl-awsprod-us-east-1/nflx-scl-ml/from_eyeline/users/cbolduc/test/synthetic/temple_anim_screen_center",
        #val_data_dir = ["/root/Data/synthetic_data/spheres_val"],
        augmentation_cfg=SpheresAugConfig(
            horizontal_flip = False,
            noise=False,
            vignetting=False,
            auto_exposure=False,
            haze=False,
            white_balance=False
        ),
        random_ev = False, #For multiple EV
        HDR = False, #For reinhard
        mask_img = True, #Mask the input image
    )
    val_dataset = SpheresDataset(config_val, target_layers=args.targets.split(","), num_frames=21, split="objects")
    #dataset = VideoDataset(args=args)
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
        output_dir=args.output_path,
        use_gradient_checkpointing=False,
        
    )

    # Checkpoint resume
    first_step = 0
    os.makedirs(args.output_path, exist_ok=True)
    all_saved_ckpt = [f for f in os.listdir(args.output_path) if f.endswith(".safetensors")]
    if len(all_saved_ckpt) > 0:
        all_saved_ckpt = sorted(all_saved_ckpt, key=lambda x: int(x.split("-")[1].split(".")[0]) if "step" in x else int(x.split("-")[1]))
        last_ckpt = all_saved_ckpt[-1]
        print(f"Resuming from checkpoint {last_ckpt}")
        state_dict = load_state_dict(os.path.join(args.output_path, last_ckpt))
        load_result = model.pipe.dit.load_state_dict(state_dict, strict=True)
        if "step" in last_ckpt:
            first_step = int(last_ckpt.split("-")[1].split(".")[0])
        print(f"Resumed checkpoint {last_ckpt}, num_steps: {first_step}")
    else:
        print("No checkpoint found. Training from scratch.")


    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    launch_training_task(
        dataset, model, model_logger, optimizer, scheduler,
        val_dataset=val_dataset,
        num_steps=args.num_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
        validation_every_steps=args.validation_every_steps,
        find_unused_parameters=args.find_unused_parameters,
        num_workers=args.dataset_num_workers,
        output_path=args.output_path
    )
