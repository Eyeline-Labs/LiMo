from dataclasses import dataclass
from typing import List, Optional, Union
from datasets.dataset_relighting import MultiLightDatasetConfig, MultiLightAugConfig
from datasets.dataset_light_estimation import HDRPanoDatasetConfig
from datasets.dataset_supres import SupResDatasetConfig, SupResAugConfig
from datasets.dataset_spheres import SpheresDatasetConfig, SpheresAugConfig

@dataclass
class OptimizerConfig:

    type: str = "adamw"
    """The optimizer type to use. Choose between ["adamw", "prodigy"]."""

    learning_rate: float = 1e-4
    """Initial learning rate (after the potential warmup period) to use."""
    
    scale_lr: bool = False
    """Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size."""
    
    lr_scheduler: str = "constant"
    """The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]."""
    
    lr_warmup_steps: int = 500
    """Number of steps for the warmup in the lr scheduler."""

    lr_num_cycles:int = 1
    """Number of hard resets of the lr in cosine_with_restarts scheduler."""
    
    lr_power: float=1.0
    """Power factor of the polynomial scheduler."""

    adam_beta1: float = 0.9
    """The beta1 parameter for the Adam optimizer."""
    
    adam_beta2: float = 0.999
    """The beta2 parameter for the Adam optimizer."""
    
    adam_weight_decay: float = 1e-2
    """Weight decay to use."""
    
    adam_epsilon: float = 1e-08
    """Epsilon value for the Adam optimizer."""
    
    max_grad_norm: float = 1.0
    """Max gradient norm."""

    use_8bit_adam: bool = False
    """Whether or not to use 8-bit Adam from bitsandbytes."""

@dataclass
class TrainConfig:
    pretrained_model_name_or_path: str
    """Path to pretrained model or model identifier from huggingface.co/models."""
    
    output_dir: str
    """The output directory where the model predictions and checkpoints will be written."""

    dataset_cfg: Union[MultiLightDatasetConfig,SupResDatasetConfig,SpheresDatasetConfig]
    """The configuration of the dataset."""

    optimizer_cfg: OptimizerConfig
    """The configuration of the optimizer."""

    seed: Optional[int] = None
    """A seed for reproducible training."""
    
    input_perturbation: float = 0
    """The scale of input perturbation. Recommended 0.1."""

    revision: Optional[str] = None
    """Revision of pretrained model identifier from huggingface.co/models."""
    
    variant: Optional[str] = None
    """Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16."""
    
    val_denoising_steps: int = 20
    """The number of denoising steps for validation."""
    
    val_dry_run: bool = False
    """Run evaluation before training."""
    
    train_batch_size: int = 16
    """Batch size (per device) for the training dataloader."""
    
    train_iter_per_epoch: int = 10_000
    """Randomly sample dataset to reduce epoch size."""
    
    max_train_steps: int = 150000 
    """Total number of training steps to perform. If provided, overrides num_train_epochs."""
    
    gradient_accumulation_steps: int = 1
    """Number of updates steps to accumulate before performing a backward/update pass."""
    
    gradient_checkpointing: bool = False
    """Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass."""
    
    snr_gamma: Optional[float] = None
    """SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. More details here: https://arxiv.org/abs/2303.09556."""

    allow_tf32: bool = False
    """Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices."""
    
    use_ema: bool = False
    """Whether to use EMA model."""
    
    non_ema_revision: Optional[str] = None
    """Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or remote repository specified with --pretrained_model_name_or_path."""
    
    dataloader_num_workers: int = 0
    """Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."""
    
    prediction_type: Optional[str] = None
    """The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen."""
    
    logging_dir: str = "logs"
    """[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."""
    
    trainable_precision: Optional[str] = None
    """Whether to use mixed precision for trainable parameters. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10 and an Nvidia Ampere GPU. Default to the value of accelerate config of the current system or the flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."""

    frozen_precision: Optional[str] = "fp16"
    """Whether to use mixed precision for frozen parameters. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10 and an Nvidia Ampere GPU. Default to the value of accelerate config of the current system or the flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."""
    
    report_to: str = "tensorboard"
    """The integration to report the results and logs to. Supported platforms are `"tensorboard"` (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations."""
    
    local_rank: int = -1
    """For distributed training: local_rank."""
    
    checkpointing_steps: int = 10000
    """Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming training using `--resume_from_checkpoint`."""
    
    checkpoints_total_limit: Optional[int] = None
    """Max number of checkpoints to store."""
    
    resume_from_checkpoint: Optional[str] = None
    """Whether training should be resumed from a previous checkpoint. Use a path saved by `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint."""
    
    enable_xformers_memory_efficient_attention: bool = False
    """Whether or not to use xformers."""
    
    validation_epochs: int = 1
    """Run validation every X epochs."""
    
    validation_steps: int = -1
    """Run validation every X steps."""

    num_validation_images: int = 32
    """Number of images that should be generated during validation."""
    
    num_validation_images_display: int = 8
    """Number of images that should be displayed during validation."""
    
    tracker_project_name: str = "finetuned_DM"
    """The `project_name` argument passed to Accelerator.init_trackers for more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator."""
    
    vae_nondeterministic: bool = False
    """Whether to directly use mean as the latent space."""
    
    condition_list: str = "composed_image"
    """List of input fed into u-net besides noise. Commas separate the conditions."""

    targets: str = "intrinsic_0_0"
    """List of targets. Commas separate the target keys to be used."""
    
    max_sequence_length: int = 77
    """Maximum sequence length to use with with the T5 text encoder"""

    weighting_scheme: str = "logit_normal"
    """The weighting scheme to use. Choose between ["sigma_sqrt", "logit_normal", "mode", "cosmap"]."""

    logit_mean: float = 0.0
    """Mean to use when using the 'logit_normal' weighting scheme."""

    logit_std: float = 1.0
    """Std to use when using the 'logit_normal' weighting scheme."""

    mode_scale: float = 1.29
    """Scale of mode weighting scheme. Only effective when using the 'mode' as the weighting_scheme."""

    precondition_outputs: int = 0
    """Flag indicating if we are preconditioning the model outputs or not as done in EDM. This affects how model `target` is calculated."""

    lotus_formulation: bool = False
    """Flag indicating if we are using the LOTUS formulation. Training only at T=1000. See https://arxiv.org/pdf/2409.18124"""

    lotus_num_steps: int = 1
    """Number of steps to take in the LOTUS formulation. Only effective when using the 'lotus_formulation' flag."""
    
    lotus_cond_decoder: bool = True
    """Flag indicating if we are using the LOTUS formulation with a conditioned decoder. Only effective when using the 'lotus_formulation' flag."""

    profile: bool = False
    """Whether to profile the training process."""

    consistency_loss: bool = False
    """Whether to use consistency loss. This loss pushed normals and intrinsics to be consistent across lighting conditions."""

    controlnet: bool = False 
    """Whether to use controlnet."""

    compile_network: bool = True
    """Whether to compile the network with DeepSpeed integration of torch.compile."""

    coarse_to_fine: bool = False
    """Whether to use coarse to fine resolution for training."""

    coarse_resolution: int = 384 
    """The coarsest resolution. Only effective when using the 'coarse_to_fine' flag."""

    fine_resolution: int = 512
    """The finest resolution. Used as the training resolution."""
    
    save_img_every_n_steps: int = 100
    """Save images every n steps during training"""
    
    
    sphere_weighting: bool = False
    """Whether to use sphere weighting. This is used to weight the loss of each sphere based"""

# Note that we could also define this library using separate YAML files (similar to
# `config_path`/`config_name` in Hydra), but staying in Python enables seamless type
# checking + IDE support.

multilight_augmentation = MultiLightAugConfig(
                        horizontal_flip = True,
                        auto_exposure = True,
                        random_exposure = True,
                        noise=True,
                        vignetting=True,
                        haze=True,
                        white_balance=True,
                    )
supres_augmentation = SupResAugConfig(
                        horizontal_flip = True,
                        random_exposure= True,
                        white_balance=True,
                        contrast=True,
                        gamma=True,
)

default_train_configs = {
    "default": (
        "",
        TrainConfig(
            pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-1",
            output_dir = "/root/projects/relighting/train_outputs/test",
            dataset_cfg = MultiLightDatasetConfig(
                data_dir = "/root/projects/synthetic_data/test_a100_v2/",
                intrinsics_max_l = 0,
            ),
            optimizer_cfg = OptimizerConfig(
                learning_rate = 3e-5,
                lr_scheduler = "cosine",
                lr_warmup_steps = 0,
            ),
        ),
    ),
    "relighting": (
        "Base relighting configuration",
        TrainConfig(
            #pretrained_model_name_or_path = "stabilityai/stable-diffusion-3.5-medium",
            pretrained_model_name_or_path = "black-forest-labs/FLUX.1-schnell",
            output_dir = "/root/projects/relighting/train_outputs/test_a_n_l_10k",
            dataset_cfg = [
                #Outdoor dataset
                MultiLightDatasetConfig(
                    data_dir = "s3://nflx-studio-algo-research-sl-awsprod-us-east-1/nflx-scl-ml/from_eyeline/users/jphilip/synthetic_data/outdoor_16k_20250227/",
                    intrinsics_max_l = 0,
                    augmentation_cfg=multilight_augmentation,
                    combine_max_lighting=1,
                    num_output_lighting=1,
                ),
                #Indoor dataset
                MultiLightDatasetConfig(
                    data_dir = "s3://nflx-studio-algo-research-sl-awsprod-us-east-1/nflx-scl-ml/from_eyeline/users/jphilip/synthetic_data/indoor_MF_20250319/",
                    intrinsics_max_l = 0,
                    augmentation_cfg=multilight_augmentation,
                    combine_max_lighting=2,
                    num_output_lighting=1,
                    sampling_weight = 5.0,
                ),
            ], 
            optimizer_cfg = OptimizerConfig(
                learning_rate = 1e-5,
                lr_scheduler = "cosine",
                use_8bit_adam = True,
                scale_lr=False,
            ),
            condition_list = "composed_image,alpha",
            targets = "intrinsic_0_0,camera_normals,lighting_r_1",
            trainable_precision = "bf16",
            frozen_precision = "fp16",
            train_iter_per_epoch = 8192,
            validation_epochs = 1,
            max_train_steps = 150000,
            gradient_accumulation_steps = 1,
            train_batch_size = 1,
            enable_xformers_memory_efficient_attention = True,
            dataloader_num_workers = 12,
            allow_tf32 = True,
            use_ema = False,
            weighting_scheme = "none",
            mode_scale = 0.0,
            controlnet = False,
            coarse_to_fine = True,
        ),
    ),
    "light_estimation": (
        "Base lighting estimation configuration",
        TrainConfig(
            pretrained_model_name_or_path = "stabilityai/stable-diffusion-3.5-medium",
            #pretrained_model_name_or_path = "black-forest-labs/FLUX.1-schnell",
            output_dir = "/root/Projects/light_estimation/train_outputs/first",
            dataset_cfg = [
                #Polyhaven
                HDRPanoDatasetConfig(
                    data_dir = "/root/Data/hdris_down"
                )
            ],
            optimizer_cfg = OptimizerConfig(
                type = "deepspeedadam",
                learning_rate = 1e-5,
                lr_scheduler = "cosine",
                use_8bit_adam = False,
                scale_lr=False,
            ),
            report_to = "wandb",
            condition_list = "image,depth,mask",
            targets = "intrinsic_0_0,camera_normals,lighting_r_1",
            trainable_precision = "bf16",
            frozen_precision = "fp16",
            train_iter_per_epoch = 8192,
            validation_epochs = 1,
            max_train_steps = 150000,
            gradient_accumulation_steps = 1,
            train_batch_size = 8,
            enable_xformers_memory_efficient_attention = True,
            dataloader_num_workers = 12,
            allow_tf32 = True,
            use_ema = False,
            weighting_scheme = "none",
            mode_scale = 0.0,
            controlnet = False,
            coarse_to_fine = True,
        ),
    ),
    "balls": (
        "Base lighting estimation configuration",
        TrainConfig(
            #pretrained_model_name_or_path = "stabilityai/stable-diffusion-3.5-medium",
            pretrained_model_name_or_path = "black-forest-labs/FLUX.1-schnell",
            output_dir = "/root/Projects/balls/train_outputs/EV_geometry",
            dataset_cfg = [
                SpheresDatasetConfig(
                    data_dir = ["/root/Data/synthetic_data/spheres_bk_indoor",
                                "/root/Data/synthetic_data/spheres_close_bk_indoor"
                                #"/root/Data/synthetic_data/spheres_bk_outdoor",
                                #"/root/Data/synthetic_data/spheres_rand_indoor",
                                "/root/Data/synthetic_data/spheres_rand_outdoor"],
                    val_data_dir = ["/root/Data/synthetic_data/spheres_val"],
                    augmentation_cfg=SpheresAugConfig(
                        horizontal_flip = True,
                        noise=True,
                        vignetting=True,
                        auto_exposure=True,
                        haze=True,
                        white_balance=True
                    ),
                    random_ev = True, #For multiple EV
                    HDR = False, #For reinhard
                    mask_img = True, #Mask the input image
                )
            ],
            optimizer_cfg = OptimizerConfig(
                #type = "deepspeedadam",
                #learning_rate = 1e-5,
                learning_rate = 1e-5,
                lr_scheduler = "cosine",
                #use_8bit_adam = False,
                use_8bit_adam = True,
                scale_lr=False,
            ),
            report_to = "wandb",
            condition_list = "image,position,normals,dist_to_sphere,dir_to_sphere",
            targets = "sphere_0,sphere_1,sphere_2",
            trainable_precision = "bf16",
            frozen_precision = "fp16",
            train_iter_per_epoch = 8192,
            validation_epochs = 1,
            max_train_steps = 150000,
            gradient_accumulation_steps = 1,
            train_batch_size = 1,
            enable_xformers_memory_efficient_attention = True,
            dataloader_num_workers = 16,
            allow_tf32 = True,
            use_ema = False,
            weighting_scheme = "none",
            mode_scale = 0.0,
            controlnet = False,
            coarse_to_fine = False,
            save_img_every_n_steps = 1000,
            compile_network = False,
            resume_from_checkpoint = "latest",
            sphere_weighting = True,
            #val_denoising_steps = 1,
        ),
    ),
    "balls_no_geo": (
        "Base lighting estimation configuration",
        TrainConfig(
            #pretrained_model_name_or_path = "stabilityai/stable-diffusion-3.5-medium",
            pretrained_model_name_or_path = "black-forest-labs/FLUX.1-schnell",
            output_dir = "/root/Projects/balls/train_outputs/EV_no_geometry_80_test",
            dataset_cfg = [
                SpheresDatasetConfig(
                    data_dir = ["/root/Data/synthetic_data/spheres_bk_indoor",
                                "/root/Data/synthetic_data/spheres_close_bk_indoor"
                                #"/root/Data/synthetic_data/spheres_bk_outdoor",
                                #"/root/Data/synthetic_data/spheres_rand_indoor",
                                "/root/Data/synthetic_data/spheres_rand_outdoor"],
                    val_data_dir = ["/root/Data/synthetic_data/spheres_val"],
                    augmentation_cfg=SpheresAugConfig(
                        horizontal_flip = True,
                        noise=True,
                        vignetting=True,
                        auto_exposure=True,
                        haze=True,
                        white_balance=True
                    ),
                    random_ev = True, #For multiple EV
                    HDR = False, #For reinhard
                    mask_img = True, #Mask the input image
                )
            ],
            optimizer_cfg = OptimizerConfig(
                #type = "deepspeedadam",
                #learning_rate = 1e-5,
                learning_rate = 1e-5,
                lr_scheduler = "cosine",
                #use_8bit_adam = False,
                use_8bit_adam = True,
                scale_lr=False,
            ),
            report_to = "wandb",
            condition_list = "image,position,normals",
            targets = "sphere_0,sphere_1,sphere_2",
            trainable_precision = "bf16",
            frozen_precision = "fp16",
            train_iter_per_epoch = 8192,
            validation_epochs = 1,
            max_train_steps = 150000,
            gradient_accumulation_steps = 1,
            train_batch_size = 1,
            enable_xformers_memory_efficient_attention = True,
            dataloader_num_workers = 8,
            allow_tf32 = True,
            use_ema = False,
            weighting_scheme = "none",
            mode_scale = 0.0,
            controlnet = False,
            coarse_to_fine = False,
            save_img_every_n_steps = 1000,
            compile_network = False,
            resume_from_checkpoint = "latest",
            sphere_weighting = True,
            #val_denoising_steps = 1,
        ),
    ),
    "balls_video": (
        "Base lighting estimation configuration",
        TrainConfig(
            #pretrained_model_name_or_path = "stabilityai/stable-diffusion-3.5-medium",
            #pretrained_model_name_or_path = "black-forest-labs/FLUX.1-schnell",
            pretrained_model_name_or_path = "./Wan2.2-T2V-A14B",
            output_dir = "/root/Projects/balls_video/train_outputs/test",
            dataset_cfg = [
                SpheresDatasetConfig(
                    data_dir = ["/root/Data/synthetic_data/spheres_bk_indoor",
                                "/root/Data/synthetic_data/spheres_close_bk_indoor"
                                #"/root/Data/synthetic_data/spheres_bk_outdoor",
                                #"/root/Data/synthetic_data/spheres_rand_indoor",
                                "/root/Data/synthetic_data/spheres_rand_outdoor"],
                    val_data_dir = ["/root/Data/synthetic_data/spheres_val"],
                    augmentation_cfg=SpheresAugConfig(
                        horizontal_flip = True,
                        noise=True,
                        vignetting=True,
                        auto_exposure=True,
                        haze=True,
                        white_balance=True
                    ),
                    random_ev = True, #For multiple EV
                    HDR = False, #For reinhard
                    mask_img = True, #Mask the input image
                )
            ],
            optimizer_cfg = OptimizerConfig(
                #type = "deepspeedadam",
                #learning_rate = 1e-5,
                learning_rate = 1e-5,
                lr_scheduler = "cosine",
                #use_8bit_adam = False,
                use_8bit_adam = True,
                scale_lr=False,
            ),
            report_to = "wandb",
            condition_list = "image,position,normals,dist_to_sphere,dir_to_sphere",
            targets = "sphere_0,sphere_1,sphere_2",
            trainable_precision = "bf16",
            frozen_precision = "fp16",
            train_iter_per_epoch = 8192,
            validation_epochs = 1,
            max_train_steps = 150000,
            gradient_accumulation_steps = 1,
            train_batch_size = 1,
            enable_xformers_memory_efficient_attention = True,
            dataloader_num_workers = 16,
            allow_tf32 = True,
            use_ema = False,
            weighting_scheme = "none",
            mode_scale = 0.0,
            controlnet = False,
            coarse_to_fine = False,
            save_img_every_n_steps = 1000,
            compile_network = False,
            resume_from_checkpoint = "latest",
            sphere_weighting = True,
            #val_denoising_steps = 1,
        ),
    ),
    "supres": (
        "Base super resolution configuration",
        TrainConfig(
            #pretrained_model_name_or_path = "stabilityai/stable-diffusion-3.5-medium",
            pretrained_model_name_or_path = "black-forest-labs/FLUX.1-schnell",
            output_dir = "/root/projects/super_resolution/train_outputs/test_lq_hq_7subjects",
            dataset_cfg = [
                #Outdoor dataset
                SupResDatasetConfig(
                    data_dir = "s3://nflx-studio-algo-research-sl-awsprod-us-east-1/nflx-scl-ml/from_eyeline/users/jphilip/supres_data/JW/",
                    augmentation_cfg=supres_augmentation,
                ),
                SupResDatasetConfig(
                    data_dir = "s3://nflx-studio-algo-research-sl-awsprod-us-east-1/nflx-scl-ml/from_eyeline/users/jphilip/supres_data/ET/",
                    augmentation_cfg=supres_augmentation,
                ),
                SupResDatasetConfig(
                    data_dir = "s3://nflx-studio-algo-research-sl-awsprod-us-east-1/nflx-scl-ml/from_eyeline/users/jphilip/supres_data/AV/",
                    augmentation_cfg=supres_augmentation,
                ),
                SupResDatasetConfig(
                    data_dir = "s3://nflx-studio-algo-research-sl-awsprod-us-east-1/nflx-scl-ml/from_eyeline/users/jphilip/supres_data/MH/",
                    augmentation_cfg=supres_augmentation,
                ),
                SupResDatasetConfig(
                    data_dir = "s3://nflx-studio-algo-research-sl-awsprod-us-east-1/nflx-scl-ml/from_eyeline/users/jphilip/supres_data/TK/",
                    augmentation_cfg=supres_augmentation,
                ),
                SupResDatasetConfig(
                    data_dir = "s3://nflx-studio-algo-research-sl-awsprod-us-east-1/nflx-scl-ml/from_eyeline/users/jphilip/supres_data/AS/",
                    augmentation_cfg=supres_augmentation,
                    sampling_weight = 0.25,
                ),
                SupResDatasetConfig(
                    data_dir = "s3://nflx-studio-algo-research-sl-awsprod-us-east-1/nflx-scl-ml/from_eyeline/users/jphilip/supres_data/EW/",
                    augmentation_cfg=supres_augmentation,
                    sampling_weight = 0.25,
                ),
            ], 
            optimizer_cfg = OptimizerConfig(
                learning_rate = 1e-5,
                lr_scheduler = "cosine",
                use_8bit_adam = True,
                scale_lr=False,
            ),
            condition_list = "lq_frame",
            targets = "hq_frame",
            trainable_precision = "bf16",
            frozen_precision = "fp16",
            train_iter_per_epoch = 8192,
            validation_epochs = 1,
            max_train_steps = 150000,
            gradient_accumulation_steps = 1,
            train_batch_size = 1,
            enable_xformers_memory_efficient_attention = True,
            dataloader_num_workers = 12,
            allow_tf32 = True,
            use_ema = False,
            weighting_scheme = "none",
            mode_scale = 0.0,
            controlnet = False,
            coarse_to_fine = True,
        ),
    ),
}
