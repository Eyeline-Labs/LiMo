from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TestConfig:
    model_path: str
    """The path containing the model to test. Either a directory to an exported pipeline or a path to a model checkpoint."""

    input_path: str
    """The path containing the input image(s) to test. Either a directory containing images or a path to a single image."""
    

    num_input_conditions: int
    """The number of input conditions for the model. TODO Make this automatic."""

    condition_default_values: List[int]
    """The default values for the input conditions after the input image."""

    test_model_dtype: str = "fp16"
    """The dtype to use for the model during testing."""

    targets: str = "intrinsic_0_0"
    """The targets to test the model on."""
    
    condition_list: str = "image,position"
    """The list of conditions to use for the model"""

    max_resolution: int = 2048 * 2048 
    """The maximum resolution to use for the input images. They will be resized so their total number of pixels is below this value."""

    force_max_resolution: bool = False
    """Whether to force the input images to be resized to match the maximum resolution when they are smaller than it."""

    num_inference_steps: int = 28
    """The number of inference steps to run."""
    
    sigmas: Optional[List[float]] = None
    """The sigmas to test the model on. Alternative to num_inference_steps, if specified. num_inference_steps will be ignored if this is specified."""

    fixed_noise: bool = False
    """Whether to use fixed noise for the input images."""

    output_type: str = "exr"
    """The type of the output images. Can be 'png' or 'exr'."""

    crop: bool= False
    """Whether to center crop the input images instead of resizing."""

    pretrained_model_name_or_path: str = "black-forest-labs/FLUX.1-schnell"
    """The base model that was used. Required for now hpefully we automate the detection in the future."""
    
    warp_previous_output: bool = False
    """Whether to warp the previous output to use as a condition."""
    
    swap_low_frequency: bool = False
    """Whether to swap the low frequency of the output image with the one of the input. This can be usefull to obtain a stable output."""
    
    pred_ev: bool = False
    """Whether to predict multiple exposure values"""

    output_path: Optional[str] = None
    """The path to save the output images to."""

default_test_configs = {
    "default": (
        "",
        TestConfig(
            model_path = "",
            input_path = "",
            num_input_conditions = 0,
            condition_default_values = [],
        ),
    ),
    "balls": (
        "Base sphere configuration",
        TestConfig(
            model_path = "../checkpoints/Flux/mp_rank_00_model_states.pt",
            input_path = "../datasets/classroom",
            output_path = "../outputs/classroom_image",
            num_input_conditions = 5,
            condition_list = "image,position,normals,dist_to_sphere,dir_to_sphere",
            targets = "sphere_0,sphere_1",
            condition_default_values = [1,0,0],
            pred_ev = True,
            num_inference_steps = 20
        ),
    ),
}
