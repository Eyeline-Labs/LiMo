import math
from typing import List
import torch

# ---- Helper functions ----


# Gaussian kernel for the Pyramid of Laplacians
def gauss_kernel(dim: int = 2, channels: int = 3, device: torch.device ="cpu"):
    kernel_x = torch.tensor(
        [1.0, 4.0, 6.0, 4.0, 1.0],device=device
    )
    kernel_x /= 8.0
    # Add empty dimensions to make the kernel compatible with dim
    for _ in range(dim - 1):
        kernel_x = kernel_x.unsqueeze(0)

    # Unsqueeze the channel dimension and the group dimension
    kernel_x = kernel_x.unsqueeze(0).unsqueeze(0)

    kernel_x = kernel_x.repeat((channels, 1) + (1,) * dim)
    single_axis_kernels = [kernel_x]
    if dim >= 2:
        kernel_y = kernel_x.swapaxes(-1, -2)
        single_axis_kernels.append(kernel_y)
    if dim >= 3:
        kernel_z = kernel_x.swapaxes(-1, -3)
        single_axis_kernels.append(kernel_z)

    all_axis_kernel = kernel_x.clone()
    for kernel in single_axis_kernels[1:]:
        all_axis_kernel = all_axis_kernel * kernel

    all_axis_kernel /= all_axis_kernel.sum() / channels

    return single_axis_kernels, all_axis_kernel


# Downsample and Upsample functions for the Pyramid of Laplacians
@torch.jit.script
def downsample(img, kernel, padding_mode: str = "reflect"):
    dim = img.dim() - 2
    img = torch.nn.functional.pad(img, (2,) * (2 * dim), mode=padding_mode)

    if dim == 3:
        return torch.nn.functional.conv3d(
            img, kernel, stride=[2, 2, 2], groups=img.shape[1]
        )
    elif dim == 2:
        return torch.nn.functional.conv2d(
            img, kernel, stride=[2, 2], groups=img.shape[1]
        )
    else:
        return torch.nn.functional.conv1d(img, kernel, stride=2, groups=img.shape[1])


@torch.jit.script
def upsample(img, kernel, padding_mode: str = "reflect"):
    dim = img.dim() - 2
    img = torch.nn.functional.pad(img, (1, 1) * dim, mode=padding_mode)
    if dim == 3:
        return torch.nn.functional.conv_transpose3d(
            img,
            kernel,
            groups=img.shape[1],
            stride=[2, 2, 2],
            padding=[4, 4, 4],
            output_padding=[1, 1, 1],
        )
    elif dim == 2:
        return torch.nn.functional.conv_transpose2d(
            img,
            kernel,
            groups=img.shape[1],
            stride=[2, 2],
            padding=[4, 4],
            output_padding=[1, 1],
        )
    else:
        return torch.nn.functional.conv_transpose1d(
            img,
            kernel,
            groups=img.shape[1],
            stride=[2],
            padding=[4],
            output_padding=[1],
        )


@torch.jit.script
def upsample_2d(img, kernel, padding_mode: str = "reflect"):
    img = torch.nn.functional.pad(img, (1, 1, 1, 1), mode=padding_mode)

    return torch.nn.functional.conv_transpose2d(
        img,
        kernel,
        groups=img.shape[1],
        stride=[2, 2],
        padding=[4, 4],
        output_padding=[1, 1],
    )


def upsample_separated(
    img,
    kernel_x,
    kernel_y=torch.empty([]),
    kernel_z=torch.empty([]),
    padding_mode: str = "reflect",
):
    dim = img.dim() - 2

    if dim == 3:
        return upsample_separated_3d(img, kernel_x, kernel_y, kernel_z, padding_mode)
    elif dim == 2:
        return upsample_separated_2d(img, kernel_x, kernel_y, padding_mode)
    else:
        return upsample_separated_1d(img, kernel_x, padding_mode)


@torch.jit.script
def upsample_separated_1d(
    img,
    kernel_x,
    padding_mode: str = "reflect",
):
    img = torch.nn.functional.pad(img, (1, 1), mode=padding_mode)
    img = torch.nn.functional.conv_transpose1d(
        img,
        kernel_x,
        groups=img.shape[1],
        stride=[2],
        padding=[4],
        output_padding=[1],
    )
    return img


@torch.jit.script
def upsample_separated_2d(
    img,
    kernel_x,
    kernel_y=torch.empty([]),
    padding_mode: str = "reflect",
):
    img = torch.nn.functional.pad(img, (0, 0, 1, 1), mode=padding_mode)
    img = torch.nn.functional.conv_transpose2d(
        img,
        kernel_y,
        groups=img.shape[1],
        stride=[2, 1],
        padding=[4, 0],
        output_padding=[1, 0],
    )
    img = torch.nn.functional.pad(img, (1, 1, 0, 0), mode=padding_mode)
    img = torch.nn.functional.conv_transpose2d(
        img,
        kernel_x,
        groups=img.shape[1],
        stride=[1, 2],
        padding=[0, 4],
        output_padding=[0, 1],
    )
    return img


@torch.jit.script
def upsample_separated_3d(
    img,
    kernel_x,
    kernel_y,
    kernel_z,
    padding_mode: str = "reflect",
):
    img = torch.nn.functional.pad(img, (0, 0, 0, 0, 1, 1), mode=padding_mode)
    img = torch.nn.functional.conv_transpose3d(
        img,
        kernel_z,
        groups=img.shape[1],
        stride=[2, 1, 1],
        padding=[4, 0, 0],
        output_padding=[1, 0, 0],
    )
    img = torch.nn.functional.pad(img, (0, 0, 1, 1, 0, 0), mode=padding_mode)
    img = torch.nn.functional.conv_transpose3d(
        img,
        kernel_y,
        groups=img.shape[1],
        stride=[1, 2, 1],
        padding=[0, 4, 0],
        output_padding=[0, 1, 0],
    )
    img = torch.nn.functional.pad(img, (1, 1, 0, 0, 0, 0), mode=padding_mode)
    img = torch.nn.functional.conv_transpose3d(
        img,
        kernel_x,
        groups=img.shape[1],
        stride=[1, 1, 2],
        padding=[0, 0, 4],
        output_padding=[0, 0, 1],
    )
    return img

@torch.jit.script
def inv_laplacian_pyramid(
    levels: List[torch.Tensor],
    kernel_us=torch.empty(0),
    kernel_x=torch.empty(0),
    kernel_y=None,
    kernel_z=None,
    padding_mode: str = "reflect",
):
    current = levels[-1]
    if kernel_us.numel() == 0 or kernel_x.numel() == 0:
        seprated_kernels, kernel_ds= gauss_kernel(
            current.dim() - 2, current.shape[1], device=current.device
        )
        kernel_x = seprated_kernels[0]
        if current.dim() - 2 >= 2:
            kernel_y = seprated_kernels[1]
        if current.dim() - 2 >= 3:
            kernel_z = seprated_kernels[2]
        kernel_us = (2**(current.dim()-2)) * kernel_ds

    for level in levels[-2::-1]:
        # If the resolution is too small, we use a full kernel and not separated ones as it incurrs less CPU overhead
        if current.numel() < 16384:
            current = upsample(current, kernel_us, padding_mode)
        else:
            current = upsample_separated(
                current, kernel_x, kernel_y, kernel_z, padding_mode
            )
        current += level
    return current

@torch.jit.script
def fwd_laplacian_pyramid(
    current,
    num_lvls: int,
    kernel_us = torch.empty(0),
    kernel_ds = torch.empty(0),
    kernel_x = torch.empty(0),
    kernel_y = None,
    kernel_z = None,
    padding_mode: str = "reflect",
):
    if kernel_us.numel() == 0 or kernel_ds.numel() == 0 or kernel_x.numel() == 0:
        separated_kernels, kernel_ds = gauss_kernel(
            current.dim() - 2, current.shape[1], device=current.device
        )
        kernel_x = separated_kernels[0]
        if current.dim() - 2 >= 2:
            kernel_y = separated_kernels[1]
        if current.dim() - 2 >= 3:
            kernel_z = separated_kernels[2]
        kernel_us = (2**(current.dim()-2)) * kernel_ds

    assert num_lvls > 0, "num_levels must be >0. A single level does not do any decomposition"
    levels=[]
    for level in range(num_lvls-1):
        down = downsample(
            current,
            kernel_ds,
            padding_mode,
        )
        if current.numel() < 16384:
            up = upsample(down, kernel_us, padding_mode)
        else:
            up = upsample_separated(
                down, kernel_x, kernel_y, kernel_z, padding_mode
            )
        diff = current - up
        levels.append(diff)
        current = down
    levels.append(current)
    return levels


@torch.jit.ignore
def __assign(param, value):
    # Using [:] instead of .data = value to force memory copy and the data to keep the same position in memory
    # This is useful for the graphed callables
    param.data[:] = value


# ---- Pyramid of Laplacians ----
class PoL(torch.nn.Module):
    def __init__(
        self,
        init_tensor,
        levels=-1,
        reprojection_interval=16,
        learning_rate=3e-3,
        weight_decay=0.0,
        padding_mode="reflect",
        graphed=True,
    ):
        super(PoL, self).__init__()

        assert (
            len(init_tensor.shape) <= 5 and len(init_tensor.shape) >= 3
        ), "init_tensor must be a 3D,4D or 5D tensor following the NCDHW convention, with N=1"

        if levels == -1:
            levels = min(
                [int(math.floor(math.log2(res))) for res in init_tensor.shape[2:]]
            )
            print(f"PoL levels automatically set to {levels}")

        assert (
            levels > 0
        ), "lvl must be >0 or equal to -1 to use the maximum number of levels"
        for res in init_tensor.shape[2:]:
            assert res >= 0, "resolution along each dimension must be >=0"
            assert (
                res % 2 ** (levels - 1) == 0
            ), "resolution must be divisible by 2**(lvl-1)"

        assert learning_rate >= 0.0, "learning_rate must be >=0.0"
        assert weight_decay >= 0.0, "weight_decay must be >=0.0"
        assert reprojection_interval > 0, "reprojection_interval must be >0"
        assert padding_mode in [
            "constant",
            "reflect",
            "replicate",
            "circular",
        ], "padding_mode must be one of 'constant', 'reflect', 'replicate' or 'circular', default is 'replicate'"

        if levels == 1:
            print("Warning: PoL with only one level is equivalent to a single texture")

        self.dim = len(init_tensor.shape) - 2
        self.resolutions = list(init_tensor.shape[2:])
        self.channels = init_tensor.shape[1]

        self.num_lvls = levels
        self.reprojection_interval = reprojection_interval

        separated_kernels, kernel = gauss_kernel(self.dim, self.channels)
        self.register_buffer("kernel_x", separated_kernels[0])
        if self.dim >= 2:
            self.register_buffer("kernel_y", separated_kernels[1])
        else:
            self.kernel_y = torch.empty([])
        if self.dim >= 3:
            self.register_buffer("kernel_z", separated_kernels[2])
        else:
            self.kernel_z = torch.empty([])

        self.register_buffer("kernel_ds", kernel)
        self.register_buffer("kernel_us", (2**self.dim) * kernel)

        if weight_decay > 0.0:
            print(
                "Warning: Using weight_decay with PoL, the Adam optimizer is not recommended, prefer AdamW."
            )
        # Create each level parameters
        self.levels = torch.nn.ParameterList()
        self.levels_lr = {}
        self.levels_weight_decay = {}
        self.__padding_mode = padding_mode

        for lvl in range(self.num_lvls):
            self.levels.append(
                torch.nn.Parameter(
                    torch.zeros(
                        [1, self.channels]
                        + [res // pow(2, lvl) for res in self.resolutions]
                    )
                )
            )

            self.levels_lr[lvl] = learning_rate

            if (
                lvl < (self.num_lvls - 1) and weight_decay > 0.0
            ):  # If not last level encourage sparsity
                self.levels_weight_decay[lvl] = weight_decay

        # Initialize the Laplacian Pyramid
        # Put the Laplacian Pyramid on the same device as the init_tensor
        self.super_to(init_tensor.device)
        self.init(init_tensor)
        print("PoL initialized")

        self.graphed = graphed
        if self.graphed:
            print("Computing graphed callables")
            self.__inv_laplacian_pyramid = torch.cuda.make_graphed_callables(
                lambda l, k_us, k_x, k_y, k_z: inv_laplacian_pyramid(
                    l, k_us, k_x, k_y, k_z, self.__padding_mode
                ),
                (
                    tuple(self.levels),
                    self.kernel_us,
                    self.kernel_x,
                    self.kernel_y,
                    self.kernel_z,
                ),
                num_warmup_iters=12,
            )
        else:
            self.__inv_laplacian_pyramid = (
                lambda l, k_us, k_x, k_y, k_z: inv_laplacian_pyramid(
                    l, k_us, k_x, k_y, k_z, self.__padding_mode
                )
            )

    # Parameters of the Laplacian Pyramid, to be used in an optimizer as a list of dictionaries
    def parameters(self):
        params = []
        for lvl in range(self.num_lvls):
            grad_params_env_dict = {
                "params": self.levels[lvl],
                "lr": self.levels_lr[lvl],
            }
            if lvl in self.levels_weight_decay:  # If not last level encourage sparsity
                grad_params_env_dict["weight_decay"] = self.levels_weight_decay[lvl]
            params.append(grad_params_env_dict)
        return params

    # Initialize the Laplacian Pyramid with a single texture
    @torch.no_grad()
    def init(self, init):
        self.reproject_to_PoL(init)

    # Reproject the internal representation to a valid Laplacian Pyramid
    @staticmethod
    @torch.jit.script
    def __reproject_laplacian_pyramid(
        current,
        levels: List[torch.nn.Parameter],
        kernel_us,
        kernel_ds,
        kernel_x,
        kernel_y,
        kernel_z,
        padding_mode: str,
    ):
        for level in levels[:-1]:
            down = downsample(
                current,
                kernel_ds,
                padding_mode,
            )
            if current.numel() < 16384:
                up = upsample(down, kernel_us, padding_mode)
            else:
                up = upsample_separated(
                    down, kernel_x, kernel_y, kernel_z, padding_mode
                )
            diff = current - up
            __assign(level, diff)
            current = down
        __assign(levels[-1], current)

    # The internal pyramid representation is not guranteed to be a valid Laplacian Pyramid
    # Every reprojection_interval iterations, we reproject the internal representation to a valid Laplacian Pyramid using this function
    # This function is also used for initialization and to replace the reconstruction by a provided signal
    @torch.no_grad()
    def reproject_to_PoL(self, override=None):
        if override is not None:
            current = override.contiguous()
        else:
            # Recompute the reconstruction for safety
            current = self.__inv_laplacian_pyramid(
                tuple(self.levels),
                self.kernel_us,
                self.kernel_x,
                self.kernel_y,
                self.kernel_z,
            )

        PoL.__reproject_laplacian_pyramid(
            current,
            self.levels,
            self.kernel_us,
            self.kernel_ds,
            self.kernel_x,
            self.kernel_y,
            self.kernel_z,
            self.__padding_mode,
        )
        # Setting composed to None so that it is recomputed
        self._reconstruction = None

    def forward(self):
        return self.reconstruction()

    # Get the reconstruction from the Laplacian Pyramid
    # The reason why we separate this from __get_reconstruction is that we want to be able to compose the map once but use it several times per iteration
    def reconstruction(self):
        if (
            not hasattr(
                self,
                "_reconstruction",
            )
            or self._reconstruction is None
        ):
            out = self.__inv_laplacian_pyramid(
                tuple(self.levels),
                self.kernel_us,
                self.kernel_x,
                self.kernel_y,
                self.kernel_z,
            )

            setattr(self, "_reconstruction", out)
        return self._reconstruction

    # This function is helpful to get the mipmaps of the Laplacian Pyramid
    # For instance for use in a differentiable renderer
    # This function returns a list of tensors.
    def mip_lvls(self):
        mips = [self.levels[-1]]
        for level in self.levels[-2::-1]:
            upsampled = upsample_separated(
                mips[-1],
                self.kernel_x,
                self.kernel_y,
                self.kernel_z,
                self.__padding_mode,
            )

            mips.append(upsampled + level)
        mips.reverse()
        return mips

    # This function returns a single tensor that can be trilinearly sampled, for instance depending on learnable blur parameters
    # It is best to compute this only once per iteration, such as for the composed map but this is left to the user
    def composed_lvls(self):
        mips = [self.levels[-1]]
        for level in self.levels[-2::-1]:
            for m in range(len(mips)):
                mips[m] = upsample_separated(
                    mips[m],
                    self.kernel_x,
                    self.kernel_y,
                    self.kernel_z,
                    self.__padding_mode,
                )

            mips.append(mips[m] + level)
        return torch.stack(mips, dim=-3)

    def super_to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        return self
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if self.graphed:
            print("Computing graphed callables")
            self.__inv_laplacian_pyramid = torch.cuda.make_graphed_callables(
                lambda l, k_us, k_x, k_y, k_z: PoL.__inv_laplacian_pyramid(
                    l, k_us, k_x, k_y, k_z, self.__padding_mode
                ),
                (
                    tuple(self.levels),
                    self.kernel_us,
                    self.kernel_x,
                    self.kernel_y,
                    self.kernel_z,
                ),
                num_warmup_iters=12,
            )
        else:
            self.__inv_laplacian_pyramid = (
                lambda l, k_us, k_x, k_y, k_z: PoL.__inv_laplacian_pyramid(
                    l, k_us, k_x, k_y, k_z, self.__padding_mode
                )
            )
        return self

    
    # ---- Modifiers ----
    @torch.no_grad()
    def clamp_(self, min=None, max=None):
        assert min is not None or max is not None, "min and max cannot be both None"
        self.reproject_to_PoL(self.reconstruction().clamp(min=min, max=max))
        self._reconstruction = None
    
    @torch.no_grad()
    def ensure_monotonus_(self, decreasing=False):
        assert self.dim == 1, "ensure_monotonus_ only works for 1D textures"
        reconstruction = self.reconstruction()
        if decreasing:
            reconstruction = torch.flip(torch.cummax(torch.flip(reconstruction, [-1]), dim=-1).values, [-1])
        else:
            reconstruction = torch.cummax(reconstruction, dim=-1).values
        self.reproject_to_PoL(reconstruction)
        self._reconstruction = None

    # ---- Callbacks ----

    # Function to be called at the end of each iteration
    @torch.jit.export
    def end_iter_callback(self, iter_n: int):
        self._reconstruction = None
        if iter_n % self.reprojection_interval == 0:
            self.reproject_to_PoL()
