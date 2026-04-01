import argparse
import os
from torchpol.torch_pol import PoL, inv_laplacian_pyramid, fwd_laplacian_pyramid

import imageio
import torch
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
import torch.profiler as profiler


def visualize_1d(pol, x, gt_data):
    save_path = "test_results/1D"
    os.makedirs(save_path, exist_ok=True)
    files = glob.glob(save_path + "/*.png")
    for f in files:
        os.remove(f)
    # visualize result
    composed_lvls = pol.composed_lvls()
    # Plot composed levels
    figure, axis = plt.subplots(pol.num_lvls + 1, 1)
    figure.set_size_inches(10, 20)

    x = x[0].cpu().detach().numpy()

    axis[0].plot(x, gt_data[0, 0].cpu().detach().numpy())
    axis[0].set_title(f"Signal")

    for l in range(pol.num_lvls):
        # For Sine Function
        axis[l + 1].plot(x, composed_lvls[0, l, 0].cpu().detach().numpy())
        axis[l + 1].set_title(f"Level {l}")

    plt.savefig(os.path.join(save_path, "levels.png"))


def visualize_2d(pol):
    save_path = "test_results/2D"
    os.makedirs(save_path, exist_ok=True)
    files = glob.glob(save_path + "/*.png")
    for f in files:
        os.remove(f)
    # visualize result
    composed_lvls = pol.composed_lvls()
    save_image(
        composed_lvls.permute(0, 2, 1, 3, 4)[0],
        os.path.join(save_path, "composed_lvls.png"),
    )

    prediction = pol.reconstruction().squeeze().permute(1, 2, 0).cpu().detach().numpy()
    prediction = np.clip(prediction, a_min=0, a_max=np.inf)
    prediction = np.uint8(np.clip(prediction, 0.0, 1.0) * 255.0)
    imageio.imwrite(os.path.join(save_path, "optimized.png"), prediction)
    levels = pol.levels
    mip_lvls = pol.mip_lvls()
    level_container = np.zeros(
        (pol.resolutions[0], int(pol.resolutions[1] * 1.5), 3), dtype=np.uint8
    )
    mip_container = np.zeros(
        (pol.resolutions[0], int(pol.resolutions[1] * 1.5), 3), dtype=np.uint8
    )
    h_offset = 0
    w_offset = 0
    for l_id, level in enumerate(levels):
        level = level.squeeze().permute(1, 2, 0).cpu().detach().numpy()
        mip_lvl = mip_lvls[l_id].squeeze().permute(1, 2, 0).cpu().detach().numpy()
        if l_id == len(levels) - 1:  # last level
            level = np.uint8(np.clip(level, 0.0, 1.0) * 255.0)
        else:
            level = np.uint8(np.clip(0.5 + level * 0.5, 0.0, 1.0) * 255.0)

        mip_lvl = np.uint8(np.clip(mip_lvl, 0.0, 1.0) * 255.0)

        level_container[
            h_offset : h_offset + level.shape[0], w_offset : w_offset + level.shape[1]
        ] = level
        mip_container[
            h_offset : h_offset + mip_lvl.shape[0],
            w_offset : w_offset + mip_lvl.shape[1],
        ] = mip_lvl

        if w_offset == 0:
            w_offset += level.shape[1]
        elif w_offset > 0:
            h_offset += level.shape[0]

    imageio.imwrite(os.path.join(save_path, "decomposed.png"), level_container)
    imageio.imwrite(os.path.join(save_path, "mipmaps.png"), mip_container)


def visualize_3d(pol):
    save_path = "test_results/3D"
    os.makedirs(save_path, exist_ok=True)
    files = glob.glob(save_path + "/*.png")
    for f in files:
        os.remove(f)

    for l, mip in enumerate(pol.mip_lvls()):
        save_image(
            mip.permute(0, 2, 1, 3, 4)[0],
            os.path.join(save_path, f"mip_lvl_{str(l)}.png"),
        )

    for l, lvl in enumerate(pol.levels):
        if l != pol.num_lvls - 1:
            # Center the signal
            save_image(
                0.5 * lvl.permute(0, 2, 1, 3, 4)[0] + 0.5,
                os.path.join(save_path, f"lvl_{str(l)}.png"),
            )
        else:
            save_image(
                lvl.permute(0, 2, 1, 3, 4)[0],
                os.path.join(save_path, f"lvl_{str(l)}.png"),
            )


def test(dim_to_test: int):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    # Path to data
    # Volumetric data from https://graphics.stanford.edu/data/voldata/
    paths = ["data/kodim23.png", "data/cthead-8bit"]
    os.makedirs("test_results", exist_ok=True)
    profile = False
    if profile:
        os.makedirs("profiling", exist_ok=True)

    # Test for different dimensions
    if dim_to_test == -1:
        dims = [1, 2, 3]
    else:
        dims = [dim_to_test]

    for dim in dims:
        # Get data
        if dim == 1:
            x = torch.linspace(0, 1, 2**15).unsqueeze(0)
            f = 8 * torch.randn(4, 1)
            gt_data = (
                torch.sin(2 * np.pi * (x * f + torch.rand_like(f)))
                .mean(0, keepdim=True)[None]
                .cuda()
            )
        elif dim == 2:
            path = paths[0]
            gt_data = imageio.imread(path)
            gt_data = torch.from_numpy(gt_data).float() / 255.0
            gt_data = gt_data.unsqueeze(0).permute(0, 3, 1, 2)
            gt_data = gt_data.cuda()
        if dim == 3:
            path = paths[1]
            images_path = glob.glob(path + "/*.tif")
            list_data = []
            for p in images_path:
                data = imageio.imread(p)
                data = torch.from_numpy(data).float() / 255.0
                data = data[None, None, None]
                data = data.cuda()
                list_data.append(data)
            gt_data = torch.cat(list_data, dim=2)
            # Resize to 128x128x128
            gt_data = torch.nn.functional.interpolate(
                gt_data, size=(128, 128, 128), mode="trilinear", align_corners=False
            )

        if dim == 1:
            threshold = 0.99
            lr = 1e-3
        else:
            threshold = 0.999
            lr = 3e-3

        pol = PoL(
            0 * gt_data,
            learning_rate=lr,
            weight_decay=0.05,
            padding_mode="reflect",
            graphed=True,  # Default
        )
        pol = pol.cuda()
        # Parameters list, with learning rates and weight decay associated
        params = pol.parameters()

        optimizer = torch.optim.AdamW(params, betas=(0.9, 0.99), fused=True)

        # loss function
        loss_fn = torch.nn.MSELoss()

        EPOCHS = 2000

        if profile:
            prof = torch.profiler.profile(
                schedule=torch.profiler.schedule(
                    wait=200, warmup=100, active=1, repeat=1
                ),
                record_shapes=False,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    f"./profiling/{dim}D"
                ),
                with_stack=False,
            )
            prof.start()

        for i in tqdm(range(EPOCHS)):
            optimizer.zero_grad(set_to_none=True)

            recon = pol()

            recon_copy = recon.detach().clone()
            mask = torch.rand_like(gt_data[0, 0]) > threshold

            recon_copy[:, :, mask] = recon[:, :, mask]
            if recon_copy.isnan().any() or recon_copy.isinf().any():
                print("NaN or Inf")
                break
            loss = loss_fn(recon_copy, gt_data)

            loss.backward()
            optimizer.step()

            if i % (EPOCHS / 10) == 0:
                print(f"{i} | loss: ", loss.item())

            pol.end_iter_callback(i)

            if profile:
                prof.step()
        if profile:
            prof.stop()

        if dim == 1:
            visualize_1d(pol, x, gt_data)
        elif dim == 2:
            visualize_2d(pol)
        elif dim == 3:
            visualize_3d(pol)

        # We now validate the inv_laplacian_pyramid and fwd_laplacian_pyramid functions
        print("Testing fwd_laplacian_pyramid and inv_laplacian_pyramid for dim ", dim)
        levels = fwd_laplacian_pyramid(gt_data, num_lvls=2)
        recon = inv_laplacian_pyramid(levels)
        assert (
            torch.allclose(
                recon,
                gt_data,
                rtol=1e-2,
                atol=1e-2,
            )
        ), "Reconstruction failed"


if __name__ == "__main__":
    # Get arguments to know which test to run, with argparser
    parser = argparse.ArgumentParser(description="Test PoL")
    parser.add_argument(
        "--dim",
        type=int,
        default=-1,
        help="Dimension of the data to test. -1 for 1D, 2D and 3D",
    )
    args = parser.parse_args()
    test(args.dim)
