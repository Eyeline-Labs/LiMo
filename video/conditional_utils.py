import math
import random
import numpy as np
import torch


def add_condition_input(dit, ntimes):
    old_conv = dit.patch_embedding
    in_old = old_conv.in_channels
    out_ch = old_conv.out_channels
    in_new = in_old * (1 + ntimes)
    
    new_conv = torch.nn.Conv3d(
        in_channels=in_new,
        out_channels=out_ch,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        bias=old_conv.bias is not None,
        groups=old_conv.groups,
        device=old_conv.weight.device,
        dtype=old_conv.weight.dtype
    )
    
    new_conv.weight.data[:,-in_old:] = old_conv.weight.data
    pad = torch.cat([old_conv.weight.data for _ in range(in_new // in_old)], dim=1)[:, :(in_new - in_old)]
    new_conv.weight.data[:,:-in_old] = (0.5 / in_new) * pad
    
    new_conv.bias.data = old_conv.bias.data

    dit.patch_embedding = new_conv
    dit.patch_embedding.requires_grad_(True)