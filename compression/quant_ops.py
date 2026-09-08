'''Quantization operator helpers.

Low-level utility functions for fake quantization, computing scales
and zero points, and quantize/dequantize helpers used by the
compression layers.
'''

import torch


def get_qmax(bits):
    return 2**(bits - 1) - 1

def ste_round(x):
    return x + (torch.round(x) - x).detach()

def compute_scale(bits, tensor=None, per_channel=False, channel_dim=0, r_min=None, r_max=None):
    q_max = get_qmax(bits)
    q_min = -q_max

    if r_min is None or r_max is None:
        if per_channel:
            t_flat = tensor.transpose(0, channel_dim).contiguous().view(tensor.shape[channel_dim], -1)
            r_min = t_flat.min(dim=1).values
            r_max = t_flat.max(dim=1).values
            shape = [1] * tensor.dim()
            shape[channel_dim] = -1
            r_min = r_min.view(shape)
            r_max = r_max.view(shape)
        else:
            r_min = tensor.min()
            r_max = tensor.max()

    scale = (r_max - r_min) / (q_max - q_min)
    return scale.clamp(min=1e-8)

def compute_zero_point(scale, bits, tensor=None, per_channel=False, channel_dim=0, r_min=None):
    q_max = get_qmax(bits)
    q_min = -q_max

    if r_min is None:
        if per_channel:
            t_flat = tensor.transpose(0, channel_dim).contiguous().view(tensor.shape[channel_dim], -1)
            r_min = t_flat.min(dim=1).values
            shape = [1] * tensor.dim()
            shape[channel_dim] = -1
            r_min = r_min.view(shape)
        else:
            r_min = tensor.min()

    zp = torch.round(q_min - (r_min / scale))
    return zp.clamp(q_min, q_max)


def quantize(tensor, scale, zero_point):
    return ste_round(tensor / scale + zero_point)


def dequantize(q_tensor, scale, zero_point):
    return (q_tensor - zero_point) * scale


def fake_quantize_weight(W, bits=8):
    qmax = get_qmax(bits)

    # Per-output-channel scale
    # Conv2d:  [out_channels, in_channels, H, W]
    # Linear:  [out_features, in_features]
    reduce_dims = tuple(range(1, W.dim()))

    max_abs = W.abs().amax(dim=reduce_dims, keepdim=True)

    scale = max_abs / qmax
    scale = scale.clamp(min=1e-8)

    q = ste_round(W / scale)
    q = q.clamp(-qmax, qmax)

    return q * scale

def fake_quantize_activation(x, scale, zero_point, bits=8):
    # asymmetric, scale/zero_point come from calibration — never recomputed here
    q_max = get_qmax(bits)
    q_tensor = quantize(x, scale, zero_point).clamp(-q_max, q_max)
    return dequantize(q_tensor, scale, zero_point)


def calibrate_activation_scale(r_min, r_max, bits):
    scale = compute_scale(bits, r_min=r_min, r_max=r_max)
    zero_point = compute_zero_point(scale, bits, r_min=r_min)
    return scale, zero_point
