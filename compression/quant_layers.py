import torch
import torch.nn as nn
import torch.nn.functional as F
from compression.quant_ops import fake_quantize_weight, fake_quantize_activation


class QuantConv2d(nn.Module):
    def __init__(self, conv_layer, bits=8, act_bits=8):
        super().__init__()
        self.conv = conv_layer
        self.bits = bits
        self.act_bits = act_bits

        self.register_buffer("act_scale", torch.tensor(1.0))
        self.register_buffer("act_zero_point", torch.tensor(0.0))
        self.act_quant_enabled = False

    def forward(self, x):
        W_hat = fake_quantize_weight(self.conv.weight, bits=self.bits)
        out = F.conv2d(x,
                       weight=W_hat,
                       bias=self.conv.bias,
                       stride=self.conv.stride,
                       padding=self.conv.padding,
                       dilation=self.conv.dilation,
                       groups=self.conv.groups)

        if self.act_quant_enabled:
            out = fake_quantize_activation(out, self.act_scale, self.act_zero_point, bits=self.act_bits)

        return out


class QuantLinear(nn.Module):
    def __init__(self, linear_layer, bits=8, act_bits=8):
        super().__init__()
        self.linear = linear_layer
        self.bits = bits
        self.act_bits = act_bits

        self.register_buffer("act_scale", torch.tensor(1.0))
        self.register_buffer("act_zero_point", torch.tensor(0.0))
        self.act_quant_enabled = False

    def forward(self, x):
        W_hat = fake_quantize_weight(self.linear.weight, bits=self.bits)
        out = F.linear(x, weight=W_hat, bias=self.linear.bias)

        if self.act_quant_enabled:
            out = fake_quantize_activation(out, self.act_scale, self.act_zero_point, bits=self.act_bits)

        return out


def replace_conv_layers(model, bits=8, act_bits=8):
    to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            to_replace.append(name)

    for name in to_replace:
        *parent_path, attr_name = name.split(".")
        parent = model
        for p in parent_path:
            parent = getattr(parent, p)
        old_conv = getattr(parent, attr_name)
        setattr(parent, attr_name, QuantConv2d(old_conv, bits=bits, act_bits=act_bits))

    return model


def replace_linear_layers(model, bits=8, act_bits=8):
    to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            to_replace.append(name)

    for name in to_replace:
        *parent_path, attr_name = name.split(".")
        parent = model
        for p in parent_path:
            parent = getattr(parent, p)
        old_linear = getattr(parent, attr_name)
        setattr(parent, attr_name, QuantLinear(old_linear, bits=bits, act_bits=act_bits))

    return model


def apply_calibration(model, scales, zero_points):
    applied = 0
    for name, module in model.named_modules():
        if hasattr(module, "act_scale") and name in scales:
            module.act_scale.copy_(scales[name])
            module.act_zero_point.copy_(zero_points[name])
            module.act_quant_enabled = True
            applied += 1
    print(f"Calibration applied to {applied} layers.")
    return model
