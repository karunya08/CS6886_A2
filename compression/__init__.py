
'''Compression package exports.

Shorthand imports to expose quantization ops, quantized layer
wrappers and calibration hooks at the package level.
'''

from .quant_ops import fake_quantize_weight, fake_quantize_activation, calibrate_activation_scale
from .quant_layers import QuantConv2d, QuantLinear, replace_conv_layers, replace_linear_layers, apply_calibration
from .hooks import calibrate_activations
