
import torch
from compression import QuantConv2d, QuantLinear

FP32_BITS = 32

def layer_weight_bits(module):
    if isinstance(module, QuantConv2d):
        n_weights = module.conv.weight.numel()
        n_out_channels = module.conv.weight.shape[0]
    elif isinstance(module, QuantLinear):
        n_weights = module.linear.weight.numel()
        n_out_channels = module.linear.weight.shape[0]
    else:
        return None

    original_bits = n_weights * FP32_BITS
    compressed_bits = n_weights * module.bits
    overhead_bits = n_out_channels * FP32_BITS

    return {
        "original_bits": original_bits,
        "compressed_bits": compressed_bits,
        "overhead_bits": overhead_bits,
        "total_compressed_bits": compressed_bits + overhead_bits,
    }

def compute_weight_compression(model):
    total_original, total_compressed, total_overhead = 0, 0, 0
    for module in model.modules():
        info = layer_weight_bits(module)
        if info is None:
            continue
        total_original += info["original_bits"]
        total_compressed += info["total_compressed_bits"]
        total_overhead += info["overhead_bits"]

    original_mb = total_original / 8 / 1e6
    compressed_mb = total_compressed / 8 / 1e6
    overhead_mb = total_overhead / 8 / 1e6

    return {
        "weight_original_mb": original_mb,
        "weight_compressed_mb": compressed_mb,
        "weight_overhead_mb": overhead_mb,
        "weight_compression_ratio": original_mb / compressed_mb,
    }

def estimate_activation_memory(model, sample_input):
    activation_bits_original = 0
    activation_bits_compressed = 0
    activation_overhead_bits = 0
    handles = []

    def make_hook(module):
        def hook(mod, inp, out):
            nonlocal activation_bits_original, activation_bits_compressed, activation_overhead_bits
            n_elements = out.numel()
            activation_bits_original += n_elements * FP32_BITS
            activation_bits_compressed += n_elements * mod.act_bits
            activation_overhead_bits += 2 * FP32_BITS
        return hook

    for module in model.modules():
        if isinstance(module, (QuantConv2d, QuantLinear)):
            handles.append(module.register_forward_hook(make_hook(module)))

    model.eval()
    with torch.no_grad():
        model(sample_input)

    for h in handles:
        h.remove()

    original_mb = activation_bits_original / 8 / 1e6
    compressed_mb = (activation_bits_compressed + activation_overhead_bits) / 8 / 1e6

    return {
        "activation_original_mb": original_mb,
        "activation_compressed_mb": compressed_mb,
        "activation_compression_ratio": original_mb / compressed_mb,
    }


def compute_model_size(model, sample_input):
    weight_stats = compute_weight_compression(model)
    act_stats = estimate_activation_memory(model, sample_input)

    total_original = weight_stats["weight_original_mb"] + act_stats["activation_original_mb"]
    total_compressed = weight_stats["weight_compressed_mb"] + act_stats["activation_compressed_mb"]

    return {
        **weight_stats,
        **act_stats,
        "total_original_mb": total_original,
        "total_compressed_mb": total_compressed,
        "overall_compression_ratio": total_original / total_compressed,
    }