import torch
from compression import QuantConv2d, QuantLinear
FP32_BITS = 32

def layer_weight_bits(original_module, compressed_module):
    if isinstance(compressed_module, QuantConv2d):
        n_weights = compressed_module.conv.weight.numel()
        n_out_channels = compressed_module.conv.weight.shape[0]
    elif isinstance(compressed_module, QuantLinear):
        n_weights = compressed_module.linear.weight.numel()
        n_out_channels = compressed_module.linear.weight.shape[0]
    else:
        return None
    original_bits = original_module.weight.numel() * FP32_BITS
    compressed_bits = n_weights * compressed_module.bits
    overhead_bits = n_out_channels * FP32_BITS
    return {
        "original_bits": original_bits,
        "compressed_bits": compressed_bits,
        "overhead_bits": overhead_bits,
        "total_compressed_bits": compressed_bits + overhead_bits,
    }

def compute_weight_compression(original_model, compressed_model):
    original_modules = [
        m for m in original_model.modules()
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))
    ]
    compressed_modules = [
        m for m in compressed_model.modules()
        if isinstance(m, (QuantConv2d, QuantLinear))
    ]
    total_original = sum(m.weight.numel() * FP32_BITS for m in original_modules)
    total_compressed = 0
    total_overhead = 0
    for module in compressed_modules:
        if isinstance(module, QuantConv2d):
            n_weights = module.conv.weight.numel()
            n_out_channels = module.conv.weight.shape[0]
            bits = module.bits
        else:
            n_weights = module.linear.weight.numel()
            n_out_channels = module.linear.weight.shape[0]
            bits = module.bits
        total_compressed += n_weights * bits
        total_overhead += n_out_channels * FP32_BITS
    original_mb = total_original / 8 / 1e6
    compressed_mb = (total_compressed + total_overhead) / 8 / 1e6
    overhead_mb = total_overhead / 8 / 1e6
    return {
        "weight_original_mb": original_mb,
        "weight_compressed_mb": compressed_mb,
        "weight_overhead_mb": overhead_mb,
        "weight_compression_ratio": original_mb / compressed_mb,
    }

def estimate_activation_memory(original_model, compressed_model, sample_input):
    original_bits = 0
    compressed_bits = 0
    compressed_overhead = 0
    original_handles = []
    compressed_handles = []

    def original_hook(mod, inp, out):
        nonlocal original_bits
        original_bits += out.numel() * FP32_BITS

    def compressed_hook(mod, inp, out):
        nonlocal compressed_bits, compressed_overhead
        compressed_bits += out.numel() * mod.act_bits
        compressed_overhead += 2 * FP32_BITS

    for module in original_model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            original_handles.append(module.register_forward_hook(original_hook))

    for module in compressed_model.modules():
        if isinstance(module, (QuantConv2d, QuantLinear)):
            compressed_handles.append(module.register_forward_hook(compressed_hook))

    original_model.eval()
    compressed_model.eval()
    with torch.no_grad():
        original_model(sample_input)
        compressed_model(sample_input)

    for h in original_handles:
        h.remove()
    for h in compressed_handles:
        h.remove()

    original_mb = original_bits / 8 / 1e6
    compressed_mb = (compressed_bits + compressed_overhead) / 8 / 1e6
    return {
        "activation_original_mb": original_mb,
        "activation_compressed_mb": compressed_mb,
        "activation_compression_ratio": original_mb / compressed_mb,
    }

def compute_model_size(original_model, compressed_model, sample_input):
    weight_stats = compute_weight_compression(original_model, compressed_model)
    act_stats = estimate_activation_memory(original_model, compressed_model, sample_input)
    total_original = weight_stats["weight_original_mb"] + act_stats["activation_original_mb"]
    total_compressed = weight_stats["weight_compressed_mb"] + act_stats["activation_compressed_mb"]
    return {
        **weight_stats,
        **act_stats,
        "total_original_mb": total_original,
        "total_compressed_mb": total_compressed,
        "overall_compression_ratio": total_original / total_compressed,
    }