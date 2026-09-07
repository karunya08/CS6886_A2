import torch

from compression import QuantConv2d, QuantLinear

FP32_BITS = 32


# ============================================================
# WEIGHT ACCOUNTING
# ============================================================

def count_original_weight_bits(model):
    """
    Count FP32 weights in the ORIGINAL, unpruned model.
    """

    total_bits = 0

    for module in model.modules():

        if isinstance(module, torch.nn.Conv2d):
            total_bits += module.weight.numel() * FP32_BITS

            if module.bias is not None:
                total_bits += module.bias.numel() * FP32_BITS

        elif isinstance(module, torch.nn.Linear):
            total_bits += module.weight.numel() * FP32_BITS

            if module.bias is not None:
                total_bits += module.bias.numel() * FP32_BITS

    return total_bits


def count_compressed_weight_bits(model):
    """
    Count weights in the FINAL compressed/pruned model.

    Quantized weights:
        n_weights * weight_bits

    Weight overhead:
        one FP32 scale per output channel
    """

    compressed_bits = 0
    overhead_bits = 0

    for module in model.modules():

        if isinstance(module, QuantConv2d):

            n_weights = module.conv.weight.numel()
            n_out_channels = module.conv.weight.shape[0]

            compressed_bits += (
                n_weights * module.bits
            )

            # One FP32 scale per output channel
            overhead_bits += (
                n_out_channels * FP32_BITS
            )

            # Bias remains FP32
            if module.conv.bias is not None:
                overhead_bits += (
                    module.conv.bias.numel()
                    * FP32_BITS
                )

        elif isinstance(module, QuantLinear):

            n_weights = module.linear.weight.numel()
            n_out_channels = module.linear.weight.shape[0]

            compressed_bits += (
                n_weights * module.bits
            )

            overhead_bits += (
                n_out_channels * FP32_BITS
            )

            # Bias remains FP32
            if module.linear.bias is not None:
                overhead_bits += (
                    module.linear.bias.numel()
                    * FP32_BITS
                )

    return compressed_bits, overhead_bits


def compute_weight_compression(
    original_model,
    compressed_model
):

    original_bits = count_original_weight_bits(
        original_model
    )

    compressed_bits, overhead_bits = (
        count_compressed_weight_bits(
            compressed_model
        )
    )

    total_compressed_bits = (
        compressed_bits
        + overhead_bits
    )

    original_mb = (
        original_bits / 8 / 1e6
    )

    compressed_mb = (
        total_compressed_bits / 8 / 1e6
    )

    overhead_mb = (
        overhead_bits / 8 / 1e6
    )

    compression_ratio = (
        original_mb / compressed_mb
    )

    return {
        "weight_original_mb": original_mb,
        "weight_compressed_mb": compressed_mb,
        "weight_overhead_mb": overhead_mb,
        "weight_compression_ratio":
            compression_ratio,
    }


# ============================================================
# ACTIVATION ACCOUNTING
# ============================================================

def count_activation_bits(
    model,
    sample_input,
    use_quantized_bits=False
):

    total_bits = 0
    overhead_bits = 0

    handles = []

    def make_hook(module):

        def hook(mod, inp, out):

            nonlocal total_bits
            nonlocal overhead_bits

            n_elements = out.numel()

            if use_quantized_bits:
                total_bits += (
                    n_elements
                    * mod.act_bits
                )

                # Activation scale + zero point
                overhead_bits += (
                    2 * FP32_BITS
                )

            else:
                total_bits += (
                    n_elements
                    * FP32_BITS
                )

        return hook

    # --------------------------------------------------------
    # Original model
    # --------------------------------------------------------

    if use_quantized_bits:

        target_types = (
            QuantConv2d,
            QuantLinear,
        )

    else:

        target_types = (
            torch.nn.Conv2d,
            torch.nn.Linear,
        )

    for module in model.modules():

        if isinstance(module, target_types):

            handles.append(
                module.register_forward_hook(
                    make_hook(module)
                )
            )

    model.eval()

    with torch.no_grad():
        model(sample_input)

    for handle in handles:
        handle.remove()

    return total_bits, overhead_bits


def estimate_activation_memory(
    original_model,
    compressed_model,
    sample_input
):

    # --------------------------------------------------------
    # ORIGINAL FP32 ACTIVATIONS
    # --------------------------------------------------------

    original_bits, _ = count_activation_bits(
        original_model,
        sample_input,
        use_quantized_bits=False
    )

    # --------------------------------------------------------
    # FINAL QUANTIZED + PRUNED ACTIVATIONS
    # --------------------------------------------------------

    compressed_bits, overhead_bits = (
        count_activation_bits(
            compressed_model,
            sample_input,
            use_quantized_bits=True
        )
    )

    total_compressed_bits = (
        compressed_bits
        + overhead_bits
    )

    original_mb = (
        original_bits / 8 / 1e6
    )

    compressed_mb = (
        total_compressed_bits / 8 / 1e6
    )

    compression_ratio = (
        original_mb / compressed_mb
    )

    return {
        "activation_original_mb":
            original_mb,

        "activation_compressed_mb":
            compressed_mb,

        "activation_overhead_mb":
            overhead_bits / 8 / 1e6,

        "activation_compression_ratio":
            compression_ratio,
    }


# ============================================================
# TOTAL MODEL ACCOUNTING
# ============================================================

def compute_model_size(
    original_model,
    compressed_model,
    sample_input
):

    weight_stats = compute_weight_compression(
        original_model,
        compressed_model
    )

    activation_stats = estimate_activation_memory(
        original_model,
        compressed_model,
        sample_input
    )

    total_original_mb = (
        weight_stats["weight_original_mb"]
        + activation_stats["activation_original_mb"]
    )

    total_compressed_mb = (
        weight_stats["weight_compressed_mb"]
        + activation_stats["activation_compressed_mb"]
    )

    overall_ratio = (
        total_original_mb
        / total_compressed_mb
    )

    return {
        **weight_stats,
        **activation_stats,

        "total_original_mb":
            total_original_mb,

        "total_compressed_mb":
            total_compressed_mb,

        "overall_compression_ratio":
            overall_ratio,
    }