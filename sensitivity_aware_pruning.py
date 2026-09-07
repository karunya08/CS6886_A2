import torch
import torch.nn as nn

from baseline import baseline, prepare_data
from calibrate import run_calibration
from compress import finetune_quantized
from eval import run_eval
from size_accounting import compute_model_size


# ============================================================
# FIND MOBILEV2 INVERTED RESIDUAL BLOCKS
# ============================================================

def get_inverted_residual_blocks(model):
    """
    Find torchvision MobileNetV2 InvertedResidual blocks.

    We use the class name rather than importing torchvision's
    internal class directly.
    """

    return [
        (name, m)
        for name, m in model.named_modules()
        if type(m).__name__ == "InvertedResidual"
    ]


def has_expand_stage(block):
    """
    Expansion blocks have:

        [expand, depthwise, project_conv, project_bn]

    while expand_ratio == 1 blocks have:

        [depthwise, project_conv, project_bn]

    We only prune blocks with an explicit expansion stage.
    """

    return len(block.conv) == 4


# ============================================================
# GET PRUNING TARGETS
# ============================================================

def get_prune_targets(block):
    """
    Return the layers associated with the internal expansion dimension.

        expand_conv
        expand_bn
        depthwise_conv
        depthwise_bn
        project_conv
    """

    expand_conv, expand_bn = block.conv[0][0], block.conv[0][1]

    dw_conv, dw_bn = block.conv[1][0], block.conv[1][1]

    project_conv = block.conv[2]

    return (
        expand_conv,
        expand_bn,
        dw_conv,
        dw_bn,
        project_conv
    )


# ============================================================
# CHANNEL IMPORTANCE
# ============================================================

def compute_channel_importance(conv_module):
    """
    Compute L2 importance for every output channel of the
    expansion convolution.

    For channel c:

        importance[c] = ||W_c||_2

    where W_c is the complete convolutional filter.
    """

    W = conv_module.conv.weight.detach()

    return W.view(
        W.shape[0], -1
    ).norm(
        p=2,
        dim=1
    )


def normalize_importance(importance):
    """
    Normalize importance within one block.

    Mean becomes approximately 1.

    This prevents blocks with inherently larger weight magnitudes
    from dominating the global ranking.
    """

    return importance / importance.mean().clamp(min=1e-8)


# ============================================================
# COLLECT GLOBAL IMPORTANCE
# ============================================================

def collect_global_importance(model):
    """
    Collect normalized channel importance from every eligible
    inverted-residual block.

    Returns a list containing:

        (score, block_name, channel_index)

    Example:

        (0.42, 'features.2', 17)
        (0.73, 'features.2', 31)
        (1.24, 'features.3', 5)
        ...
    """

    records = []

    for name, block in get_inverted_residual_blocks(model):

        if not has_expand_stage(block):
            continue

        expand_conv, _, _, _, _ = get_prune_targets(block)

        importance = compute_channel_importance(
            expand_conv
        )

        normalized = normalize_importance(
            importance
        )

        for channel_idx, score in enumerate(normalized):

            records.append(
                (
                    float(score.item()),
                    name,
                    channel_idx
                )
            )

    return records


# ============================================================
# GLOBAL CHANNEL SELECTION
# ============================================================

def select_global_keep_indices(
    model,
    sparsity,
    min_keep_ratio=0.25
):
    """
    Globally rank all normalized channel importance scores.

    Target sparsity:
        e.g. 0.50 means remove 50% of all eligible expansion
        channels across the network.

    min_keep_ratio:
        Prevents a single block from being completely destroyed.

        Example:
            min_keep_ratio=0.25

        means every block must retain at least 25% of its
        original expansion channels.
    """

    records = collect_global_importance(model)

    if len(records) == 0:
        return {}, 0, 0

    # --------------------------------------------------------
    # Count original channels
    # --------------------------------------------------------

    n_total = len(records)

    n_to_prune = int(
        round(sparsity * n_total)
    )

    # --------------------------------------------------------
    # Determine minimum number of channels each block must keep
    # --------------------------------------------------------

    block_sizes = {}

    for _, block_name, _ in records:

        if block_name not in block_sizes:
            block_sizes[block_name] = 0

        block_sizes[block_name] += 1

    min_keep = {
        name: max(
            1,
            int(round(size * min_keep_ratio))
        )
        for name, size in block_sizes.items()
    }

    # --------------------------------------------------------
    # Sort globally from least important -> most important
    # --------------------------------------------------------

    records.sort(
        key=lambda x: x[0]
    )

    # --------------------------------------------------------
    # Select channels to prune
    # --------------------------------------------------------

    prune_by_block = {
        name: set()
        for name in block_sizes
    }

    current_keep = {
        name: block_sizes[name]
        for name in block_sizes
    }

    n_pruned = 0

    for score, block_name, channel_idx in records:

        if n_pruned >= n_to_prune:
            break

        # Do not allow this block to fall below
        # its minimum number of channels.
        if current_keep[block_name] <= min_keep[block_name]:
            continue

        prune_by_block[block_name].add(
            channel_idx
        )

        current_keep[block_name] -= 1

        n_pruned += 1

    # --------------------------------------------------------
    # Convert prune indices -> keep indices
    # --------------------------------------------------------

    keep_indices = {}

    for block_name, n_channels in block_sizes.items():

        prune_set = prune_by_block[block_name]

        keep = [
            idx
            for idx in range(n_channels)
            if idx not in prune_set
        ]

        keep_indices[block_name] = torch.tensor(
            keep,
            dtype=torch.long
        )

    actual_sparsity = (
        n_pruned / n_total
        if n_total > 0
        else 0.0
    )

    return (
        keep_indices,
        n_pruned,
        n_total
    )


# ============================================================
# SLICE CONV OUTPUT CHANNELS
# ============================================================

def slice_conv_out(conv, keep_idx):
    """
    Expand convolution:

        [Cexp, Cin, 1, 1]

    becomes:

        [Cexp_new, Cin, 1, 1]
    """

    new_conv = nn.Conv2d(
        conv.in_channels,
        len(keep_idx),
        conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None
    )

    new_conv.weight.data = (
        conv.weight.data[keep_idx].clone()
    )

    if conv.bias is not None:
        new_conv.bias.data = (
            conv.bias.data[keep_idx].clone()
        )

    return new_conv.to(
        conv.weight.device
    )


# ============================================================
# SLICE CONV INPUT CHANNELS
# ============================================================

def slice_conv_in(conv, keep_idx):
    """
    Project convolution:

        [Cout, Cexp, 1, 1]

    becomes:

        [Cout, Cexp_new, 1, 1]
    """

    new_conv = nn.Conv2d(
        len(keep_idx),
        conv.out_channels,
        conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None
    )

    new_conv.weight.data = (
        conv.weight.data[:, keep_idx].clone()
    )

    if conv.bias is not None:
        new_conv.bias.data = (
            conv.bias.data.clone()
        )

    return new_conv.to(
        conv.weight.device
    )


# ============================================================
# SLICE DEPTHWISE CONV
# ============================================================

def slice_depthwise_conv(conv, keep_idx):
    """
    Depthwise convolution:

        [Cexp, 1, K, K]

    becomes:

        [Cexp_new, 1, K, K]

    with:

        in_channels = out_channels = groups = Cexp_new
    """

    new_channels = len(keep_idx)

    new_conv = nn.Conv2d(
        new_channels,
        new_channels,
        conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=new_channels,
        bias=conv.bias is not None
    )

    new_conv.weight.data = (
        conv.weight.data[keep_idx].clone()
    )

    if conv.bias is not None:
        new_conv.bias.data = (
            conv.bias.data[keep_idx].clone()
        )

    return new_conv.to(
        conv.weight.device
    )


# ============================================================
# SLICE BATCHNORM
# ============================================================

def slice_bn(bn, keep_idx):

    new_bn = nn.BatchNorm2d(
        len(keep_idx),
        eps=bn.eps,
        momentum=bn.momentum,
        affine=bn.affine,
        track_running_stats=bn.track_running_stats
    )

    if bn.affine:

        new_bn.weight.data = (
            bn.weight.data[keep_idx].clone()
        )

        new_bn.bias.data = (
            bn.bias.data[keep_idx].clone()
        )

    if bn.track_running_stats:

        new_bn.running_mean.data = (
            bn.running_mean.data[keep_idx].clone()
        )

        new_bn.running_var.data = (
            bn.running_var.data[keep_idx].clone()
        )

    return new_bn.to(
        bn.weight.device
        if bn.affine
        else next(bn.parameters()).device
    )


# ============================================================
# PRUNE ONE BLOCK
# ============================================================

def prune_block(block, keep_idx):
    """
    Physically shrink the internal expansion dimension.

    Data path:

        Cin
          ↓
        Cexp
          ↓
        Cexp
          ↓
        Cout

    becomes:

        Cin
          ↓
        Cexp_new
          ↓
        Cexp_new
          ↓
        Cout
    """

    (
        expand_conv,
        expand_bn,
        dw_conv,
        dw_bn,
        project_conv
    ) = get_prune_targets(block)

    n_original = (
        expand_conv.conv.weight.shape[0]
    )

    # --------------------------------------------------------
    # Expand: shrink output channels
    # --------------------------------------------------------

    expand_conv.conv = slice_conv_out(
        expand_conv.conv,
        keep_idx
    )

    # --------------------------------------------------------
    # Depthwise: shrink input/output/groups
    # --------------------------------------------------------

    dw_conv.conv = slice_depthwise_conv(
        dw_conv.conv,
        keep_idx
    )

    # --------------------------------------------------------
    # Project: shrink input channels
    # --------------------------------------------------------

    project_conv.conv = slice_conv_in(
        project_conv.conv,
        keep_idx
    )

    # --------------------------------------------------------
    # Corresponding BatchNorm layers
    # --------------------------------------------------------

    block.conv[0][1] = slice_bn(
        expand_bn,
        keep_idx
    )

    block.conv[1][1] = slice_bn(
        dw_bn,
        keep_idx
    )

    return len(keep_idx), n_original


# ============================================================
# APPLY GLOBAL STRUCTURED PRUNING
# ============================================================

def apply_structured_pruning(
    model,
    sparsity,
    min_keep_ratio=0.25
):
    """
    Global importance-based structured pruning.

    Unlike the previous version, we DO NOT prune the same
    percentage from every block.

    Instead:

        1. Calculate L2 importance per channel.
        2. Normalize importance within each block.
        3. Globally rank every channel.
        4. Prune the least important global channels.
        5. Keep the remaining channels in each block.
    """

    (
        keep_indices,
        n_pruned,
        n_total
    ) = select_global_keep_indices(
        model,
        sparsity,
        min_keep_ratio=min_keep_ratio
    )

    if n_total == 0:

        print("No eligible expansion channels found.")

        return model, 0.0

    n_blocks_pruned = 0
    n_kept_total = 0
    n_original_total = 0

    for name, block in get_inverted_residual_blocks(model):

        if not has_expand_stage(block):
            continue

        keep_idx = keep_indices[name]

        # Put indices on the same device as the weights.
        expand_conv, _, _, _, _ = get_prune_targets(block)

        keep_idx = keep_idx.to(
            expand_conv.conv.weight.device
        )

        n_kept, n_original = prune_block(
            block,
            keep_idx
        )

        n_kept_total += n_kept
        n_original_total += n_original

        n_blocks_pruned += 1

    actual_sparsity = (
        1.0 -
        n_kept_total / n_original_total
    )

    print(
        f"Globally pruned {n_blocks_pruned} blocks | "
        f"{n_kept_total}/{n_original_total} expand channels kept | "
        f"{actual_sparsity * 100:.2f}% sparsity"
    )

    # Show how pruning was distributed.
    print("\nPer-block pruning:")
    
    for name, block in get_inverted_residual_blocks(model):

        if not has_expand_stage(block):
            continue

        keep_idx = keep_indices[name]

        # Original count can be recovered from the
        # number of records for that block.
        expand_conv, _, _, _, _ = get_prune_targets(block)

        current_channels = expand_conv.conv.out_channels

        print(
            f"  {name}: "
            f"kept {len(keep_idx)} channels"
        )

    return model, actual_sparsity


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def run_structured_pruned_compression(
    checkpoint_path,
    weight_bits=8,
    act_bits=4,
    sparsity=0.5,
    min_keep_ratio=0.25,

    num_calib_batches=10,

    quant_finetune_epochs=3,
    quant_finetune_lr=1e-4,

    prune_finetune_epochs=3,
    prune_finetune_lr=1e-4,

    device="cuda"
):
    """
    Complete pipeline:

        FP32 baseline
             ↓
        replace with QuantConv/QuantLinear
             ↓
        calibration
             ↓
        quantization fine-tuning
             ↓
        global importance-based structured pruning
             ↓
        recalibration
             ↓
        pruning fine-tuning
             ↓
        recalibration
             ↓
        evaluation + size/MAC accounting
    """

    # ========================================================
    # LOAD BASELINE
    # ========================================================

    model = baseline()

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device
        )
    )

    model = model.to(device)
    model.eval()

    train_loader, test_loader = prepare_data()

    # ========================================================
    # QUANTIZATION
    # ========================================================

    from compression import (
        replace_conv_layers,
        replace_linear_layers
    )

    model = replace_conv_layers(
        model,
        bits=weight_bits,
        act_bits=act_bits
    )

    model = replace_linear_layers(
        model,
        bits=weight_bits,
        act_bits=act_bits
    )

    # ========================================================
    # INITIAL CALIBRATION
    # ========================================================

    model = run_calibration(
        model,
        train_loader,
        bits=act_bits,
        num_batches=num_calib_batches,
        device=device
    )

    # ========================================================
    # QUANTIZATION FINE-TUNING
    # ========================================================

    model = finetune_quantized(
        model,
        train_loader,
        epochs=quant_finetune_epochs,
        lr=quant_finetune_lr,
        device=device
    )

    # Recalibrate after QAT
    model = run_calibration(
        model,
        train_loader,
        bits=act_bits,
        num_batches=num_calib_batches,
        device=device
    )

    # ========================================================
    # GLOBAL STRUCTURED PRUNING
    # ========================================================

    model, actual_sparsity = apply_structured_pruning(
        model,
        sparsity=sparsity,
        min_keep_ratio=min_keep_ratio
    )

    model = model.to(device)

    # ========================================================
    # RECALIBRATION AFTER STRUCTURAL PRUNING
    # ========================================================

    model = run_calibration(
        model,
        train_loader,
        bits=act_bits,
        num_batches=num_calib_batches,
        device=device
    )

    # ========================================================
    # FINE-TUNE PRUNED MODEL
    # ========================================================

    model = finetune_quantized(
        model,
        train_loader,
        epochs=prune_finetune_epochs,
        lr=prune_finetune_lr,
        device=device
    )

    # Final calibration
    model = run_calibration(
        model,
        train_loader,
        bits=act_bits,
        num_batches=num_calib_batches,
        device=device
    )

    # ========================================================
    # EVALUATION
    # ========================================================

    eval_results = run_eval(
        model,
        test_loader
    )

    # ========================================================
    # SIZE / PARAMETER / MAC ACCOUNTING
    # ========================================================

    sample_input, _ = next(
        iter(test_loader)
    )

    # IMPORTANT:
    # use one image so activation size and MACs
    # represent a single inference.
    sample_input = sample_input[:1].to(device)

    size_results = compute_model_size(
        model,
        sample_input
    )

    # ========================================================
    # RETURN EVERYTHING
    # ========================================================

    return {
        **eval_results,
        **size_results,

        "weight_bits": weight_bits,
        "act_bits": act_bits,

        "target_sparsity": sparsity,
        "actual_sparsity": actual_sparsity,

        "min_keep_ratio": min_keep_ratio,

        "quant_finetune_epochs": quant_finetune_epochs,
        "prune_finetune_epochs": prune_finetune_epochs,
    }


# ============================================================
# TEST RUN
# ============================================================

if __name__ == "__main__":

    result = run_structured_pruned_compression(
        "baseline_best.pt",

        weight_bits=4,
        act_bits=8,

        sparsity=0.5,

        min_keep_ratio=0.25,

        quant_finetune_epochs=3,
        quant_finetune_lr=1e-4,

        prune_finetune_epochs=3,
        prune_finetune_lr=1e-4,

        device="cuda"
    )

    print("\nFinal result:")
    print(result)