import torch
import torch.nn as nn

from calibrate import run_calibration
from eval import run_eval
from size_accounting import compute_model_size
from pipeline.quant import finetune_quantized


# ============================================================
# FIND INVERTED RESIDUAL BLOCKS
# ============================================================

def get_inverted_residual_blocks(model):

    return [
        (name, module)
        for name, module in model.named_modules()
        if type(module).__name__ == "InvertedResidual"
    ]


def has_expand_stage(block):
    return len(block.conv) == 4


# ============================================================
# PRUNING TARGETS
# ============================================================

def get_prune_targets(block):
    expand_conv = block.conv[0][0]
    expand_bn = block.conv[0][1]

    dw_conv = block.conv[1][0]
    dw_bn = block.conv[1][1]

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
    W = conv_module.conv.weight.detach()
    return W.view(W.shape[0], -1).norm(p=2, dim=1)


def normalize_importance(importance):
    mean = importance.mean().clamp(min=1e-8)
    return importance / mean


# ============================================================
# GLOBAL IMPORTANCE
# ============================================================

def collect_global_importance(model):
    records = []
    for name, block in get_inverted_residual_blocks(model):
        if not has_expand_stage(block):
            continue

        expand_conv, _, _, _, _ = get_prune_targets(block)
        importance = compute_channel_importance(expand_conv)
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
# SELECT CHANNELS
# ============================================================

def select_global_keep_indices(model, sparsity, min_keep_ratio=0.25):
    records = collect_global_importance(model)

    if not records:
        return {}, 0, 0

    n_total = len(records)
    n_to_prune = int(round(sparsity * n_total))

    block_sizes = {}
    for _, block_name, _ in records:
        block_sizes.setdefault(block_name, 0)
        block_sizes[block_name] += 1

    min_keep = {
        name: max(
            1,
            int(round(
                size * min_keep_ratio
            ))
        )
        for name, size in block_sizes.items()
    }

    records.sort(key=lambda x: x[0])

    prune_by_block = {name: set() for name in block_sizes}

    current_keep = {name: size for name, size in block_sizes.items()}

    n_pruned = 0
    for _, block_name, channel_idx in records:
        if n_pruned >= n_to_prune:
            break

        if (current_keep[block_name] <= min_keep[block_name]):
            continue

        prune_by_block[block_name].add(channel_idx)
        current_keep[block_name] -= 1
        n_pruned += 1

    keep_indices = {}

    for block_name, n_channels in block_sizes.items():
        prune_set = prune_by_block[block_name]

        keep = [idx for idx in range(n_channels) if idx not in prune_set]
        keep_indices[block_name] = torch.tensor(
            keep,
            dtype=torch.long
        )

    return (
        keep_indices,
        n_pruned,
        n_total
    )


# ============================================================
# SLICE CONV OUTPUT
# ============================================================

def slice_conv_out(conv, keep_idx):

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

    new_conv.weight.data = (conv.weight.data[keep_idx].clone())

    if conv.bias is not None:
        new_conv.bias.data = (conv.bias.data[keep_idx].clone())

    return new_conv.to(conv.weight.device)


# ============================================================
# SLICE CONV INPUT
# ============================================================

def slice_conv_in(conv, keep_idx):

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

    new_conv.weight.data = (conv.weight.data[:, keep_idx].clone())

    if conv.bias is not None:
        new_conv.bias.data = (conv.bias.data.clone())

    return new_conv.to(conv.weight.device)


# ============================================================
# SLICE DEPTHWISE CONV
# ============================================================

def slice_depthwise_conv(conv, keep_idx):

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

    new_conv.weight.data = (conv.weight.data[keep_idx].clone())

    if conv.bias is not None:
        new_conv.bias.data = (conv.bias.data[keep_idx].clone())

    return new_conv.to(conv.weight.device)


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
        new_bn.weight.data = (bn.weight.data[keep_idx].clone())
        new_bn.bias.data = (bn.bias.data[keep_idx].clone())

    if bn.track_running_stats:
        new_bn.running_mean.data = (bn.running_mean.data[keep_idx].clone())
        new_bn.running_var.data = (bn.running_var.data[keep_idx].clone())

    return new_bn.to(keep_idx.device)


# ============================================================
# PRUNE ONE BLOCK
# ============================================================

def prune_block(block, keep_idx):
    (
        expand_conv,
        expand_bn,
        dw_conv,
        dw_bn,
        project_conv
    ) = get_prune_targets(block)

    n_original = (expand_conv.conv.weight.shape[0])

    # Expand
    expand_conv.conv = slice_conv_out(expand_conv.conv, keep_idx)

    # Depthwise
    dw_conv.conv = slice_depthwise_conv(dw_conv.conv, keep_idx)

    # Projection
    project_conv.conv = slice_conv_in(project_conv.conv, keep_idx)

    # BatchNorm
    block.conv[0][1] = slice_bn(expand_bn, keep_idx)
    block.conv[1][1] = slice_bn(dw_bn, keep_idx)

    return len(keep_idx), n_original


# ============================================================
# APPLY STRUCTURED PRUNING
# ============================================================

def apply_structured_pruning(
    model,
    sparsity,
    min_keep_ratio=0.25
):
    (
        keep_indices,
        n_pruned,
        n_total
    ) = select_global_keep_indices(model, sparsity, min_keep_ratio)

    if n_total == 0:
        print("No eligible expansion channels found.")
        return model, 0.0

    n_kept_total = 0
    n_original_total = 0
    n_blocks = 0

    for name, block in get_inverted_residual_blocks(model):
        if not has_expand_stage(block):
            continue

        keep_idx = keep_indices[name]
        expand_conv, _, _, _, _ = (get_prune_targets(block))

        keep_idx = keep_idx.to(expand_conv.conv.weight.device)

        n_kept, n_original = prune_block(block, keep_idx)

        n_kept_total += n_kept
        n_original_total += n_original
        n_blocks += 1

    actual_sparsity = (
        1.0 -
        n_kept_total / n_original_total
    )

    print(
        f"Pruned {n_blocks} blocks | "
        f"Kept {n_kept_total}/{n_original_total} "
        f"expansion channels | "
        f"Actual sparsity: "
        f"{actual_sparsity * 100:.2f}%"
    )

    return model, actual_sparsity


# ============================================================
# PRUNING PIPELINE
# ============================================================

def run_pruning(
    model,
    original_model,
    train_loader,
    test_loader,

    sparsity=0.5,
    min_keep_ratio=0.25,

    act_bits=8,

    num_calib_batches=10,

    finetune_epochs=3,
    finetune_lr=1e-4,
    finetune_momentum=0.9,
    finetune_weight_decay=0.0,

    device="cuda"
):

    # --------------------------------------------------------
    # STRUCTURED PRUNING
    # --------------------------------------------------------

    model, actual_sparsity = (
        apply_structured_pruning(
            model,
            sparsity=sparsity,
            min_keep_ratio=min_keep_ratio
        )
    )

    model = model.to(device)

    # --------------------------------------------------------
    # RECALIBRATION
    # --------------------------------------------------------

    model = run_calibration(
        model,
        train_loader,
        bits=act_bits,
        num_batches=num_calib_batches,
        device=device
    )

    # --------------------------------------------------------
    # FINE-TUNING
    # --------------------------------------------------------

    model = finetune_quantized(
        model,
        train_loader,
        epochs=finetune_epochs,
        lr=finetune_lr,
        momentum=finetune_momentum,
        weight_decay=finetune_weight_decay,
        device=device
    )

    # --------------------------------------------------------
    # FINAL CALIBRATION
    # --------------------------------------------------------

    model = run_calibration(
        model,
        train_loader,
        bits=act_bits,
        num_batches=num_calib_batches,
        device=device
    )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    eval_results = run_eval(model, test_loader)

    # --------------------------------------------------------
    # SIZE ACCOUNTING
    # --------------------------------------------------------

    sample_input, _ = next(iter(test_loader))
    sample_input = sample_input[:1].to(device)

    size_results = compute_model_size(original_model, model, sample_input)

    results = {
        **eval_results,
        **size_results,

        "target_sparsity": sparsity,
        "actual_sparsity": actual_sparsity,

        "min_keep_ratio": min_keep_ratio,

        "prune_finetune_epochs": finetune_epochs,
        "prune_finetune_lr": finetune_lr,
        "prune_finetune_momentum": finetune_momentum,
        "prune_finetune_weight_decay": finetune_weight_decay,

        "prune_calib_batches": num_calib_batches
    }

    return model, results