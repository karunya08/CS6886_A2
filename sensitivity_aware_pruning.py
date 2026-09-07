import torch
import torch.nn as nn

from baseline import baseline
from compression import (
    replace_with_quantized_layers,
    calibrate_model,
    enable_activation_quantization,
)
from train import train_model
from size_accounting import compute_model_size


# ============================================================
# BLOCK DISCOVERY
# ============================================================

def get_inverted_residual_blocks(model):
    blocks = []

    for name, module in model.named_modules():
        if module.__class__.__name__ == "InvertedResidual":
            blocks.append((name, module))

    return blocks


def has_expand_stage(block):
    return hasattr(block, "conv") and len(block.conv) == 4


def get_prune_targets(model):
    targets = []

    for name, block in get_inverted_residual_blocks(model):

        if not has_expand_stage(block):
            continue

        expand = block.conv[0][0]

        if isinstance(expand, nn.Conv2d):
            targets.append({
                "name": name,
                "block": block,
                "expand": expand,
            })

    return targets


# ============================================================
# CHANNEL IMPORTANCE
# ============================================================

def compute_channel_importance(conv):
    W = conv.weight.detach()

    return torch.norm(
        W.reshape(W.shape[0], -1),
        p=2,
        dim=1
    )


def normalize_importance(importance):
    return importance / importance.mean().clamp(min=1e-12)


def collect_global_importance(model):

    targets = get_prune_targets(model)

    all_scores = []

    for target in targets:

        importance = compute_channel_importance(
            target["expand"]
        )

        normalized = normalize_importance(importance)

        target["importance"] = normalized

        for idx, score in enumerate(normalized):
            all_scores.append({
                "block": target["name"],
                "channel": idx,
                "score": score.item(),
            })

    return targets, all_scores


# ============================================================
# GLOBAL CHANNEL SELECTION
# ============================================================

def select_global_keep_indices(
    targets,
    all_scores,
    sparsity,
    min_keep_ratio=0.25,
):

    total_channels = len(all_scores)

    target_remove = int(
        round(total_channels * sparsity)
    )

    ranked = sorted(
        all_scores,
        key=lambda x: x["score"]
    )

    keep = {
        target["name"]: set(
            range(target["expand"].out_channels)
        )
        for target in targets
    }

    min_keep = {}

    for target in targets:

        n = target["expand"].out_channels

        min_keep[target["name"]] = max(
            1,
            int(round(n * min_keep_ratio))
        )

    removed = 0

    for item in ranked:

        if removed >= target_remove:
            break

        block_name = item["block"]
        channel = item["channel"]

        if len(keep[block_name]) <= min_keep[block_name]:
            continue

        keep[block_name].remove(channel)
        removed += 1

    actual_sparsity = removed / total_channels

    return keep, actual_sparsity


# ============================================================
# TENSOR SLICING
# ============================================================

def slice_conv_out(conv, keep_indices):

    idx = torch.tensor(
        sorted(keep_indices),
        dtype=torch.long,
        device=conv.weight.device
    )

    new_conv = nn.Conv2d(
        conv.in_channels,
        len(idx),
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    ).to(conv.weight.device)

    new_conv.weight.data.copy_(conv.weight.data[idx])

    if conv.bias is not None:
        new_conv.bias.data.copy_(conv.bias.data[idx])

    return new_conv


def slice_conv_in(conv, keep_indices):

    idx = torch.tensor(
        sorted(keep_indices),
        dtype=torch.long,
        device=conv.weight.device
    )

    new_conv = nn.Conv2d(
        len(idx),
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    ).to(conv.weight.device)

    new_conv.weight.data.copy_(
        conv.weight.data[:, idx]
    )

    if conv.bias is not None:
        new_conv.bias.data.copy_(conv.bias.data)

    return new_conv


def slice_depthwise_conv(conv, keep_indices):

    idx = torch.tensor(
        sorted(keep_indices),
        dtype=torch.long,
        device=conv.weight.device
    )

    n = len(idx)

    new_conv = nn.Conv2d(
        n,
        n,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        groups=n,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    ).to(conv.weight.device)

    new_conv.weight.data.copy_(
        conv.weight.data[idx])

    if conv.bias is not None:
        new_conv.bias.data.copy_(
            conv.bias.data[idx])

    return new_conv


def slice_bn(bn, keep_indices):

    idx = torch.tensor(
        sorted(keep_indices),
        dtype=torch.long,
        device=bn.weight.device
    )

    new_bn = nn.BatchNorm2d(
        len(idx),
        eps=bn.eps,
        momentum=bn.momentum,
        affine=bn.affine,
        track_running_stats=bn.track_running_stats,
    ).to(bn.weight.device)

    if bn.affine:
        new_bn.weight.data.copy_(
            bn.weight.data[idx]
        )
        new_bn.bias.data.copy_(
            bn.bias.data[idx]
        )

    if bn.track_running_stats:
        new_bn.running_mean.data.copy_(
            bn.running_mean.data[idx]
        )
        new_bn.running_var.data.copy_(
            bn.running_var.data[idx]
        )

    return new_bn


# ============================================================
# PRUNE ONE BLOCK
# ============================================================

def prune_block(block, keep_indices):

    conv = block.conv

    # Expansion
    expand = conv[0]

    expand[0] = slice_conv_out(
        expand[0],
        keep_indices
    )

    expand[1] = slice_bn(
        expand[1],
        keep_indices
    )

    # Depthwise
    depthwise = conv[1]

    depthwise[0] = slice_depthwise_conv(
        depthwise[0],
        keep_indices
    )

    depthwise[1] = slice_bn(
        depthwise[1],
        keep_indices
    )

    # Projection
    conv[2] = slice_conv_in(
        conv[2],
        keep_indices
    )


# ============================================================
# APPLY GLOBAL STRUCTURED PRUNING
# ============================================================

def apply_structured_pruning(
    model,
    sparsity,
    min_keep_ratio=0.25,
):

    targets, all_scores = collect_global_importance(model)

    keep_indices, actual_sparsity = (
        select_global_keep_indices(
            targets,
            all_scores,
            sparsity,
            min_keep_ratio,
        )
    )

    print("\n" + "=" * 60)
    print("GLOBAL STRUCTURED PRUNING")
    print("=" * 60)

    print(f"Requested sparsity : {sparsity:.2%}")
    print(f"Actual sparsity    : {actual_sparsity:.2%}")
    print(f"Blocks             : {len(targets)}")

    for target in targets:

        name = target["name"]
        block = target["block"]

        old_n = target["expand"].out_channels
        keep = keep_indices[name]

        print(
            f"{name}: "
            f"{old_n} -> {len(keep)}"
        )

        prune_block(
            block,
            keep
        )

    return model, actual_sparsity


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_structured_pruned_compression(
    checkpoint_path,
    weight_bits=4,
    act_bits=8,
    sparsity=0.5,
    min_keep_ratio=0.25,
    qat_epochs=3,
    prune_finetune_epochs=3,
    device=None,
):

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    print("\n" + "=" * 70)
    print("STRUCTURED PRUNING + QUANTIZATION")
    print("=" * 70)

    print(
        f"W{weight_bits}A{act_bits} | "
        f"Sparsity={sparsity:.0%} | "
        f"MinKeep={min_keep_ratio:.0%}"
    )


    # ========================================================
    # ORIGINAL MODEL
    # ========================================================

    # NEVER modify this model.
    # Used only for correct original-size accounting.

    original_model = baseline()

    original_model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device
        )
    )

    original_model = original_model.to(device)
    original_model.eval()


    # ========================================================
    # WORKING MODEL
    # ========================================================

    model = baseline()

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device
        )
    )

    model = model.to(device)


    # ========================================================
    # QUANTIZATION
    # ========================================================

    model = replace_with_quantized_layers(
        model,
        weight_bits=weight_bits,
        act_bits=act_bits,
    )

    model = model.to(device)


    # ========================================================
    # CALIBRATION
    # ========================================================

    model.eval()

    calibrate_model(
        model,
        device=device,
    )

    enable_activation_quantization(model)


    # ========================================================
    # QAT
    # ========================================================

    print("\nStarting QAT...")

    train_model(
        model,
        epochs=qat_epochs,
        device=device,
    )


    # ========================================================
    # STRUCTURED PRUNING
    # ========================================================

    model, actual_sparsity = apply_structured_pruning(
        model,
        sparsity=sparsity,
        min_keep_ratio=min_keep_ratio,
    )

    model = model.to(device)


    # ========================================================
    # PRUNING FINE-TUNING
    # ========================================================

    print("\nStarting pruning fine-tuning...")

    train_model(
        model,
        epochs=prune_finetune_epochs,
        device=device,
    )


    # ========================================================
    # FINAL CALIBRATION
    # ========================================================

    model.eval()

    calibrate_model(
        model,
        device=device,
    )

    enable_activation_quantization(model)


    # ========================================================
    # FINAL EVALUATION
    # ========================================================

    model.eval()

    # --------------------------------------------------------
    # USE THE SAME TEST LOADER / SAMPLE INPUT CODE THAT
    # WAS ALREADY IN YOUR ORIGINAL FILE.
    #
    # The important part is:
    #
    # sample_input = sample_input[:1].to(device)
    #
    # --------------------------------------------------------

    with torch.no_grad():

        correct = 0
        total = 0

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    accuracy = 100.0 * correct / total

    print(
        f"\nFinal accuracy: {accuracy:.2f}%"
    )


    # ========================================================
    # ACTIVATION SIZE INPUT
    # ========================================================

    # Take ONE image from the existing test batch.

    sample_input = images[:1].to(device)


    # ========================================================
    # CORRECT SIZE ACCOUNTING
    # ========================================================

    size_results = compute_model_size(
        original_model,
        model,
        sample_input,
    )


    # ========================================================
    # RESULTS
    # ========================================================

    results = {
        "weight_bits": weight_bits,
        "act_bits": act_bits,

        "sparsity": sparsity,
        "actual_sparsity": actual_sparsity,

        "accuracy": accuracy,

        **size_results,
    }


    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    for key, value in results.items():
        print(f"{key}: {value}")

    return model, results