import torch
import torch.nn as nn

from baseline import baseline, prepare_data
from calibrate import run_calibration
from compress import finetune_quantized
from eval import run_eval
from size_accounting import compute_model_size


def get_inverted_residual_blocks(model):
    """torchvision's InvertedResidual blocks stay untouched by replace_conv_layers
    (only the nn.Conv2d leaves inside them get wrapped as QuantConv2d), so we can
    still walk model.features and match on class name -- duck-typed rather than
    imported, since the exact torchvision internal class isn't guaranteed stable
    across versions."""
    return [(name, m) for name, m in model.named_modules() if type(m).__name__ == "InvertedResidual"]


def has_expand_stage(block):
    # expand_ratio == 1 (only the very first InvertedResidual block in MobileNetV2)
    # has no expand conv -- conv Sequential is [depthwise, project_conv, project_bn]
    # instead of [expand, depthwise, project_conv, project_bn]. We skip these blocks:
    # their "hidden dim" IS the block's input channel count, so pruning it would mean
    # touching the input side of the block (and, for the first block specifically,
    # the stem's output) -- exactly the cross-layer bookkeeping we're avoiding.
    return len(block.conv) == 4


def get_prune_targets(block):
    """Pull out the three quantized conv wrappers + two batchnorms that define a
    block's expand dimension. Indices follow torchvision's InvertedResidual.conv
    Sequential layout: [0]=expand ConvBNReLU, [1]=depthwise ConvBNReLU, [2]=project
    conv (plain, no BN/ReLU fused since the paper's linear bottleneck), [3]=project BN."""
    expand_conv, expand_bn = block.conv[0][0], block.conv[0][1]
    dw_conv, dw_bn = block.conv[1][0], block.conv[1][1]
    project_conv = block.conv[2]
    return expand_conv, expand_bn, dw_conv, dw_bn, project_conv


def compute_channel_importance(conv_module):
    # L2 norm per output-channel filter of the expand conv's weight -- same
    # magnitude-based criterion as the unstructured pruning, just aggregated
    # per-channel instead of per-weight.
    W = conv_module.conv.weight.detach()
    return W.view(W.shape[0], -1).norm(p=2, dim=1)


def select_keep_indices(importance, sparsity):
    n = importance.numel()
    n_keep = max(1, int(round((1 - sparsity) * n)))
    keep = torch.topk(importance, n_keep).indices
    return keep.sort().values


def slice_conv_out(conv, keep_idx):
    """Rebuild conv with fewer output channels, copying over the kept filters."""
    new_conv = nn.Conv2d(conv.in_channels, len(keep_idx), conv.kernel_size,
                          stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
                          groups=conv.groups, bias=conv.bias is not None)
    new_conv.weight.data = conv.weight.data[keep_idx].clone()
    if conv.bias is not None:
        new_conv.bias.data = conv.bias.data[keep_idx].clone()
    return new_conv.to(conv.weight.device)


def slice_conv_in(conv, keep_idx):
    """Rebuild conv with fewer input channels (project conv: only the input side
    shrinks, out_channels/oup is unchanged since that's the block's fixed output)."""
    new_conv = nn.Conv2d(len(keep_idx), conv.out_channels, conv.kernel_size,
                          stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
                          groups=conv.groups, bias=conv.bias is not None)
    new_conv.weight.data = conv.weight.data[:, keep_idx].clone()
    if conv.bias is not None:
        new_conv.bias.data = conv.bias.data.clone()
    return new_conv.to(conv.weight.device)


def slice_depthwise_conv(conv, keep_idx):
    # depthwise: groups == in_channels == out_channels, weight shape [C, 1, k, k].
    # Slicing along dim 0 both selects the surviving channels AND is the correct
    # per-group weight subset, since each group only ever owned 1 input channel.
    new_conv = nn.Conv2d(len(keep_idx), len(keep_idx), conv.kernel_size,
                          stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
                          groups=len(keep_idx), bias=conv.bias is not None)
    new_conv.weight.data = conv.weight.data[keep_idx].clone()
    if conv.bias is not None:
        new_conv.bias.data = conv.bias.data[keep_idx].clone()
    return new_conv.to(conv.weight.device)


def slice_bn(bn, keep_idx):
    new_bn = nn.BatchNorm2d(len(keep_idx), eps=bn.eps, momentum=bn.momentum,
                             affine=bn.affine, track_running_stats=bn.track_running_stats)
    new_bn.weight.data = bn.weight.data[keep_idx].clone()
    new_bn.bias.data = bn.bias.data[keep_idx].clone()
    new_bn.running_mean.data = bn.running_mean.data[keep_idx].clone()
    new_bn.running_var.data = bn.running_var.data[keep_idx].clone()
    return new_bn.to(bn.weight.device)


def prune_block(block, sparsity):
    """Physically shrink one inverted-residual block's expand dimension in place.
    Returns (n_kept, n_original) for sparsity bookkeeping."""
    expand_conv, expand_bn, dw_conv, dw_bn, project_conv = get_prune_targets(block)

    importance = compute_channel_importance(expand_conv)
    keep_idx = select_keep_indices(importance, sparsity)
    n_original = importance.numel()

    # expand_conv/project_conv are QuantConv2d wrappers -- only their inner .conv
    # (the real nn.Conv2d) gets rebuilt; .bits/.act_bits and the wrapper class stay
    # put, so the quantization forward logic needs zero changes.
    expand_conv.conv = slice_conv_out(expand_conv.conv, keep_idx)
    dw_conv.conv = slice_depthwise_conv(dw_conv.conv, keep_idx)
    project_conv.conv = slice_conv_in(project_conv.conv, keep_idx)

    block.conv[0][1] = slice_bn(expand_bn, keep_idx)
    block.conv[1][1] = slice_bn(dw_bn, keep_idx)

    return len(keep_idx), n_original


def apply_structured_pruning(model, sparsity):
    """Prune the expand dimension of every eligible InvertedResidual block to the
    same target sparsity. Skips blocks with expand_ratio == 1 (see has_expand_stage)."""
    n_kept_total, n_original_total = 0, 0
    n_blocks_pruned = 0

    for name, block in get_inverted_residual_blocks(model):
        if not has_expand_stage(block):
            continue
        n_kept, n_original = prune_block(block, sparsity)
        n_kept_total += n_kept
        n_original_total += n_original
        n_blocks_pruned += 1

    actual_sparsity = 1 - (n_kept_total / n_original_total) if n_original_total else 0.0
    print(f"Structurally pruned {n_blocks_pruned} blocks | "
          f"{n_kept_total}/{n_original_total} expand channels kept "
          f"({actual_sparsity * 100:.2f}% sparsity)")
    return model, actual_sparsity


def run_structured_pruned_compression(checkpoint_path, weight_bits=8, act_bits=4, sparsity=0.5,
                                       num_calib_batches=10,
                                       quant_finetune_epochs=3, quant_finetune_lr=1e-4,
                                       prune_finetune_epochs=2, prune_finetune_lr=1e-4,
                                       device="cuda"):
    """Same pipeline shape as pruning.run_pruned_compression, swapping in structural
    channel pruning for the mask-based version. No mask/hooks needed here -- the
    model is physically smaller after apply_structured_pruning, so compute_model_size
    reports a real reduction rather than the idealized one pruning.py needed."""
    model = baseline()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    train_loader, test_loader = prepare_data()

    from compression import replace_conv_layers, replace_linear_layers
    model = replace_conv_layers(model, bits=weight_bits, act_bits=act_bits)
    model = replace_linear_layers(model, bits=weight_bits, act_bits=act_bits)

    model = run_calibration(model, train_loader, bits=act_bits, num_batches=num_calib_batches, device=device)
    model = finetune_quantized(model, train_loader, epochs=quant_finetune_epochs, lr=quant_finetune_lr, device=device)
    model = run_calibration(model, train_loader, bits=act_bits, num_batches=num_calib_batches, device=device)

    model, actual_sparsity = apply_structured_pruning(model, sparsity)
    model = model.to(device)

    # recalibrate before this fine-tune too: every act_scale downstream of a pruned
    # block is now stale (channel counts changed, so min/max stats no longer apply)
    model = run_calibration(model, train_loader, bits=act_bits, num_batches=num_calib_batches, device=device)
    model = finetune_quantized(model, train_loader, epochs=prune_finetune_epochs, lr=prune_finetune_lr, device=device)
    model = run_calibration(model, train_loader, bits=act_bits, num_batches=num_calib_batches, device=device)

    eval_results = run_eval(model, test_loader)

    sample_input, _ = next(iter(test_loader))
    sample_input = sample_input.to(device)
    size_results = compute_model_size(model, sample_input)

    return {
        **eval_results,
        **size_results,
        "weight_bits": weight_bits,
        "act_bits": act_bits,
        "target_sparsity": sparsity,
        "actual_sparsity": actual_sparsity,
    }


if __name__ == "__main__":
    result = run_structured_pruned_compression("baseline_best.pt", weight_bits=4, act_bits=8, sparsity=0.5)
    print(result)