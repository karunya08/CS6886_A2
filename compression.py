'''Top-level compression pipeline CLI.

Entrypoint script that runs quantization followed by
global importance based structured pruning and prints final results.
'''

import argparse
import torch

from baseline import baseline, prepare_data
from pipeline.quant import run_quantization
from pipeline.prune import run_pruning


def main():

    parser = argparse.ArgumentParser(
        description="Quantization + Sensitivity-Aware Structured Pruning Pipeline"
    )

    # --------------------------------------------------
    # Model / device
    # --------------------------------------------------
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="baseline_best.pt",
        help="Path to baseline checkpoint"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: cuda or cpu (default: auto)"
    )

    # --------------------------------------------------
    # Quantization parameters
    # --------------------------------------------------
    parser.add_argument(
        "--weight-bits",
        type=int,
        choices=[2, 4, 8],
        default=4,
        help="Weight quantization bits"
    )

    parser.add_argument(
        "--act-bits",
        type=int,
        choices=[2, 4, 8],
        default=8,
        help="Activation quantization bits"
    )

    parser.add_argument(
        "--quant-calib-batches",
        type=int,
        default=10,
        help="Number of batches used for activation calibration"
    )

    parser.add_argument(
        "--quant-epochs",
        type=int,
        default=3,
        help="QAT fine-tuning epochs"
    )

    parser.add_argument(
        "--quant-lr",
        type=float,
        default=1e-4,
        help="QAT learning rate"
    )

    parser.add_argument(
        "--quant-momentum",
        type=float,
        default=0.9,
        help="QAT SGD momentum"
    )

    parser.add_argument(
        "--quant-weight-decay",
        type=float,
        default=0.0,
        help="QAT weight decay"
    )

    # --------------------------------------------------
    # Pruning parameters
    # --------------------------------------------------
    parser.add_argument(
        "--sparsity",
        type=float,
        required=True,
        help="Target global channel sparsity (e.g. 0.25, 0.50, 0.70)"
    )

    parser.add_argument(
        "--min-keep-ratio",
        type=float,
        default=0.25,
        help="Minimum fraction of channels retained per block"
    )

    parser.add_argument(
        "--prune-calib-batches",
        type=int,
        default=10,
        help="Number of batches used for post-pruning calibration"
    )

    parser.add_argument(
        "--prune-epochs",
        type=int,
        default=3,
        help="Fine-tuning epochs after pruning"
    )

    parser.add_argument(
        "--prune-lr",
        type=float,
        default=1e-4,
        help="Post-pruning fine-tuning learning rate"
    )

    parser.add_argument(
        "--prune-momentum",
        type=float,
        default=0.9,
        help="Post-pruning SGD momentum"
    )

    parser.add_argument(
        "--prune-weight-decay",
        type=float,
        default=0.0,
        help="Post-pruning weight decay"
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # Device
    # --------------------------------------------------
    device = args.device

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("FINAL COMPRESSION PIPELINE")
    print("=" * 60)

    print(f"Checkpoint       : {args.checkpoint}")
    print(f"Device           : {device}")
    print(f"Weight bits      : {args.weight_bits}")
    print(f"Activation bits  : {args.act_bits}")
    print(f"Sparsity         : {args.sparsity * 100:.0f}%")
    print(f"Min keep ratio   : {args.min_keep_ratio}")
    print("=" * 60)

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    train_loader, test_loader = prepare_data()

    # --------------------------------------------------
    # Load original FP32 baseline
    # Used later for compression accounting
    # --------------------------------------------------
    original_model = baseline()
    original_model.load_state_dict(
        torch.load(args.checkpoint, map_location=device)
    )
    original_model = original_model.to(device)
    original_model.eval()

    # --------------------------------------------------
    # Stage 1: Quantization
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 1: QUANTIZATION")
    print("=" * 60)

    quantized_model, quant_results = run_quantization(
        checkpoint_path=args.checkpoint,
        weight_bits=args.weight_bits,
        act_bits=args.act_bits,
        num_calib_batches=args.quant_calib_batches,
        finetune_epochs=args.quant_epochs,
        finetune_lr=args.quant_lr,
        finetune_momentum=args.quant_momentum,
        finetune_weight_decay=args.quant_weight_decay,
        device=device,
    )

    # --------------------------------------------------
    # Stage 2: Sensitivity-Aware Structured Pruning
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 2: SENSITIVITY-AWARE PRUNING")
    print("=" * 60)

    pruned_model, prune_results = run_pruning(
        model=quantized_model,
        original_model=original_model,
        train_loader=train_loader,
        test_loader=test_loader,
        sparsity=args.sparsity,
        min_keep_ratio=args.min_keep_ratio,
        num_calib_batches=args.prune_calib_batches,
        finetune_epochs=args.prune_epochs,
        finetune_lr=args.prune_lr,
        finetune_momentum=args.prune_momentum,
        finetune_weight_decay=args.prune_weight_decay,
        device=device,
    )

    # --------------------------------------------------
    # Final results
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print("\nQuantization:")
    for key, value in quant_results.items():
        print(f"{key}: {value}")

    print("\nPruning:")
    for key, value in prune_results.items():
        print(f"{key}: {value}")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()