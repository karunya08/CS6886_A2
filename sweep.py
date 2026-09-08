import sys
import wandb

from compression import (
    run_compression,
    run_structured_pruned_compression
)


CHECKPOINT_PATH = "baseline_best.pt"


# ============================================================
# QUANTIZATION SWEEP
# ============================================================

quant_sweep_config = {
    "method": "grid",

    "metric": {
        "name": "accuracy",
        "goal": "maximize"
    },

    "parameters": {
        "weight_bits": {
            "values": [8, 4, 2]
        },

        "act_bits": {
            "values": [8, 4, 2]
        },
    },
}


def quant_sweep_run():

    wandb.init()

    config = wandb.config

    result = run_compression(
        CHECKPOINT_PATH,
        weight_bits=config.weight_bits,
        act_bits=config.act_bits,
        finetune_epochs=2,
    )

    wandb.log(result)


# ============================================================
# SENSITIVITY-AWARE PRUNING SWEEP
# ============================================================

pruning_sweep_config = {
    "method": "grid",

    "metric": {
        "name": "accuracy",
        "goal": "maximize"
    },

    "parameters": {
        "sparsity": {
            "values": [0.25, 0.40, 0.50, 0.60, 0.70]
        }
    },
}


def pruning_sweep_run():

    wandb.init()

    config = wandb.config

    result = run_structured_pruned_compression(
        CHECKPOINT_PATH,

        # Selected quantization configuration
        weight_bits=4,
        act_bits=8,

        # Swept pruning parameter
        sparsity=config.sparsity,

        # Fixed pruning parameters
        min_keep_ratio=0.25,

        num_calib_batches=10,

        quant_finetune_epochs=3,
        quant_finetune_lr=1e-4,

        prune_finetune_epochs=3,
        prune_finetune_lr=1e-4,

        device="cuda"
    )

    wandb.log(result)


# ============================================================
# SELECT SWEEP
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("Usage:")
        print("  python sweep.py quant")
        print("  python sweep.py prune")

        sys.exit(1)

    mode = sys.argv[1].lower()

    # --------------------------------------------------------
    # QUANTIZATION SWEEP
    # --------------------------------------------------------

    if mode == "quant":

        sweep_id = wandb.sweep(
            quant_sweep_config,
            project="mobilenetv2-cifar10-compression"
        )

        wandb.agent(
            sweep_id,
            function=quant_sweep_run
        )

    # --------------------------------------------------------
    # SENSITIVITY-AWARE PRUNING SWEEP
    # --------------------------------------------------------

    elif mode == "prune":
        sweep_id = wandb.sweep(
            pruning_sweep_config,
            project="mobilenetv2-cifar10-compression"
        )

        wandb.agent(
            sweep_id,
            function=pruning_sweep_run
        )

    # --------------------------------------------------------
    # INVALID MODE
    # --------------------------------------------------------

    else:
        print(f"Unknown mode: {mode}")
        print("Use either:")
        print("  python sweep.py quant")
        print("  python sweep.py prune")