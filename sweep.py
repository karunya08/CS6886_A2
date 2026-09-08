'''Sweep runners for hyperparameter search.

Defines WandB sweep configurations and runner functions used to
explore quantization and pruning hyperparameters for the pipeline.
'''

import sys
import torch
import wandb

from baseline import baseline, prepare_data
from pipeline.quant import run_quantization
from pipeline.prune import run_pruning


CHECKPOINT_PATH = "baseline_best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_NAME = "mobilenetv2-cifar10-compression"


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

    _, result = run_quantization(
        CHECKPOINT_PATH,

        weight_bits=config.weight_bits,
        act_bits=config.act_bits,

        num_calib_batches=10,

        finetune_epochs=2,
        finetune_lr=1e-4,
        finetune_momentum=0.9,
        finetune_weight_decay=0.0,

        device=DEVICE
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

    # --------------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------------

    train_loader, test_loader = prepare_data()

    # --------------------------------------------------------
    # LOAD ORIGINAL FP32 MODEL
    # --------------------------------------------------------

    original_model = baseline()

    original_model.load_state_dict(
        torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE
        )
    )

    original_model = original_model.to(DEVICE)
    original_model.eval()

    # --------------------------------------------------------
    # STAGE 1: FIXED W4A8 QUANTIZATION
    # --------------------------------------------------------

    quantized_model, quant_results = run_quantization(
        CHECKPOINT_PATH,

        weight_bits=4,
        act_bits=8,

        num_calib_batches=10,

        finetune_epochs=3,
        finetune_lr=1e-4,
        finetune_momentum=0.9,
        finetune_weight_decay=0.0,

        device=DEVICE
    )

    # --------------------------------------------------------
    # STAGE 2: SENSITIVITY-AWARE STRUCTURED PRUNING
    # --------------------------------------------------------

    pruned_model, prune_results = run_pruning(
        model=quantized_model,

        original_model=original_model,

        train_loader=train_loader,
        test_loader=test_loader,

        sparsity=config.sparsity,

        min_keep_ratio=0.25,

        act_bits=8,

        num_calib_batches=10,

        finetune_epochs=3,
        finetune_lr=1e-4,
        finetune_momentum=0.9,
        finetune_weight_decay=0.0,

        device=DEVICE
    )

    # --------------------------------------------------------
    # COMBINE RESULTS
    # --------------------------------------------------------

    result = {
        **quant_results,
        **prune_results,

        "weight_bits": 4,
        "act_bits": 8,
        "sparsity": config.sparsity
    }

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
            project=PROJECT_NAME
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
            project=PROJECT_NAME
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