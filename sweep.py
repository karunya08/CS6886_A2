
import wandb
from compress import run_compression

CHECKPOINT_PATH = "baseline_best.pt"

sweep_config = {
    "method": "grid",
    "metric": {"name": "accuracy", "goal": "maximize"},
    "parameters": {
        "weight_bits": {"values": [8, 4, 2]},
        "act_bits": {"values": [8, 4, 2]},
    },
}


def sweep_run():
    wandb.init()
    config = wandb.config

    result = run_compression(
        CHECKPOINT_PATH,
        weight_bits=config.weight_bits,
        act_bits=config.act_bits,
        finetune_epochs=2,
    )

    wandb.log(result)


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="mobilenetv2-cifar10-compression")
    wandb.agent(sweep_id, function=sweep_run)
