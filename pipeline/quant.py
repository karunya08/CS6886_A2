'''Quantization pipeline utilities.

Functions to create a quantized model from a checkpoint, perform
calibration, QAT fine-tuning, re-calibration, evaluation and size
accounting. Exposes `run_quantization` and a QAT helper.
'''

import torch
import torch.nn as nn
import torch.optim as optim

from baseline import baseline, prepare_data
from compression import replace_conv_layers, replace_linear_layers
from calibrate import run_calibration
from eval import run_eval
from size_accounting import compute_model_size


# ============================================================
# QUANTIZATION FINE-TUNING
# ============================================================

def finetune_quantized(
    model,
    train_loader,
    epochs=3,
    lr=1e-4,
    momentum=0.9,
    weight_decay=0.0,
    device="cuda"
):
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )

    model.train()

    for epoch in range(epochs):

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        print(
            f"[QAT] Epoch {epoch + 1}/{epochs} | "
            f"Loss: {running_loss / total:.4f} | "
            f"Train Acc: {100.0 * correct / total:.2f}%"
        )

    model.eval()

    return model


# ============================================================
# QUANTIZATION PIPELINE
# ============================================================

def run_quantization(
    checkpoint_path,
    weight_bits=4,
    act_bits=8,
    num_calib_batches=10,
    finetune_epochs=3,
    finetune_lr=1e-4,
    finetune_momentum=0.9,
    finetune_weight_decay=0.0,
    device="cuda"
):
    """
    Stage 1:

        FP32 baseline
            ↓
        Quantized layers
            ↓
        Calibration
            ↓
        QAT fine-tuning
            ↓
        Recalibration
            ↓
        Evaluation
            ↓
        Size accounting
    """

    # --------------------------------------------------------
    # LOAD ORIGINAL FP32 MODEL
    # --------------------------------------------------------

    original_model = baseline()

    original_model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device
        )
    )

    original_model = original_model.to(device)
    original_model.eval()

    # --------------------------------------------------------
    # CREATE QUANTIZED MODEL
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # REPLACE CONV / LINEAR
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # INITIAL CALIBRATION
    # --------------------------------------------------------

    model = run_calibration(
        model,
        train_loader,
        bits=act_bits,
        num_batches=num_calib_batches,
        device=device
    )

    # --------------------------------------------------------
    # QAT
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
    # EVALUATION
    # --------------------------------------------------------

    eval_results = run_eval(
        model,
        test_loader
    )

    # --------------------------------------------------------
    # SIZE ACCOUNTING
    # --------------------------------------------------------

    sample_input, _ = next(
        iter(test_loader)
    )

    sample_input = sample_input[:1].to(device)

    size_results = compute_model_size(
        original_model,
        model,
        sample_input
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    results = {
        **eval_results,
        **size_results,

        "weight_bits": weight_bits,
        "act_bits": act_bits,

        "quant_finetune_epochs": finetune_epochs,
        "quant_finetune_lr": finetune_lr,
        "quant_finetune_momentum": finetune_momentum,
        "quant_finetune_weight_decay": finetune_weight_decay,

        "quant_calib_batches": num_calib_batches
    }

    return model, results