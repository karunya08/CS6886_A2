# MobileNetV2 Compression on CIFAR-10

This project implements a two-stage model compression pipeline for **MobileNetV2 trained on CIFAR-10**.

The compression pipeline combines:

1. **Quantization-Aware Training (QAT)** with configurable weight and activation precision.
2. **Global importance based structured channel pruning** starting from the selected quantized model.

The objective is to reduce model weight storage and activation memory while retaining classification accuracy.

---

## Repository Structure

```text
CS6886_A2/
│
├── compression/
│   ├── __init__.py
│   ├── hooks.py
│   ├── quant_layers.py
│   └── quant_ops.py
│
├── pipeline/
│   ├── quant.py
│   └── prune.py
│
├── baseline.py
├── baseline_best.pt
├── calibrate.py
├── compression.py
├── eval.py
├── size_accounting.py
├── sweep.py
└── .gitignore
```

### File Organization

| File / Directory              | Description                                                              |
| ----------------------------- | ------------------------------------------------------------------------ |
| `baseline.py`                 | MobileNetV2 definition, CIFAR-10 data preparation, and baseline training |
| `baseline_best.pt`            | Trained FP32 baseline checkpoint                                         |
| `compression/`                | Low-level quantization implementation                                    |
| `compression/quant_layers.py` | Quantized convolution and linear layers                                  |
| `compression/quant_ops.py`    | Quantization and fake-quantization operations                            |
| `compression/hooks.py`        | Activation calibration hooks                                             |
| `calibrate.py`                | Activation calibration interface                                         |
| `pipeline/quant.py`           | Stage 1: quantization and QAT                                            |
| `pipeline/prune.py`           | Stage 2: sensitivity-aware structured channel pruning                    |
| `compression.py`              | Main end-to-end compression pipeline and command-line interface          |
| `eval.py`                     | Model evaluation                                                         |
| `size_accounting.py`          | Weight, activation, and overall compression statistics                   |
| `sweep.py`                    | Weights & Biases quantization and pruning sweeps                         |

---

# Compression Pipeline

The complete pipeline is:

```text
                    FP32 Baseline
                         │
                         ▼
                  Quantized Model
                         │
                         ▼
              Activation Calibration
                         │
                         ▼
                         QAT
                         │
                         ▼
                  Recalibration
                         │
                         ▼
                  Quantized Model
                         │
                         ▼
          Sensitivity-Aware Structured
                     Pruning
                         │
                         ▼
                Channel Removal
                         │
                         ▼
                  Recalibration
                         │
                         ▼
                    Fine-Tuning
                         │
                         ▼
                 Final Recalibration
                         │
                         ▼
                Evaluation + Size
                   Accounting
```

---

# Stage 1 — Quantization

The first stage converts the FP32 MobileNetV2 model into a quantized model using configurable:

* Weight precision
* Activation precision
* Activation calibration
* Quantization-aware fine-tuning

For example, **W4A8** means:

```text
Weights     : 4 bits
Activations : 8 bits
```

### Quantization Scheme

**Symmetric quantization** is used for weights because their distributions are generally centered around zero, making a zero-point of zero appropriate and simplifying the quantization process.

**Asymmetric quantization** is used for activations because activations are typically non-negative and may have a non-zero minimum. This allows the available quantization range to better match the activation distribution and reduce quantization error.

The quantization sweep evaluates 8-, 4-, and 2-bit representations for both weights and activations.

The sweep showed that **4-bit weights with 8-bit activations (W4A8)** provided the best trade-off between accuracy retention and compression.

---

# Stage 2 — Sensitivity-Aware Structured Pruning

The selected quantized model is subsequently subjected to **global importance-based structured channel pruning**.

Channel importance is estimated using the **L2 norm of the weights of the expansion-layer output channels** in MobileNetV2 inverted residual blocks.

```text
                 Expansion Layer
                       │
                       ▼
              Output Channels
                       │
                       ▼
                 L2 Weight Norm
                       │
                       ▼
              Channel Importance
```

Channels with smaller importance scores are considered better pruning candidates.

The importance scores are normalized within each eligible block and then **globally ranked across blocks**. The least-important channels are removed according to a configurable target sparsity.

A **minimum keep ratio of 25%** is enforced for each prunable block to prevent excessive reduction of individual layers.

## First Inverted Residual Block

The **inverted residual block with an expansion ratio of 1 is excluded from pruning and retained unchanged**.

This block is located at the beginning of the network and is responsible for processing the initial low-level representations. These features include fundamental visual patterns such as edges, textures, and local structures that form the input representations for subsequent layers.

Pruning this block could therefore remove important low-level features and cause a disproportionate loss in accuracy. Excluding it provides a safeguard against aggressively modifying the earliest feature representations while allowing deeper blocks to undergo structured compression.

## Physical Channel Removal

Unlike unstructured pruning, the selected channels are **physically removed from the network dimensions**.

For each pruned channel, the corresponding parameters are removed from:

* Expansion convolution
* Depthwise convolution
* Projection convolution
* Batch-normalization layers

This reduces the actual parameter count and computational dimensions rather than simply setting weights to zero.

---

# Pruning Levels

The pruning sweep evaluates the following target sparsities:

```text
25%
40%
50%
60%
70%
```

The minimum keep ratio is fixed at:

```text
25%
```

This allows the effect of increasing global sparsity on accuracy and compression to be evaluated systematically.

---

# Running the Compression Pipeline

`compression.py` provides the main command-line interface.

## Example: W4A8 with 25% pruning

```bash
python compression.py --weight-bits 4 --act-bits 8 --sparsity 0.25
```

# Running Weights & Biases Sweeps

The repository includes two W&B sweeps:


Run:

```bash
python sweep.py quant
```

```bash
python sweep.py prune
```


# Recommended Workflow

First run the quantization sweep:

```bash
python sweep.py quant
```

Use the resulting accuracy–compression trade-off to select the quantization configuration.

For the current pipeline, **W4A8** is used for the pruning stage.

Then run the pruning sweep:

```bash
python sweep.py prune
```

This evaluates the effect of increasing structured sparsity.

Individual configurations can then be reproduced directly using:

```bash
python compression.py --weight-bits 4 --act-bits 8 --sparsity 0.50
```

---


# Reproducibility

The main compression parameters can be configured from the command line.

For example:

```bash
python compression.py \
    --weight-bits 4 \
    --act-bits 8 \
    --sparsity 0.50
```

Supported weight and activation bit-widths are:

```text
2, 4, 8
```

The pruning sparsity can be configured independently.

For W&B experiments:

```bash
python sweep.py quant
```

and:

```bash
python sweep.py prune
```

---

# Requirements

The implementation uses:

* Python
* PyTorch
* Torchvision
* Weights & Biases

A CUDA-capable GPU is recommended for QAT, pruning, and fine-tuning experiments.


The approach combines low-precision representation with physically removing less-important channels, providing reductions in both **memory footprint and computational dimensions** while attempting to preserve model accuracy.
