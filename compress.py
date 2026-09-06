
import torch
import torch.nn as nn
import torch.optim as optim
from baseline import baseline, prepare_data
from compression import replace_conv_layers, replace_linear_layers
from calibrate import run_calibration
from eval import run_eval
from size_accounting import compute_model_size


def finetune_quantized(model, train_loader, epochs=3, lr=1e-4, device="cuda"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    model.train()
    for epoch in range(epochs):
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        print(f"[QAT] epoch {epoch+1}/{epochs} | loss {running_loss/total:.4f} | train acc {100.*correct/total:.2f}%")

    model.eval()
    return model


def run_compression(checkpoint_path, weight_bits=8, act_bits=8, num_calib_batches=10, finetune_epochs=3, finetune_lr=1e-4, device="cuda"):
    model = baseline()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    train_loader, test_loader = prepare_data()

    model = replace_conv_layers(model, bits=weight_bits, act_bits=act_bits)
    model = replace_linear_layers(model, bits=weight_bits, act_bits=act_bits)

    # initial calibration -- needed before QAT can run a forward pass with quant active
    model = run_calibration(model, train_loader, bits=act_bits, num_batches=num_calib_batches, device=device)

    # QAT
    model = finetune_quantized(model, train_loader, epochs=finetune_epochs, lr=finetune_lr, device=device)

    # recalibrate post-finetune, since weight distributions shifted during training
    model = run_calibration(model, train_loader, bits=act_bits, num_batches=num_calib_batches, device=device)

    n_conv = sum(1 for m in model.modules() if type(m).__name__ == "QuantConv2d")
    n_linear = sum(1 for m in model.modules() if type(m).__name__ == "QuantLinear")
    print("QuantConv2d count:", n_conv)
    print("QuantLinear count:", n_linear)
    
    eval_results = run_eval(model, test_loader)

    sample_input, _ = next(iter(test_loader))
    sample_input = sample_input.to(device)
    size_results = compute_model_size(model, sample_input)

    return {**eval_results, **size_results, "weight_bits": weight_bits, "act_bits": act_bits,
            "finetune_epochs": finetune_epochs}


if __name__ == "__main__":
    result = run_compression("baseline_best.pt", weight_bits=8, act_bits=8, finetune_epochs=5)
    print(result)
