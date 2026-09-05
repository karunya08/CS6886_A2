
from compression.hooks import calibrate_activations
from compression.quant_layers import apply_calibration


def run_calibration(model, calib_loader, bits=8, num_batches=10, device="cuda"):
    layers_to_calibrate = {
        name: module for name, module in model.named_modules()
        if hasattr(module, "act_scale")
    }
    print(f"Calibrating {len(layers_to_calibrate)} layers...")

    scales, zero_points = calibrate_activations(model, calib_loader, layers_to_calibrate, bits, num_batches, device)
    model = apply_calibration(model, scales, zero_points)
    return model
