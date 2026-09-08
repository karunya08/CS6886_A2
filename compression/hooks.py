import torch
from compression.quant_ops import calibrate_activation_scale

def make_calibration_hook(name, stats_store):
    def hook(module, input, output):
        out = output.detach()
        batch_min = out.min()
        batch_max = out.max()

        if name not in stats_store:
            stats_store[name] = {"min": batch_min, "max": batch_max}
        else:
            stats_store[name]["min"] = torch.min(stats_store[name]["min"], batch_min)
            stats_store[name]["max"] = torch.max(stats_store[name]["max"], batch_max)
    return hook

def calibrate_activations(model, calib_loader, layers_to_calibrate, bits, num_batches=10, device="cuda"):
    stats_store = {}
    handles = []

    for name, layer in layers_to_calibrate.items():
        handles.append(layer.register_forward_hook(make_calibration_hook(name, stats_store)))
    
    model.eval()

    for i, (images, _) in enumerate(calib_loader):
        if(i >= num_batches):
            break
        images = images.to(device)
        model(images)
    
    for h in handles:
        h.remove()
    
    scales, zero_points = {}, {}
    for name, stats in stats_store.items():
        scale, zero_point = calibrate_activation_scale(stats["min"], stats["max"], bits)
        scales[name] = scale
        zero_points[name] = zero_point

    return scales, zero_points
