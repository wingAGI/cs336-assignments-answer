import torch

def _to_device_and_compile(model):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)

    if device.type == "mps":
        model = torch.compile(model, backend="aot_eager")
    else:
        model = torch.compile(model)

    return model, device