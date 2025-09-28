import torch

def test_gpu_available_for_dl_models():
    # Only run if PyTorch is installed
    try:
        import torch
    except ImportError:
        return  # not a DL project, skip

    assert torch.cuda.is_available(), "GPU not available for DL models!"

def test_model_on_gpu_if_dl(load_model):
    model = load_model
    if hasattr(model, "to"):  # PyTorch model
        assert next(model.parameters()).is_cuda, "Model not on GPU!"
