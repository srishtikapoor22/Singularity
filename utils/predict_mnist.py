# utils/predict_mnist.py
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
import numpy as np
from models.mlp import MLP
import io

MODEL_PATH = "models/mnist.pt"   # adjust if different

def load_model(model_path: str = MODEL_PATH, device="cpu"):
    model = MLP(input_dim=28*28, hidden_dims=[128, 64], output_dim=10)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def _preprocess_pil(img: Image.Image):
    """
    Expects a PIL image. Returns a torch tensor shaped (1, 28*28), dtype float32,
    normalized with MNIST mean/std.
    """
    # ensure grayscale
    img = img.convert("L")

    # resize to 28x28 with antialias
    img = img.resize((28, 28), Image.LANCZOS)

    # convert to numpy [0..1]
    arr = np.asarray(img).astype(np.float32) / 255.0  # shape (28,28)

    # heuristic: if background is white and digit is dark (common photo), invert so digit is white on black
    # MNIST digits are white (high pixel value) on black (low). If mean is > 0.5, we assume white background and invert.
    if arr.mean() > 0.5:
        arr = 1.0 - arr

    # normalize with MNIST stats
    mean = 0.1307
    std = 0.3081
    arr = (arr - mean) / std

    # flatten and convert to tensor
    tensor = torch.from_numpy(arr).view(1, -1).float()
    return tensor

def predict_image(model, pil_image=None, image_bytes: bytes = None, device="cpu"):
    """
    Predicts from a PIL Image or raw image bytes.
    Returns (pred_label:int, probs: list[float]) where probs are softmaxed probabilities for classes 0..9.
    """
    if pil_image is None and image_bytes is not None:
        pil_image = Image.open(io.BytesIO(image_bytes))

    if pil_image is None:
        raise ValueError("Provide a PIL image or image_bytes")

    x = _preprocess_pil(pil_image)
    x = x.to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))

    # return predicted label and probability vector
    return pred, probs.tolist()
