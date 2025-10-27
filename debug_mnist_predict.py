# debug_mnist_predict.py
import os
import traceback
import torch
import numpy as np
from PIL import Image
import torchvision

# ---- adjust these paths if your structure differs ----
MODEL_PATH = "models/mnist.pt"         # trained weights
DEBUG_IMAGE = "debug_digit.png"        # optional: put a custom image here (28x28 or larger)
# -----------------------------------------------------

# Import loader + preprocess from your project utils
try:
    from utils.predict_mnist import load_model, _preprocess_pil, predict_image
except Exception as e:
    print("Failed to import from utils.predict_mnist:", e)
    raise

def make_debug_image_from_mnist(save_path="debug_digit.png"):
    """If user didn't supply debug image, create one from MNIST test set (first sample)."""
    print("No debug image found — creating one from torchvision MNIST test set...")
    ds = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=torchvision.transforms.ToTensor())
    img_tensor, label = ds[0]  # first test sample
    arr = (img_tensor.squeeze().numpy() * 255).astype("uint8")
    pil = Image.fromarray(arr, mode="L")
    pil.save(save_path)
    print(f"Saved sample MNIST image to {save_path} (true label: {label})")
    return save_path, int(label)

def main():
    print("Debug script started.")
    # 1) Ensure model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at {MODEL_PATH}. Aborting.")
        return

    # 2) Ensure debug image exists; if not, create from MNIST test sample
    if not os.path.exists(DEBUG_IMAGE):
        img_path, true_label = make_debug_image_from_mnist(DEBUG_IMAGE)
    else:
        img_path = DEBUG_IMAGE
        true_label = None

    # 3) Load model using your project's loader
    try:
        print("Loading model via utils.predict_mnist.load_model(...)")
        model = load_model(model_path=MODEL_PATH, device="cpu")
        print("Loaded model object:", type(model))
    except Exception as e:
        print("FAILED to load model:")
        traceback.print_exc()
        return

    # 4) Open the image and run the exact preprocessing used in app
    try:
        pil = Image.open(img_path).convert("L")
    except Exception:
        print("Failed to open debug image:", img_path)
        traceback.print_exc()
        return

    try:
        print("Running _preprocess_pil() ...")
        tensor = _preprocess_pil(pil)   # expected shape [1, 784] or similar
        print("Preprocessed tensor:", type(tensor), tensor.shape, tensor.dtype)
        # numeric stats
        arr = tensor.detach().cpu().numpy().reshape(-1)
        print("Tensor stats -> min, max, mean, std:", float(arr.min()), float(arr.max()), float(arr.mean()), float(arr.std()))
    except Exception:
        print("Preprocessing failed:")
        traceback.print_exc()
        return

    # 5) Check model call and types — isolate any 'int is not callable' issues
    try:
        model.eval()
        print("About to call model(...) with tensor of shape", tensor.shape)
        # ensure float and right shape for MLP (flatten)
        t_in = tensor.to("cpu").float().view(1, -1)
        print("Input passed to model:", t_in.shape, t_in.dtype)

        with torch.no_grad():
            out = model(t_in)
            print("Model output type:", type(out), "shape:", out.shape)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
        print(f"Prediction OK -> pred: {pred}, top3: {np.argsort(probs)[-3:][::-1].tolist()}, max_prob: {probs.max():.4f}")
    except Exception as e:
        print("Prediction failed during model(...) call:")
        traceback.print_exc()
        return

    # 6) Cross-check using predict_image helper (if available)
    try:
        print("Cross-check: running predict_image(model, pil_image=pil)...")
        pred2, probs2 = predict_image(model, pil_image=pil, device="cpu")
        print("predict_image returned -> pred:", pred2, "top3:", sorted(range(len(probs2)), key=lambda i:probs2[i], reverse=True)[:3])
    except Exception as e:
        print("predict_image helper failed:")
        traceback.print_exc()

    # 7) Print final note
    if true_label is not None:
        print(f"True label (from MNIST sample): {true_label}")

if __name__ == "__main__":
    main()
