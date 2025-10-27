# utils/predict_mnist.py
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
import numpy as np
from models.mlp import MLP
import io
from torchvision import transforms
import numpy as np
from PIL import Image,ImageOps,ImageFilter 

MODEL_PATH = "models/mnist.pt"   # adjust if different

def load_model(model_path: str = MODEL_PATH, device="cpu"):
    model = MLP(input_dim=28*28, hidden_dims=[128, 64], output_dim=10)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def _preprocess_pil(img: Image.Image):
    #convert to grayscale
    img=img.convert("L")

    #invert if bg is white
    arr=np.array(img)/255.0
    if arr.mean()>0.3:
        img=ImageOps.invert(img)

    img=img.filter(ImageFilter.GaussianBlur(radius=0.5))

    bw=np.array(img)
    coords=np.column_stack(np.where(bw>10))
    if coords.size>0:
        y0,x0=coords.min(axis=0)
        y1,x1=coords.max(axis=0)
        img=img.crop((x0,y0,x1+1,y1+1))
        img=img.resize((20,20),Image.Resampling.LANCZOS)
        new_img=Image.new("L",(28,28),0)
        new_img.paste(img,((28-20)//2,(28-20)//2))
        img=new_img
    else:
        img=img.resize((28,28),Image.Resampling.LANCZOS)

    #convert to tensor and normalize
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])
    
    return transform(img).view(1,-1)



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
    x = x.to(device).float()

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))

    # return predicted label and probability vector
    return pred, probs.tolist()
