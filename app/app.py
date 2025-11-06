import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from models.mlp import MLP
from models.xor_model import XORModel
from utils.activations import ActivationRecorder
import sys, os
import torchvision
from data.mnist import get_mnist_loaders
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from utils.predict_mnist import load_model, predict_image,_preprocess_pil
from PIL import Image
import io
from viz.mnist_3d import mnist_3d_viz

PROJECT_ROOT = r"C:\Users\Srishti\Singularity"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
  

st.markdown("""
    <style>
    /* Main background */
    .main {
        background-color: #0b0c10;
        color: #c5c6c7;
        font-family: 'Orbitron', sans-serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #101820;
    }

    /* Titles */
    h1, h2, h3 {
        color: #66fcf1;
        text-shadow: 0 0 10px #45a29e;
    }

    /* Buttons */
    button[kind="primary"] {
        background-color: #1f2833 !important;
        color: #66fcf1 !important;
        border-radius: 8px;
        border: 1px solid #45a29e;
    }
    button[kind="primary"]:hover {
        background-color: #45a29e !important;
        color: #0b0c10 !important;
        box-shadow: 0 0 10px #66fcf1;
    }

    /* Slider track color */
    .stSlider > div[data-baseweb="slider"] > div {
        background: linear-gradient(to right, #66fcf1, #1f2833);
    }
    </style>
""", unsafe_allow_html=True)


class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(28*28,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)


st.set_page_config(page_title="Singularity", layout="wide")
st.title("Singularity")
tab1,tab2=st.tabs(["XOR Demo","MNIST Demo"])




with tab1:
    st.header("XOR - Neural Network Vizualization")
    if "xor_model" not in st.session_state:
        st.session_state["xor_model"] = XORModel()
        st.session_state["xor_trained"] = False

    model = st.session_state["xor_model"]
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)


    x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=torch.float32)
    y = torch.tensor([[0.], [1.], [1.], [0.]], dtype=torch.float32)

    st.sidebar.header("Training Settings")
    epochs = st.sidebar.slider("Epochs", 100, 2000, 500, step=100)
    train_btn = st.sidebar.button("Train XOR Model")

    if train_btn:
        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        st.success(f"Training Complete! Final Loss: {loss.item():.4f}")
        os.makedirs("models", exist_ok=True)
        save_path = "models/xor_streamlit.pth"
        torch.save(model.state_dict(), save_path)
        st.session_state["xor_trained"] = True
        st.info(f"Model weights saved to {save_path}")

        
    st.sidebar.header("Test Inputs")
    a = st.sidebar.selectbox("Input A", [0.0, 1.0])
    b = st.sidebar.selectbox("Input B", [0.0, 1.0])

    if st.sidebar.button("Run XOR Prediction"):
        save_path = "models/xor_streamlit.pth"
        if os.path.exists(save_path):
            model.load_state_dict(torch.load(save_path, map_location="cpu"))
            model.eval()
            st.success("Loaded trained weights for prediction.")
        else:
            st.warning("No saved model weights found — predictions will use current model (may be untrained).")

        with torch.no_grad():
            test_input = torch.tensor([[float(a), float(b)]])
            pred = model(test_input)
            prob = float(pred.item()) if pred.numel() == 1 else float(pred.detach().cpu().numpy().flatten()[0])
        st.write(f"**Prediction for ({a}, {b}):** {prob:.4f}  → class: {1 if prob > 0.5 else 0}")




    if st.button("Generate 3D XOR Visualization"):
        from viz.xor_3d import xor_3d_viz
        save_path = "models/xor_streamlit.pth"
        if os.path.exists(save_path):
            model.load_state_dict(torch.load(save_path, map_location="cpu"))
            model.eval()
            st.success("Loaded trained model for visualization.")
        else:
            st.warning("No saved model found — visualizing current model (may be untrained).")

        input_tensor = torch.tensor([[float(a), float(b)]])
        fig3d = xor_3d_viz(model, input_tensor,animate=True)
        st.plotly_chart(fig3d, use_container_width=True)




with tab2:
    st.header("MNIST Digit Classifier")

    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
    train_dataset=datasets.MNIST(root='data',train=True,download=True,transform=transform)
    test_dataset=datasets.MNIST(root='data',train=False,download=True,transform=transform)

    train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=64,shuffle=True)
    test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=10,shuffle=False)


    mnist_model=MLP(28*28,[128,64],10)
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(mnist_model.parameters(),lr=0.001,weight_decay=1e-4)

    st.sidebar.header("MNIST Training Settings")
    mnist_epochs=st.sidebar.slider("Epochs",1,15,5)
    mnist_btn=st.sidebar.button("Train MNIST Model")
     
    if mnist_btn:
        mnist_model.train()
        losses=[]
        for epoch in range(mnist_epochs):
            total_loss=0
            for data, target in train_loader:
                data=data.view(data.size(0),-1)
                optimizer.zero_grad()
                out=mnist_model(data)
                loss=criterion(out,target)
                loss.backward()
                optimizer.step()
                total_loss+=loss.item()
            losses.append(total_loss/len(train_loader))
            
            st.write(f"Epoch {epoch+1}: loss: {losses[-1]:.4f}")

        torch.save(mnist_model.state_dict(),"models/mnist.pt")
        st.success(f"MNIST Training Complete! final loss: {loss.item():.4f}")
        

        fig,ax=plt.subplots()
        ax.plot(losses)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        
        st.pyplot(fig)

    st.subheader("Upload & Predict a Digit")

    def get_predict_model(path="models/mnist.pt"):
        try:
            return load_model(model_path=path, device="cpu")
        except Exception as e:
            return None

    predict_model = get_predict_model("models/mnist.pt")

    uploaded = st.file_uploader("Upload an image (png/jpg) of a digit to predict", type=["png","jpg","jpeg","bmp","gif"])

    if uploaded is not None:
        try:
            pil_img = Image.open(uploaded).convert("RGB")
        except Exception as e:
            st.error(f"Unable to open image: {e}")
            pil_img = None

        if pil_img is not None:
            st.image(pil_img, caption="Uploaded image", width=150)

            if st.button("Predict uploaded image"):
                

                if predict_model is None:
                    try:
                        mnist_model.eval()
                        local_model = mnist_model
                        st.info("Using in-session MNIST model (saved model not found).")
                    except Exception as e:
                        st.error("No model available for prediction. Train the model or save 'models/mnist.pt'.")
                        local_model = None
                else:
                    local_model = predict_model
                    st.success("Loaded saved model for prediction.")

                if local_model is not None:
                    try:
                        img_bytes=uploaded.getvalue()
                        pred_label,probs=predict_image(local_model,image_bytes=img_bytes,device="cpu")
                        st.subheader(f"Predicted digit: {pred_label}")

                        #top 3 predictions
                        sorted_idx=sorted(range(len(probs)),key=lambda i:probs[i],reverse=True)
                        st.write("Top predictions:")
                        for i in sorted_idx[:3]:
                            st.write(f"{i}:{probs[i]*100:.2f}%")
                        st.bar_chart({str(i):probs[i] for i in range(len(probs))})

                        pil_img=Image.open(io.BytesIO(img_bytes)).convert("L")
                        preproc_tensor=_preprocess_pil(pil_img)

                        # save for viz
                        st.session_state['test_image_tensor'] = preproc_tensor
                        st.session_state['test_label_pred'] = int(pred_label)

                        st.success("Prediction saved for 3D visualisation")

                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

    #testing and viz
    st.subheader("MNIST Model Testing and Visualization")

    if st.button("Test MNIST Model"):

        #load trained model
        mnist_model.load_state_dict(torch.load("models/mnist.pt"))
        mnist_model.eval()
        
        rec=ActivationRecorder()
        rec.register(mnist_model,['net.0','net.2'])

        images,labels=next(iter(test_loader))
        x=images[0].view(1,-1)
        y=labels[0].item()


        with torch.no_grad():
            output=mnist_model(x)
            pred = torch.argmax(output, dim=1).item()
        activations=rec.get_activations()
        
        
        img = images[0].squeeze().cpu().numpy()
        img_denorm = img * 0.3081 + 0.1307
        st.image(img_denorm, caption=f"Predicted: {pred}", width=150, clamp=True)

        for layer_name,act in activations.items():
            fig,ax=plt.subplots(figsize=(6,3))
            ax.bar(range(len(act.flatten())),act.flatten().numpy())
            ax.set_title(f"Layer Activations: {layer_name}")
            st.pyplot(fig)
        
        # Store images and labels in session state for 3D viz
        st.session_state['test_images'] = images
        st.session_state['test_labels'] = labels

    if st.button("Generate 3D Visualization"):
        #loading trained model
        mnist_model.load_state_dict(torch.load("models/mnist.pt",map_location="cpu"))
        mnist_model.eval()

        #case1: custom image
        if 'test_image_tensor' in st.session_state:
            pil_img=st.session_state['test_image_tensor'].view(1,-1).float()
            label=st.session_state.get('test_label_pred',None)

            st.info("Generating 3D Vizualisation")
            x=pil_img.view(1,-1).float()
            fig3d = mnist_3d_viz(x, torch.tensor([st.session_state['test_label_pred']]), mnist_model,animate=True)
            st.plotly_chart(fig3d,use_container_width=True)

        #case2: fallback to sample input
        elif 'test_images' in st.session_state:
            images=st.session_state['test_images']
            labels=st.session_state['test_labels']
            x=images[0].view(1,-1)
            label=labels[0].item()

            st.info(f"Generating 3D Visualization for test sample (True Label: {label})...")
            fig3d = mnist_3d_viz(x, torch.tensor([label]), mnist_model,animate=True)
            st.plotly_chart(fig3d, use_container_width=True)
        
        #case3: no input
        else:
            st.warning("Please upload an image or run 'Test MNIST Model' first!")