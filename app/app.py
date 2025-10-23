import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from models.mlp import MLP
from models.xor_model import XORModel
from viz.animate_fwd import animate_fwd
from utils.activations import ActivationRecorder
import sys, os
import torchvision
from viz.interactive_net import build_network_fig, forward_pass_act
from data.mnist import get_mnist_loaders
from viz.mnist_viz import viz_mnistfwd
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

PROJECT_ROOT = r"C:\Users\Srishti\Singularity"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
  

#css
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
    st.header("XOR Neural Network Playground")

    # --- Model Setup ---
    # --- Model Setup (XORModel with session persistence) ---
    if "xor_model" not in st.session_state:
        st.session_state["xor_model"] = XORModel()          # model instance persisted across reruns
        st.session_state["xor_trained"] = False

    model = st.session_state["xor_model"]

    # Use BCELoss because XORModel ends with Sigmoid
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)


    # --- XOR Data ---
    x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=torch.float32)
    y = torch.tensor([[0.], [1.], [1.], [0.]], dtype=torch.float32)

    # --- Sidebar: Training Settings ---
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

        st.success(f"✅ Training Complete! Final Loss: {loss.item():.4f}")
            # persist weights so future reruns / viz use the trained model
        os.makedirs("models", exist_ok=True)
        save_path = "models/xor_streamlit.pth"
        torch.save(model.state_dict(), save_path)
        st.session_state["xor_trained"] = True
        st.info(f"Model weights saved to {save_path}")

        

    # --- Sidebar: Test Inputs ---
    st.sidebar.header("Test Inputs")
    a = st.sidebar.selectbox("Input A", [0.0, 1.0])
    b = st.sidebar.selectbox("Input B", [0.0, 1.0])

    if st.sidebar.button("Run XOR Prediction"):
    # If saved weights exist, load them into the session model (ensures we use trained weights)
        save_path = "models/xor_streamlit.pth"
        if os.path.exists(save_path):
            model.load_state_dict(torch.load(save_path, map_location="cpu"))
            model.eval()
            st.success("Loaded trained weights for prediction.")
        else:
            st.warning("No saved model weights found — predictions will use current model (may be untrained).")

        with torch.no_grad():
            test_input = torch.tensor([[float(a), float(b)]])
            pred = model(test_input)  # XORModel already applies Sigmoid in forward
            # ensure scalar prob
            prob = float(pred.item()) if pred.numel() == 1 else float(pred.detach().cpu().numpy().flatten()[0])
        st.write(f"**Prediction for ({a}, {b}):** {prob:.4f}  → class: {1 if prob > 0.5 else 0}")




    if st.button("Generate 3D XOR Visualization"):
        from viz.xor_3d import xor_3d_viz
        # load trained weights if available
        save_path = "models/xor_streamlit.pth"
        if os.path.exists(save_path):
            model.load_state_dict(torch.load(save_path, map_location="cpu"))
            model.eval()
            st.success("Loaded trained model for visualization.")
        else:
            st.warning("No saved model found — visualizing current model (may be untrained).")

        input_tensor = torch.tensor([[float(a), float(b)]])
        fig3d = xor_3d_viz(model, input_tensor)
        st.plotly_chart(fig3d, use_container_width=True)




with tab2:
    st.header("MNIST Digit Classifier")
     
     #data setup
    transform=transforms.Compose([transforms.ToTensor()])
    train_dataset=datasets.MNIST(root='data',train=True,download=True,transform=transform)
    test_dataset=datasets.MNIST(root='data',train=False,download=True,transform=transform)

    train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=64,shuffle=True)
    test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=10,shuffle=False)


     #model setup
    mnist_model=MLP(28*28,[128,64],10)
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(mnist_model.parameters(),lr=0.001)

     #sidebar
    st.sidebar.header("MNIST Training Settings")
    mnist_epochs=st.sidebar.slider("Epochs",1,5,1)
    mnist_btn=st.sidebar.button("Train MNIST Model")
     
     #training
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
        

        #plot training loss
        fig,ax=plt.subplots()
        ax.plot(losses)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        
        st.pyplot(fig)

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
        
        img = images[0].squeeze().numpy()
        true_label = labels[0].item()
        st.image(img, caption=f"True Label: {true_label}, Predicted: {pred}", width=150)

        for layer_name,act in activations.items():
            fig,ax=plt.subplots(figsize=(6,3))
            ax.bar(range(len(act.flatten())),act.flatten().numpy())
            ax.set_title(f"Layer Activations: {layer_name}")
            st.pyplot(fig)
        
        # Store images and labels in session state for 3D viz
        st.session_state['test_images'] = images
        st.session_state['test_labels'] = labels

    if st.button("Generate 3D Visualization"):
        # Check if we have test data in session state
        if 'test_images' not in st.session_state:
            st.warning("Please run 'Test MNIST Model' first to load test data!")
        else:
            # Load model
            mnist_model.load_state_dict(torch.load("models/mnist.pt"))
            mnist_model.eval()
            
            # Get stored test data
            images = st.session_state['test_images']
            labels = st.session_state['test_labels']
            
            # Generate 3D visualization
            from viz.mnist_3d import mnist_3d_viz
            fig3d = mnist_3d_viz(images, labels, mnist_model)
            st.plotly_chart(fig3d, use_container_width=True)



