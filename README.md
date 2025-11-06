# Singularity: 3D Neural Network Visualizer

## Overview

Singularity is designed for learners, educators, and researchers to explore and understand neural networks beyond the black-box.  
It provides an interactive 3D visualization where neuron activations and connection strengths are animated in real time during forward passes.  
The project supports simple XOR networks as well as more complex MNIST digit classifiers, making it ideal for both educational and research purposes.

Key points:  
- Visualizes activations layer by layer in 3D.  
- Supports multiple models: XOR and MNIST digit classifier.  
- Highlights changes in activations and weights during inference.  
- Provides annotations and explanations for each layer and neuron.

## Features

- **Interactive 3D Visualization** – Explore neurons, layers, and connections in an intuitive 3D environment.  
- **Layer-wise Forward Pass Animation** – Watch activations propagate through the network step by step.  
- **Supports Multiple Models** – XOR network (2–2–1 MLP) and MNIST digit classifier.  
- **Activation and Weight Insights** – Observe how activations and connection strengths change during inference.  
- **Detailed Annotations** – Each layer and neuron includes informative labels and explanations.  
- **Easy-to-use Interface** – Simple controls to play, pause, and explore network behavior.  

## Project Structure

```text
Singularity/
│
├── .gitignore
├── pyproject.toml
├── README.md
├── requirements.txt
├── setup.cfg
│
├── app.py
│
├── assets/
│   ├── 3d_layout.png
│   ├── hidden_layer_heatmap.png
│   └── xor_fwd.mp4
│
├── data/
│   ├── __init__.py
│   └── mnist.py
│
├── models/
│   ├── mlp.py
│   ├── mnist.pt
│   ├── xor.pth
│   ├── xor_model.py
│   └── xor_streamlit.pth
│
├── train/
│   ├── mnist_train.py
│   └── xor_train.py
│
├── utils/
│   ├── activations.py
│   └── predict_mnist.py
│
└── viz/
    ├── mnist_3d.py
    ├── network_viz.py
    └── xor_3d.py

## Live Demo
Try out the **Singularity** app live on Streamlit: [https://srishtikapoor22-singularity-app-qa2yle.streamlit.app/](https://srishtikapoor22-singularity-app-qa2yle.streamlit.app/)


