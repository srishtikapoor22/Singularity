# debug_xor.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
from models.xor_model import XORModel  # ensure this exists per earlier instructions

torch.manual_seed(42)

# Data
X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]], dtype=torch.float32)
Y = torch.tensor([[0.],[1.],[1.],[0.]], dtype=torch.float32)

# Model
model = XORModel()
print("Model summary:\n", model)
print("Initial output (raw):", model(X).detach().cpu().numpy().round(4))

# Loss & optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Train
n_epochs = 5000
for epoch in range(1, n_epochs+1):
    model.train()
    optimizer.zero_grad()
    out = model(X)
    loss = criterion(out, Y)
    loss.backward()
    optimizer.step()
    if epoch % 500 == 0 or epoch == 1:
        with torch.no_grad():
            preds = model(X)
            probs = preds.detach().cpu().numpy().flatten()
            print(f"Epoch {epoch:4d} loss={loss.item():.6f} probs={probs.round(4)}")

# Save
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(),"models/xor_debug.pth")
print("Saved model to models/xor_debug.pth")

# Final checks (per-sample)
model.eval()
with torch.no_grad():
    out = model(X).detach().cpu().numpy().flatten()
    print("\nFinal outputs per input (probabilities):")
    for inp, p in zip(X.numpy(), out):
        print(f" Input {inp.tolist()} -> prob={p:.4f} -> class={(1 if p>0.5 else 0)}")
