import torch
from torch import nn, optim
from models.mlp import MLP
import os
import matplotlib.pyplot as plt
from utils.activations import ActivationRecorder


def train_xor(save_path: str = "models/xor.pth", epochs: int = 1000, lr: float = 0.01):
    torch.manual_seed(42)

    # XOR data
    X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=torch.float32)
    y = torch.tensor([[0.], [1.], [1.], [0.]], dtype=torch.float32)

    # Model + recorder
    model = MLP(input_dim=2, hidden_dims=[8], output_dim=1)
    recorder = ActivationRecorder()
    recorder.register(model.net, ["0", "2"])  # hook into Linear layers

    # Loss + optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        preds = model(X)
        loss = loss_fn(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} - loss: {loss.item():.6f}")

            model.eval()
            with torch.no_grad():
                out = model(X)
                probs = torch.sigmoid(out)
                print("Preds (logits):", out.numpy().round(3))
                print("Preds (probs):", probs.numpy().round(3))

            # Access recorded activations
            _ = model(X)  # forward pass to record
            activations = recorder.data  # <-- corrected

            hidden_layer = list(activations.keys())[0]
            hidden_acts = activations[hidden_layer].detach().numpy()

            fig, ax = plt.subplots()
            im = ax.imshow(hidden_acts, cmap="viridis", aspect="auto")
            ax.set_title("Hidden Layer Activations XOR")
            ax.set_xlabel("Neurons")
            ax.set_ylabel("Samples")
            ax.set_yticks(range(4))
            ax.set_yticklabels(["00","01","10","11"])
            fig.colorbar(im, ax=ax)
            
            os.makedirs("assets",exist_ok=True)
            plt.savefig("assets/hidden_layer_heatmap.png",dpi=200)
            plt.close()

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    train_xor()
