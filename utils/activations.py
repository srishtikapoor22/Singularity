from typing import Dict, List, Callable
import torch

class ActivationRecorder:
    """
    Records intermediate activations of specified layers in a PyTorch model.
    """

    def __init__(self):
        self.data: Dict[str, torch.Tensor] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    def register(self, model: torch.nn.Module, layer_names: List[str]):
        """
        Attach forward hooks to layers in the model whose names match `layer_names`.
        """
        # Clear old hooks if any
        self.remove()

        for name, module in model.named_modules():
            if name in layer_names:
                handle = module.register_forward_hook(self._make_hook(name))
                self.handles.append(handle)

    def _make_hook(self, name: str) -> Callable:
        """
        Internal helper to generate hook function for a given layer name.
        """
        def hook(module, input, output):
            try:
                self.data[name] = output.detach().cpu()
            except Exception:
                # fallback in case output is not a tensor
                self.data[name] = output
        return hook

    def remove(self):
        """
        Remove all hooks and clear recorded activations.
        """
        self.data.clear()
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []
