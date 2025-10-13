from typing import Dict, List, Callable
import torch

class ActivationRecorder:
    """
    Records intermediate activations from specific layers during the forward pass.
    """

    def __init__(self):
        self.data: Dict[str, torch.Tensor] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.layer_order: List[str] = []

    def register(self, model, layer_names: List[str] = None):
        """
        Register forward hooks to capture activations from specific layers.
        If `layer_names` is None, automatically attach to all Linear/Conv2d layers.
        """
        self.remove()

        for name, module in model.named_modules():
            if layer_names is None:
                # Automatically register for Linear or Conv2d layers
                if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                    handle = module.register_forward_hook(self._make_hook(name))
                    self.handles.append(handle)
                    self.layer_order.append(name)
            elif name in layer_names:
                handle = module.register_forward_hook(self._make_hook(name))
                self.handles.append(handle)
                self.layer_order.append(name)

    def _make_hook(self, name: str) -> Callable:
        """Internal helper to create the forward hook."""
        def hook(module, input, output):
            try:
                self.data[name] = output.detach().cpu()
            except Exception:
                self.data[name] = output
        return hook

    def remove(self):
        """Remove all hooks and clear stored activations."""
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []
        self.data.clear()
        self.layer_order = []

    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Return recorded activations in layer order."""
        ordered_data = {k: self.data[k] for k in self.layer_order if k in self.data}
        return ordered_data
