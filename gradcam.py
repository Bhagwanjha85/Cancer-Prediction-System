import numpy as np
import torch
import cv2
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GradCAM")

class GradCAM:
    """
    Computes PyTorch Grad-CAM (Gradient-weighted Class Activation Mapping) heatmaps
    to visualize deep model decision focus regions.
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module = None):
        self.model = model
        self.model.eval()
        
        # If target layer is not provided, automatically find the last Conv2d layer
        if target_layer is None:
            self.target_layer = self.find_last_conv_layer(self.model)
            logger.info(f"Automatically identified last conv layer for Grad-CAM: {self.target_layer}")
        else:
            self.target_layer = target_layer
            
            
        self.gradients = None
        self.activations = None
        self.handlers = []
        
        # Register hooks
        self._register_hooks()

    def find_last_conv_layer(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Dynamically finds the last Conv2d module in a PyTorch model.
        """
        for module in reversed(list(model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                return module
        raise ValueError("No Conv2d layers found in the PyTorch model.")

    def _register_hooks(self):
        """
        Registers forward hooks to capture activations and registers a tensor-level 
        backward hook to capture gradients.
        """
        def forward_hook(module, input, output):
            # Save activations
            self.activations = output.detach()
            
            # Register a hook directly on the output tensor to capture its gradient
            # when backward() is called. This is extremely robust and avoids
            # module-level backward hook inplace/view errors.
            def tensor_backward_hook(grad):
                self.gradients = grad.detach()
                
            output.register_hook(tensor_backward_hook)

        self.handlers.append(self.target_layer.register_forward_hook(forward_hook))

    def remove_hooks(self):
        """
        Removes all registered forward hooks from the model.
        """
        for handler in self.handlers:
            handler.remove()
        self.handlers = []

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Generates a 2D Grad-CAM heatmap normalized to [0, 1].
        """
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Target score selection
        score = output[0][class_idx]
        
        # Backward pass
        score.backward()
        
        # Get activations and gradients
        # activations shape: [1, C, H, W]
        # gradients shape: [1, C, H, W]
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hook variables failed to capture. Check hook registrations.")
            
        gradients = self.gradients[0]  
        activations = self.activations[0]  
        
        # Channel-wise global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # [C]
        
        # Linear combination of activations and weights (only using positive gradients to highlight features of class interest)
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            if w > 0:
                cam += w * activations[i]
            
        # Apply ReLU to consider only features that positively contribute to the class score
        cam = torch.clamp(cam, min=0)
        
        # Enhance spatial alignment by multiplying with the mean activation across channels
        # which naturally highlights physical boundaries and high-contrast lesion zones
        mean_activation = torch.mean(torch.clamp(activations, min=0), dim=0)
        if mean_activation.max() > 0:
            mean_activation = mean_activation / mean_activation.max()
            cam = cam * mean_activation
        
        # Transfer to CPU numpy
        cam = cam.cpu().numpy()
        
        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
            
        return cam

    @staticmethod
    def overlay_heatmap(
        heatmap: np.ndarray, 
        original_img: np.ndarray, 
        alpha: float = 0.55, 
        colormap: int = cv2.COLORMAP_JET
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Overlays the computed Grad-CAM heatmap on top of the original BGR image.
        Returns:
            colored_heatmap: Heatmap converted to RGB for visualization.
            superimposed_img: Heatmap overlaid on original BGR image, returned in RGB.
        """
        # Resize heatmap
        heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        
        # Apply power scaling to suppress noise/background highlights and focus on peak areas
        heatmap_resized = np.power(heatmap_resized, 1.8)
        
        # Re-normalize after power scaling
        if heatmap_resized.max() > 0:
            heatmap_resized = heatmap_resized / heatmap_resized.max()
        
        # Scale to 255
        heatmap_255 = np.uint8(255 * heatmap_resized)
        
        # Apply color map
        colored_heatmap_bgr = cv2.applyColorMap(heatmap_255, colormap)
        
        # Superimpose
        superimposed_bgr = cv2.addWeighted(original_img, 1.0 - alpha, colored_heatmap_bgr, alpha, 0)
        
        # Convert to RGB
        colored_heatmap_rgb = cv2.cvtColor(colored_heatmap_bgr, cv2.COLOR_BGR2RGB)
        superimposed_rgb = cv2.cvtColor(superimposed_bgr, cv2.COLOR_BGR2RGB)
        
        return colored_heatmap_rgb, superimposed_rgb
