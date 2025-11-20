from typing import Tuple
import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        def fwd_hook(_, __, output):
            self.activations = output.detach()

        def bwd_hook(_, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.fh = target_layer.register_forward_hook(fwd_hook)
        self.bh = target_layer.register_full_backward_hook(bwd_hook)

    def __del__(self):
        try:
            self.fh.remove(); self.bh.remove()
        except Exception:
            pass

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[:, class_idx]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # Global average pool gradients over spatial dims
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() + 1e-6)
        return cam


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    heat = (heatmap * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    if image.max() <= 1.0:
        img = (image * 255).astype(np.uint8)
    else:
        img = image.astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    overlay = cv2.addWeighted(heat, alpha, img, 1 - alpha, 0)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return overlay


def grad_cam(model, image, target_layer):
    grad_cam = GradCAM(model, target_layer)
    heatmap = grad_cam.generate(image)
    return heatmap
