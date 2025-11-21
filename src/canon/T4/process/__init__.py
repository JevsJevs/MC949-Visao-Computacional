from .base_model import BaseInpaintingModel
from .stable_diffusion_inpainting import StableDiffusionInpainting
from .controlnet_inpainting import ControlNetInpainting
from .paint_by_example import PaintByExample
from .kandinsky_inpainting import KandinskyInpainting
from .resshift_inpainting import ResShiftInpainting
from .model_registry import get_model, list_available_models, run_all_inpainting_models

__all__ = [
    "BaseInpaintingModel",
    "StableDiffusionInpainting",
    "ControlNetInpainting",
    "PaintByExample",
    "KandinskyInpainting",
    "ResShiftInpainting",
    "get_model",
    "list_available_models",
    "run_all_inpainting_models",
]
