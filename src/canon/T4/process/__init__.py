from .base_model import BaseInpaintingModel
from .stable_diffusion_inpainting import StableDiffusionInpainting
from .paint_by_example import PaintByExample
from .kandinsky_inpainting import KandinskyInpainting
from .resshift_inpainting import ResShiftInpainting
from .model_registry import (
    get_model, 
    list_available_models, 
    list_core_models,
    list_optional_models,
    check_model_availability,
    run_all_inpainting_models
)

__all__ = [
    "BaseInpaintingModel",
    "StableDiffusionInpainting",
    "PaintByExample",
    "KandinskyInpainting",
    "ResShiftInpainting",
    "get_model",
    "list_available_models",
    "list_core_models",
    "list_optional_models",
    "check_model_availability",
    "run_all_inpainting_models",
]
