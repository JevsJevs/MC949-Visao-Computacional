from typing import Dict, Type, List
from .base_model import BaseInpaintingModel
from .stable_diffusion_inpainting import StableDiffusionInpainting
from .controlnet_inpainting import ControlNetInpainting
from .paint_by_example import PaintByExample
from .kandinsky_inpainting import KandinskyInpainting
from .resshift_inpainting import ResShiftInpainting


MODEL_REGISTRY: Dict[str, Type[BaseInpaintingModel]] = {
    "stable_diffusion": StableDiffusionInpainting,
    "controlnet": ControlNetInpainting,
    "paint_by_example": PaintByExample,
    "kandinsky": KandinskyInpainting,
    "resshift": ResShiftInpainting,
}


def get_model(model_name: str, **kwargs) -> BaseInpaintingModel:
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{model_name}' not found. Available models: {available}")
    
    model_class = MODEL_REGISTRY[model_name]
    return model_class(**kwargs)


def list_available_models() -> List[str]:
    return list(MODEL_REGISTRY.keys())


def run_all_inpainting_models(image, mask, models_list: List[str] = None, **kwargs) -> Dict[str, Dict]:
    if models_list is None:
        models_list = list_available_models()
    
    results = {}
    for model_name in models_list:
        try:
            model = get_model(model_name, **kwargs)
            result = model.inpaint(image, mask)
            results[model_name] = result
            model.unload_model()
        except Exception as e:
            results[model_name] = {
                "error": str(e),
                "model_name": model_name
            }
    
    return results
