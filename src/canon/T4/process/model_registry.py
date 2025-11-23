from typing import Dict, Type, List, Tuple
from .base_model import BaseInpaintingModel
from .stable_diffusion_inpainting import StableDiffusionInpainting
from .paint_by_example import PaintByExample
from .kandinsky_inpainting import KandinskyInpainting
from .resshift_inpainting import ResShiftInpainting


# Modelos principais (sempre disponíveis)
CORE_MODELS: Dict[str, Type[BaseInpaintingModel]] = {
    "stable_diffusion": StableDiffusionInpainting,
    "paint_by_example": PaintByExample,
    "kandinsky": KandinskyInpainting,
}

# Modelos opcionais (requerem instalação adicional)
OPTIONAL_MODELS: Dict[str, Type[BaseInpaintingModel]] = {
    "resshift": ResShiftInpainting,
}

MODEL_REGISTRY: Dict[str, Type[BaseInpaintingModel]] = {
    **CORE_MODELS,
    **OPTIONAL_MODELS
}


def get_model(model_name: str, **kwargs) -> BaseInpaintingModel:
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{model_name}' not found. Available models: {available}")
    
    model_class = MODEL_REGISTRY[model_name]
    return model_class(**kwargs)


def list_available_models(include_optional: bool = True) -> List[str]:
    """Lista modelos disponíveis.
    
    Args:
        include_optional: Se True, inclui modelos opcionais que podem
                         requerer instalação adicional (como ResShift)
    
    Returns:
        Lista de nomes de modelos disponíveis
    """
    if include_optional:
        return list(MODEL_REGISTRY.keys())
    else:
        return list(CORE_MODELS.keys())


def list_core_models() -> List[str]:
    """Lista apenas modelos principais (sem dependências externas)."""
    return list(CORE_MODELS.keys())


def list_optional_models() -> List[str]:
    """Lista modelos opcionais que requerem instalação adicional."""
    return list(OPTIONAL_MODELS.keys())


def check_model_availability(model_name: str) -> Tuple[bool, str]:
    """Verifica se um modelo pode ser carregado.
    
    Args:
        model_name: Nome do modelo
    
    Returns:
        Tuple (disponivel: bool, mensagem: str)
    """
    if model_name not in MODEL_REGISTRY:
        return False, f"Modelo '{model_name}' não encontrado no registro"
    
    try:
        model_class = MODEL_REGISTRY[model_name]
        # Tenta instanciar sem carregar o modelo
        model = model_class(device="cpu")
        return True, "Modelo disponível"
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Erro ao verificar modelo: {str(e)}"


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
