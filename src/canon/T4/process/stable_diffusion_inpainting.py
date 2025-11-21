from PIL import Image
from .base_model import BaseInpaintingModel


class StableDiffusionInpainting(BaseInpaintingModel):
    def __init__(self, model_id: str = "runwayml/stable-diffusion-inpainting", 
                 device: str = None,
                 num_inference_steps: int = 50,
                 guidance_scale: float = 7.5,
                 **kwargs):
        super().__init__("StableDiffusionInpainting", device, **kwargs)
        self.model_id = model_id
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.config.update({
            "model_id": model_id,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale
        })

    def load_model(self) -> None:
        if self.is_loaded:
            return
        
        from diffusers import StableDiffusionInpaintPipeline
        import torch
        
        self.model = StableDiffusionInpaintPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None
        )
        self.model = self.model.to(self.device)
        self.is_loaded = True

    def _inpaint_impl(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        result = self.model(
            prompt="",
            image=image,
            mask_image=mask,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale
        ).images[0]
        
        return result
