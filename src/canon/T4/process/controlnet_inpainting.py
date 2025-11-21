from PIL import Image
from .base_model import BaseInpaintingModel


class ControlNetInpainting(BaseInpaintingModel):
    def __init__(self, 
                 controlnet_id: str = "lllyasviel/control_v11p_sd15_inpaint",
                 base_model_id: str = "runwayml/stable-diffusion-v1-5",
                 device: str = None,
                 num_inference_steps: int = 50,
                 guidance_scale: float = 7.5,
                 controlnet_conditioning_scale: float = 1.0,
                 **kwargs):
        super().__init__("ControlNetInpainting", device, **kwargs)
        self.controlnet_id = controlnet_id
        self.base_model_id = base_model_id
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.controlnet_conditioning_scale = controlnet_conditioning_scale
        self.config.update({
            "controlnet_id": controlnet_id,
            "base_model_id": base_model_id,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "controlnet_conditioning_scale": controlnet_conditioning_scale
        })

    def load_model(self) -> None:
        if self.is_loaded:
            return
        
        from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline
        import torch
        
        controlnet = ControlNetModel.from_pretrained(
            self.controlnet_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        self.model = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            self.base_model_id,
            controlnet=controlnet,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None
        )
        self.model = self.model.to(self.device)
        self.is_loaded = True

    def _inpaint_impl(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        import numpy as np
        
        mask_array = np.array(mask)
        control_image = np.array(image).copy()
        control_image[mask_array > 128] = 255
        control_image = Image.fromarray(control_image)
        
        result = self.model(
            prompt="",
            image=image,
            mask_image=mask,
            control_image=control_image,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            controlnet_conditioning_scale=self.controlnet_conditioning_scale
        ).images[0]
        
        return result
