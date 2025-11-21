from PIL import Image
from .base_model import BaseInpaintingModel


class KandinskyInpainting(BaseInpaintingModel):
    def __init__(self, 
                 model_id: str = "kandinsky-community/kandinsky-2-2-decoder-inpaint",
                 prior_id: str = "kandinsky-community/kandinsky-2-2-prior",
                 device: str = None,
                 num_inference_steps: int = 50,
                 guidance_scale: float = 4.0,
                 **kwargs):
        super().__init__("KandinskyInpainting", device, **kwargs)
        self.model_id = model_id
        self.prior_id = prior_id
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.config.update({
            "model_id": model_id,
            "prior_id": prior_id,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale
        })

    def load_model(self) -> None:
        if self.is_loaded:
            return
        
        from diffusers import KandinskyV22InpaintPipeline, KandinskyV22PriorPipeline
        import torch
        
        self.prior = KandinskyV22PriorPipeline.from_pretrained(
            self.prior_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.prior = self.prior.to(self.device)
        
        self.model = KandinskyV22InpaintPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.model = self.model.to(self.device)
        self.is_loaded = True

    def _inpaint_impl(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        prompt = ""
        negative_prompt = ""
        
        image_embeds, negative_image_embeds = self.prior(
            prompt=prompt,
            negative_prompt=negative_prompt
        ).to_tuple()
        
        result = self.model(
            image=image,
            mask_image=mask,
            image_embeds=image_embeds,
            negative_image_embeds=negative_image_embeds,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale
        ).images[0]
        
        return result

    def unload_model(self) -> None:
        import torch
        
        if self.model is not None:
            del self.model
            self.model = None
        
        if hasattr(self, 'prior') and self.prior is not None:
            del self.prior
            self.prior = None
        
        self.is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
