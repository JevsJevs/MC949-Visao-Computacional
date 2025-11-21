from PIL import Image
import numpy as np
from .base_model import BaseInpaintingModel


class ResShiftInpainting(BaseInpaintingModel):
    def __init__(self, 
                 device: str = None,
                 num_inference_steps: int = 15,
                 **kwargs):
        super().__init__("ResShiftInpainting", device, **kwargs)
        self.num_inference_steps = num_inference_steps
        self.config.update({
            "num_inference_steps": num_inference_steps,
            "task": "inpainting"
        })

    def load_model(self) -> None:
        if self.is_loaded:
            return
        
        try:
            import torch
            from pathlib import Path
            import sys
            
            resshift_path = Path(__file__).parent.parent.parent.parent.parent / "external" / "ResShift"
            if resshift_path.exists():
                sys.path.insert(0, str(resshift_path))
            
            from resshift.sampler import ResShiftSampler
            
            self.model = ResShiftSampler(
                task="inpainting",
                device=self.device
            )
            self.is_loaded = True
        except ImportError as e:
            raise ImportError(
                f"ResShift not found. Please clone the repository: "
                f"git clone https://github.com/zsyOAOA/ResShift.git external/ResShift"
            ) from e

    def _inpaint_impl(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        import torch
        
        image_array = np.array(image).astype(np.float32) / 255.0
        mask_array = np.array(mask).astype(np.float32) / 255.0
        
        if len(mask_array.shape) == 2:
            mask_array = mask_array[:, :, None]
        
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(mask_array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            result_tensor = self.model.inference(
                image_tensor,
                mask_tensor,
                num_inference_steps=self.num_inference_steps
            )
        
        result_array = result_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        result_array = np.clip(result_array * 255, 0, 255).astype(np.uint8)
        result = Image.fromarray(result_array)
        
        return result
