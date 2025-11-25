from PIL import Image
import numpy as np
from .base_model import BaseInpaintingModel


class ResShiftInpainting(BaseInpaintingModel):
    """
    ResShift inpainting model with color transfer correction.
    
    Implementa correções críticas para inpainting com ResShift:
    1. Máscara invertida: ResShift espera -1 para área desconhecida, não +1
    2. Color transfer: Ajusta distribuição de cor da área gerada para match com área conhecida
    3. Composição correta: Combina área gerada com imagem original preservando área conhecida
    
    Nota: Requer GPU para melhor desempenho. Tempo de inferência ~1-2s em GPU T4.
    """
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
            from omegaconf import OmegaConf
            from basicsr.utils.download_util import load_file_from_url
            
            # Adicionar ResShift ao PYTHONPATH antes de importar
            resshift_path = Path(__file__).parent.parent.parent.parent.parent / "external" / "ResShift"
            if not resshift_path.exists():
                raise ImportError(f"ResShift not found at {resshift_path}")
            
            # Adicionar ao início do sys.path se ainda não estiver
            resshift_str = str(resshift_path)
            if resshift_str not in sys.path:
                sys.path.insert(0, resshift_str)
            
            # Importar ResShiftSampler
            from sampler import ResShiftSampler
            
            # Preparar configurações para inpainting
            config_path = resshift_path / 'configs' / 'inpaint_lama256_imagenet.yaml'
            configs = OmegaConf.load(str(config_path))
            
            # Preparar diretórios e checkpoints
            ckpt_dir = resshift_path / 'weights'
            ckpt_dir.mkdir(exist_ok=True)
            
            ckpt_url = 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_inpainting_imagenet_s4.pth'
            ckpt_path = ckpt_dir / 'resshift_inpainting_imagenet_s4.pth'
            
            vqgan_url = 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth'
            vqgan_path = ckpt_dir / 'autoencoder_vq_f4.pth'
            
            # Download checkpoints se necessário
            if not ckpt_path.exists():
                print(f"Downloading ResShift checkpoint...")
                load_file_from_url(
                    url=ckpt_url,
                    model_dir=str(ckpt_dir),
                    progress=True,
                    file_name=ckpt_path.name,
                )
            
            if not vqgan_path.exists():
                print(f"Downloading VQGAN checkpoint...")
                load_file_from_url(
                    url=vqgan_url,
                    model_dir=str(ckpt_dir),
                    progress=True,
                    file_name=vqgan_path.name,
                )
            
            # Configurar paths nos configs
            configs.model.ckpt_path = str(ckpt_path)
            configs.autoencoder.ckpt_path = str(vqgan_path)
            configs.diffusion.params.sf = 1  # scale factor = 1 para inpainting
            
            # Criar sampler
            self.model = ResShiftSampler(
                configs,
                sf=1,
                chop_size=512,
                chop_stride=448,  # 512 - 64
                chop_bs=1,
                use_amp=True,
                seed=12345,
                padding_offset=configs.model.params.get('lq_size', 64),
            )
            
            self.is_loaded = True
        except Exception as e:
            raise RuntimeError(f"Failed to load ResShift: {str(e)}") from e

    def _inpaint_impl(self, image: Image.Image, mask: Image.Image, **kwargs) -> Image.Image:
        import torch
        import torch.nn.functional as F
        
        # Converter imagem para tensor [-1, 1]
        image_array = np.array(image).astype(np.float32) / 255.0  # [0, 1]
        image_array = image_array * 2.0 - 1.0  # [-1, 1]
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
        
        # Converter máscara para tensor [-1, 1]
        # CORREÇÃO: ResShift espera -1 = área desconhecida (preencher), 1 = área conhecida
        # Nossa máscara: branco (255) = preencher, preto (0) = manter
        # Então precisamos INVERTER: branco vira -1, preto vira 1
        mask_array = np.array(mask).astype(np.float32) / 255.0  # [0, 1] - branco=1, preto=0
        mask_array = 1.0 - mask_array  # INVERTER: branco=0, preto=1
        mask_array = mask_array * 2.0 - 1.0  # [-1, 1] - branco=-1, preto=1
        
        if len(mask_array.shape) == 2:
            mask_array = mask_array[:, :, None]
        
        mask_tensor = torch.from_numpy(mask_array).permute(2, 0, 1).unsqueeze(0)
        
        # Mover para device correto
        if self.device == 'cuda':
            image_tensor = image_tensor.cuda()
            mask_tensor = mask_tensor.cuda()
        
        # Executar inferência usando sample_func diretamente
        with torch.no_grad():
            result_tensor = self.model.sample_func(
                image_tensor,
                noise_repeat=False,
                mask=mask_tensor
            )
        
        # Normalizar resultado, máscara e imagem para [0, 1]
        result_normalized = result_tensor * 0.5 + 0.5  # [-1, 1] -> [0, 1]
        mask_normalized = mask_tensor * 0.5 + 0.5      # [-1, 1] -> [0, 1]
        image_normalized = image_tensor * 0.5 + 0.5    # [-1, 1] -> [0, 1]
        
        # Expandir máscara para 3 canais (RGB) se necessário
        if mask_normalized.shape[1] == 1:
            mask_normalized = mask_normalized.expand(-1, 3, -1, -1)
        
        # AJUSTE DE COR: Transferir distribuição de cor da área conhecida para área gerada
        # Calcular máscaras binárias
        known_mask = mask_normalized > 0.5  # Área conhecida (valores próximos de 1)
        inpaint_mask = mask_normalized < 0.5  # Área a preencher (valores próximos de 0)
        
        # Ajustar cor de cada canal RGB
        result_adjusted = result_normalized.clone()
        for c in range(3):
            if known_mask[0, c].sum() > 0 and inpaint_mask[0, c].sum() > 0:
                # Estatísticas da área conhecida na imagem original
                known_mean = image_normalized[0, c][known_mask[0, c]].mean()
                known_std = image_normalized[0, c][known_mask[0, c]].std()
                
                # Estatísticas da área gerada
                gen_mean = result_normalized[0, c][inpaint_mask[0, c]].mean()
                gen_std = result_normalized[0, c][inpaint_mask[0, c]].std()
                
                # Transferir distribuição de cor (color transfer)
                if gen_std > 0:
                    result_adjusted[0, c][inpaint_mask[0, c]] = (
                        (result_normalized[0, c][inpaint_mask[0, c]] - gen_mean) * 
                        (known_std / gen_std) + known_mean
                    )
        
        result_adjusted = torch.clamp(result_adjusted, 0, 1)
        
        # Composição com máscara invertida:
        # mask_normalized perto de 0 = usar resultado ajustado
        # mask_normalized perto de 1 = usar imagem original
        final_result = result_adjusted * (1 - mask_normalized) + image_normalized * mask_normalized
        
        # Converter para imagem [0, 255]
        result_array = final_result.squeeze(0).permute(1, 2, 0).cpu().numpy()
        result_array = np.clip(result_array * 255, 0, 255).astype(np.uint8)
        result = Image.fromarray(result_array)
        
        return result
