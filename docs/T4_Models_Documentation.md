# Documentação Técnica - Modelos de Inpainting

Esta documentação detalha a implementação dos modelos de inpainting configurados na Etapa 2 do projeto T4. O sistema fornece uma interface unificada para trabalhar com diferentes modelos de difusão aplicados à tarefa de restauração de imagens.

## Arquitetura

### Classe Base: `InpaintingModel`

Todos os modelos herdam de `InpaintingModel` (`base_model.py`), que define a interface padrão:

**Métodos principais:**
- `load_model()`: Carrega o modelo na memória
- `inpaint(image, mask, **kwargs)`: Executa inpainting
- `preprocess(image, mask)`: Pré-processamento (conversões de tipo)
- `postprocess(result, original_size)`: Pós-processamento (redimensionamento)
- `unload_model()`: Libera memória GPU
- `get_model_info()`: Retorna metadados do modelo

**Características:**
- Conversão automática PIL ↔ numpy ↔ torch
- Tracking de tempo de inferência
- Gerenciamento de memória GPU
- Validação de tipos de entrada

## Modelos Implementados

1. **Stable Diffusion Inpainting** - Modelo base, equilibrado
2. **ControlNet Inpainting** - Controle estrutural fino
3. **Paint-by-Example** - Baseado em imagens de referência
4. **Kandinsky 2.2** - Arquitetura alternativa
5. **ResShift** - Inpainting + super-resolution

### 1. Stable Diffusion Inpainting

**Classe:** `StableDiffusionInpainting`  
**Model ID:** `runwayml/stable-diffusion-inpainting`  
**Uso:** Modelo geral, bom equilíbrio qualidade/velocidade

**Parâmetros configuráveis:**
```python
model = get_model("stable_diffusion", 
    device="cuda",
    num_inference_steps=50,     # Padrão
    guidance_scale=7.5           # Padrão
)
```

**Características:**
- Baseado em Stable Diffusion v1.5
- Treinado especificamente para inpainting
- Suporta prompts textuais
- Tempo médio: ~8-10s por imagem (GPU)

### 2. Paint-by-Example

**Classe:** `PaintByExample`  
**Model ID:** `Fantasy-Studio/Paint-by-Example`  
**Uso:** Inpainting baseado em imagens de referência

**Parâmetros configuráveis:**
```python
model = get_model("paint_by_example",
    device="cuda",
    num_inference_steps=50,
    guidance_scale=5.0              # Padrão
)

# Uso com imagem de referência
result = model.inpaint(image, mask, example_image=reference_img)
```

**Características:**
- Pode usar imagem de exemplo
- Copia estilo/textura da referência
- Útil para restauração com contexto
- Tempo médio: ~9-11s por imagem (GPU)

### 3. Kandinsky 2.2 Inpainting

**Classe:** `KandinskyInpainting`  
**Model IDs:**
- Prior: `kandinsky-community/kandinsky-2-2-prior`
- Decoder: `kandinsky-community/kandinsky-2-2-decoder-inpaint`

**Uso:** Arquitetura alternativa, bons resultados em alguns casos

**Parâmetros configuráveis:**
```python
model = get_model("kandinsky",
    device="cuda",
    num_inference_steps=50,
    guidance_scale=4.0              # Padrão
)
```

**Características:**
- Arquitetura de dois estágios (prior + decoder)
- Baseado em CLIP para entendimento semântico
- Diferentes estilos de geração
- Tempo médio: ~10-13s por imagem (GPU)

### 4. ResShift

**Classe:** `ResShiftInpainting`  
**Repository:** `zsyOAOA/ResShift`  
**Uso:** Inpainting + super-resolution unificado

**Parâmetros configuráveis:**
```python
model = get_model("resshift",
    device="cuda",
    num_inference_steps=15,         # Padrão
    guidance_scale=7.5
)
```

**Características:**
- Modelo unificado (inpainting + upscale)
- Inferência mais rápida (15 steps padrão)
- Requer repositório externo
- Tempo médio: ~5-7s por imagem (GPU)
- Requer instalação adicional:

```bash
git clone https://github.com/zsyOAOA/ResShift.git external/ResShift
```

## Sistema de Registro

O `model_registry.py` implementa um padrão factory para instanciação de modelos.

### Funções principais:

```python
from canon.T4 import get_model, list_available_models

# Listar modelos
models = list_available_models()  # ['stable_diffusion', 'controlnet', ...]

# Instanciar modelo
model = get_model("stable_diffusion", device="cuda")

# Executar inpainting
result = model.inpaint(image, mask, prompt="high quality")
```

## Configuração

### Arquivo YAML (`config/model_configs.yaml`)

```yaml
stable_diffusion:
  model_id: "runwayml/stable-diffusion-inpainting"
  device: "cuda"
  num_inference_steps: 50
  guidance_scale: 7.5

controlnet:
  controlnet_id: "lllyasviel/control_v11p_sd15_inpaint"
  base_model_id: "runwayml/stable-diffusion-v1-5"
  device: "cuda"
  num_inference_steps: 50
  guidance_scale: 7.5
  controlnet_conditioning_scale: 1.0
```

### Formato de Saída

Cada resultado de `inpaint()` retorna um dicionário:

```python
result = {
    "image": PIL.Image,           # Imagem resultante
    "model_name": str,            # Nome do modelo
    "inference_time": float,      # Tempo em segundos
    "device": str,                # "cuda" ou "cpu"
    "config": dict                # Configuração usada
}
```

### Utilitários
- `batch_utils.py` - Processamento em lote, comparação visual
- `test_models.py` - Testes automatizados
- `example_usage.py` - Exemplos práticos com dados reais
- Configuração YAML para parâmetros padrão

## Referências

- Stable Diffusion: https://github.com/runwayml/stable-diffusion
- ControlNet: https://github.com/lllyasviel/ControlNet
- Paint-by-Example: https://github.com/Fantasy-Studio/Paint-by-Example
- Kandinsky: https://github.com/ai-forever/Kandinsky-2
- ResShift: https://github.com/zsyOAOA/ResShift
- Diffusers: https://github.com/huggingface/diffusers
