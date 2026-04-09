#para crear funciones que ayuden al resto de la aplicacion

import numpy as np
import torch
from huggan.pytorch.lightweight_gan.lightweight_gan import LightweightGAN

from huggan.pytorch.lightweight_gan.lightweight_gan import LightweightGAN

# ---Patching---
_old_from_pretrained = LightweightGAN._from_pretrained

@classmethod
def _new_from_pretrained(cls, model_id, use_auth_token=False, **kwargs):
    # Interceptamos la llamada interna y le inyectamos el argumento obligatorio
    return _old_from_pretrained(model_id, use_auth_token=use_auth_token, **kwargs)

LightweightGAN._from_pretrained = _new_from_pretrained
# --- Patching end ---

# Tu función original puede quedar limpia de nuevo
def carga_modelo(model_name):
    # Ya no necesitas pasarle el parámetro aquí, el parche lo maneja
    gan = LightweightGAN.from_pretrained(model_name)
    return gan


## Cargamos el modelo desde el Hub de Hugging Face
def carga_modelo(model_name="ceyda/butterfly_cropped_uniq1K_512", model_version=None):
    gan = LightweightGAN.from_pretrained(model_name, use_auth_token=False)
    gan.eval()
    return gan


## Usamos el modelo GAN para generar imágenes
def genera(gan, batch_size=1):
    with torch.no_grad():
        ims = gan.G(torch.randn(batch_size, gan.latent_dim)).clamp_(0.0, 1.0) * 255
        ims = ims.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)
    return ims