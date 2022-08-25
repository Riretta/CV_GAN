import sys
sys.path.append(".")

# also disable grad to save memory
import torch
torch.set_grad_enabled(False)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import yaml
import torch
from omegaconf import OmegaConf
from taming.models.vcgan import VQModel, GumbelVQ
from taming.models.vqgan import VQModel as VQModel_VQ
from taming.models.vqgan import GumbelVQ as GumbelVQ_VQ



def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config

def load_vcgan(config, ckpt_path=None, is_gumbel=False):
  if is_gumbel:
    model = GumbelVQ(**config.model.params)
  else:
    model = VQModel(**config.model.params)
  if ckpt_path is not None:
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
  return model.eval()

def load_vqgan(config, ckpt_path=None, is_gumbel=False):
  if is_gumbel:
    model = GumbelVQ_VQ(**config.model.params)
  else:
    model = VQModel_VQ(**config.model.params)
  if ckpt_path is not None:
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
  return model.eval()

def preprocess_vqgan(x):
  x = 2.*x - 1.
  return x

def custom_to_pil(x):
  x = x.detach().cpu()
  x = torch.clamp(x, -1., 1.)
  x = (x + 1.)/2.
  x = x.permute(1,2,0).numpy()
  x = (255*x).astype(np.uint8)
  x = Image.fromarray(x)
  if not x.mode == "RGB":
    x = x.convert("RGB")
  return x

def reconstruct_with_vqgan(x, model):
  # could also use model(x) for reconstruction but use explicit encoding and decoding here
  z, _, _ = model.encode(x)
  print(z.size())
  print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}")
  xrec = model.decode(z)
  return xrec

# configRP = load_config("logs/2022-01-11T12-53-39_imagenet_vqgan/configs/2022-01-28T10-37-57-project.yaml", display=False)
configCVGAN = load_config("configs/imagenet_vcgan.yaml", display=False)
modelCVGAN = load_vcgan(configCVGAN, ckpt_path="logs/2022-01-11T12-53-39_imagenet_vqgan/checkpoints/last.ckpt").to(DEVICE)
configVQGAN = load_config("configs/imagenet_vqgan.yaml", display=False)
modelVQGAN = load_vqgan(configVQGAN, ckpt_path="logs/2022-01-17T11-45-59_imagenet_vqgan/checkpoints/last.ckpt").to(DEVICE)
import io
import os, sys
import requests
import PIL
from PIL import Image
from PIL import ImageDraw, ImageFont
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf", 22)

def download_image(url):
  resp = requests.get(url)
  resp.raise_for_status()
  return PIL.Image.open(io.BytesIO(resp.content))


def preprocess(img, target_image_size=256, map_dalle=True):
  s = min(img.size)

  if s < target_image_size:
    raise ValueError(f'min dim for image {s} < {target_image_size}')

  r = target_image_size / s
  s = (round(r * img.size[1]), round(r * img.size[0]))
  img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
  img = TF.center_crop(img, output_size=2 * [target_image_size])
  img = torch.unsqueeze(T.ToTensor()(img), 0)
  print(img.size())
  return img

def stack_reconstructions(input, x0,x1, titles=[]):
  assert input.size == x0.size == x1.size
  w, h = input.size[0], input.size[1]
  img = Image.new("RGB", (3*w, h))
  img.paste(input, (0,0))
  img.paste(x0, (1*w,0))
  img.paste(x1, (2 * w, 0))

  for i, title in enumerate(titles):
    ImageDraw.Draw(img).text((i*w, 0), f'{title}', (255, 255, 255), font=font) # coordinates, text, color, font
  return img


titles=["Input", "CVGAN", "VQGAN"]
def reconstruction_pipeline(url, size=320):
  x_vqgan = preprocess(download_image(url), target_image_size=size, map_dalle=False)
  x_vqgan = x_vqgan.to(DEVICE)
  x1 = reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), modelCVGAN)
  x2 = reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), modelVQGAN)


  img = stack_reconstructions(custom_to_pil(preprocess_vqgan(x_vqgan[0])),
                              custom_to_pil(x1[0]), custom_to_pil(x2[0]),titles=titles)
  return img

img = reconstruction_pipeline(url='https://heibox.uni-heidelberg.de/f/7bb608381aae4539ba7a/?dl=1', size=256)

img.save("samples/squereel.png")

img = reconstruction_pipeline("https://heibox.uni-heidelberg.de/f/e41f5053cbd34f11a8d5/?dl=1", size=256)

img.save("samples/birds.png")

img = reconstruction_pipeline(url='https://assets.bwbx.io/images/users/iqjWHBFdfxIU/iKIWgaiJUtss/v2/1000x-1.jpg', size=256)

img.save("samples/penguin.png")
