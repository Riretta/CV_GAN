import sys
sys.path.append(".")

# also disable grad to save memory
import torch
from tqdm import tqdm
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
  # print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}")
  xrec = model.decode(z)
  return xrec

# configRP = load_config("logs/2022-01-11T12-53-39_imagenet_vqgan/configs/2022-01-28T10-37-57-project.yaml", display=False)
# configCVGAN = load_config("configs/denoisefish_vcgan.yaml", display=False)
# modelCVGAN = load_vcgan(configCVGAN, ckpt_path="logs_2/denoise_GDL/checkpoints/Epoch12_dgdl.ckpt").to(DEVICE)
# modelCVGAN_GDL = load_vcgan(configCVGAN, ckpt_path="logs_2/denoise_GDL/checkpoints/Epoch12_dgdl.ckpt").to(DEVICE)
# modelCVGAN_sobel = load_vcgan(configCVGAN, ckpt_path="logs_2/denoise_GDL/checkpoints/Epoch12_dgdl.ckpt").to(DEVICE)
# modelCVGAN_sobel_GDL = load_vcgan(configCVGAN, ckpt_path="logs_2/denoise_GDL/checkpoints/Epoch12_dgdl.ckpt").to(DEVICE)
#folder_name = "CSobel_GDL_sobel_200e"

#configVQGAN = load_config("configs/denoisefish_vqgan.yaml", display=False)
#modelVQGAN = load_vqgan(configVQGAN, ckpt_path="logs/denoise_2022-01-17T11-45-59_imagenet_vqgan/checkpoints/last.ckpt").to(DEVICE)

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

def open_image(path):
  return PIL.Image.open(path)


def preprocess(img, target_image_size=256, map_dalle=True):
  s = min(img.size)

  # if s < target_image_size:
  #   raise ValueError(f'min dim for image {s} < {target_image_size}')

  r = target_image_size / s
  s = (round(r * img.size[1]), round(r * img.size[0]))
  img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
  img = TF.center_crop(img, output_size=2 * [target_image_size])
  img = torch.unsqueeze(T.ToTensor()(img), 0)
  if img.size()[1] < 3:
    img = torch.cat([img,img,img],dim=1)
  return img

def stack_reconstructions(input, x0,x1,x2,x3, titles=[]):
  assert input.size == x0.size == x1.size == x2.size == x3.size
  w, h = input.size[0], input.size[1]
  img = Image.new("RGB", (w, 5*h))
  img.paste(input, (0,0))
  img.paste(x0, (0,1*h))
  img.paste(x1, (0,2 * h))
  img.paste(x2, (0,3 * h))
  img.paste(x3, (0,4 * h))
  for i, title in enumerate(titles):
    ImageDraw.Draw(img).text((0,i*h), f'{title}', (255, 255, 255), font=font) # coordinates, text, color, font
  return img

def single_reconstructions(input, x0, titles=[]):
  assert input.size == x0.size
  w, h = input.size[0], input.size[1]
  img = Image.new("RGB", (1*w, h))
  img.paste(x0, (0,0))
  # for i, title in enumerate(titles):
  #   ImageDraw.Draw(img).text((i*w, 0), f'{title}', (255, 255, 255), font=font) # coordinates, text, color, font
  return img


# titles=["Input", "UW-CVGAN w/o SL", "UW-CVGAN", "VQGAN"]
titles=["", "", "", "",""]
def reconstruction_pipeline(path, size=320):
  x_vqgan = preprocess(open_image(path), target_image_size=size, map_dalle=False)
  x_vqgan = x_vqgan.to(DEVICE)
  x1 = reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), modelCVGAN)
  x2 = reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), modelCVGAN_GDL)
  x3 = reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), modelCVGAN_sobel)
  x4 = reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), modelCVGAN_sobel_GDL)
  img = stack_reconstructions(custom_to_pil(preprocess_vqgan(x_vqgan[0])),
                              custom_to_pil(x1[0]), custom_to_pil(x2[0]), custom_to_pil(x3[0]),custom_to_pil(x4[0]) ,titles=titles)
  return img

def reconstruction_single(path, model, size=320):
  x_vqgan = preprocess(open_image(path), target_image_size=size, map_dalle=False)
  x_vqgan = x_vqgan.to(DEVICE)
  x0 = reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), model)
  img = single_reconstructions(custom_to_pil(preprocess_vqgan(x_vqgan[0])),custom_to_pil(x0[0]),titles=titles)
  return img

""" Human benchmark """
# root, folder = "/media/TBData2/Datasets/ImageNet10k/Test/ctest10k/", "ImageNet"
# root, folder  = "/media/TBData2/Datasets/COCO_stuff/val/val2017/", "COCO"
# root, folder  = "/media/TBData2/Datasets/ImageNet/val/", "ImageNet_val"
# root, folder   = "/media/TBData2/Datasets/FFHQ/images1024x1024/", "FFHQ"
""" Dataset of fish """
# root,folder, target =  "/media/TBData2/Datasets/HICRD/HICRD_paired/lr", "HICRD_paired" ,"/media/TBData2/Datasets/HICRD/HICRD_paired/hr"
# root,folder, target = "/media/TBData2/Datasets/UIEB/UIEB/", "UIEB", " "
# root,folder,target = "/media/TBData2/Datasets/EUVP/Paired/train_A", "EUVP","/media/TBData2/Datasets/EUVP/Paired/train_B"
# root,folder,target = "/media/TBData2/Datasets/EUVP/test_samples/testA/", "EUVP_test","/media/TBData2/Datasets/EUVP/test_samples/testB"
root,folder,target = "/media/TBData2/Datasets/USR_248/TEST/testA", "USR248_test", "/media/TBData2/Datasets/USR_248/TEST/testB"
# root,folder,target = "/media/TBData2/Datasets/UFO_120/UFO-120/TEST/test", "UFO120", "/media/TBData2/Datasets/UFO_120/UFO-120/TEST/GT"
# appendix= '_Epoch8'
# pretuningfolder = 'PreTuning_Epoch_8'
# if not os.path.exists(os.path.join('samples', folder + appendix)):
#   os.mkdir(os.path.join('samples',pretuningfolder, folder + appendix + '_group'))
#   os.mkdir(os.path.join('samples',pretuningfolder, folder + appendix))
#   os.mkdir(os.path.join('samples',pretuningfolder, folder + appendix + '_GDL'))
#   os.mkdir(os.path.join('samples',pretuningfolder, folder + appendix + '_sobel'))
#   os.mkdir(os.path.join('samples',pretuningfolder, folder + appendix + '_sobel_GDL'))
#   os.mkdir(os.path.join('samples',pretuningfolder, folder + appendix + '_INPUT'))

list_dir = os.listdir(root)
print(f"There are {len(list_dir)} folders dataset {folder}")
# for img_dir in tqdm(list_dir):
#   list_imgs = os.listdir(os.path.join(root,img_dir))
#   print(f"Start the list of N: {len(list_imgs)} - {img_dir}")
#   for img_path in tqdm(list_imgs):
#       if ".png" in img_path or ".jpg" in img_path or "JPEG" in img_path:
#         dataset = os.path.join(root,img_dir)
        # img = reconstruction_pipeline(path=os.path.join(dataset,img_path), size=256)
        # img.save(f"samples/{pretuningfolder}/{folder}{appendix}_group/{img_path}")
        #
        # img = reconstruction_single(os.path.join(dataset,img_path), modelCVGAN, size=256)
        # img.save(f"samples/{pretuningfolder}/{folder}{appendix}/{img_path}")
        # del img

        # img = reconstruction_single(os.path.join(dataset,img_path), modelCVGAN_GDL, size=256)
        # img.save(f"samples/{pretuningfolder}/{folder}{appendix}_GDL/{img_path}")
        # del img
        #
        # img = reconstruction_single(os.path.join(dataset, img_path), modelCVGAN_sobel, size=256)
        # img.save(f"samples/{pretuningfolder}/{folder}{appendix}_sobel/{img_path}")
        # del img
        #
        # img = reconstruction_single(os.path.join(dataset,img_path), modelCVGAN_sobel_GDL, size=256)
        # img.save(f"samples/{pretuningfolder}/{folder}{appendix}_sobel_GDL/{img_path}")
        # del img
        # img = preprocess(open_image(os.path.join(dataset,img_path),), target_image_size=256, map_dalle=False)
        # img = img.to(DEVICE)
        # img = single_reconstructions(custom_to_pil(preprocess_vqgan(img[0])),custom_to_pil(preprocess_vqgan(img[0])))
        # img.save(f"samples/{pretuningfolder}/{folder}{appendix}_INPUT/{img_path}")
        # del img
print("END.. with models")
if not target == " ":
  list_dir = os.listdir(target)
  print(f"TARGET:: There are {len(list_dir)} folders")
  for img_dir in list_dir:
    list_imgs = os.listdir(os.path.join(target,img_dir))
    print(f"Start the list of N: {len(list_imgs)} - {img_dir}")
    for img_path in list_imgs:
        if ".png" in img_path or ".jpg" in img_path or "JPEG" in img_path:
          dataset = os.path.join(target,img_dir)
          img = preprocess(open_image(os.path.join(dataset,img_path),), target_image_size=256, map_dalle=False)
          img = img.to(DEVICE)
          img = single_reconstructions(custom_to_pil(preprocess_vqgan(img[0])),custom_to_pil(preprocess_vqgan(img[0])))
          if 'test' in img_path:
            img.save(f"samples/{folder}_256/{img_path.replace('_ref','')}")
          else:
            img.save(f"samples/{folder}_256/{img_path.replace('ref', 'good')}")
          del img
  print('END')
