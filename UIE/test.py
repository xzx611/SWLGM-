## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881



import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
import utils

from natsort import natsorted
from glob import glob
from basicsr.models.archs.restormer_arch import MambaU as Restormer
from skimage.util import img_as_ubyte
from pdb import set_trace as stx

parser = argparse.ArgumentParser(description='Image Deraining using Restormer')

parser.add_argument('--input_dir', default='../', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
# parser.add_argument('--weights', default='../experiments/UIE_MambaU_111/models/net_g_202000.pth', type=str, help='Path to weights')
parser.add_argument('--weights', default='../experiments/Now_best_l1+SSIM/net_g_latest.pth', type=str, help='Path to weights')

args = parser.parse_args()

####### Load yaml #######
yaml_file = '../UIE/Options/UIE_MambaU.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

model_restoration = Restormer(**x['network_g'])

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()


factor = 8
# datasets = ['Rain100L', 'Rain100H', 'Test100', 'Test1200', 'Test2800']
datasets = ['UIEB']
for dataset in datasets:
    result_dir  = os.path.join(args.result_dir, dataset)
    os.makedirs(result_dir, exist_ok=True)

    inp_dir = os.path.join(args.input_dir, dataset, 'test', 'input')
    #print(inp_dir)
    files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.jpg')))
    with torch.no_grad():
        for file_ in tqdm(files):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            img = np.float32(utils.load_img(file_))/255.
            img = torch.from_numpy(img).permute(2,0,1)
            input_ = img.unsqueeze(0).cuda()

            # Padding in case images are not multiples of 8
            h,w = input_.shape[2], input_.shape[3]
            H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

            restored = model_restoration(input_)

            '''
            # Unpad images to original dimensions
            restored = restored[:,:,:h,:w]

            restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0]+'.png')), img_as_ubyte(restored))
            
            # 假设模型返回两个张量：restored1 和 restored2
            restored1, restored2 = restored

            # 处理第一个图像
            restored1_processed = restored1[:, :, :32, :32]  # 根据第一个图像的尺寸裁剪
            restored1_processed = torch.clamp(restored1_processed, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(
                0).numpy()
            if restored1_processed.shape[-1] > 3:
                restored1_processed = restored1_processed[:, :, :3]  # 取前3个通道
                print("截取前3个通道，新形状:", restored1_processed.shape)

            # 处理第二个图像
            #restored2_processed = restored2[:, :, :h, :w]  # 根据第二个图像的尺寸裁剪
            #restored2_processed = torch.clamp(restored2_processed, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(
                #0).numpy()

            # 分别保存两个图像
            targetdir = r'E:/xzx\SWGLM/UIE/results\latent'
            #base_name = os.path.splitext(os.path.split(file_)[-1])[0]
            utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0]+'.png')), img_as_ubyte(restored1_processed))
            #utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0]+'.png')), img_as_ubyte(restored2_processed))
            '''
            # 假设模型返回两个张量：restored1 和 restored2
            restored1, restored2 = restored

            # 处理第一个图像
            restored1_processed = restored1[:, :, :256, :256]  # 根据第一个图像的尺寸裁剪
            restored1_processed = torch.clamp(restored1_processed, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(
                0).numpy()
            if restored1_processed.shape[-1] > 3:
                restored1_processed = np.mean(restored1_processed, axis=2, keepdims=True)
                # restored1_processed = torch.mean(restored1_processed,dim=1,keepdim=True).cpu().detach()
                restored1_processed = restored1_processed[:, :, :1]  # 取前1个通道
                print("截取前3个通道，新形状:", restored1_processed.shape)

            # 处理第二个图像
            # restored2_processed = restored2[:, :, :h, :w]  # 根据第二个图像的尺寸裁剪
            # restored2_processed = torch.clamp(restored2_processed, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(
            # 0).numpy()

            # 分别保存两个图像
            targetdir = r'E:/xzx/SWGLM/UIE/results\latent'
            # base_name = os.path.splitext(os.path.split(file_)[-1])[0]
            utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0] + '.png')),
                           img_as_ubyte(restored1_processed))