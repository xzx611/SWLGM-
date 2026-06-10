import torch
import torch.nn as nn
from mamba_ssm import Mamba

import time

class convmamba(nn.Module):
    def __init__(self,stride=1,c=3):
        super().__init__()
        #self.mamba = VSSBlock(c)
        self.mamba = Mamba(c)

    def forward(self, x):
        B, C, H, W = x.shape
        img_z = torch.zeros([B, C, H, W]).cuda()
        for i in range(0,H,32):
            for j in range(0,W,32):
                img_mask = torch.zeros([B, C, H, W]).cuda()
                img_mask[:, :, i:i+32, j:j+32] = 1
                img = torch.mul(x, img_mask)
                '''
                img = img.permute(0, 2, 3, 1)
                #print(img.shape) #[1, 3, 256, 256]
                img_m = self.mamba(img).permute(0, 3, 1, 2)
                #print(img_m.shape)
                '''
                img = img.reshape(B, -1, C)
                img_m = self.mamba(img).reshape(B, C, H, W)
                img_z = img_z + img_m
        return img_z





if __name__ == '__main__':
    start_time = time.time()
    data = torch.randn([1, 3, 256, 256]).cuda()
    #model = Gmamba(nodes_num=1024).cuda()
    #model = Gmambablock(nodes_num=1024,os = (256, 256),ks = (8, 8),st = (8, 8),).cuda()
    #model = convmamba(c=32).cuda()
    model = convmamba().cuda()
    #model = mambaBlock(inputs_dim=1024).cuda()
    out = model(data)
    print('out=', out.shape)
    print('time', time.time() - start_time)