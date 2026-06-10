import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import numpy as np
from einops import rearrange
from mamba_ssm import Mamba
#from .VSS import VSSBlock


##########################################################################
def window_partition(x, window_size: int,h,w):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    pad_l = pad_t = 0
    pad_r = (window_size - w % window_size) % window_size
    pad_b = (window_size - h % window_size) % window_size
    x = F.pad(x, [ pad_l, pad_r, pad_t, pad_b])
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return windows

def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    pad_l = pad_t = 0
    pad_r = (window_size - W % window_size) % window_size
    pad_b = (window_size - H % window_size) % window_size
    H = H+pad_b
    W = W+pad_r
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, -1, H // window_size, W // window_size, window_size, window_size)
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, -1, H, W)
    windows = F.pad(x, [pad_l, -pad_r, pad_t, -pad_b])
    return windows
##########################################################################


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = BiasFree_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## FFN
class FeedForward(nn.Module):
    def __init__(self, dim, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*3)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.relu(x1) * x2
        x = self.project_out(x)
        return x


class SWPSA(nn.Module):
    def __init__(self,
                 dim=24,
                 #drop_path=0.,
                 #attn_drop_rate=0.,
                 #resolution=[64, 64],
                 #d_state=16,
                 #expand=2,
                 #d_conv=4,
                 #h=64,
                 #w=64,
                 window_size=32,
                 shift_size=3,
                 bias=False):
        super(SWPSA, self).__init__()
        #h = resolution[0]
        #w = resolution[1]
        self.window_size = window_size
        self.shift_size = shift_size

        self.qkv_conv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1,groups=dim * 3, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=3, padding=1,bias=bias)
        self.project_out1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1,bias=bias)

        self.qkv_conv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv1 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1,groups=dim * 3, bias=bias)
        self.mamba = Mamba(dim)
        #self.VSSBlock = VSSBlock(hidden_dim=dim, d_state=d_state, expand=expand, d_conv=d_conv,drop_path=drop_path, attn_drop_rate=attn_drop_rate)

    def window_partitions(self,x, window_size: int):
        """
        Args:
            x: (B, H, W, C)
            window_size (int): window size(M)

        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows

    def create_mask(self, x):

        n,c,H,W = x.shape
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        print(Hp)
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = self.window_partitions(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def create_mask1(self, x):

        n,c,H,W = x.shape
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = self.window_partitions(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        shortcut = x
        b, c, h, w = x.shape

        x = window_partition(x,self.window_size,h,w)
        #print('window1', x.shape)
        B, C, H, W = x.shape

        img = x.reshape(B, -1, C)
        #print(img.shape)
        out = self.mamba(img).reshape(B, C, H, W)
        #out = self.VSSBlock(x)
        #print('mamba1',out.shape)
        #out = rearrange(out, 'b c (h w) -> b c h w', h=int(self.window_size),
                        #w=int(self.window_size))
        #print("out",out.shape)
        out = self.project_out(out)
        #print('out',out.shape)
        out = window_reverse(out,self.window_size,h,w)
        #shift = out
        shift = torch.roll(out,shifts=(-self.shift_size,-self.shift_size),dims=(2,3))   #切换滚动窗口
        #shift_window = window_partition(shift,self.window_size,h,w)
        #print('shift_window',shift.shape)
        # qkv = self.qkv_dwconv1(self.qkv_conv1(shift_window))
        # q, k,v  = qkv.chunk(3, dim=1)
        # q = rearrange(q, 'b c h w -> b c (h w)')
        # k = rearrange(k, 'b c h w -> b c (h w)')
        # v = rearrange(v, 'b c h w -> b c (h w)')
        #
        # q = torch.nn.functional.normalize(q, dim=-1)
        # k = torch.nn.functional.normalize(k, dim=-1)
        #
        # attn = (q.transpose(-2,-1) @ k)/self.window_size
        out2 = shift.reshape(B, -1, C)
        #print('out2',out2.shape)
        out2 = self.mamba(out2).reshape(B, C, H, W)
        #print('out2',out2.shape)
        #out2 = self.VSSBlock(shift_window)
        #print('out2',out2.shape)
        # mask = self.create_mask(shortcut).narrow(2,0,1)
        # print('mask',mask.shape)
        # #print(attn.shape,mask.unsqueeze(0).shape)
        # attn = attn.view(b,-1,self.window_size,self.window_size)# + mask.unsqueeze(0)
        # print(attn.shape)
        # attn = attn.view(-1,self.window_size*self.window_size,self.window_size*self.window_size)
        # attn = attn.softmax(dim=-1)
        #
        # out = (v @ attn)
        #
        # out = rearrange(out, 'b c (h w) -> b c h w', h=int(self.window_size),
        #                 w=int(self.window_size))
        #print(out2.shape, "out2")
        out2 = self.project_out1(out2)
        #print(out2.shape,"out2")
        out2 = window_reverse(out2,self.window_size,h,w)
        out2 = torch.roll(out2,shifts=(self.shift_size,self.shift_size),dims=(2,3))

        return out2

class WindowmambaBlock(nn.Module):
    def __init__(self, dim=24,
                 #drop_path=0.,
                 #attn_drop_rate=0.,
                 #resolution=[64, 64],
                 #d_state=16,
                 #expand=2,
                 #d_conv=4,
                 h=64,
                 w=64,
                 window_size=32,
                 shift_size=3,
                 bias=False):
        super(WindowmambaBlock, self).__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = SWPSA(dim, window_size, shift_size, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim,bias)

    def forward(self, x):

        y = self.attn(self.norm1(x))
        diffY = x.size()[2] - y.size()[2]
        diffX = x.size()[3] - y.size()[3]

        y = F.pad(y, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = x+y
        x = x + self.ffn(self.norm2(x))

        return x



if __name__ == '__main__':
    data = torch.randn([2, 24, 256, 256]).cuda()
    model = WindowmambaBlock().cuda()
    print("model:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    out = model(data)
    print('out', out.shape)