#DCGAN
import torch.nn as nn
import torch.nn.functional as F
import torch

class Mlps(nn.Module):
    
    def __init__(self, inc, outc=128):
        super(Mlps, self).__init__()
        
        self.mlp = nn.Sequential()
        self.mlp.add_module("mlp", nn.Conv2d(inc, outc, 1))
        self.mlp.add_module('bn', nn.BatchNorm2d(outc))
        self.mlp.add_module('Relu', nn.ReLU(inplace=True))
    
    def forward(self, x):
        x = self.mlp(x)
        return x
    
class Acn(nn.Module):
    
    def __init__(self, inc, eps=1e-3):
        super(Acn, self).__init__()
        
        self.eps = eps
        self.att = nn.Conv2d(inc, 1, 1)
        
    def forward(self, x):
        b, _, n, _ = x.shape
        a = F.softmax(self.att(x), dim=2)
        mean = (x * a).sum(dim=2, keepdim=True)
        out = x - mean
        std = torch.sqrt(
                (a*out**2).sum(dim=2, keepdim=True) + self.eps
            )
        out = out/std
        out = x
        return out
    

class Generator(nn.Module):
    """
    Implementation of DCGAN generator is learnt from https://github.com/eriklindernoren/PyTorch-GAN.git
    """
    def __init__(self,conf_data):
        super(Generator, self).__init__()

        self.init_size = conf_data['generator']['input_shape'] // 4
        self.l1 = nn.Sequential(nn.Linear(conf_data['generator']['latent_dim'], 128*self.init_size**2))
        self.channels = conf_data['generator']['channels']

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            Acn(128),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            Acn(64),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True) ,
            
            # nn.Upsample(scale_factor=2),
            # nn.Conv2d(64, 32, 3, stride=1, padding=1),
            # Acn(32),
            # nn.BatchNorm2d(32, 0.8),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, self.channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
