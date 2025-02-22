import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class ConvBlock(nn.Module):
    
    def __init__(self, channels_in, channels_out, kernel_size, **kwargs):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(channels_out)
        
        self.relu = nn.LeakyReLU(0.02)
        
    def forward(self, input):
        
        input = self.conv(input)
        input = self.bn(input) 
        
        return self.relu(input)
    
    def weight_init(self):
        
        nn.init.xavier_uniform_(self.conv.weight.data)
        self.bn.weight.data.normal_(1.0, 0.02)
        self.bn.bias.data.fill_(0)
    
class DeconvBlock(nn.Module):
    
    def __init__(self, channels_in, channels_out, kernel_size, batch_norm=True, **kwargs):
        super(DeconvBlock, self).__init__()
        
        self.batch_norm = batch_norm
        
        self.deconv = nn.ConvTranspose2d(channels_in, channels_out, kernel_size, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(channels_out)
        
    def forward(self, input):
        
        input = F.leaky_relu(input)
        input = self.deconv(input)
        if self.batch_norm:
            input = self.bn(input)
            
        return input
    
    def weight_init(self):
        
        nn.init.xavier_uniform_(self.deconv.weight.data)
        
        self.bn.weight.data.normal_(1.0, 0.02)
        self.bn.bias.data.fill_(0)