import torch
import torch.nn as nn
import torch.nn.functional as F

from network.network_block import *

class Encoder(nn.Module):
    
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.conv1 = ConvBlock(1,   32,  7, stride=2, padding=3)
        self.conv2 = ConvBlock(32,  64,  5, stride=2, padding=2)
        self.conv3 = ConvBlock(64,  128, 5, stride=2, padding=2)
        self.conv4 = ConvBlock(128, 256, 3, stride=2, padding=1)
        
        self.linear = nn.Linear(256*4*4, 512)
        
    def forward(self, input):
        
        output = input.view(-1, 1,  64, 64)
        
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        
        output = output.view(-1, 5, 256*4*4)
        output = torch.mean(output, 1)
        output = self.linear(output.view(-1, 256*4*4))

        return output
    
    def weight_init(self):
        
        self.conv1.weight_init()
        self.conv2.weight_init()
        self.conv3.weight_init()
        self.conv4.weight_init()        
        
        nn.init.xavier_uniform_(self.linear.weight.data)
    
    
class ViewTransformLayer(nn.Module):
    
    def __init__(self):
        super(ViewTransformLayer, self).__init__()
        
        self.view_trans = nn.Linear(14, 512, bias=False)
        
    def forward(self, input, view_encode):
        
        latent_z = input + self.view_trans(view_encode)
        
        return latent_z
    
    def weight_init(self):
        
        nn.init.xavier_uniform_(self.view_trans.weight.data)            
    
        
class Generator(nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
        
        self.deconv1 = DeconvBlock(512+5, 256, 4, stride=2)
        self.deconv2 = DeconvBlock(256  , 128, 4, stride=2, padding=1)
        self.deconv3 = DeconvBlock(128  ,  64, 4, stride=2, padding=1)
        self.deconv4 = DeconvBlock(64   ,  32, 4, stride=2, padding=1)
        self.deconv5 = DeconvBlock(32   ,   1, 4, stride=2, padding=1, batch_norm = False)
        
        self.tanh = nn.Tanh()
        
    def forward(self, input):
        
        input = input.view(-1, 512+5, 1, 1)
        
        input = self.deconv1(input)
        input = self.deconv2(input)
        input = self.deconv3(input)
        input = self.deconv4(input)
        input = self.deconv5(input)
        
        input = self.tanh(input)
        input = input.view(-1, 64, 64)
        
        return input
    
    def weight_init(self):
        
        self.deconv1.weight_init()
        self.deconv2.weight_init()
        self.deconv3.weight_init()
        self.deconv4.weight_init()    
        self.deconv5.weight_init()         
    
class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv1 = ConvBlock(1,    32, 7, stride=2, padding=3)
        self.conv2 = ConvBlock(32,   64, 5, stride=2, padding=2)
        self.conv3 = ConvBlock(64,  128, 5, stride=2, padding=2)
        self.conv4 = ConvBlock(128, 256, 3, stride=2, padding=1)
        
        self.linear = nn.Linear(256*4*4, 5 + 14 + 1000)
        
    def forward(self, input):
        
        input = input.view(-1, 1, 64, 64)
        
        input = self.conv1(input)
        input = self.conv2(input)
        input = self.conv3(input)
        input = self.conv4(input)
        input = input.view(-1, 256*4*4)
        
        output = self.linear(input)
        
        return output
    
    def weight_init(self):
        
        self.conv1.weight_init()
        self.conv2.weight_init()
        self.conv3.weight_init()
        self.conv4.weight_init()        
        
        nn.init.xavier_uniform_(self.linear.weight.data)