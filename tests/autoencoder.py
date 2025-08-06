import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

device = torch.device('cuda')
#device = torch.device('cpu')

#DHT transform
def DiscreteHartleyTransform(input):
    fft = torch.fft.fft2(input, norm="forward")
    fft = torch.fft.fftshift(fft)
    return fft.real - fft.imag
#Inverse DHT transform
def InverseDiscreteHartleyTransform(input):
    dht = torch.fft.ifftshift(input)
    fft = torch.fft.fft2(dht, norm="backward")
    return fft.real - fft.imag


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.act_fn = nn.LeakyReLU(0.01)
        self.act_fn_out = nn.Softmax(dim=1)
        self.flatten = nn.Flatten()
        self.unflatten = nn.Unflatten(1, (25, 2, 2))
        self.pool = nn.MaxPool2d(2)
        self.unpool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(1,100,3,2)
        self.conv2 = nn.Conv2d(100,50,3,2)
        self.conv3 = nn.Conv2d(50,25,4,2)
        self.conv4 = nn.Conv2d(25,10,2,1)
        self.conv4t = nn.ConvTranspose2d(10,25,2,1)
        self.conv3t = nn.ConvTranspose2d(25,50,4,2)
        self.conv2t = nn.ConvTranspose2d(50,100,4,2)
        self.conv1t = nn.ConvTranspose2d(100,1,4,2)
        
    def encode(self, x):
        #x= DiscreteHartleyTransform(x)
        x = self.conv1(x)
        x = self.act_fn(x)
        #x = self.pool(x)
        x = self.conv2(x)
        x = self.act_fn(x)
        #x = self.pool(x)
        x = self.conv3(x)
        x = self.act_fn(x)
        #x = self.conv4(x)
        #x = self.act_fn(x)
        #x = self.pool(x)
        x = self.flatten(x)
        return x
        
    def decode(self, x):
        x = self.unflatten(x)
        #x = self.conv4t(x)
        #x = self.act_fn(x)
        #x = self.unpool(x)
        x = self.conv3t(x)
        x = self.act_fn(x)
        #x = self.unpool(x)
        x = self.conv2t(x)
        x = self.act_fn(x)
        #x = self.unpool(x)
        x = self.conv1t(x)
        x = F.pad(x, (1,1,1,1), value=0)
        #x = InverseDiscreteHartleyTransform(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

