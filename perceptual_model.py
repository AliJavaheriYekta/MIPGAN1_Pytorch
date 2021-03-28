import torch 
from torchvision import models 
import torch.nn as nn 


class VGG16_for_Perceptual(torch.nn.Module):
    def __init__(self,requires_grad=False,n_layers):
        super(VGG16_for_Perceptual,self).__init__()
        vgg_pretrained_features=models.vgg16(pretrained=True).features 

        self.slices = []
        for i in n_layers:
            slice = torch.nn.Sequential()
            for j in range(i):
                slice.add_module(str(j),vgg_pretrained_features[j])
            self.slices.append(slice)
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad=False
        
        self.results = []
    
    def forward(self,x):
        for slice in self.slices:
            x = slice(x)
            self.results.append(x)

        return self.results


