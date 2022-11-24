import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import pytorch_model_summary
from util import weight_init 
 
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__() 

        self.project = nn.Sequential(
                    nn.ConvTranspose2d(in_channels= 100, out_channels= 1024, kernel_size= 4, stride= 2, padding=0, bias=False), 
                    nn.BatchNorm2d(1024),
                    nn.ReLU()
        )
                    
        self.conv1 = nn.Sequential(
                    nn.ConvTranspose2d(in_channels= 1024, out_channels= 512, kernel_size= 4, stride= 2, padding=1, bias=False), 
                    nn.BatchNorm2d(512),
                    nn.ReLU()
                    )
        
        self.conv2 = nn.Sequential(
                    nn.ConvTranspose2d(in_channels= 512, out_channels= 256, kernel_size= 4, stride= 2, padding=1, bias=False), 
                    nn.BatchNorm2d(256),
                    nn.ReLU()
                    )
   
        self.conv3 = nn.Sequential(
                    nn.ConvTranspose2d(in_channels= 256, out_channels= 128, kernel_size= 4, stride= 2, padding=1, bias=False), 
                    nn.BatchNorm2d(128),
                    nn.ReLU()
                    )        

        self.conv4 =nn.Sequential(
                    nn.ConvTranspose2d(in_channels= 128, out_channels= 3, kernel_size= 4, stride= 2, padding=1, bias=False), 
                    # nn.BatchNorm2d(3),
                    nn.Tanh()
                    )

    def forward(self, x): # input noise z
        
        x = self.project(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        return x

class Discriminator(nn.Module): # check ^2 channels --> paper 
    def __init__(self):
        super(Discriminator, self).__init__() 
                    
        self.conv1 = nn.Sequential(
                    nn.Conv2d(in_channels= 3, out_channels= 64, kernel_size= 4, stride= 2, padding=1, bias=False), 
                    nn.LeakyReLU(0.2, inplace=True)     # from paper
                    )
        
        self.conv2 = nn.Sequential(
                    nn.Conv2d(in_channels= 64, out_channels= 64 * 2, kernel_size= 4, stride= 2, padding=1, bias=False), 
                    nn.BatchNorm2d(64 * 2),
                    nn.LeakyReLU(0.2, inplace=True) 
                    )
   
        self.conv3 = nn.Sequential(
                    nn.Conv2d(in_channels= 64 * 2, out_channels= 64 * 4, kernel_size= 4, stride= 2, padding=1, bias=False), 
                    nn.BatchNorm2d(64 * 4),
                    nn.LeakyReLU(0.2, inplace=True) 
                    )        

        self.conv4 =nn.Sequential(
                    nn.Conv2d(in_channels= 64 * 4, out_channels= 64 * 8, kernel_size= 4, stride= 2, padding=1, bias=False), 
                    nn.BatchNorm2d(64 * 8),
                    nn.LeakyReLU(0.2, inplace=True) 
                    )

        self.conv5 = nn.Sequential(
                    nn.Conv2d(in_channels= 64 * 8, out_channels= 1, kernel_size= 4, stride= 2, padding=0, bias=False), 
                    nn.Sigmoid()
                    )

    def forward(self, x): 

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x) 
        
        return x                # true or false 
       
if __name__=="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"    
    print(device)
    
    model = Generator().to(device)
    weight_init(model)

    # print(model) 
    
    # radom normalization 
    # input =  torch.randn(1, 100, 1, 1, device= device) # noise z 1 x 100 x 1 x 1
    # print(input)
    
    summary(model, (100, 1, 1), device =device)  
    
    
    discrimin = Discriminator().to(device)
    weight_init(discrimin)
    
    summary(discrimin, (3, 64, 64), device =device)  