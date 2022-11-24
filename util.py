import os
import torch 
import torchvision
import torchvision.transforms as transforms
from args import parser 
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
import torch.nn as nn

def show_result(dataloader, img_list):
    batch = next(iter(dataloader))
    
    it = iter(dataloader)
    batch = it.next()
    
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Image")
    plt.imshow(np.transpose(torchvision.utils.make_grid(batch[0][:64], padding=5, normalize=True), (1,2,0))) # batch size 128
    
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    
    plt.savefig("./output/output")
 
def show_img(dataloader): 
    it = iter(dataloader)
    batch = it.next()
    
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Image")
    plt.imshow(np.transpose(torchvision.utils.make_grid(batch[0][:64], padding=2, normalize=True), (1,2,0))) # batch size 128
    plt.savefig("./output/input")

def save_checkpoint(epoch, model, optimizer, filename, lr):
    state ={
        'Epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr' : lr
    }
    
    torch.save(state, filename)

def get_dir_name():
    arg = parser.parse_args()
    
    split_symbol = '~' if os.name == 'nt' else ':'
    model_name_template = split_symbol.join(['S:{}_mini_batch', '{}_layer', '{}_id'])
    model_name = model_name_template.format(arg.mini_batch, arg.layer, arg.id)
    
    dir_name = os.path.join(model_name)
    
    return dir_name 

def get_data(mode):    
    arg = parser.parse_args()

    batch_size = arg.mini_batch 
    img_size = arg.img_size

    data_transform = transforms.Compose([transforms.Resize(size= img_size), # according to DCGAN tutorial in pytorh 
                                         transforms.CenterCrop(img_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])   # do not change sequence! 
    
    trainset = torchvision.datasets.ImageFolder(root='./data/celeba', 
                                            transform = data_transform,
                                            )

    dataloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle=True)   
        
    return dataloader

def get_file_name(PATH):
    arg = parser.parse_args()
    
    if(arg.mode == 'resume'):
        model_path = os.path.join('./save/', PATH)
        file_path_G = model_path + '/DCGAN_G.pth'
        file_path_D = model_path + '/DCGAN_D.pth'
        
        tb_pth_train = os.path.join('./logs/train/', PATH)
        tb_pth_valid = os.path.join('./logs/valid/', PATH)
        tb_pth_test = os.path.join('./logs/test/', PATH)
    
    else:
        model_path = os.path.join('./save/', PATH)
        
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        file_path_G = model_path + '/DCGAN_G.pth'
        file_path_D = model_path + '/DCGAN_D.pth'
        
        # train
        tb_pth_train = os.path.join('./logs/train/', PATH)
        if not os.path.isdir(tb_pth_train):
            os.makedirs(tb_pth_train)
        
        # valid 
        tb_pth_valid = os.path.join('./logs/valid/', PATH)
        if not os.path.isdir(tb_pth_valid):
            os.makedirs(tb_pth_valid)        
        
        # test 
        tb_pth_test = os.path.join('./logs/test/', PATH)
        if not os.path.isdir(tb_pth_test):
            os.makedirs(tb_pth_test)        
    
    return file_path_G, file_path_D, tb_pth_train, tb_pth_valid, tb_pth_test

# initialize the weight 
def weight_init(model): 
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02) # mean 0, std 0.02 
                                                      # no bias is set in ConvTranspose
    
    elif classname.find('BatchNorm')!=-1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)         # set bias 0 
        
if __name__=="__main__":
    dataload = get_data('train')
    show_img(dataload)