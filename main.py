import torch 
import torchvision
import torch.optim as optim
import torch.nn as nn  
from model import Generator, Discriminator
import numpy as np
from torchvision.transforms.functional import resize 

from args import parser 
from util  import save_checkpoint, get_dir_name, get_data, get_file_name, show_img, weight_init, show_result

from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
    
def train(train):    
    arg = parser.parse_args()
    epoch_num = arg.epoch
    lr = arg.lr
    batch_num = arg.mini_batch
    
    path = get_dir_name()
    file_path_G, file_path_D, tb_pth_train, tb_pth_valid, tb_pth_test  = get_file_name(path)
 
    # model declaration 
    modelG = Generator().to(device)
    weight_init(modelG)
    
    modelD = Discriminator().to(device)
    weight_init(modelD)

    if(arg.mode == 'resume'):
        checkpoint_G = torch.load(file_path_G)
        checkpoint_D = torch.load(file_path_D)
        
        lr = checkpoint_G['lr']
        
        modelG.load_state_dict(checkpoint_G['state_dict'])
        modelD.load_state_dict(checkpoint_D['state_dict'])
        
        epoch_loadG = checkpoint_G['Epoch']
        
        optimizerD = optim.Adam(modelD.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizerG = optim.Adam(modelG.parameters(), lr=lr, betas=(0.5, 0.999))
        
        optimizerD.load_state_dict(checkpoint_D['optimizer'])  
        optimizerG.load_state_dict(checkpoint_G['optimizer'])  

    else: 
        optimizerD = optim.Adam(modelD.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizerG = optim.Adam(modelG.parameters(), lr=lr, betas=(0.5, 0.999))
    
    criterion = nn.BCELoss().to(device) # discriminate 1 or 0 
    
    true_label = 1
    false_label = 0 
               
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    writer = SummaryWriter(tb_pth_train)     
    for epoch in range(epoch_num):
        if(arg.mode=='resume' and epoch_num==0):
            epoch = epoch_loadG
 
        for i, data in enumerate(train, 0):
            modelD.zero_grad() 
            
            # Update Discriminator 
            
            # train discriminator real data
            real_data = data[0].to(device)
            b_size = real_data.size(0)
            label = torch.full((b_size,), true_label, dtype=torch.float, device=device) # create tensor filled true_label
            
            output = modelD(real_data).view(-1)
            lossD_real = criterion(output, label)
            lossD_real.backward()                        # calculate gradient 
            predD_real = output.mean().item()
            
            # train generator fake data 
            noise = torch.randn(b_size, 100, 1, 1, device=device)
            fake = modelG(noise)
            label.fill_(false_label)
            
            # disciminate whether fake or not
            output= modelD(fake.detach()).view(-1) # detach from device?
            
            lossD_fake= criterion(output, label)
            lossD_fake.backward()                  # calculate gradient 
            predD_fake = output.mean().item()
            
            # total loss for real and fake
            lossD= lossD_real + lossD_fake
            
            optimizerD.step()
 
            # Update Generator 
            modelG.zero_grad()
            label.fill_(true_label) # for generator, fake images are real
            
            output= modelD(fake).view(-1)
            lossG = criterion(output, label)
            lossG.backward()
            
            predG = output.mean().item()  # check
            optimizerG.step()
            
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (epoch, epoch_num, i, len(train), lossD.item(), lossG.item(), predD_real, predD_fake, predG))
   
            writer.add_scalar("Loss_D", lossD.item() , i )
            writer.add_scalar("Loss_G", lossG.item() , i )
            writer.add_scalar("D(x)", predD_real, i )
            writer.add_scalar("D(G(z))", predD_fake, i )
            writer.add_scalar("G(z)",predG, i )
            
            G_losses.append(lossG.item())
            D_losses.append(lossD.item())
            
            if (iters % 500 == 0) or ((epoch == epoch_num-1) and (i == len(train)-1)):   
                with torch.no_grad():
                    fake = modelG(noise).detach().cpu() 
                img_list.append(torchvision.utils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    show_result(train, img_list)

    writer.close()

    save_checkpoint(epoch, modelG, optimizerG, file_path_G, lr) 
    save_checkpoint(epoch, modelD, optimizerD, file_path_D, lr) 
    
def main():
    print("Start")
    torch.cuda.empty_cache()
    
    global device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainload = get_data('train')
    train(trainload)
    
if __name__=="__main__":
    main()