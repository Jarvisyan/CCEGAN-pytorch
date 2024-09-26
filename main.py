# pip install munkres
# pip install lightly
import os
import torch
from utils import setup_seed, init_weights
from networks import Backbone, extended_SimSiam
from networks import Generator, Discriminator, Finetune_net
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, ConcatDataset
from training_funcs import train_GAN, train_simsiam
from filtering_funcs import firt_filter, second_filter



def train_mnist(p):
    setup_seed(42)
    M, class_num = 5, 10
    z_dim, num_epochs, batch_size = 62, 12, 70
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    out_dim = class_num
    num_ftrs, proj_hidden_dim, pred_hidden_dim = 256 * 3 * 3, 128, 128
   

    cluster_size = int(200 / M / class_num) 
    iterations = 1000
    
    cluster_size_chosen = int(cluster_size * iterations * p * 0.125) #500 * 5
    print(f'M = {M}, p = {p} | cluster_size = {cluster_size}, cluster_size_chosen = {cluster_size_chosen}\n')
    
    model_dir = f'ckpt/M{M}/p{p}'
    os.makedirs(model_dir, exist_ok=True)
    
    backbone = Backbone()
    my_SimSiam = extended_SimSiam(backbone, num_ftrs, proj_hidden_dim, pred_hidden_dim, out_dim)
    my_SimSiam.to(device)
    Nets = locals()  
    for m in range(M):
        Nets[f'G_{m}'] = Generator(class_num, z_dim)
        Nets[f'D_{m}'] = Discriminator(class_num)
        Nets[f'E_{m}'] = Finetune_net(backbone, class_num)

        Nets[f'G_{m}'].apply(init_weights)
        Nets[f'D_{m}'].apply(init_weights)
        
        Nets[f'G_{m}'].to(device) 
        Nets[f'D_{m}'].to(device) 
        Nets[f'E_{m}'].to(device)
    
    train_dataset = MNIST(root = './data', train = True, download = True, transform = transforms.ToTensor())
    test_dataset = MNIST(root = './data', train = False, download = True, transform = transforms.ToTensor())
    combined_dataset = ConcatDataset([train_dataset, test_dataset])
    dataloader = DataLoader(combined_dataset, batch_size, shuffle = True, num_workers = 2)     

    first_epoch = True 
    for epoch in range(num_epochs):
        decay = 0.98 ** epoch
        #M-step: train GANs
        for m in range(M):
            Nets[f'G_{m}'].train()
            Nets[f'D_{m}'].train()
            Nets[f'E_{m}'].train()

            Nets[f'optimizer_G_{m}'] = torch.optim.Adam(Nets[f'G_{m}'].parameters(), lr = decay * 3 * 0.0002, betas = (0.5, 0.999))
            Nets[f'optimizer_D_{m}'] = torch.optim.Adam(Nets[f'D_{m}'].parameters(), lr = decay * 0.0002, betas = (0.5, 0.999))
            train_GAN(Nets[f'D_{m}'], Nets[f'G_{m}'], Nets[f'E_{m}'], Nets[f'optimizer_D_{m}'], Nets[f'optimizer_G_{m}'], 
                dataloader, class_num, z_dim, first_epoch, m, epoch, device)
                
        # Ensemble and filtering step
        ## train my_SimSiam
        assert backbone.conv[0].weight[0].requires_grad == True #check whether free the backbone
        optimizer_my_SimSiam = torch.optim.Adam(my_SimSiam.parameters(), lr = decay * 3 * 0.0002, betas = (0.5, 0.99)) #M = 1 or M = 5
        train_simsiam(Nets, my_SimSiam, optimizer_my_SimSiam, dataloader, cluster_size, class_num, iterations, z_dim, M, epoch, device)

        ##first filtering & train Es
        for param in backbone.parameters(): #freeze the backbone
            param.requires_grad = False
        assert Nets[f'E_{m}'].features.conv[0].weight[0].requires_grad == False  #check whether freeze the backbone        
        fake_samples_filtered_all, fake_samples_num_ls = firt_filter(my_SimSiam, Nets, decay, dataloader, class_num, cluster_size_chosen, M, epoch, device)
        ## fake_samples_filtered_all: filtered images
        
        ##second filtering & E-step: retrain Es
        second_filter(Nets, decay, dataloader, class_num, fake_samples_filtered_all, fake_samples_num_ls, first_epoch, M, epoch, device)
        
        for param in backbone.parameters(): #free the backbone
            param.requires_grad = True

        first_epoch = False
    
        # save model
        print('saved model...')
        torch.save(my_SimSiam.state_dict(), f'ckpt/M{M}/p{p}/model_{epoch}.pth')



if __name__ == "__main__":

    train_mnist(p = 5)