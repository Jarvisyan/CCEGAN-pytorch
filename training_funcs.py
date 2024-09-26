import torch
from torch import nn
from tqdm import tqdm
from utils import gen_cond_label
from torch.nn import functional as F
from utils import eval_by_E

def train_GAN(D, G, E, optimizer_D, optimizer_G, dataloader, class_num, z_dim, first_epoch, m, epoch, device):
    for epoch_GAN in range(5):
        pbar = tqdm(dataloader)
        loss_D_ema = None
        loss_G_ema = None
        for X, _ in pbar:
            #train D first
            optimizer_D.zero_grad()
            X = X.to(device)
            X_size = X.shape[0]
            #real_images_loss 
            D_real = D(X)
            if first_epoch:
                Y_real = 1/class_num * torch.ones(X_size, class_num, device = device)
            else:
                Y_real = nn.Softmax(dim = 1)(E(X)).detach()
            Y_real = torch.cat([Y_real, 2 * torch.ones(X_size, 1, device = device)], 1)
            ones = torch.ones_like(Y_real)
            ones[:, -1] = 0
            D_real_loss = torch.nn.BCEWithLogitsLoss(weight = Y_real)(D_real, ones)
      
            #fake_images_loss
            G_input, _ = gen_cond_label(X_size, class_num, z_dim)
            G_input = G_input.to(device)
            X_fake = G(G_input).detach() 
            D_fake = D(X_fake)
            Y_fake = torch.zeros_like(Y_real)
            Y_fake[:, -1] = 1
            D_fake_loss = torch.nn.BCEWithLogitsLoss()(D_fake, Y_fake)

            #total loss
            D_loss = D_real_loss + D_fake_loss
            D_loss.backward()
            optimizer_D.step()
            
      
            #train G
            #G.train()
            #D.eval()
            optimizer_G.zero_grad()
            G_input, cond_label = gen_cond_label(X_size, class_num, z_dim)
            G_input = G_input.to(device)
            X_fake = G(G_input)
            D_fake = D(X_fake)
            Y_fake = torch.cat([cond_label, torch.zeros(X_size, 1)], 1)
            Y_fake = Y_fake.to(device)

            G_loss = torch.nn.BCEWithLogitsLoss()(D_fake, Y_fake)
            G_loss.backward()
            optimizer_G.step()
            
            if loss_D_ema is None:
                loss_D_ema = D_loss.item()
                loss_G_ema = G_loss.item()
            else:
                loss_D_ema = 0.95 * loss_D_ema + 0.05 * D_loss.item()            
                loss_G_ema = 0.95 * loss_G_ema + 0.05 * G_loss.item()
            pbar.set_description(f"Epoch_{epoch} | GAN_{m}_{epoch_GAN}: loss_D = {loss_D_ema:.4f} | loss_G = {loss_G_ema:.4f}")


def train_simsiam(Nets, my_SimSiam, optimizer_my_SimSiam, dataloader, cluster_size, class_num, iterations, z_dim, M, epoch, device):

    fake_num = cluster_size * class_num
    for m in range(M):
        Nets[f'fake_samples_{m}'] = torch.zeros(0, device = device)
        Nets[f'fake_labels_{m}'] = torch.zeros(0, device = device)


    loss_sim_ema = None
    pbar = tqdm(range(iterations))
    for i in pbar:
        #generate
        X_to_fuse = torch.zeros(0, device = device)   ##batch_size = 200
        for m in range(M):
            G_input, cond_label = gen_cond_label(fake_num, class_num, z_dim)
            pseudo_label = torch.argmax(cond_label, 1)
            G_input, pseudo_label = G_input.to(device), pseudo_label.to(device)
            X_fake = Nets[f'G_{m}'](G_input).detach()
          
            #save
            X_to_fuse = torch.cat([X_to_fuse, X_fake], 0)
            Nets[f'fake_samples_{m}'] = torch.cat([Nets[f'fake_samples_{m}'], X_fake], 0)
            Nets[f'fake_labels_{m}'] = torch.cat([Nets[f'fake_labels_{m}'], pseudo_label], 0)

        #contrast learning
        my_SimSiam.train()
        optimizer_my_SimSiam.zero_grad()
        z, p, H1, H2 = my_SimSiam(X_to_fuse)
        my_SimSiam_loss = 0
        for m in range(M):
            for i in range(class_num):
                idx1, idx2 = cluster_size * i + fake_num * m, cluster_size * (i + 1) + fake_num * m
                z_i, p_i = z[idx1 : idx2], p[idx1 : idx2]
                z_i_norm, p_i_norm = F.normalize(z_i), F.normalize(p_i)
                my_SimSiam_loss -= torch.mm(z_i_norm, p_i_norm.T).sum() / (cluster_size ** 2)
        total_loss = my_SimSiam_loss + 5 * M * (H1 - H2)
        total_loss.backward()
        optimizer_my_SimSiam.step()
        
        if loss_sim_ema is None:
            loss_sim_ema = total_loss.item()
        else:
            loss_sim_ema = 0.95 * loss_sim_ema + 0.05 * total_loss.item()
        pbar.set_description(f"Epoch_{epoch} | iter_{i}: loss_E = {loss_sim_ema:.4f}, H1 = {H1:.4f}, H2 = {H2:.4f}")        


    ##evaluate by SIMSIAM
    SIMSIAM_flag = True
    eval_by_E(my_SimSiam, 'my_SimSiam', epoch, dataloader, SIMSIAM_flag, device)




def train_E(G, E, optimizer_E, class_num, z_dim, epoch, device):
    loss_E_ema = None
    pbar = tqdm(range(1000))
    for i in pbar:
        ### generate fake samples & pseudo-labels
        G_input, cond_label = gen_cond_label(200, class_num, z_dim)
        pseudo_label = torch.argmax(cond_label, 1)
        G_input, pseudo_label = G_input.to(device), pseudo_label.to(device)
        X_fake = G(G_input).detach()

        optimizer_E.zero_grad()
        pred = E(X_fake)
        E_loss = torch.nn.CrossEntropyLoss()(pred, pseudo_label)
        E_loss.backward()
        optimizer_E.step()
        
        if loss_E_ema is None:
            loss_E_ema = E_loss.item()
        else:
            loss_E_ema = 0.95 * loss_E_ema + 0.05 * E_loss.item()
        pbar.set_description(f"Epoch_{epoch}, iter_{i}: loss_E = {loss_E_ema:.4f}")





