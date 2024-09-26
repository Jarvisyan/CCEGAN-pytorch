import torch
import numpy as np
from tqdm import tqdm
from scipy import stats
from utils import eval_by_E, label_assignment_by_base_E, label_map

def filter_images(my_SimSiam, fake_samples, fake_labels, class_num, cluster_size_chosen, device):
    my_SimSiam.eval()
    fake_samples_filtered = []
    cluster_size = []
    with torch.no_grad():
        for i in range(class_num):  
            cluster_idx = fake_labels == i
            images_cluster_i = fake_samples[cluster_idx]
            loader_i = torch.utils.data.DataLoader(images_cluster_i, batch_size = 256, shuffle = False, num_workers = 0)
            
            pred_conf = []
            pred_label = []
            for x in loader_i:
                pred = my_SimSiam(x)[0]
                label = torch.argmax(pred, 1)
                pred_conf.append(pred)
                pred_label.append(label)
            pred_conf = torch.cat(pred_conf, 0) 
            pred_label = torch.cat(pred_label, 0)
            
            #pred_conf = pred_conf.cpu()
            label_mode, _ = pred_label.mode()
            
            idx_chosen = pred_label == label_mode
            pred_conf = pred_conf[idx_chosen]
            images_cluster_i = images_cluster_i[idx_chosen]
            mode_num = int(sum(idx_chosen))
            if mode_num > cluster_size_chosen: 
                v, _ = pred_conf.max(dim = 1)
                _, idx_high = v.sort(descending = True)
                idx_high_chosen = idx_high[ : cluster_size_chosen]
                images_cluster_i_chosen = images_cluster_i[idx_high_chosen]
                fake_samples_filtered.append(images_cluster_i_chosen)
                cluster_size.append(cluster_size_chosen)
            else:
                fake_samples_filtered.append(images_cluster_i)
                cluster_size.append(mode_num)
    
    return torch.cat(fake_samples_filtered, 0), torch.tensor(np.repeat(range(10), cluster_size), device = device).long()

def firt_filter(my_SimSiam, Nets, decay, dataloader, class_num, cluster_size_chosen, M, epoch, device):
    fake_samples_filtered_all = []
    fake_samples_num_ls = [] 
  
    for m in range(M):
        Nets[f'optimizer_E_{m}'] = torch.optim.Adam((param for param in Nets[f'E_{m}'].parameters()
                              if param.requires_grad), lr = decay * 0.3 * 0.0002, betas = (0.5, 0.999))
        fake_samples_filtered, fake_labels_filtered = filter_images(my_SimSiam, Nets[f'fake_samples_{m}'], Nets[f'fake_labels_{m}'],
                                      class_num, cluster_size_chosen, device)
        fake_samples_filtered_all.append(fake_samples_filtered)
        fake_samples_num_ls.append(fake_samples_filtered.shape[0])
        torch_fake = torch.utils.data.TensorDataset(fake_samples_filtered, fake_labels_filtered)
        train_fake = torch.utils.data.DataLoader(dataset = torch_fake, batch_size = 200, shuffle = True, drop_last = True, num_workers = 0)
        
        for epoch_E in range(5):
            loss_E_ema = None
            pbar = tqdm(train_fake)
            for X_fake, y_fake in pbar:          
                Nets[f'optimizer_E_{m}'].zero_grad()
                pred = Nets[f'E_{m}'](X_fake)
                E_loss = torch.nn.CrossEntropyLoss()(pred, y_fake)
                E_loss.backward()
                Nets[f'optimizer_E_{m}'].step()

                if loss_E_ema is None:
                    loss_E_ema = E_loss.item()
                else:
                    loss_E_ema = 0.95 * loss_E_ema + 0.05 * E_loss.item()            
                pbar.set_description(f"First filtering, E_{m} | Epoch_{epoch}_{epoch_E}: loss_E = {loss_E_ema:.4f}")            
        ##evaluate by Es
        SIMSIAM_flag = False
        eval_by_E(Nets[f'E_{m}'], m, epoch, dataloader, SIMSIAM_flag, device)
    return torch.cat(fake_samples_filtered_all, 0), fake_samples_num_ls
    
def second_filter(Nets, decay, dataloader, class_num, fake_samples_filtered_all, fake_samples_num_ls, first_epoch, M, epoch, device):
    base_partitions = torch.zeros(0)
    #generate pseudo_labels
    fake_iter = torch.utils.data.DataLoader(dataset = fake_samples_filtered_all, batch_size = 200,
                        shuffle = False, num_workers = 0)
    for m in range(M):
        #label assignment by base E
        pred = label_assignment_by_base_E(Nets[f'E_{m}'], fake_iter, device)
        base_partitions = torch.cat([base_partitions, pred.reshape(-1, 1)], 1).long()

    #label alignment
    base_fake_aligned = torch.zeros_like(base_partitions) ## N*M
    anchor = base_partitions[:, 0]
    base_fake_aligned[:, 0] = anchor
    if first_epoch: 
        for m in range(1, M):
            base = base_partitions[:, m]
            base_aligned, base_to_anchor, anchor_to_base = label_map(anchor, base)
            base_fake_aligned[:, m] = torch.tensor(base_aligned)
            Nets[f'Base_to_anchor_{m}'] = base_to_anchor #save map, base_label : anchor_label
            Nets[f'Anchor_to_base_{m}'] = anchor_to_base #save map, anchor_label : base_label
    else:
        for m in range(1, M):
            base = base_partitions[:, m]
            for i in range(class_num):
                idx_base = base == i
                base_fake_aligned[:, m][idx_base] = Nets[f'Base_to_anchor_{m}'][i]
    label_fused, count_vote = stats.mode(base_fake_aligned, axis = 1) #voted pred_label
    idx_high_confidence = (count_vote > M/2).squeeze()

    SIMSIAM_flag = False
    assert Nets[f'E_{m}'].features.conv[0].weight[0].requires_grad == False  #check whether freeze the backbone
    idx_start, idx_end = 0, 0
    for m in range(M):
        Nets[f'E_{m}'].train()
        Nets[f'optimizer_E_{m}'] = torch.optim.Adam((param for param in Nets[f'E_{m}'].parameters()
                              if param.requires_grad), lr = decay * 0.3 * 0.0002, betas = (0.5, 0.999))
        idx_end += fake_samples_num_ls[m] 
        fake_samples_chosen = fake_samples_filtered_all[idx_start : idx_end]
        label_fused_chosen = label_fused[idx_start : idx_end]
        ##标签还原：aligned labels -> pseudo_labels
        label_reset = np.zeros_like(label_fused_chosen)
        if m == 0:
            label_reset = label_fused_chosen.squeeze()
        else:
            for i in range(class_num):
                idx_cluster = label_fused_chosen == i
                label_reset[idx_cluster] = Nets[f'Anchor_to_base_{m}'][i]
            label_reset = label_reset.squeeze()
        idx_high_confidence_chosen = idx_high_confidence[idx_start : idx_end]
        fake_dataset = torch.utils.data.TensorDataset(fake_samples_chosen[idx_high_confidence_chosen],
                                torch.tensor(label_reset)[idx_high_confidence_chosen])
        fake_iter = torch.utils.data.DataLoader(dataset = fake_dataset,
                            batch_size = 200, shuffle = True, num_workers = 0)
        for epoch_E in range(10):
            loss_E_ema = None
            pbar = tqdm(fake_iter)
            for X, y in pbar: 
                X, y = X.to(device), y.to(device)
                Nets[f'optimizer_E_{m}'].zero_grad()
                pred = Nets[f'E_{m}'](X)
                E_loss = torch.nn.CrossEntropyLoss()(pred, y)
                E_loss.backward()
                Nets[f'optimizer_E_{m}'].step()
                if loss_E_ema is None:
                    loss_E_ema = E_loss.item()
                else:
                    loss_E_ema = 0.95 * loss_E_ema + 0.05 * E_loss.item()            
                pbar.set_description(f"Second filtering, E_{m} | Epoch_{epoch}_{epoch_E}: loss_E = {loss_E_ema:.4f}")
        
        ##evaluate by Es
        SIMSIAM_flag = False
        eval_by_E(Nets[f'E_{m}'], m, epoch, dataloader, SIMSIAM_flag, device)        