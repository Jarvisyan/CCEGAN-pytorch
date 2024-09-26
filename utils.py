import torch
import random
import numpy as np
from torch import nn
from munkres import Munkres
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()      

def gen_cond_label(batch_size, class_num, z_dim):
    conditional_label = torch.zeros(batch_size, class_num)
    cluster_size = round(batch_size / class_num)
    for i in range(class_num):
        if i == class_num - 1:
            conditional_label[i * cluster_size : , i] = 1
        else:
            conditional_label[i * cluster_size : (i + 1) * cluster_size, i] = 1
    G_input = torch.cat([conditional_label, torch.rand(batch_size, z_dim)], 1)
    return G_input, conditional_label

# def Purity(true_label, pred_label):
    # k_set = torch.unique(pred_label)
    # correct_num = 0
    # for i in k_set:
        # idx = pred_label == i
        # cluster_i = true_label[idx]
        # correct_num += torch.max(torch.bincount(cluster_i.int()))
    # purity = correct_num / len(true_label)
    # return float(purity)
    
def cluster_acc(Y_pred, Y):
    #from sklearn.utils.linear_assignment_ import linear_assignment
    from scipy.optimize import linear_sum_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_sum_assignment(w.max() - w) #linear_assignment(w.max() - w)
    return sum([w[i,j] for i,j in zip(ind[0], ind[1])])*1.0/Y_pred.size, w #sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w

def eval_by_E(E, m, epoch, dataloader, SIMSIAM_falg, device):
    E.eval()
    pred_label = torch.zeros(0, device = device)
    true_label = torch.zeros(0)
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            if SIMSIAM_falg:
                pred = E(X)[0]
            else:
                pred = nn.Softmax(dim =1)(E(X))
            label = torch.argmax(pred, 1)
            pred_label = torch.cat([pred_label, label])
            true_label = torch.cat([true_label, y])
    pred_label = pred_label.to('cpu')
    # purity = Purity(true_label, pred_label)
    acc, _ = cluster_acc(pred_label.long().numpy(), true_label.long().numpy())
    nmi = NMI(true_label, pred_label)
    print(f'Epoch_{epoch} | E_{m}: acc = {acc}, nmi = {nmi}')
    with open("./logs.txt", 'a') as f:
        f.write(f'Epoch_{epoch} | E_{m}: acc = {acc}, nmi = {nmi}\n')
    return true_label.long(), pred_label.long()
    
def label_assignment_by_base_E(E, dataloder, device):
    E.eval()
    pred_label = torch.zeros(0, device = device)
    with torch.no_grad():
        for X in dataloder:
            X = X.to(device)
            pred = nn.Softmax(dim =1)(E(X))
            label = torch.argmax(pred, 1)
            pred_label = torch.cat([pred_label, label], 0)
    return pred_label.cpu().long()


def map_cost(pred, k, idx_true):
    idx_pred = (pred == k).astype(int)
    union_num = sum((idx_pred +  idx_true) > 0)
    intersection_num = sum((idx_pred +  idx_true) == 2)
    return union_num - intersection_num

def label_map(true_label, pred):
    
    true_label, pred = np.array(true_label), np.array(pred)
    assert len(np.unique(true_label)) == (max(true_label) + 1)
    assert len(np.unique(pred)) == (max(pred) + 1)
    

    n = len(true_label)
    k_true = max(true_label) + 1
    k_pred = max(pred) + 1
    cost_matrix = np.zeros([k_pred, k_true], int)
    for j in range(k_true):
        idx_true = (true_label == j).astype(int)
        col_val = map(map_cost,
                      np.tile(pred, (k_pred, 1)),
                      [i for i in range(k_pred)],
                      np.tile(idx_true, (k_pred, 1)))
        cost_matrix[:, j] = list(col_val)
    count = 0
    if k_pred < k_true:
        while k_pred + count < k_true:
            cost_matrix = np.concatenate((cost_matrix, np.repeat(0, k_true).reshape(1, -1)), 0)
            count += 1
    elif k_pred > k_true:
        while k_true + count < k_pred:
            cost_matrix = np.concatenate((cost_matrix, np.repeat(0, k_pred).reshape(-1, 1)), 1)
            count += 1
    assert cost_matrix.shape[0] == max(k_pred, k_true) and cost_matrix.shape[1] == max(k_pred, k_true)
    

    solve = Munkres()
    solution_map = solve.compute(cost_matrix)
    pred_to_true = {k : v for (k, v) in solution_map} #pred_label : true_label
    true_to_pred = {v : k for (k, v) in solution_map} #true_label : pred_label
    pred_aligned = [pred_to_true[i] for i in pred] 
    return pred_aligned, pred_to_true, true_to_pred
    
    
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True