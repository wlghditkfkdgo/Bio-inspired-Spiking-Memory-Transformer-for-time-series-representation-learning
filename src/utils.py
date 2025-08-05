import torch
import pandas as pd
    
import math
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
from timm.scheduler import create_scheduler, create_scheduler_v2

import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from spikingjelly.clock_driven import functional

# visualization
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from tqdm import tqdm
import os

class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    



def undersampling(data, sampling_size):

    num_classes = len(data.iloc[:, -1].unique())
    
    if sampling_size:
        min_value = sampling_size
    else:
        min_value = data.iloc[:, -1].value_counts().min()
    
    X_sampled = []
    y_sampled = []

    start_idx = 0
    for i in range(0, num_classes):
        
        max_idx = len(data[data.iloc[:, -1] == i])
        mask = np.random.permutation(range(start_idx, start_idx + max_idx))
        mask = mask[:min_value]
        sampled_data = data.iloc[mask, :]
        
        X_sampled.append(sampled_data.iloc[:, :-1].values)
        y_sampled.append(sampled_data.iloc[:, -1].values)
        
        start_idx = start_idx + max_idx
        
    df_X = np.vstack(X_sampled)
    df_y = np.hstack(y_sampled)

    df_X = pd.DataFrame(df_X)
    df_y = pd.DataFrame(df_y)
    
    df = pd.concat([df_X, df_y], axis=1, ignore_index=False)
    
    return df

    
def get_scheduler(scheduler:str, optimizer, **kargs):
    
    step_size = 10
    gamma = 0.5 
    T_max = 10
    patience = 2

    if scheduler == 'step' : scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
    elif scheduler == 'exponential' : scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    # elif scheduler == 'cosine' : scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0)
    # elif scheduler == 'cosine' : scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=0)
    elif scheduler == 'cosine' : 
        max_lr = kargs["max_lr"]
        max_epochs = kargs["max_epochs"]
        min_lr = kargs["min_lr"]
        scheduler, _ = create_scheduler_v2(optimizer, sched='cosine', num_epochs=max_epochs, decay_epochs=int(max_epochs * 0.3), warmup_epochs=int(max_epochs * 0.2), cooldown_epochs=int(max_epochs * 0.1), min_lr=min_lr, noise_pct=0.67, warmup_lr=0.00001)
        
        # scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=max_epochs, cycle_mult=1, min_lr=1e-6, max_lr=max_lr, gamma=1, warmup_steps=5)
        # scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=100, cycle_mult=1, min_lr=1e-5, max_lr=max_lr, gamma=1, warmup_steps=5)

        # max_lr = kargs["max_lr"]
        # # scheduler = create_scheduler()
        # # scheduler = torch.optim.lr_scheduler.cos
        # # scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=10, cycle_mult=1, min_lr=1e-10, max_lr=max_lr, gamma=0.9, warmup_steps=3)
        # scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=10, cycle_mult=1, min_lr=1e-10, max_lr=max_lr, gamma=0.5, warmup_steps=2)
    elif scheduler == 'reduce' : scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=0.1, cooldown=1, min_lr=1e-6)
    else:
        assert NotImplementedError(scheduler)
    
    return scheduler

def get_pred(output, topk=(1,)):
    
    maxk = min(max(topk), output.shape[1])
    _, pred = output.topk(maxk, 1, True, True)
    
    return pred.squeeze(-1)
        
        
def get_class_weights(class_num_samples:torch.Tensor,) -> torch.Tensor :
    
    num_classes = class_num_samples.shape[0]
    
    ## Class-aware loss
    num_max = class_num_samples.max()
    tot_num_samples = class_num_samples.sum()
    
    mu = num_max / tot_num_samples
    
    # weights = mu * tot_num_samples / class_num_samples
    weights = torch.log(mu * tot_num_samples / class_num_samples)
    weights = torch.max(torch.ones_like(weights), weights)
    
    # weights = tot_num_samples / class_num_samples
    print(weights)
    
    ## Context-aware loss
    # total = np.sum(class_num_samples)
    # class_weights = dict()
    # num_classes = len(class_num_samples)
    
    # factor = 1 / num_classes
    # mu = [factor]
    
    return weights


def model_info(model, verbose=False, img_size=640):
    """ (C) copyright https://github.com/BICLab/EMS-YOLO

    Args:
        model (_type_): _description_
        verbose (bool, optional): _description_. Defaults to False.
        img_size (int, optional): _description_. Defaults to 640.
    """
    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPs
        from thop import profile
        
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
        img = torch.zeros((1, model.yaml.get('ch', 3), stride, stride), device=next(model.parameters()).device)  # input
        flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2  # stride GFLOPs
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  # expand if int/float
        fs = ', %.1f GFLOPs' % (flops * img_size[0] / stride * img_size[1] / stride)  # 640x640 GFLOPs
    except (ImportError, Exception):
        fs = ''
        
    
def get_energy_consumption(O_ac, O_mac, E_ac=0.9, E_mac=4.6, unit=None) -> float:
    
    E_ac = E_ac * 1e-12
    E_mac = E_mac * 1e-12
    
    energy = E_ac * O_ac + E_mac * O_mac
    
    if unit is None or unit == 'p':
        return energy / 1e-12
    
    elif unit == 'm' :
        return energy / 1e-3
    
    elif unit == 'u' :
        return energy / 1e-6
    
def tsne_visual(feature:list, actual_label, metric, save_path, num_classes):

    tsne = TSNE(n_components=2, metric=metric, perplexity=20)
    cluster = np.array(tsne.fit_transform(np.array(feature)))
    actual = np.array(actual_label)

    plt.figure(figsize=(10, 10))
    labels = [str(i) for i in range(num_classes)]
    
    color = ["#66C5CC","#F6CF71","#F89C74","#DCB0F2","#87C55F","#9EB9F3","#FE88B1","#C9DB74","#8BE0A4","#B497E7","#D3B484","#B3B3B3"]
    
    for l, label in tqdm(enumerate(labels)):
        idx = np.where(actual == l)
        plt.scatter(cluster[idx, 0], cluster[idx, 1], marker='.', label=label, c=color[l])

    plt.savefig(save_path)
    plt.close()
    
    
def heatmap_visual(features:np.array, save_path):
    
    os.makedirs(save_path, exist_ok=True)
    for batch in range(len(features)):
        for sample in tqdm(range(features[batch].shape[1])):
            feature = features[batch].sum(0)[sample].sum(0)
            
            fig, ax = plt.subplots()
            heatmap = ax.pcolor(feature, cmap=plt.cm.Blues)
            
            ax.set_xticks(np.arange(feature.shape[1]) + 0.5, minor=False)
            ax.set_yticks(np.arange(feature.shape[0]) + 0.5, minor=False)
            
            ax.set_xlim(0, int(feature.shape[1]))
            ax.set_ylim(0, int(feature.shape[0]))
            
            ax.invert_yaxis()
            ax.xaxis.tick_top()
            
            ax.set_xlabel('key')
            ax.set_ylabel('query')
            
            plt.xticks(rotation=45)
            plt.savefig(save_path+f'/{batch}_{sample}_attn_map.png')
            plt.close()
    
        
def plot_eval(model, loader, num_classes, save_path, device):

    actual = []
    embedding = {'attn_map' : list(),}

    with torch.no_grad():
        for data, label in loader:
            
            data, label = data.to(device), label.to(device)
            
            feature = model(data)
            
            for i, emb_k in enumerate(feature):
                
                if not emb_k in embedding.keys():
                        embedding[emb_k] = feature[emb_k].cpu().numpy().tolist()
                else:
                    if emb_k == 'attn_map':
                        embedding[emb_k].append(feature[emb_k].cpu().data.numpy())
                    else:
                        embedding[emb_k] += feature[emb_k].cpu().numpy().tolist()
                    
                
            actual += label.cpu().numpy().tolist()
            
            functional.reset_net(model)

    
    for i, emb_k in enumerate(embedding):    
        if emb_k == 'attn_map' :
            # pass
            heatmap_visual(embedding[emb_k], save_path=save_path + f'/attn_heatmap_result')
        else:
            tsne_visual(embedding[emb_k], actual_label=actual, metric='euclidean', save_path=save_path + f'/tsne_result_{emb_k}.png', num_classes=num_classes)
        

def print_epoch_info(epoch, lr, num_classes:int, train_result:dict, val_result:dict=None):
    bar = "-" * 30
    
    print(bar)
    print(f"{'epoch':15s}{epoch:>15d} (lr={lr})")
    for k, v in train_result.items():
            value = v if isinstance(v, float) else v.mean()
            print(f"{'train_' + k:15s}{value:>15.5f}")
    if val_result is not None:
        for k, v in val_result.items():
            value = v if isinstance(v, float) else v.mean()
            print(f"{'val_' + k:15s}{value:>15.5f}")
        print(bar) 
    if num_classes > 0:
        for i in range(num_classes): print(f"{'val_' + 'acc' + '_' + str(i):15s}{val_result['acc'][i]:>15.5f}")
    print(bar)
    
    
# class OrthogonalProjectionLoss(nn.Module):
#     def __init__(self, no_norm=False, use_attention=False, gamma=2):
#         super(OrthogonalProjectionLoss, self).__init__()

#         self.no_norm = no_norm
#         self.gamma = gamma
#         self.use_attention = use_attention


#     def forward(self, features, labels=None):
        
#         device = features.device

#         if self.use_attention:
#             features_weights = torch.matmul(features, features.T)
#             features_weights = F.softmax(features_weights, dim=1)
#             features = torch.matmul(features_weights, features)

#         #  features are normalized
#         if not self.no_norm:
#             features = F.normalize(features, p=2, dim=1)

#         labels = labels[:, None]  # extend dim
#         mask = torch.eq(labels, labels.t()).bool().to(device)
#         eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)

#         mask_pos = mask.masked_fill(eye, 0).float()
#         mask_neg = (~mask).float()
#         dot_prod = torch.matmul(features, features.t())

#         pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
#         neg_pairs_mean = torch.abs(mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)

#         loss = (1.0 - pos_pairs_mean) + (self.gamma * neg_pairs_mean)
#         # loss = neg_pairs_mean

#         return loss, pos_pairs_mean, neg_pairs_mean

def random_masking_3D(xb, mask_ratio):
    # xb: [bs x num_patch x dim]
    # xb = xb.flatten(0, 1).contiguous()
    bs, L, D = xb.shape
    x = xb.clone()

    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(bs, L, device=xb.device)  # noise in [0, 1], bs x L

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)  # ids_restore: [bs x L]

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]  # ids_keep: [bs x len_keep]
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))  # x_kept: [bs x len_keep x dim]

    # removed x
    x_removed = torch.zeros(bs, L - len_keep, D, device=xb.device)  # x_removed: [bs x (L-len_keep) x dim]
    x_ = torch.cat([x_kept, x_removed], dim=1)  # x_: [bs x L x dim]

    # combine the kept part and the removed one
    x_masked = torch.gather(x_, dim=1,
                            index=ids_restore.unsqueeze(-1).repeat(1, 1, D))  # x_masked: [bs x num_patch x dim]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, L], device=x.device)  # mask: [bs x num_patch]
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)  # [bs x num_patch]
    
    return x_masked, x_kept, mask, ids_restore



def create_temporal_proximity_mask(num_tokens: int, sharpness: float = 1.0):
    """
    Time step 간 거리가 가까울수록 가중치가 높아지는 proximity mask 생성.

    Args:
        seq_len (int): 시계열 길이 (time step 수)
        sharpness (float): 거리 증가에 따른 감쇠 속도 (높을수록 가까운 것에 집중)

    Returns:
        mask (torch.Tensor): (seq_len, seq_len) 형태의 proximity mask
    """
    time = torch.arange(num_tokens).unsqueeze(0)  # (1, T)
    distance = torch.abs(time.T - time)        # (T, T) → |i - j|
    mask = torch.exp(-sharpness * distance.float())  # proximity → 가까울수록 1에 가까움
    return mask

def consolidation_loss(current_state_dict, previous_state_dict):
    loss = 0.0
    for name in current_state_dict:
        cur = current_state_dict[name]
        prev = previous_state_dict[name]
        if torch.is_floating_point(cur):  # avoid buffers like int types
            loss += F.mse_loss(cur, prev)
    return loss

def square_tokenize(x: torch.Tensor, T: int | None = None, pad_mode: str = "zero"):
    """
    Rearrange (B, C, L) → (T, B, N=T, C) so that token length == token count.
    
    Args
    ----
    x        : Input tensor of shape (B, C, L)
    T        : Desired square edge; if None, choose ceil(sqrt(L))
    pad_mode : 'zero' | 'replicate'  (how to pad when L < T*T)
    
    Returns
    -------
    out      : Tensor (T, B, N=T, C)
    """
    B, C, L = x.shape
    if T is None:
        T = math.ceil(math.sqrt(L))          # make it square

    target_len = T * T

    if L < target_len:                       # need padding
        pad_len = target_len - L
        if pad_mode == "zero":
            x = F.pad(x, (0, pad_len))
        elif pad_mode == "replicate":
            last = x[..., -1:].expand(-1, -1, pad_len)
            x = torch.cat((x, last), dim=-1)
        else:
            raise ValueError("pad_mode must be 'zero' or 'replicate'")
    elif L > target_len:                     # truncate excess
        x = x[..., :target_len]

    # reshape → (B, C, T, T) and permute
    out = x.view(B, C, T, T).permute(2, 0, 3, 1).contiguous()
    # shape : (T, B, N=T, C)   (matches your [T, B, D, N] with D=C)
    return out

# def interpolate_missing_np(arr):
#     """
#     NaN 값을 선형 보간으로 채우는 함수 (numpy 기반)
#     """
#     if not np.isnan(arr).any():
#         return arr
    
#     # NaN이 있는 index
#     nans = np.isnan(arr)
#     not_nans = ~nans
#     indices = np.arange(len(arr))
    
#     # 양쪽 방향 보간
#     arr[nans] = np.interp(indices[nans], indices[not_nans], arr[not_nans])
#     return arr

def interpolate_missing_np(arr):
    """
    arr: 2D np.ndarray with shape (C, L)
    Applies linear interpolation per channel (row-wise)
    """
    if arr.shape[0] == 1:
        if not np.isnan(arr).any():
            return arr
        
        # NaN이 있는 index
        nans = np.isnan(arr)
        not_nans = ~nans
        indices = np.arange(len(arr))
        
        # 양쪽 방향 보간
        arr[nans] = np.interp(indices[nans], indices[not_nans], arr[not_nans])
        return arr
    else:
        for c in range(arr.shape[0]):
            row = arr[c]
            nans = np.isnan(row)
            not_nans = ~nans
            indices = np.arange(len(row))
            if nans.any() and not_nans.any():
                row[nans] = np.interp(indices[nans], indices[not_nans], row[not_nans])
            arr[c] = row
        return arr
    
def slice_and_flatten_with_padding(x: torch.Tensor, y: torch.Tensor, segment_len: int) -> torch.Tensor:
    """
    Args:
        x (Tensor): [B, C, L] input tensor
        segment_len (int): Length of each segment

    Returns:
        Tensor: [B * num_segments, C, segment_len] tensor after padding and slicing
    """
    B, C, L = x.shape
    remainder = L % segment_len

    # Zero padding if needed
    if remainder != 0:
        pad_len = segment_len - remainder
        # Pad on the right (last dimension)
        x = F.pad(x, (0, pad_len), mode='constant', value=0)
        L = L + pad_len

    num_segments = L // segment_len

    # [B, C, L] → [B, C, num_segments, segment_len]
    x = x.unfold(dimension=2, size=segment_len, step=segment_len)
    x = x.permute(0, 2, 1, 3).contiguous()  # [B, num_segments, C, segment_len]
    x = x.view(B * num_segments, C, segment_len)
    y = y.repeat_interleave(num_segments)
    
    return x, y
