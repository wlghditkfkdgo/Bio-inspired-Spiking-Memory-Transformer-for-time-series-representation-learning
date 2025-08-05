import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler, Subset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold

import pandas as pd
import numpy as np

from typing import Optional, Callable, Tuple
from imblearn.over_sampling import SMOTE
from copy import deepcopy

from config import set_seed_worker
from utils import interpolate_missing_np


class CustomDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, oversampling=False, transform=None, segment_len=0) -> None:
        super().__init__()

        self.segment_len = segment_len
        self.transform = transform

        # Z-score normalization across all samples and time
        self.mean = X.mean(axis=(0, 2), keepdims=True)
        self.std = X.std(axis=(0, 2), keepdims=True)
        X = (X - self.mean) / (self.std + 1e-8)

        if oversampling:
            B, C, L = X.shape
            X = X.reshape(B, -1)  # Flatten to [B, C * L] for SMOTE
            X, y = SMOTE().fit_resample(X, y)
            X = X.reshape(-1, C, L)

        self.x = X
        self.y = y

        if segment_len > 0:
            B, C, L = self.x.shape
            remainder = L % segment_len
            pad_len = segment_len - remainder if remainder != 0 else 0

            if pad_len > 0:
                self.x = np.pad(self.x, ((0, 0), (0, 0), (0, pad_len)), mode='constant')
                L += pad_len

            self.num_segments = L // segment_len
            self.x = self.x.reshape(B, C, self.num_segments, segment_len)
            self.x = self.x.transpose(0, 2, 1, 3).reshape(B * self.num_segments, C, segment_len)
            self.y = np.repeat(self.y, self.num_segments)

    def __getitem__(self, idx):
        data = torch.tensor(self.x[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)

        if self.transform:
            data = self.transform(data)

        return data, label

    def __len__(self):
        return self.x.shape[0]
    
class CustomCSVDataset(Dataset):
    def __init__(self, X, y, oversampling=False, transform=None) -> None:
        super().__init__()
        
        """
            train=True : True -> train/val Dataset, False -> test Dataset
            root : Path to load data
            class_imbalance : True -> larger class undersampling
        """
        
        self.x = X
        self.y = y
        
        self.transform = transform
        
        if oversampling:
            smote = SMOTE()
            self.x, self.y = smote.fit_resample(self.x, self.y)
    
    def __getitem__(self, idx):
        
        data = torch.tensor(self.x[idx][:], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        
        if len(self.x.shape) < 3 : data = data.unsqueeze(-2)
            
        if self.transform:
            data = self.transform(data)
        
        return data, label
    
    def __len__(self):
        return self.x.shape[0] 
    
class SmoteSampler(Dataset):
    def __init__(self, trainset:Subset, transform=None) -> None:
        super().__init__()
        
        smote = SMOTE()
        
        self.transform = transform
        
        self.x, self.y = smote.fit_resample(trainset.dataset.x, trainset.dataset.y)
        
        
    def __getitem__(self, idx):
        data = torch.tensor(self.x[idx][:], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)
            
        if self.transform:
            data = self.transform(data)
        
        return data, label
    
    def __len__(self):
        
        return self.x.shape[0] 
        


class TestCustomDataset(Dataset):
    def __init__(self, root=None, class_imbalance=True, transform=None, window_size=192) -> None:
        """
            root: csv file root,
            class_imbalance: True => take resample to make samples of lack class same with largest class
        """
        super().__init__()
        
        if root is None:
            root = './MIT_BIH_ECG/bin_3_mitbih_test.csv'
        
        f = root.split('.')[-1]
        if f == 'csv' : 
            self.data = pd.read_csv(root, header=None)
            
        elif f == 'pt':
            self.data = torch.load(root)
            
        self.transform = transform
        self.undersampling_size = 3000
        
        if class_imbalance:
            # self.data = oversampling(self.data)
            self.data = self.undersampling(self.data, self.undersampling_size)
        
        num_steps = int(len(self.data.columns) - 1)
        
        self.x = self.data.iloc[:, :num_steps].to_numpy(dtype=np.float64) 
        self.y = self.data.iloc[:, num_steps].to_numpy(dtype=np.compat.long) 
        
    def __getitem__(self, idx):
            
        img = torch.tensor(self.x[idx][:], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)
            
        if self.transform:
            img = self.transform(img)
        
        return img, label
    
    def __len__(self):
        
        return self.x.shape[0] 
    
    @staticmethod
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
    

class BinaryUndersamplingSampler(Sampler):
    
    def __init__(self, dataset:Subset):
        self.dataset = dataset

        label0_idx = []
        label1_idx = []
        
        labels = dataset.dataset.y[dataset.indices]

        for i, label in zip(dataset.indices, labels):
            if label == 0: label0_idx.append(i)
            elif label == 1: label1_idx.append(i)
            
        sampling_size = min(len(label0_idx), len(label1_idx))
        idx0 = torch.randperm(len(label0_idx))[:sampling_size]
        idx1 = torch.randperm(len(label1_idx))[:sampling_size]
        
        sampled_label0_idx = []
        sampled_label1_idx = []
        
        for i0, i1 in zip(idx0, idx1):
            sampled_label0_idx.append(label0_idx[i0])
            sampled_label1_idx.append(label1_idx[i1])
        
        self.indices = sampled_label0_idx + sampled_label1_idx
        print(f"undersampling >> [0 : {len(sampled_label0_idx)}] + [1 : {len(sampled_label1_idx)}] = {len(self.indices)}")
        
    def __iter__(self):
        for i in torch.randperm(len(self.indices)):
            yield self.indices[i]
    
    def __len__(self):
        return len(self.indices)

class MultiClassUndersamplingSampler(Sampler):
    
    def __init__(self, dataset:Dataset, num_classes=5, sampling=['avg', 'min', 'cut', 'smote', 'None']):
        self.dataset = dataset
        
        idx_dict = {i: list() for i in range(num_classes)}
        indices = np.arange(len(dataset))
        
        if isinstance(dataset, Subset):
            labels = np.array(dataset.dataset.y)[dataset.indices]
        else:
            labels = dataset.y

        for i, label in zip(indices, labels):
            for c in range(num_classes):
                if label == c: idx_dict[c].append(i)

        if sampling == 'avg' :
            sampling_size = []       
            for c in range(0, num_classes): # class 0 -> ignore
                sampling_size.append(len(idx_dict[c]))
            sampling_size_rank = sorted(sampling_size)
            sampling_size = np.array(sampling_size_rank[:-2]).mean().astype(np.int16)
            
        elif sampling == 'min':
            sampling_size = float("inf")
            for c in range(num_classes): # class 4 -> ignore
                idx_len = len(idx_dict[c])
                if idx_len < sampling_size:
                    sampling_size = idx_len
        
        elif sampling == 'cut':
            # cut largest data
            sampling_size = 0       
            for c in range(1, num_classes): # class 0 -> ignore
                sampling_size += len(idx_dict[c])
            sampling_size = int(sampling_size/(num_classes - 1))
        
        elif sampling == 'smote':
            sampling_size = len(dataset)

        
        label_idx_dict = dict()
        if sampling == 'cut':
            label_idx_dict[0] = torch.randperm(len(idx_dict[0])) [:sampling_size]
            for c in range(1, num_classes):
                label_idx_dict[c] = torch.randperm(len(idx_dict[c]))
        elif sampling == 'None':
            for c in range(num_classes):
                # shuffled = np.random.permutation(idx_dict[c])
                # label_idx_dict[c] = shuffled.tolist()
                label_idx_dict[c] = idx_dict[c]
        else:
            for c in range(num_classes):
                label_idx_dict[c] = torch.randperm(len(idx_dict[c]), generator=torch.Generator().manual_seed(42))[:sampling_size]
        
        self.sampled_label_idx_dict = {i : list() for i in range(num_classes)}
        for c in range(num_classes):
            # for i in label_idx_dict[c]:
            #     self.sampled_label_idx_dict[c].append(idx_dict[c][i])
            self.sampled_label_idx_dict[c] = idx_dict[c]
        
        self.indices = deepcopy(self.sampled_label_idx_dict[0])
        for c in range(1, num_classes):
            self.indices += self.sampled_label_idx_dict[c]
            
        print(f"sampling >> {sampling}")
        for c in range(num_classes):
            self.sampled_label_idx_dict[c] = len(self.sampled_label_idx_dict[c])
            print(f"[ {c} : {self.sampled_label_idx_dict[c]} ]", end=" ")
        print(f"=> {len(self.indices)}")
        
    def __iter__(self):
        for i in torch.randperm(len(self.indices)):
            yield self.indices[i]
    
    def __len__(self):
        return len(self.indices)
    
    def get_num_samples(self):
        
        class_num_sample = np.array(list(self.sampled_label_idx_dict.values()))
        class_num_sample = torch.tensor(class_num_sample)
        
        return class_num_sample
    
def create_dataset(train:bool, data_root:Optional[str]=None, split:bool=False, data_window_size:int=192, train_val_ratio=list[7, 3]):
    pass
    
def create_loader(train:bool, batch_size:int, data_root:Optional[str]=None, transform=None, num_classes=2, train_val_ratio:list=None, sampling=['avg', 'min', 'cut', 'smote', 'None'], data_window_size=192, segment_len=0, n_folds=5, fold_id=0, num_workers:int=16):
    
    f = data_root.split('.')[-1]
    
    if f == 'csv':
        data = pd.read_csv(data_root, header=None)
        
        len_data = int(len(data.columns) - 1)
        
        X = data.iloc[:, :len_data].to_numpy(dtype=np.float64) 
        y = data.iloc[:, len_data].to_numpy(dtype=np.compat.long)
        
        for i in range(X.shape[0]):
            X[i] = interpolate_missing_np(X[i][np.newaxis, :])
    
    elif f == 'pt' : 
        data = torch.load(data_root)
        
        X = data['train_X'] if train else data['test_X']
        y = data['train_y'] if train else data['test_y']
        
        X = X.numpy() if isinstance(X, torch.Tensor) else X
        y = y.numpy() if isinstance(y, torch.Tensor) else y
        for i in range(X.shape[0]):
            X[i] = interpolate_missing_np(X[i])
        
        len_data = X.shape[-1] #[N C L]

    else:
        raise NotImplementedError
   
    if len_data < data_window_size:
        pad = np.zeros((X.shape[0], int(data_window_size-len_data)), dtype=np.float64)
        X = np.concatenate((X, pad), axis=1)
    
    if train:
        X_train, y_train = X, y
        oversampling = sampling == 'smote'
        
        if train_val_ratio is not None:
            sum_ratio = train_val_ratio[0] + train_val_ratio[1]
            # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=train_val_ratio[1]/sum_ratio, stratify=y)
            if f == 'csv' : 
                trainset = CustomCSVDataset(X_train, y_train, oversampling=oversampling)
                # valset = CustomCSVDataset(X_val, y_val, oversampling=oversampling)
            elif f == 'pt' : 
                trainset = CustomDataset(X_train, y_train, oversampling=oversampling, segment_len=segment_len)
                # valset = CustomDataset(X_val, y_val, oversampling=oversampling, segment_len=segment_len)
                
            skf = StratifiedKFold(n_splits=n_folds,
                              shuffle=True, random_state=42)
            all_indices = np.arange(len(trainset))
            labels_full = trainset.y
            folds = list(skf.split(all_indices, labels_full))
            train_idx, val_idx = folds[fold_id]   # 선택한 fold
            
            train_subset = Subset(trainset, train_idx)
            val_subset   = Subset(trainset, val_idx)
            
            train_sampler = MultiClassUndersamplingSampler(train_subset, num_classes=num_classes, sampling=sampling)
            train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=train_sampler, shuffle=False, num_workers=num_workers, worker_init_fn=np.random.seed(42))
            class_num_samples = train_sampler.get_num_samples()
            
            val_sampler = MultiClassUndersamplingSampler(val_subset, num_classes=num_classes, sampling=sampling)
            val_loader = DataLoader(val_subset, batch_size=batch_size, sampler=val_sampler, shuffle=False, num_workers=num_workers, worker_init_fn=np.random.seed(42))
            # class_num_samples = train_sampler.get_num_samples()

            return (train_loader, val_loader), class_num_samples
        
        else:
            if f == 'csv' : 
                trainset = CustomCSVDataset(X_train, y_train, oversampling=oversampling)
            elif f == 'pt' : 
                trainset = CustomDataset(X_train, y_train, oversampling=oversampling, segment_len=segment_len)
                
            train_sampler = MultiClassUndersamplingSampler(trainset, num_classes=num_classes, sampling=sampling)
            
            train_loader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, shuffle=False, num_workers=num_workers, worker_init_fn=np.random.seed(42))
            class_num_samples = train_sampler.get_num_samples()
            
            return train_loader, class_num_samples
        
    else:
        # test
        if f == 'csv':
            testset = CustomCSVDataset(X, y)
        elif f == 'pt' : 
            testset = CustomDataset(X, y, segment_len=segment_len)
            
        test_sampler = MultiClassUndersamplingSampler(testset, num_classes=num_classes, sampling='None')

        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=set_seed_worker)
        class_num_samples = test_sampler.get_num_samples()

        return test_loader, class_num_samples
        