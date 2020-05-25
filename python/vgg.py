from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import cv2
from pathlib import Path
from torch.utils.data import DataLoader,Dataset
from torch.hub import load_state_dict_from_url


def read_cifar(path):
    parent_cifar = Path(path)
    query = np.load(parent_cifar / 'cifar10_64_manual_query.npz')
    db = np.load(parent_cifar / 'cifar10_64_manual_db.npz')
    train = np.load(parent_cifar/ 'cifar10_64_manual_train.npz')
    query_x = query['x']
    db_x = db['x']
    train_x = train['x']
    return query_x , db_x , train_x 


class VGG16(nn.Module):

    def __init__(self,in_channels, output_dim):
        super(VGG16,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(256,512,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(512,512,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
 
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, output_dim),
        )

    def forward(self,x):
        feat = self.features(x)
        feat_pooled = self.avgpool(feat)
        feat_flat= torch.flatten(feat_pooled, 1)
        return self.classifier(feat_flat)
    
    def get_fc7(self, x, idx=1):
        BLOCK_IDX = {1: 2, 2: 5, 3: 6} #1 for first block, 2 for second block, 3 for final, classification block
        assert idx <= 3
        
        features = self.features(x)
        features = self.avgpool(features)
        features = torch.flatten(features,1)
        out = features
        for i in range(BLOCK_IDX[idx]):
            out = self.classifier[i](out)
        return out    


# ===================== The rest was just for testing, remove later ==============
# class CIFARDataset(Dataset):
    
#     def __init__(self,np_array):
#         self.data = np_array
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self,idx):
#         if idx is not list:
#             idx = idx.tolist()
#         return self.data[idx]

# def main():
#     query_x, db_x , train_x = read_cifar('./data')
#     in_channels = 3
#     out_dim = 1000
#     model = VGG16(in_channels, out_dim)
#     state_dict = load_state_dict_from_url(url=
#     'https://download.pytorch.org/models/vgg16-397923af.pth')
#     model.load_state_dict(state_dict,strict=False)
#     print (query_x.shape)
#     query_x = torch.Tensor(query_x.reshape(-1,3,64,64))
#     train_x = torch.Tensor(train_x.reshape(-1,3,64,64))
#     train_dataset,query_dataset, db_dataset = CIFARDataset(train_x),\
#         CIFARDataset(query_x), CIFARDataset(db_x)
#     train_loader = DataLoader(train_dataset,batch_size=24,num_workers=4)
#     query_loader = DataLoader(query_dataset,batch_size=24,num_workers=4)
#     db_loader = DataLoader(db_dataset,batch_size=24,num_workers=4)

#     for train in iter(train_loader):
#         train_features = model.get_fc7(train)
#     query_features = model.get_fc7(query_x)
#     np.savez_compressed('data/fc7_query.npz')
#     print ('query done')

#     np.savez_compressed('data/fct7_train.npz')
#     print ('train done')

# if __name__ == '__main__':
#     main()
