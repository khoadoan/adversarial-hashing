from os import path
import pickle

import cv2
import numpy as np
import torch
import torchvision
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.parallel

DATA_CACHE = {}

def load_or_reload(data_path, image_size, clear_cache=False):
    if clear_cache: #this option is useful when we work on multiple datasets, prevent OOM
        DATA_CACHE = {}
        
    if data_path in DATA_CACHE:
        (x, y) = DATA_CACHE[data_path]
    else:
        loaded = np.load(data_path)
        x, y = loaded['x'], loaded['y']
        print(x.shape)
        x = x.reshape([x.shape[0], -1, image_size, image_size])
        DATA_CACHE[data_path] = (x, y)
    return x, y

def load_or_reload_vgg(data_path, clear_cache = False):
    if clear_cache: #this option is useful when we work on multiple datasets, prevent OOM
        DATA_CACHE = {}
        
    if data_path in DATA_CACHE:
        (x, y) = DATA_CACHE[data_path]
    else:
        loaded = np.load(data_path)
        x, y = loaded['x'], loaded['y']
        DATA_CACHE[data_path] = (x, y)
    return x, y

def get_dataloader(dataset, image_size, batch_size,
                   dataroot='data/', workers=0, data_transforms=None, type='train', shuffle=True, clear_cache=False):
    if data_transforms is None:
        data_transforms = transforms.Compose([
                                transforms.Scale(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
                            ])
    data_path = path.join(dataroot, '{}_{}_manual_{}.npz'.format(dataset, image_size, type))
    x, y = load_or_reload(data_path, image_size, clear_cache=clear_cache)
    print('Manually loaded {} {} examples ({})'.format(x.shape[0], type, len(y)))
    assert x.shape[0] == len(y)
    
    dataset = torch.utils.data.TensorDataset(torch.tensor(x), torch.tensor(y))
    if type == 'train':
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=shuffle, num_workers=workers)
    elif type == 'db':
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                    shuffle=False, num_workers=workers)
    elif type == 'query':
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=False, num_workers=workers)
    return dataloader

def get_cifar10_dataloader(image_size, batch_size,
                           dataroot='data/', workers=0, data_transforms=None, type='train', shuffle=True):
   return get_dataloader('cifar10', image_size, batch_size,
                   dataroot='data/', workers=0, data_transforms=None, type='train', shuffle=True)

def get_place365_dataloader(image_size, batch_size,
                           dataroot='data/', workers=0, data_transforms=None, type='train', shuffle=True):
   return get_dataloader('place365', image_size, batch_size,
                   dataroot='data/', workers=0, data_transforms=None, type='train', shuffle=True)

def get_cifar10_0_1_dataloader(image_size, batch_size,
                           dataroot='data/cifar10', workers=0, data_transforms=None, type='train', shuffle=True):
    if data_transforms is None:
        data_transforms = transforms.Compose([
                                transforms.Scale(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ])
    data_path = path.join(dataroot, 'cifar10_{}_manual_{}.npz'.format(image_size, type))
    x, y = load_or_reload(data_path, image_size)
    x = (x + 1) / 2
    print('Manually loaded {} {} examples'.format(x.shape[0], type))
    
    dataset = torch.utils.data.TensorDataset(torch.tensor(x), torch.tensor(y))
    if type == 'train':
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=shuffle, num_workers=workers)
    elif type == 'db':
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                    shuffle=False, num_workers=workers)
    elif type == 'query':
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=False, num_workers=workers)
    print(type, dataloader)
    return dataloader

def get_mnist_dataloader(image_size, batch_size,
                           dataroot='data/mnist', workers=2, data_transforms=None, type='train', shuffle=True):
    if data_transforms is None:
        data_transforms = transforms.Compose([
                                transforms.Scale(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, ), (0.5, ))
                            ])
    data_path = path.join(dataroot, 'mnist_{}_manual_{}.npz'.format(image_size, type))
    
    x, y = load_or_reload(data_path, image_size)
    x = (x + 1) / 2
    
    dataset = torch.utils.data.TensorDataset(torch.tensor(x), torch.tensor(y))
    if type == 'train':
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=shuffle, num_workers=workers)
    elif type == 'db':
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                    shuffle=False, num_workers=workers)
    elif type == 'query':
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=False, num_workers=workers)
    return dataloader

# def one_hot_to_index(a):
#   """Convert 1-hot encoding to index"""
#   return np.argmax(a, axis=-1)

# db, test, R = pickle.load(open('../code/HashGAN/db_test.pkl', 'rb'))
# db.input = db.input.reshape([-1, 3072])[:54000, :]
# test.input = test.input.reshape([-1, 3072])[:1000, :]
# db.label = one_hot_to_index(db.label)
# test.label = one_hot_to_index(test.label)

# # db.input = torchvision.transforms.functional.resize(db.input, 64, interpolation=2)
# # test.input = torchvision.transforms.functional.resize(test.input, 64, interpolation=2)
# # db.input = db.input.reshape([-1, 3, 32, 32])
# # test.input = test.input.reshape([-1, 3, 32, 32])
# print(db.input.shape)
# db.input = np.array([cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA) for img in db.input])
# test.input = np.array([cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA) for img in test.input])
# db.input = (db.input / 255)
# test.input = (test.input / 255)
# print(db.input.shape)
# db.input = db.input.reshape([-1, 3, 64, 64])
# test.input = test.input.reshape([-1, 3, 64, 64])

# def get_cifar10_dataloader(image_size, batch_size,
#                            dataroot='data/cifar10', workers=2, data_transforms=None, type='train', shuffle=True):
#     if type == 'train':
#         dataset = torch.utils.data.TensorDataset(torch.tensor(db.input), torch.tensor(db.label))
#         dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                             shuffle=shuffle, num_workers=workers)
#     elif type == 'db':
#         dataset = torch.utils.data.TensorDataset(torch.tensor(db.input), torch.tensor(db.label))
#         dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                     shuffle=False, num_workers=workers)
#     elif type == 'query':
#         dataset = torch.utils.data.TensorDataset(torch.tensor(test.input), torch.tensor(test.label))
#         dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                             shuffle=False, num_workers=workers)
#     return dataloader