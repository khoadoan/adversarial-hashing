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

# def get_cifar10_dataloader(image_size, batch_size,
#                            dataroot='data/cifar10', workers=2, data_transforms=None, type='train', shuffle=True):
#     if data_transforms is None:
#         data_transforms = transforms.Compose([
#                                 transforms.Scale(image_size),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                             ])
#     if type == 'train':
#         train_dataset = dset.CIFAR10(root=dataroot, download=True, train=True, transform=data_transforms)
#         dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
#                                             shuffle=shuffle, num_workers=workers)
#     elif type == 'test':
#         test_dataset = dset.CIFAR10(root=dataroot, download=True, train=False, transform=data_transforms)
#         dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
#                                             shuffle=shuffle, num_workers=workers)
#     return dataloader

def get_cifar10_dataloader(image_size, batch_size,
                           dataroot='data/cifar10', workers=2, data_transforms=None, type='train', shuffle=True):
    if data_transforms is None:
        data_transforms = transforms.Compose([
                                transforms.Scale(image_size),
                                transforms.ToTensor()
                            ])
    data_path = path.join(dataroot, 'cifar10_manual_{}.npz'.format(type))
    loaded = np.load(data_path)
    x, y = loaded['x'], loaded['y']
    x = x.reshape([x.shape[0], -1, image_size, image_size])
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


def get_mnist_dataloader(image_size, batch_size,
                           dataroot='data/cifar10', workers=2, data_transforms=None, type='train', shuffle=True):
    if data_transforms is None:
        data_transforms = transforms.Compose([
                                transforms.Scale(image_size),
                                transforms.ToTensor()
                            ])
        
    if data_transforms is None:
        data_transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    # Get train and test data
    train_data = dset.MNIST(dataroot, train=True, download=True,
                                transform=data_transforms)
    test_data = dset.MNIST(dataroot, train=False,
                               transform=data_transforms)
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    if type == 'train':
        return train_loader
    elif type == 'db':
        return train_loader
    elif type == 'query':
        return test_loader
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