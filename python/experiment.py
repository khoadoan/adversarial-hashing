import torch
from torch import nn
import torchvision.transforms as transforms

from tqdm import tqdm
import numpy as np

from train import model_helper
from models import mlp, dcgan
import data_manual

def get_numpy_data(dataloader):
    x, y = [], []
    for batch_x, batch_y in tqdm(iter(dataloader)):
        x.append(batch_x.numpy())
        y.append(batch_y.numpy())
    x = np.vstack(x)
    y = np.concatenate(y)
    
    return x, y

def get_mnist_dataloaders(image_size, batch_size, dataroot, workers=0, data_transforms=None):
    if data_transforms is None:
        data_transforms = transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.ToTensor()
                            ])
        
    train_dataloader =  data_manual.get_dataloader('mnist', image_size, batch_size,
                   dataroot=dataroot, workers=0, data_transforms=None, type='train', shuffle=True, clear_cache=True)
    db_dataloader =  data_manual.get_dataloader('mnist', image_size, batch_size,
                       dataroot=dataroot, workers=0, data_transforms=None, type='db', shuffle=False, clear_cache=True)
    query_dataloader =  data_manual.get_dataloader('mnist', image_size, batch_size,
                       dataroot=dataroot, workers=0, data_transforms=None, type='query', shuffle=False, clear_cache=True)
    
    return train_dataloader, db_dataloader, query_dataloader


def create_mlp_encoder_nobn(args, device):
    net = mlp.Encoder(args.image_size, args.nc, args.enc_layers, args.nz, 
                     activation=nn.LeakyReLU(0.2), use_bn=False, dropout=0)
    net.apply(model_helper.weights_init)
    print(net)
    optimizer = model_helper.get_optimizer(args, net.parameters())

    if torch.cuda.is_available():
        net = net.type(torch.cuda.FloatTensor)
    return net, optimizer

def create_mlp_decoder_nobn(args, device):
    net = mlp.Decoder(args.nz, args.dec_layers, args.nc, args.image_size,
                     activation=nn.LeakyReLU(0.2), 
                      output_activation=nn.Tanh(), use_bn=False, dropout=0)
    net.apply(model_helper.weights_init)
    print(net)
    optimizer = model_helper.get_optimizer(args, net.parameters())

    if torch.cuda.is_available():
        net = net.type(torch.cuda.FloatTensor)
    return net, optimizer

def create_mlp_encoder(args, device):
    net = mlp.Encoder(args.image_size, args.nc, args.enc_layers, args.nz, 
                     activation=nn.LeakyReLU(0.2), use_bn=True, dropout=0)
    net.apply(model_helper.weights_init)
    print(net)
    optimizer = model_helper.get_optimizer(args, net.parameters())

    if torch.cuda.is_available():
        net = net.type(torch.cuda.FloatTensor)
    return net, optimizer

def create_mlp_decoder(args, device):
    net = mlp.Decoder(args.nz, args.dec_layers, args.nc, args.image_size,
                      activation=nn.LeakyReLU(0.2), 
                      output_activation=nn.Tanh(), use_bn=True, dropout=0)
    net.apply(model_helper.weights_init)
    print(net)
    optimizer = model_helper.get_optimizer(args, net.parameters())

    if torch.cuda.is_available():
        net = net.type(torch.cuda.FloatTensor)
    return net, optimizer

def create_dcgan_encoder(args, device):
    net = dcgan.Encoder(args.image_size, args.nc, args.ndf, args.nz, args.n_extra_layers)
    net.apply(model_helper.weights_init)
    print(net)
    optimizer = model_helper.get_optimizer(args, net.parameters())

    if torch.cuda.is_available():
        net = net.type(torch.cuda.FloatTensor)
    return net, optimizer

def create_dcgan_decoder(args, device):
    net = dcgan.Decoder(args.nz, args.ngf, args.nc, args.image_size, args.n_extra_layers)
    net.apply(model_helper.weights_init)
    
    print(net)
    optimizer = model_helper.get_optimizer(args, net.parameters())

    if torch.cuda.is_available():
        net = net.type(torch.cuda.FloatTensor)
    return net, optimizer

def summarize_results(loss, metrics, ncols=5, figsize=(5 * 4, 3)):
    if type(loss) != dict:
        loss = loss.__dict__
        metrics = metrics.__dict__
    
    nrows = np.ceil(len(loss) / ncols).astype(int)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    for i, (k, v) in enumerate(loss.items()):
        if len(axes.shape) > 1:
            ax = axes[int(i / nrows), i % ncols]
        else:
            ax = axes[i % ncols]
        if len(v) > 0:
            x, y = list(zip(*v.items()))
            if 'grad' in k:
                y = [e[0] for e in y] #only get the max norm
            ax = sns.lineplot(x[10:], y[10:], ax=ax)
            ax.set_title('{}: {:.4f}'.format(k, np.min(y)))
    fig.suptitle('Losses')
    plt.tight_layout()
    plt.show()
    
    