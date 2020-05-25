import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(self, nz, hidden_dims, nc, isize, 
                 activation=nn.LeakyReLU(0.2),
                 output_activation=None,
                 use_bn=False, dropout=0):
        super(Decoder, self).__init__()
        self.nc = nc
        self.isize = isize

        output_dim = nc * isize * isize
        
        self.main = []
        current_dim = nz
        for idx, dim in enumerate(hidden_dims):
            self.main += [nn.Linear(current_dim, dim)]
            if use_bn:
                self.main += [nn.BatchNorm1d(dim)]
            self.main += [activation]
            if dropout:
                self.main += [nn.Dropout(dropout)]
            current_dim = dim
            
        self.main += [nn.Linear(current_dim, output_dim)]
        if output_activation:
            self.main += [output_activation]

        for idx, module in enumerate(self.main):
            self.add_module(str(idx), module)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.main:
            x = layer(x)
        return x.view(x.size(0), self.nc, self.isize, self.isize)

class Encoder(nn.Module):
    def __init__(self, isize, nc, hidden_dims, nz, 
                 output_activation=None,
                 activation=nn.LeakyReLU(0.2),
                 use_bn=False, dropout=0):
        super(Encoder, self).__init__()

        input_dim = nc * isize * isize
        
        self.main = []
        current_dim = input_dim
        for idx, dim in enumerate(hidden_dims):
            self.main += [nn.Linear(current_dim, dim)]
            if use_bn:
                self.main += [nn.BatchNorm1d(dim)]
            self.main += [activation]
            if dropout:
                self.main += [nn.Dropout(dropout)]
            current_dim = dim
            
        self.main.append(nn.Linear(current_dim, nz))
        if output_activation:
            slef.main.append(output_activation)
        
        for idx, module in enumerate(self.main):
            self.add_module(str(idx), module)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.main:
            x = layer(x)
        return x