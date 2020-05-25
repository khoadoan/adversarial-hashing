import os
from os import path

import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch import nn

from .utils import checkpoint_exists

def weights_init(m):
    """weight initialization"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None and m.bias.data is not None:
            nn.init.constant_(m.bias.data, 0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        # nn.init.normal_(m.weight.data, 0.0, 1)
        if m.bias is not None and m.bias.data is not None:
            nn.init.constant(m.bias.data, 0)
            
class LambdaLR:
    """https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/utils.py"""
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

def sample_z_uniform(batch_size, nz, requires_grad=False):
    # Uniform in [-1, 1]
    return torch.rand(batch_size, nz, 1, 1, requires_grad=requires_grad) * 2 - 1

def sample_z_normal(batch_size, nz, requires_grad=False):
    # Gaussian Normal
    return torch.randn(batch_size, nz, 1, 1, requires_grad=requires_grad)

def get_sample_z(sampling_type, batch_size, nz, requires_grad=False):
    if sampling_type == 'uniform':
        return sample_z_uniform(batch_size, nz, requires_grad=requires_grad)
    elif sampling_type == 'normal':
        return sample_z_normal(batch_size, nz, requires_grad=requires_grad)
    elif sampling_type == 'sign':
        return torch.sign(sample_z_normal(batch_size, nz, requires_grad=requires_grad))
    else:
        raise Exception('Unsupported sampling type: {}'.format(sampling_type))

def save_checkpoint(model_dir, iters, models, optimizers, losses, extra_args=None):
    saved_path = path.join(model_dir, 'checkpoint-{:09d}.pt'.format(iters))
    print('Save checkpoint in: {}'.format(saved_path))
    torch.save({
        'iters': iters,
        'models': {
            model_name: model.state_dict() for model_name, model in models.items()
        },
        'optimizers': {
            optimizer_name: optimizer.state_dict() for optimizer_name, optimizer in optimizers.items()
        },
        'losses': {
            loss_name: loss for loss_name, loss in losses.items()
        },
        'extra_args': extra_args
    }, saved_path)


def restore_checkpoint(saved_path, models, optimizers):
    checkpoint = torch.load(saved_path)

    iters = checkpoint['iters']

    for model_name, model in models.items():
        model.load_state_dict(checkpoint['models'][model_name])
    for optimizer_name, optimizer in optimizers.items():
        optimizer.load_state_dict(checkpoint['optimizers'][optimizer_name])
    losses = checkpoint['losses']
    extra_args = checkpoint['extra_args'] if 'extra_args' in checkpoint else None

    return iters, models, optimizers, losses, extra_args


def restore_model(saved_path, model, model_name, optimizer, optimizer_name):
    checkpoint = torch.load(saved_path)

    if model_name in checkpoint['models']:
        model.load_state_dict(checkpoint['models'][model_name])
    else:
        raise Exception('Model {} does not exist'.format(model_name))

    if optimizer_name in checkpoint['optimizers']:
        optimizer.load_state_dict(checkpoint['optimizers'][optimizer_name])
    else:
        raise Exception('Optimizer {} does not exist'.format(optimizer_name))

    return model, optimizer


def restore_latest_checkpoint(model_dir, models, optimizers):
    latest_checkpoint = sorted(filter(lambda filename: 'checkpoint' in filename, os.listdir(model_dir)))[-1]
    print('Restore from checkpoint: {}'.format(latest_checkpoint))
    return restore_checkpoint(path.join(model_dir, latest_checkpoint), models, optimizers)


def checkpoint_exists(model_dir):
    return len(list(filter(lambda filename: 'checkpoint' in filename, os.listdir(model_dir)))) > 0


def get_optimizer(args, params, epoch=0):
    if 'Adam' in args.optimizer:
        print('Create Adam optimizer: lr={}, beta1={}, beta2={}'.format(args.lr, args.beta1, args.beta2))
        _, beta1, beta2 = args.optimizer.split('_')
        optimizer = optim.Adam(params, lr=args.lr, betas=(float(beta1), float(beta2)))
    elif 'RMSprop' in args.optimizer:
        print('Create Adam optimizer: lr={}'.format(args.lr))
        optimizer = optim.RMSprop(params, lr=args.lr)
    else:
        raise Exception('Optimizer not supported: {}'.format(args.optimizer))
    
    return optimizer

def get_lr_decay(args, optimizer, epoch ):
    if optimizer:
        if args.decay_type is None:
            return None
        elif args.decay_type == 'exponential':
            if args.decay_rate < 1.0:
                print('Setting exponential lr decay: current={}, decay_rate={}'.format(epoch, args.decay_rate))
                lr_scheduler = ExponentialLR(optimizer, args.decay_rate, last_epoch=epoch)
                return lr_scheduler
        elif args.decay_type == 'lambda':
                    # Learning rate update schedulers
    #         lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    #             optimizer, lr_lambda=LambdaLR(args.num_epochs, epoch, args.decay_epoch).step
    #         )
            return None #not handling right now
        else:
            raise Exception('not supported lr decay type: {}'.format(type))
    return None

def create_conv_encoder(args, device, encoder_creator):
    net = encoder_creator(args.image_size, args.nc, args.ndf, args.nz)
    print(net)
    optimizer = get_optimizer(args, net.parameters())

    if torch.cuda.is_available():
        net = net.type(torch.cuda.FloatTensor)
    return net, optimizer

def create_conv_decoder(args, device, decoder_creator):
    net = decoder_creator(args.nz, args.ngf, args.nc, args.image_size)
    print(net)
    optimizer = get_optimizer(args, net.parameters())

    if torch.cuda.is_available():
        net = net.type(torch.cuda.FloatTensor)
    return net, optimizer


def create_any_encoder(args, device, encoder_creator):
    net = encoder_creator(args)
    print(net)
    optimizer = get_optimizer(args, net.parameters())

    if torch.cuda.is_available():
        net = net.type(torch.cuda.FloatTensor)
    return net, optimizer

def create_any_decoder(args, device, decoder_creator):
    net = decoder_creator(args)
    print(net)
    optimizer = get_optimizer(args, net.parameters())

    if torch.cuda.is_available():
        net = net.type(torch.cuda.FloatTensor)
    return net, optimizer

