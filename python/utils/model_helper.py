import os
from os import path

import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR

from .utils import checkpoint_exists

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
    else:
        raise Exception('Unsupported sampling type: {}'.format(sampling_type))

def get_arg(args, name):
    if name in args and args.__dict__[name] is not None:
        return args.__dict__[name]
    else:
        return None

def save_checkpoint(model_dir, iters, models, optimizers, losses, extra_args=None):
    saved_path = path.join(model_dir, 'checkpoint-{0:05d},{1:05d}.pt'.format(iters[0],iters[1]))
    print('Save checkpoint in: {}'.format(saved_path))
    torch.save({
        'iters': iters,
        'models': {
            model_name: model.state_dict() for model_name, model in models.items()
             if model is not None
        },
        'optimizers': {
            optimizer_name: optimizer.state_dict() for optimizer_name, optimizer in optimizers.items()
             if optimizer is not None
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
        if model is not None:
            model.load_state_dict(checkpoint['models'][model_name])
    for optimizer_name, optimizer in optimizers.items():
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizers'][optimizer_name])
    losses = checkpoint['losses']
    extra_args = checkpoint['extra_args'] if 'extra_args' in checkpoint else None

    return iters, models, optimizers, losses, extra_args

def get_wgan_D_update_frequency(args, epoch, g_iters, d_iters):
    # train the discriminator Diters times
    if g_iters < 25 or g_iters % 500 == 0:
        return 100
    else:
        return args.freqD

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
        _, beta1, beta2 = args.optimizer.split('_')
        optimizer = optim.Adam(params, lr=args.lr, betas=(float(beta1), float(beta2)))
    elif 'RMSprop' in args.optimizer:
        optimizer = optim.RMSprop(params, lr=args.lr)
    else:
        raise Exception('Optimizer not supported: {}'.format(args.optimizer))
    
    return optimizer

def get_lr_decay(args, optimizer, epochs):
    if optimizer:
        if args.decay_type is None:
            return None
        elif args.decay_type == 'exponential':
            if args.decay_rate < 1.0:
                print('Setting exponential lr decay: current={}, decay_rate={}'.format(epochs, args.decay_rate))
                lr_scheduler = ExponentialLR(optimizer, args.decay_rate, last_epoch=-1)
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

def get_training_paths(args, get_training_dir=None, feature_extractor=False):
    if get_training_dir is None:
        training_dir = path.join(args.training_dir, args.dataset,
                                 'Only_A={}-isize={}-nz={}-ngf={}-d_hidden_layers={}-bsize={}-lr={}-decay={}-optim={}-useD={}-clamp={}-clip={}-LAMBDA={}-\
denoising={}-feature_extractor={}'.format(
                                     args.only_A, args.image_size, args.nz, args.ngf, '_'.join(map(lambda e: str(e), args.d_dims)),
                                     args.batch_size, args.lr, args.decay_rate, args.optimizer,
                                     args.useD,
                                     args.clamp_value if args.clamp_value else None,
                                     args.clip_value if args.clip_value else None,
                                     args.LAMBDA,args.denoising,feature_extractor
                                 ))
    else:
        training_dir = get_training_dir(args)

    model_dir = path.join(training_dir, 'saved_model')
    result_dir = path.join(training_dir, 'results')

    return model_dir, result_dir
