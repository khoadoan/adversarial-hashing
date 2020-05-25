import os
from os import path
from argparse import Namespace

import numpy as np
import torch
from torch import optim

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

def get_optimizer(args, params):
    if 'Adam' in args.optimizer:
        _, beta1, beta2 = args.optimizer.split('_')
        optimizer = optim.Adam(params, lr=args.lr, betas=(float(beta1), float(beta2)))
    elif 'RMSprop' in args.optimizer:
        optimizer = optim.RMSprop(params, lr=args.lr)
    else:
        raise Exception('Optimizer not supported: {}'.format(args.optimizer))

    return optimizer

def init_metrics(args):
    """Init any additional training metrics other than required losses"""
    return Namespace(precision={}, mAP={})

def reset_grad(*optimizers):
    for optimizer in optimizers:
        optimizer.zero_grad()

def get_avg_loss(loss, num_steps):
    return np.mean([loss[k] for k in sorted(loss.keys())[-num_steps:]])