import os
from os import path
from argparse import Namespace
from matplotlib import pyplot as plt
import pickle

from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import seaborn as sns

from .utils import checkpoint_exists, save_checkpoint
from .model_helper import restore_latest_checkpoint, get_sample_z, get_lr_decay
from metrics import ranking
import viz

def get_arg(args, name):
    if name in args and args.__dict__[name] is not None:
        return args.__dict__[name]
    else:
        return None

def get_training_paths(args, get_training_dir=None):
    if get_training_dir is None:
        ae_layers = '' if args.enc_layers is None else '_enc={}'.format('_'.join([str(e) for e in args.enc_layers]))
        adv_layers = '_'.join(map(lambda e: str(e), args.d_dims))
        clamp_value = args.clamp_value if args.clamp_value else None
        clip_value = args.clip_value if args.clip_value else None

        if 'ngf' in args and args.ngf:
            assert 'enc_layers' not in args or args.enc_layers is not None
            ae_layers = 'ngf={}-ndf={}'.format(args.ngf, args.ndf)
            
        training_dir = path.join(args.training_dir, args.dataset,
                                 'isize={}-nz={}_{}-ae={}-freqA={}-freqG={}-useD={}-adv={}-clamp={}-clip={}-freqD={}-bsize={}-lr={}-decay={}_{}-optim={}-ALPHA={}-LAMBDA={}-BETA={}'.format(
                                     args.image_size, args.nz, args.sampling_type, ae_layers,
                                     args.freqA, args.freqG, 
                                     args.useD, adv_layers, clamp_value, clip_value, args.freqD,
                                     args.batch_size, args.lr, args.decay_type, args.decay_rate, args.optimizer,
                                     args.ALPHA, args.LAMBDA, args.BETA))
    else:
        training_dir = get_training_dir(args)

    model_dir = path.join(training_dir, 'saved_model')
    result_dir = path.join(training_dir, 'results')

    return model_dir, result_dir

def init_metrics(args):
    """Init any additional training metrics other than required losses"""
    return Namespace(precision={}, mAP={}, mymAP={})

def init_losses(args):
    """Init losses during training: A-reconstruction loss, D-discriminator loss, E-generator loss"""
    return Namespace(A={}, D={}, G={}, T={}, S={}, enc_grad_norm={})


def create_or_load(args, device, create_encoder_func, create_decoder_func, create_discriminator_func,
                   init_training_metrics_func, iters_per_epoch, get_training_dir=None):
    model_dir, result_dir = get_training_paths(args, get_training_dir=get_training_dir)

    netEnc, optimizerEnc = create_encoder_func(args, device)
    netDec, optimizerDec = create_decoder_func(args, device)

    if args.useD:
        print('Using discriminator!')
        netDisc, optimizerDisc = create_discriminator_func(args, device)
    else:
        netDisc, optimizerDisc = None, None

    if not (path.exists(model_dir) and checkpoint_exists(model_dir)):
        print('Creating fresh model in {}'.format(model_dir))
        if not path.exists(model_dir):
            os.makedirs(model_dir)
            os.makedirs(result_dir)
        metrics = init_training_metrics_func(args)
        loss = init_losses(args)
        iters = 0
    else:
        print('Loading from existing model in {}'.format(model_dir))
        if args.useD and not args.pretrainedD:
            iters, models, optimizers, losses, extra_args = restore_latest_checkpoint(
                model_dir,
                {'netEnc': netEnc, 'netDec': netDec, 'netDisc': netDisc},
                {'optimizerEnc': optimizerEnc, 'optimizerDec': optimizerDec, 'optimizerDisc': optimizerDisc})
            loss = Namespace(**losses)
        else:
            iters, models, optimizers, losses, extra_args = restore_latest_checkpoint(
                model_dir, {'netEnc': netEnc, 'netDec': netDec}, {'optimizerEnc': optimizerEnc, 'optimizerDec': optimizerDec})
            loss = Namespace(**losses)

        if extra_args and 'metrics' in extra_args:
            metrics = extra_args['metrics']
        else:
            metrics = init_training_metrics_func(args)

    epoch = iters // iters_per_epoch - 1
    lrSchedulerEnc = get_lr_decay(args, optimizerEnc, epoch )
    lrSchedulerDec = get_lr_decay(args, optimizerDec, epoch)
    lrSchedulerDisc = get_lr_decay(args, optimizerDisc, epoch)
        
    return model_dir, result_dir, iters, netEnc, netDec, netDisc, optimizerEnc, optimizerDec, optimizerDisc, lrSchedulerEnc, lrSchedulerDec, lrSchedulerDisc, loss, metrics

def get_batch_data(data_iter, nz, sampling_type):
    batch_indices, batch_x, batch_ss, batch_y = data_iter.next()
    z = get_sample_z(sampling_type, batch_x.size(0), nz)

    return batch_indices, batch_x, batch_ss, z

def reset_grad(*optimizers):
    for optimizer in optimizers:
        optimizer.zero_grad()

def get_avg_loss(loss, num_steps):
    return np.mean([loss[k] for k in sorted(loss.keys())[-num_steps:]])

def save(model_dir, iters, netEnc, optimizerEnc, netDec, optimizerDec, netDisc, optimizerDisc, loss, metrics=None):
    if netDisc and optimizerDisc:
        save_checkpoint(model_dir, iters,
            {'netEnc': netEnc, 'netDec': netDec, 'netDisc': netDisc},
            {'optimizerEnc': optimizerEnc, 'optimizerDec': optimizerDec, 'optimizerDisc': optimizerDisc},
            loss.__dict__,
            {'metrics': metrics}
        )
    else:
        save_checkpoint(model_dir, iters,
            {'netEnc': netEnc, 'netDec': netDec},
            {'optimizerEnc': optimizerEnc, 'optimizerDec': optimizerDec},
            loss.__dict__,
            {'metrics': metrics}
        )

def get_code_and_label(device, dataloader, netEnc, threshold):
    codes, labels = [], []
    for batch_indices, batch_x, batch_ss, batch_y in tqdm(iter(dataloader)):
        batch_b = netEnc(batch_x.to(device))
        batch_b = batch_b.view(batch_b.size(0), -1)
        codes.append(np.greater_equal(batch_b.detach().cpu().numpy(), threshold))
        labels.append(batch_y.cpu().numpy())
    codes = np.vstack(codes)
    labels = np.concatenate(labels)
    return codes, labels

def get_activation_and_label(device, dataloader, netEnc, threshold, size=999999999):
    activations, labels = [], []
    i = 0
    for batch_indices, batch_x, batch_ss, batch_y in tqdm(iter(dataloader)):
        batch_x = batch_x.to(device)
        if i + batch_x.size(0) > size:
            batch_x = batch_x[:size - i, :, :, :]
            batch_y = batch_y[:size - i]
        batch_b = netEnc(batch_x)
        batch_b = batch_b.view(batch_b.size(0), -1)
        activations.append(batch_b.detach().cpu().numpy())
        labels.append(batch_y.cpu().numpy())
        i += batch_x.size(0)
        if i >= size:
            break
    activations = np.vstack(activations)
    labels = np.concatenate(labels)
    return activations, labels

def run_internal_eval_by_batch(device, dataloader, netEnc, netDec, threshold, size=999999999, denoising_std=None):
    inputs, activations, labels, reconstructs = [], [], [], []
    i = 0
    for batch_indices, batch_x, batch_ss, batch_y in tqdm(iter(dataloader)):
        batch_x = batch_x.to(device)
        if denoising_std:
            batch_x = batch_x + torch.randn(batch_x.shape).to(device) * denoising_std
#         inputs.append(batch_x.detach().cpu().numpy())
        if i + batch_x.size(0) > size:
            batch_x = batch_x[:size - i, :, :, :]
            batch_y = batch_y[:size - i]
        
        batch_b = netEnc(batch_x)
        batch_reconstruct = torch.sigmoid(netDec(batch_b))
        
        batch_b = batch_b.view(batch_b.size(0), -1)
        
        activations.append(batch_b.detach().cpu().numpy())
        labels.append(batch_y.cpu().numpy())
        inputs.append(batch_x.detach().cpu())
        reconstructs.append(batch_reconstruct.detach().cpu())
#         reconstructs.append(batch_reconstruct.detach().cpu().numpy())
        
        i += batch_x.size(0)
        if i >= size:
            break
#     inputs = inputs.vstack(inputs)
    activations = np.vstack(activations)
    labels = np.concatenate(labels)
#     pickle.dump((inputs, reconstructs), open('/tmp/data.pkl', 'wb'))
#     reconstructs = np.vstack(reconstructs)
    return inputs, activations, labels, reconstructs

def run_internal_eval(args, device, result_dir, iters, netEnc, netDec, netDisc, metrics, get_dataloader_func, compute_A_loss_func,
                           compute_G_loss_func, compute_D_loss_func, data_transforms=None, threshold=0):
    test_dataloader = get_dataloader_func(args.dataset, args.image_size, args.batch_size, args.dataroot,
                                     data_transforms=data_transforms, type='query')
    test_inputs, test_activations, test_labels, test_reconstruct = run_internal_eval_by_batch(device, test_dataloader, netEnc, netDec, threshold, denoising_std = (0 if not ('denoising' in args and args.denoising) else args.denoising_std))
    z = get_sample_z(args.sampling_type, test_activations.shape[0], args.nz)

#     fig, axes = plt.subplots(1, 2, figsize=(8, 3))
#     sns.distplot(z.view(-1), kde=True, ax=axes[0])
#     sns.distplot(test_activations.reshape([-1]), kde=True, ax=axes[1])
#     axes[0].set_title('Real')
#     axes[1].set_title('Generated')
    
    if test_activations.shape[1] == 2:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        if args.sampling_type == 'sign':
            sns.distplot(z, kde=False, hist=True, ax=ax, color='blue', label='z')
            sns.distplot(np.tanh(test_activations), hist=False, kde=True, ax=ax, color='red', label='b')
        else:
            sns.distplot(z, kde=True, hist=False, ax=ax, color='blue', label='z')
            sns.distplot(test_activations, hist=False, kde=True, ax=ax, color='red', label='b')
        plt.tight_layout()
        plt.show()
    else:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        if args.sampling_type == 'sign':
            sns.distplot(z.view(-1), kde=False, hist=True, ax=ax, color='blue', label='Real')
            sns.distplot(np.tanh(test_activations.reshape([-1])), hist=True, kde=False, ax=ax, color='red', label='Generated')
        else:
            sns.distplot(z.view(-1), kde=True, hist=False, ax=ax, color='blue', label='Real')
            sns.distplot(test_activations.reshape([-1]), hist=False, kde=True, ax=ax, color='red', label='Generated')

        plt.show()
        
        
    
    # viz.plot_torch_images(test_inputs[0][:10, :, :, :], figsize=(16, 16), n=10, nrow=10, title='Input', normalize=True)
    # viz.plot_torch_images(test_reconstruct[0][:10, :, :, :], figsize=(16, 16), n=10, nrow=10, title='Reconstruct', normalize=True)

def run_external_eval(args, device, result_dir, iters, netEnc, metrics, get_dataloader_func, data_transforms=None, threshold=0):
    db_dataloader = get_dataloader_func(args.dataset, args.image_size, args.batch_size, args.dataroot,
                                     data_transforms=data_transforms, type='db')
    test_dataloader = get_dataloader_func(args.dataset, args.image_size, args.batch_size, args.dataroot,
                                     data_transforms=data_transforms, type='query')

    db_activations, db_labels = get_activation_and_label(device, db_dataloader, netEnc, threshold)
    test_activations, test_labels = get_activation_and_label(device, test_dataloader, netEnc, threshold)

    db_codes = (db_activations > threshold).astype(np.float) * 2 - 1
    test_codes = (test_activations > threshold).astype(np.float) * 2 - 1
    db_labels = ranking.one_hot_label(db_labels)
    test_labels = ranking.one_hot_label(test_labels)

    prec, mAP = ranking.calculate_all_metrics(db_codes, db_labels, test_codes, test_labels, args.R, dist_type='hamming')
    
    filename = path.join(result_dir, 'evaluation_{}.pkl'.format(iters))
    pickle.dump((args, db_codes, db_labels, test_codes, test_labels, args.R, prec, mAP), open(filename, 'wb'))
            
    for R in args.R:
        print('[{0}] EXTERNAL (Skip=Y): \tPrec@{1} {2:.5f}, mAP@{1} {3:.5f} (Size {4}/{5})'.format(iters, R, prec[R][0], mAP[R][0], test_activations.shape[0], db_activations.shape[0]))
    for R in args.R:
        print('[{0}] EXTERNAL (Skip=N): \tPrec@{1} {2:.5f}, mAP@{1} {3:.5f} (Size {4}/{5})'.format(iters, R, prec[R][1], mAP[R][1], test_activations.shape[0], db_activations.shape[0]))

    metrics.precision[iters] = prec
    metrics.mAP[iters] = mAP

def get_numpy_data(dataloader):
    x, y = [], []
    for _, batch_x, _, batch_y in tqdm(iter(dataloader)):
        x.append(batch_x.numpy())
        y.append(batch_y.numpy())
    x = np.vstack(x)
    y = np.concatenate(y)
    
    return x, y

def run_external_eval_with_original_data(args, device, result_dir, iters, netEnc, metrics, get_dataloader_func, data_transforms=None, threshold=0):
    """Use this function for some analysis"""
    db_dataloader = get_dataloader_func(args.dataset, args.image_size, args.batch_size, args.dataroot,
                                     data_transforms=data_transforms, type='db')
    test_dataloader = get_dataloader_func(args.dataset, args.image_size, args.batch_size, args.dataroot,
                                     data_transforms=data_transforms, type='query')

    db_activations, db_labels = get_activation_and_label(device, db_dataloader, netEnc, threshold)
    test_activations, test_labels = get_activation_and_label(device, test_dataloader, netEnc, threshold)

    db_codes = (db_activations > threshold).astype(np.float) * 2 - 1
    test_codes = (test_activations > threshold).astype(np.float) * 2 - 1
    db_labels = ranking.one_hot_label(db_labels)
    test_labels = ranking.one_hot_label(test_labels)

    prec, mAP = ranking.calculate_all_metrics(db_codes, db_labels, test_codes, test_labels, args.R, dist_type='hamming')
    
    filename = path.join(result_dir, 'evaluation_{}.pkl'.format(iters))
    pickle.dump((args, 
                 get_numpy_data(db_dataloader)[0], db_codes, db_labels, 
                 get_numpy_data(test_dataloader)[0], test_codes, test_labels, args.R, prec, mAP), open(filename, 'wb'))
            
    for R in args.R:
        print('[{0}] EXTERNAL: \tPrec@{1} {2:.5f}, mAP@{1} {3:.5f} (Size {4}/{5})'.format(iters, R, prec[R][1], mAP[R][1], test_activations.shape[0], db_activations.shape[0]))
        
    metrics.precision[iters] = prec
    metrics.mAP[iters] = mAP

        
def step(optimizer, lrScheduler=None):
    if lrScheduler:
        lrScheduler.step()
    else:
        optimizer.step()

def get_inf_random_batch(dataloader):
    while(True):
        for batch_x, batch_y in iter(dataloader):
            yield batch_x

def flip_w(images):
    return images.flip(2)
def flip_h(images):
    return images.flip(3)
def flip_all(images):
    return images.flip((2, 3))
def rotate_90(images, k):
    return images.rot90(k, (2,3))
PAIRWISE_DIST = nn.PairwiseDistance(2)            

def get_wgan_D_update_frequency(args, epoch, iters, g_iters, d_iters):
    # train the discriminator Diters times
    if g_iters < 25 or g_iters % 500 == 0:
        return 100
    else:
        return args.freqD
def collect_grad(net):
    """ Collect gradient information of a net after backward call """
    grads = []
    for p in net.parameters():
        if p.grad is not None:
            grads.append(p.grad.data.norm().item())
    return np.max(grads), np.mean(grads), np.std(grads)

def train(args, create_encoder_func, create_decoder_func, create_discriminator_func, get_dataloader_func,
          compute_A_loss_func, compute_G_loss_func, compute_D_loss_func,
          run_internal_eval_func=None,
          run_external_eval_func=None,
          data_transforms=None,
          device=None,
          get_D_update_frequency_func = lambda args, epoch, iters, g_iters, d_iters: 1,
          init_training_metrics_func=init_metrics, get_training_dir=None):
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    dataloader = get_dataloader_func(args.dataset, args.image_size, args.batch_size, args.dataroot, data_transforms=data_transforms, type='train')
    if get_arg(args, 'triplet'):
        random_data_generator = get_inf_random_batch(
            get_dataloader_func(args.dataset, args.image_size, args.batch_size, args.dataroot, data_transforms=data_transforms, type='train'))
    
    iters_per_epoch = len(dataloader)
            
    # Load or create new model
    model_dir, result_dir, iters, netEnc, netDec, netDisc, optimizerEnc, optimizerDec, optimizerDisc, \
    lrSchedulerEnc, lrSchedulerDec, lrSchedulerDisc, loss, metrics \
        = create_or_load(args, device, create_encoder_func, create_decoder_func, create_discriminator_func,
                         init_training_metrics_func, iters_per_epoch, get_training_dir=get_training_dir)


    print('BEGIN TRAINING...')
    a_iters = 0
    d_iters = 0
    g_iters = 0
    last_stat_iter = 0
    last_internal_eval_iter = 0
    last_external_eval_iter = 0

    def compute_pair_loss(indices, z_x, ss):
        z_x = torch.tanh(z_x)
        input_label = ss[:, indices].to(device)
        H = z_x @ torch.transpose(z_x, 1, 0) / z_x.shape[1]
        loss = torch.mul(torch.abs(input_label),  (H - input_label) ** 2).mean()
        return loss * args.BETA if args.BETA is not None else 0
    for epoch in range(int(np.ceil(iters / len(dataloader))), args.num_epochs):
        data_iter = iter(dataloader)
        i = 0  # counter for batch number
                
        while i < len(dataloader):
            ##############################################
            ################# Train G ####################

            # netEnc.train()
#             netDec.train()
            
            ### Train Autoencoder
            if i < len(dataloader):
                if args.ALPHA is not None and args.ALPHA > 0:
                    a_iters = args.freqA if 'freqA' in args and args.freqA else 1
                    for j in range(a_iters):
                        if i < len(dataloader):
                            a_iters += 1
                            iters += 1
                            i += 1

                            # Get data
                            indices, x, ss, z = get_batch_data(data_iter, args.nz, args.sampling_type)
                            if get_arg(args, 'denoising') and args.denoising_std > 0:
                                x_tilde = x + torch.randn(x.shape) * args.denoising_std
                            else:
                                x_tilde = x

                            # Encode
                            b = netEnc(x_tilde.to(device))

                            # Decode
                            xhat = netDec(b)
                            a_loss = args.ALPHA * compute_A_loss_func(args, device, x, xhat)               
                            
                            total_loss = a_loss
                            if args.BETA is not None and args.BETA > 0:
                                pair_loss = compute_pair_loss(indices, b, ss)
                                loss.S[iters] = pair_loss.item() 
                                total_loss = a_loss + pair_loss

                            reset_grad(optimizerEnc, optimizerDec)
                            total_loss.backward()
                            optimizerDec.step()
                            optimizerEnc.step()
                            loss.A[iters] = a_loss.item()

                            #Triplet loss if define
                            if get_arg(args, 'triplet'):
                                x_pos = (x + torch.randn(x.shape) * 0.3).to(device)
                                x_neg = next(random_data_generator).to(device)

                                b = netEnc(x.to(device))
                                b_pos = netEnc(x_pos)
                                b_neg = netEnc(x_neg)

                                neg_dist = PAIRWISE_DIST(b, b_neg)
                                noise_dist = PAIRWISE_DIST(b, b_pos)
                                if get_arg(args, 'triplet_rotate'):
                                    b_pos_flip1 = netEnc(flip_w(x_pos))
                                    b_pos_flip2 = netEnc(flip_h(x_pos))
                                    b_pos_flip3 = netEnc(flip_all(x_pos))

                                    b_pos_rotate1 = netEnc(rotate_90(x_pos, 1))
                                    b_pos_rotate2 = netEnc(rotate_90(x_pos, 2))
                                    b_pos_rotate3 = netEnc(rotate_90(x_pos, 3))

                                    rotate_dist = (PAIRWISE_DIST(b, b_pos_flip1) + PAIRWISE_DIST(b, b_pos_flip2) + PAIRWISE_DIST(b, b_pos_flip3) + 
                                                  PAIRWISE_DIST(b, b_pos_rotate1) + PAIRWISE_DIST(b, b_pos_rotate2) + PAIRWISE_DIST(b, b_pos_rotate3)) / 6

                                    t_loss = args.triplet_lambda * F.relu(args.triplet_hinge -  neg_dist + noise_dist + rotate_dist).mean()
                                else:
                                    t_loss = args.triplet_lambda * F.relu(args.triplet_hinge -  neg_dist + noise_dist).mean()

                                reset_grad(optimizerEnc, optimizerDec)
                                t_loss.backward()
                                optimizerEnc.step()
                                loss.T[iters] = t_loss.item()                    

            if args.LAMBDA is not None and args.LAMBDA > 0: #We run adversarial unless we need to
                ### Train Discriminator
                if args.useD and args.trainD:
                    # Get data
                    freqD = get_D_update_frequency_func(args, epoch, iters,g_iters,d_iters)
                    for _ in range(args.freqD):
                        if i < len(dataloader):
                            d_iters += 1
                            iters += 1
                            i += 1

                            # clamp parameters to a cube
                            if args.clamp_value:
                                for p in netDisc.parameters():
                                    p.data.clamp_(-args.clamp_value, args.clamp_value)

                            indices, x, ss, z = get_batch_data(data_iter, args.nz, args.sampling_type)
                            # Encode
                            b = netEnc(x.to(device))
                            if args.useD:
                                # Through D
                                netDisc.train()
                                gen_features, gen_output = netDisc(b.view(b.size(0), -1))  # get image features from Discriminator
                                real_features, real_output = netDisc(z.view(z.size(0), -1).to(device))
                            else:
                                real_features, gen_features = z.to(device).view(z.size(0), -1), b.view(b.size(0), -1)
                                real_output, gen_output = None, None

                            d_loss = compute_D_loss_func(args, device, real_features, gen_features, real_output, gen_output)

                            reset_grad(optimizerDisc)
                            d_loss.backward()
                            optimizerDisc.step()
                            loss.D[iters] = d_loss.item()

                ### Train Encoder/Generator
                # if i < len(dataloader):
                for _ in range(args.freqG):
                    if i < len(dataloader):
                        g_iters += 1
                        iters += 1
                        i += 1

                        indices, x, ss, z = get_batch_data(data_iter, args.nz, args.sampling_type)
                        b = netEnc(x.to(device))
                        if args.useD:
                            reset_grad(optimizerDisc)
                            # Through D
                            netDisc.train()
                            gen_features, gen_output = netDisc(b.view(b.size(0), -1))  # get image features from Discriminator
                            real_features, real_output = netDisc(z.view(z.size(0), -1).to(device))
                        else:
                            real_features, gen_features = z.to(device).view(z.size(0), -1), b.view(b.size(0), -1)
                            real_output, gen_output = None, None

                        g_loss = compute_G_loss_func(args, device, real_features, gen_features, 
                                                    real_output, gen_output)
                        total_loss = g_loss
                        if args.BETA is not None and args.BETA > 0:
                            pair_loss = compute_pair_loss(indices, b, ss)
                            loss.S[iters] = pair_loss.item() 
                            total_loss = g_loss + pair_loss
                        
                        reset_grad(optimizerEnc)
                        total_loss.backward()
                        optimizerEnc.step()

                        loss.enc_grad_norm[iters] = collect_grad(netEnc)
                        loss.G[iters] = g_loss.item()

            if iters - last_stat_iter > args.iters_per_stat:
                netEnc.eval()
                netDec.eval()
                if args.useD:
                    netDisc.eval()
                last_stat_iter = iters
                print('[%d/%d][%d/%d][%d]\tLoss(A/G/D/S/T): %.4f/%.4f/%.4f/%.4f/%.4f, lr %.6f'
                      % (epoch, args.num_epochs, i, len(dataloader), iters,
                         get_avg_loss(loss.A, args.iters_per_stat),
                         get_avg_loss(loss.G, args.iters_per_stat), 
                         get_avg_loss(loss.D, args.iters_per_stat),
                         get_avg_loss(loss.S, args.iters_per_stat),
                         get_avg_loss(loss.T, args.iters_per_stat), 
                         args.lr if not lrSchedulerEnc else lrSchedulerEnc.get_lr()[0]))

            if run_internal_eval_func and iters - last_internal_eval_iter > args.iters_per_internal_eval:
                netEnc.eval()
                netDec.eval()
                if args.useD:
                    netDisc.eval()
                
                last_internal_eval_iter = iters
                if args.useD and not args.pretrainedD:
                    save(model_dir, iters, netEnc, optimizerEnc, netDec, optimizerDec, netDisc, optimizerDisc, loss, metrics)
                else:
                    save(model_dir, iters, netEnc, optimizerEnc, netDec, optimizerDec, None, None, loss, metrics)

                run_internal_eval_func(args, device, result_dir, iters, netEnc, netDec, netDisc, metrics,
                                       get_dataloader_func, compute_A_loss_func, compute_G_loss_func, compute_D_loss_func,
                                       data_transforms=data_transforms, threshold=args.threshold)

            if run_external_eval_func and iters - last_external_eval_iter > args.iters_per_external_eval:
                netEnc.eval()
                netDec.eval()
                if args.useD:
                    netDisc.eval()
                
                last_external_eval_iter = iters
                run_external_eval_func(args, device, result_dir, iters, netEnc, metrics, get_dataloader_func,
                                       data_transforms=data_transforms, threshold=args.threshold)
        if lrSchedulerEnc:
            lrSchedulerEnc.step()
        if lrSchedulerDec:
            lrSchedulerDec.step()
        if lrSchedulerDisc:
            lrSchedulerDisc.step()

    #### Print the stats the last time
    if args.useD and not args.pretrainedD:
        save(model_dir, iters, netEnc, optimizerEnc, netDec, optimizerDec, netDisc, optimizerDisc, loss, metrics)
    else:
        save(model_dir, iters, netEnc, optimizerEnc, netDec, optimizerDec, None, None, loss, metrics)

    print('[FINAL]\tLoss(A/G/D/S/T): %.4f/%.4f/%.4f/%.4f/%.4f'
          % (
             get_avg_loss(loss.A, args.iters_per_stat),
             get_avg_loss(loss.G, args.iters_per_stat),
             get_avg_loss(loss.D, args.iters_per_stat),
             get_avg_loss(loss.S, args.iters_per_stat),
             get_avg_loss(loss.T, args.iters_per_stat)))

    netEnc.eval()
    netDec.eval()
    if args.useD:
        netDisc.eval()
    run_internal_eval_func(args, device, result_dir, iters, netEnc, netDec, netDisc, metrics,
                                       get_dataloader_func, compute_A_loss_func, compute_G_loss_func, compute_D_loss_func,
                                       data_transforms=data_transforms, threshold=args.threshold)
    if run_external_eval_func:
        run_external_eval_func(args, device, result_dir, iters, netEnc, metrics, get_dataloader_func,
                                       data_transforms=data_transforms, threshold=args.threshold)

    print('FINISH TRAINING!!!')
    return loss, metrics