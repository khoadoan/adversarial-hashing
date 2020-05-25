import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix

from lap import lapjv
from lap import lapmod

def compute_frobenius_pairwise_distances_torch(X, Y, device, p=1, normalized=True):
    """Compute pairwise distances between 2 sets of points"""
    assert X.shape[1] == Y.shape[1]

    d = X.shape[1]
    dists = torch.zeros(X.shape[0], Y.shape[0], device=device)

    for i in range(X.shape[0]):
        if p == 1:
            dists[i, :] = torch.sum(torch.abs(X[i, :] - Y), dim=1)
        elif p == 2:
            dists[i, :] = torch.sum((X[i, :] - Y) ** 2, dim=1)
        else:
            raise Exception('Distance type not supported: p={}'.format(p))

        if normalized:
            dists[i, :] = dists[i, :] / d

    return dists

def wasserstein1d(x, y):
    """Compute wasserstein loss in 1D"""
    x1, _ = torch.sort(x, dim=0)
    y1, _ = torch.sort(y, dim=0)
    z = (x1-y1).view(-1)
    n = x.size(0)
    return torch.dot(z, z)/n

def compute_g_sliced_component_wise_wasserstein_loss(args, device, real_features, gen_features, real_output, gen_output, eval_metrics=None, netG=None):
    b_size = real_features.size(0)
    real_features = real_features.view(b_size, -1)
    gen_features = gen_features.view(b_size, -1)
    dim = real_features.size(1)
    
    x1, _ = torch.sort(real_features, dim=0)
    y1, _ = torch.sort(gen_features, dim=0)
    z = (x1-y1).view(-1)
    
    gloss = torch.dot(z, z)/(dim * b_size)
    if args.LAMBDA is not None:
        gloss = gloss * args.LAMBDA
    
    return gloss


def compute_g_sliced_wasserstein_loss(args, device, real_features, gen_features, real_output, gen_output, eval_metrics=None, netG=None):
    bsize = gen_features.size(0)
    if len(gen_features.shape) > 2:
        gen_features = gen_features.view(bsize, -1)
        real_features = real_features.view(bsize, -1)
        
    dim = gen_features.size(1)

    xgen = gen_features
    xreal = real_features
        
    theta = torch.randn((dim, args.nprojections),
                            requires_grad=False,
                            device=device)
    theta = theta/torch.norm(theta, dim=0)[None, :]
    xgen_1d = xgen.view(-1, dim)@theta
    xreal_1d = xreal.view(-1, dim)@theta
    
    gloss = wasserstein1d(xreal_1d, xgen_1d) / (args.nprojections * bsize)# / dim #normaliz by dim and bsize
    
    return gloss

def compute_g_max_sliced_wasserstein_loss(args, device, real_features, gen_features, real_output, gen_output):
    gloss = wasserstein1d(real_output, gen_output)
    return gloss
    

def compute_d_xentropy_loss(args, device, real_features, gen_features, real_output, gen_output):
    criterion = nn.BCEWithLogitsLoss()
    dloss_gen = criterion(gen_output, torch.zeros_like(gen_output))
    dloss_real = criterion(real_output, torch.ones_like(real_output))
    dloss = dloss_gen.mean() + dloss_real.mean()
    
    return dloss   

def compute_g_ns_gan_loss(args, device, real_features, gen_features, real_output, gen_output):
    criterion = nn.BCEWithLogitsLoss()
    gloss = criterion(gen_output, torch.ones_like(gen_output))
    return gloss.mean()

def compute_d_wgan_loss(args, device, real_features, gen_features, real_output, gen_output):
    return gen_output.mean() - real_output.mean()   

def compute_g_wgan_loss(args, device, real_features, gen_features, real_output, gen_output):
    return - compute_d_wgan_loss(args, device, real_features, gen_features, real_output, gen_output)

def compute_g_primal_loss(args, device, real_features, gen_features, real_output, gen_output):
    b_size = real_features.size(0)
    real_features = real_features.view(b_size, -1)
    gen_features = gen_features.view(b_size, -1)
    
    C = compute_frobenius_pairwise_distances_torch(real_features, gen_features, device, p=2, normalized=False)
    C_cpu = C.detach().cpu().numpy()

    # Solve for M*
    row_ind, col_ind = linear_sum_assignment(C_cpu)
    values = np.asarray([1.0 for _ in range(b_size)])
    M_star_cpu = csr_matrix((values, (row_ind, col_ind)), dtype=np.float).todense()  # TODO: change this
    M_star = torch.tensor(M_star_cpu, device=device, dtype=C.dtype)
    
    gloss = torch.sum(M_star * C) / b_size
    
    return gloss

EPSILON = 1e-12
def dense_wasserstein_distance(cost_matrix, device):
    num_pts = len(cost_matrix);
    C_cpu = cost_matrix.detach().cpu().numpy();
    C_cpu *= 100000 / (C_cpu.max() + EPSILON)
    lowest_cost, col_ind_lapjv, row_ind_lapjv = lapjv(C_cpu);
    loss = 0.0;
    for i in range(num_pts):
        loss += cost_matrix[i,col_ind_lapjv[i]];
    return loss/num_pts;

def compute_g_primal_loss_v2(args, device, real_features, gen_features, real_output, gen_output, eval_metrics=None, netG=None):
    b_size = real_features.size(0)
    real_features = real_features.view(b_size, -1)
    gen_features = gen_features.view(b_size, -1)
    C = losses.compute_frobenius_pairwise_distances_torch(real_features, gen_features, device, p=2, normalized=True)
    gloss = dense_wasserstein_distance(C, device)
    if args.LAMBDA:
        gloss *= args.LAMBDA
    return gloss