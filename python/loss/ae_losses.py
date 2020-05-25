from torch import nn

def compute_L1_reconstruction_loss(args, device, x, xhat):
    criterion = nn.L1Loss()
    return criterion(xhat, x.to(device)).mean()

def compute_MSE_reconstruction_loss(args, device, x, xhat):
    criterion = nn.MSELoss()
    return criterion(xhat, x.to(device)).mean()

def compute_BCE_reconstruction_loss(args, device, x, xhat):
    criterion = nn.BCEWithLogitsLoss()
    return criterion(xhat, x.to(device)).mean()