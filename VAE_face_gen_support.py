import torch
from torch import nn

def VAE_loss_fuction(x, y, z_log_var, z_mean):

    kl_div_loss = -0.5 * torch.sum(1 + z_log_var - z_mean ^ 2 - torch.exp(z_log_var))
    mse = nn.MSELoss()(x,y)

    return mse + kl_div_loss

