import numpy as np
import torch
from torch import nn
from VAE_face_gen_settings import DEVICE

ENC_OUT_DIM = 1800


class SamplingLayer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, z_mean, z_log_var):
        batch_size = z_mean.shape[0]
        dim = z_mean.shape[1]
        normal_distribution = torch.randn(size=(batch_size, dim)).to(DEVICE)

        return z_mean + torch.exp(0.5 * z_log_var) * normal_distribution


class VAEEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.cnn_enc_model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.flat = nn.Flatten(start_dim=1, end_dim=-1)
        self.z_mean_layer = nn.Linear(41472, ENC_OUT_DIM)
        self.z_log_var_layer = nn.Linear(41472, ENC_OUT_DIM)

        self.samp_nd_layer = SamplingLayer()

    def forward(self, x):
        out0 = self.cnn_enc_model(x)
        out1 = self.flat(out0)

        # print('Reshape/view size in decoder = ', out0.shape)  # torch.Size([batch_size, 128, 18, 18])

        z_mean = self.z_mean_layer(out1)
        z_log_var = self.z_log_var_layer(out1)

        return self.samp_nd_layer(z_mean, z_log_var), z_mean, z_log_var


class VAEDecoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.linear_layer = nn.Linear(ENC_OUT_DIM, 41472)

        self.trans_cnn_dec_model = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out0 = self.linear_layer(x[0])
        out1 = out0.view(-1, 128, 18, 18)

        return self.trans_cnn_dec_model(out1), x[1], x[2]


class VAE_FaceGen(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        out_enc = self.encoder(x)
        out_dec = self.decoder(out_enc)

        return out_dec[0], out_enc[1], out_enc[2]


if __name__ == '__main__':
    print('Unit Testing .. .. ..')

    x = torch.zeros(size=(2, 3, 80, 80)).to(DEVICE)

    model_enc = VAEEncoder()
    model_dec = VAEDecoder()

    '''
    out_z_ = model_enc(x)

    out_image_z = model_dec(out_z_)

    print(out_z_[0].shape)
    print(out_image_z[0].shape)
    '''

    print('Test 2')

    vae_model = VAE_FaceGen(model_enc, model_dec).to(DEVICE)

    outs = vae_model(x)

    print(f'outVAE shape = {outs[0].shape} {outs[1].shape} {outs[2].shape}')

    from VAE_face_gen_support import VAE_loss_fuction

    loss = VAE_loss_fuction(x, outs[0], outs[1], outs[2])
    print(loss.item())
