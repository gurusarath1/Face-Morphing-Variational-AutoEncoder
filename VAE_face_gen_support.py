import torch
from torch import nn
from datetime import date
from VAE_model import VAE_FaceGen, VAEEncoder, VAEDecoder, ENC_OUT_DIM
from ml_utils import ImagesDataset
from torch.utils.data import DataLoader
from VAE_face_gen_settings import CUTFACE_DATASET_DIR, BATCH_SIZE, DEVICE, NUM_EPOCHS, MODEL_SAVE_PATH
from ml_utils import save_torch_model, load_torch_model, display_image
import matplotlib.pyplot as plt
import random


def VAE_loss_fuction(x, y, z_mean, z_log_var):
    kl_div_loss = -0.5 * torch.sum(1 + z_log_var - torch.pow(z_mean, 2) - torch.exp(z_log_var))
    mse = nn.MSELoss()(x, y)

    return (100000 * mse) + (kl_div_loss)


def VAE_loss_fuction_2(x, y, code_x):
    kl_div_loss = nn.KLDivLoss(reduction='batchmean')(code_x, torch.randn_like(code_x))
    mse = nn.MSELoss()(x, y)

    return (mse)


def vae_train_face_gen_loop():
    # Dataset
    faces_dataset = ImagesDataset(CUTFACE_DATASET_DIR, dataset_size=191500, device=DEVICE)
    train_dataloader = DataLoader(faces_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # DL Model
    model_enc = VAEEncoder()
    model_dec = VAEDecoder()
    vae_model = VAE_FaceGen(model_enc, model_dec).to(DEVICE)
    # Get the last check point model weights
    load_torch_model(vae_model, file_name='vae_net', path=MODEL_SAVE_PATH, load_latest=True)
    vae_model.train()

    # Optimizer
    optimz = torch.optim.Adam(vae_model.parameters(), lr=0.0001)#weight_decay=1e-2

    torch.cuda.empty_cache()

    loss_history = []
    for epoch in range(NUM_EPOCHS):

        for batch_idx, batch in enumerate(train_dataloader):

            # display_image(batch.detach().to('cpu'), batch_dim_exist=True)
            # while(1):

            outFace_zMean_zLogStd = vae_model(batch)

            out_faces = outFace_zMean_zLogStd[0]
            z_mean = outFace_zMean_zLogStd[1]
            z_log_std = outFace_zMean_zLogStd[2]
            z = outFace_zMean_zLogStd[3]

            loss = VAE_loss_fuction(batch, out_faces, z_mean, z_log_std)
            #loss = VAE_loss_fuction_2(batch, out_faces, z)

            # Back Prop
            loss.backward()
            optimz.step()
            optimz.zero_grad()

            # print(loss.item())
            # display_image(out_faces.detach().to('cpu'), batch_dim_exist=True)

            if batch_idx == 10 or batch_idx % 500 == 0:
                print(f'epoch = {epoch}  batch_idx = {batch_idx}   loss = {loss.item()}')
                loss_history.append(loss.cpu().item())
                file_name_info = '_' + str(date.today()) + '_' + str(loss.item())
                save_torch_model(vae_model, file_name='vae_net',
                                 additional_info=file_name_info,
                                 path=MODEL_SAVE_PATH,
                                 two_copies=True)

    plt.plot(loss_history)
    plt.show()


def generate_some_random_face():
    # DL Model
    model_enc = VAEEncoder()
    model_dec = VAEDecoder()
    vae_model = VAE_FaceGen(model_enc, model_dec).to(DEVICE)
    # Get the last check point model weights
    load_torch_model(vae_model, file_name='vae_net', path=MODEL_SAVE_PATH, load_latest=True)
    vae_model.eval()

    while (1):
        x = [torch.randn(size=(1, ENC_OUT_DIM)).to(DEVICE), None, None]

        out_dec = vae_model.decoder(x)

        face_image = out_dec[0].detach().to('cpu')

        display_image(face_image, batch_dim_exist=True)


def visually_compare_decoder_output_with_input():
    # Dataset
    faces_dataset = ImagesDataset(CUTFACE_DATASET_DIR, dataset_size=179500, device=DEVICE)

    # DL Model
    model_enc = VAEEncoder()
    model_dec = VAEDecoder()
    vae_model = VAE_FaceGen(model_enc, model_dec).to(DEVICE)
    # Get the last check point model weights
    load_torch_model(vae_model, file_name='vae_net', path=MODEL_SAVE_PATH, load_latest=True)
    vae_model.eval()

    while (1):
        idx = random.randint(0, 1000)
        x = faces_dataset[idx].unsqueeze(dim=0)
        display_image(x.detach().to('cpu'), batch_dim_exist=True)
        print(x.shape)
        out_dec = vae_model(x)
        print(out_dec[0].shape)
        face_image = out_dec[0].detach().to('cpu')
        display_image(face_image, batch_dim_exist=True)
