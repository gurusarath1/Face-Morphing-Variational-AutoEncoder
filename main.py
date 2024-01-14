import numpy as np
import torch
from VAE_face_gen_settings import RUN_MODE, CUT_TRAIN_IMAGE_FACE_AND_RESIZE, DATASET_DIR, CUTFACE_DATASET_DIR, TRAIN_LOOP, BATCH_SIZE, DEVICE
from ml_utils import ImagesDataset
from face_detection_support import process_train_images, init_face_detection
from torch.utils.data import DataLoader


if __name__ == '__main__':

    print('Running VAE face gen .. .. .. ')

    if RUN_MODE == CUT_TRAIN_IMAGE_FACE_AND_RESIZE:
        init_face_detection()
        process_train_images(DATASET_DIR, CUTFACE_DATASET_DIR)  # Extract faces from people images and store them
        exit()

    if RUN_MODE == TRAIN_LOOP:
        print('Training Loop .. .. ..')
        faces_dataset = ImagesDataset(CUTFACE_DATASET_DIR, dataset_size=179500, device=DEVICE)
        train_dataloader = DataLoader(faces_dataset, batch_size=BATCH_SIZE, shuffle=True)
        exit()