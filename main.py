import numpy as np
import torch
from VAE_face_gen_settings import RUN_MODE, CUT_TRAIN_IMAGE_FACE_AND_RESIZE, DATASET_DIR, CUTFACE_DATASET_DIR, TRAIN_LOOP, GEN_FACE, INSPECT_DECODER_OUTPUT
from face_detection_support import process_train_images, init_face_detection
from VAE_face_gen_support import vae_train_face_gen_loop, generate_some_random_face, visually_compare_decoder_output_with_input



if __name__ == '__main__':

    print('Running VAE face gen .. .. .. ')

    if RUN_MODE == CUT_TRAIN_IMAGE_FACE_AND_RESIZE:
        init_face_detection()
        process_train_images(DATASET_DIR, CUTFACE_DATASET_DIR)  # Extract faces from people images and store them
        exit()

    if RUN_MODE == TRAIN_LOOP:
        print('Training Loop .. .. ..')
        vae_train_face_gen_loop()
        exit()

    if RUN_MODE == GEN_FACE:
        print('Generating Random Face Using VAE Decoder .. .. ..')
        generate_some_random_face()
        exit()

    if RUN_MODE == INSPECT_DECODER_OUTPUT:
        print('Generating Random Face Using VAE Decoder .. .. ..')
        visually_compare_decoder_output_with_input()
        exit()