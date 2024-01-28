CUT_TRAIN_IMAGE_FACE_AND_RESIZE = 'GET_IMAGES_OF_FACES'
TRAIN_LOOP = 'TRAIN_ENCODER_DECODER_FROM_IMAGES'
GEN_FACE = 'GENERATE_RANDOM_FACE'
INSPECT_DECODER_OUTPUT = 'COMPARE_I/P_AND_O/P'

RUN_MODE = TRAIN_LOOP
DEVICE = 'cuda'

DATASET_DIR = 'G:/Guru_Sarath/Study/1_Project_PhD/1_git_repos/0_Datasets/Celeb_Dataset/img_align_celeba'
CUTFACE_DATASET_DIR = './faces'
MODEL_SAVE_PATH = './saved_model'

BATCH_SIZE = 100
NUM_EPOCHS = 100

IMAGE_SIZE = (32,32)