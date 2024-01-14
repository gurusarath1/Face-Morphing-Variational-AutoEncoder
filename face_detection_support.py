import cv2
import numpy as np
from face_detection_settings import FACE_CASCADE_FILE
import copy
import os

face_cascade = None


def init_face_detection():
    global face_cascade

    # Load the face cascade file
    # https://github.com/kipr/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_FILE)


def run_face_detection(cv2_img):
    assert face_cascade is not None

    image = copy.deepcopy(cv2_img)

    # Convert to GrayScale (Haar Cascade algorithm works on gray scale images)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Run the face detection algorithm
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)

    image_with_boxes = image
    # Draw bounding boxes
    for (x, y, w, h) in faces:
        image_with_boxes = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray_img[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]

    return faces, image_with_boxes

def extract_1face_image(image: np.ndarray, face_bounding_box: tuple) -> np.ndarray:
    (x, y, w, h) = face_bounding_box
    face_cut = image[y:y + w, x:x + h, :]
    return face_cut

def extract_1face_and_preprocess(image: np.ndarray, face_bounding_box: tuple, ouput_size: tuple = (80,80)) -> np.ndarray:

    assert (face_bounding_box[2] >= ouput_size[0] and face_bounding_box[2] >= ouput_size[1])

    face_cut = extract_1face_image(image, face_bounding_box) # get only the face
    train_size_face = cv2.resize(face_cut, ouput_size, interpolation=cv2.INTER_AREA) # resize the image to training size
    return train_size_face

def process_train_images(input_images_dir, output_images_dir, output_size=(80,80), image_format='.jpg'):
    for face_file in os.listdir(input_images_dir):

        full_image_path = os.path.join(input_images_dir, face_file)
        print('Processing file -- ', full_image_path)

        image = cv2.imread(full_image_path)
        faces, box_image = run_face_detection(image) #

        # Skip files with more than 2 faces or no faces
        if len(faces) > 1 or len(faces) == 0:
            print('Skip file -- ', full_image_path)
            continue

        w = faces[0][2]
        h = faces[0][3]

        if w < output_size[0] and h < output_size[1]:
            print('Skip file (small) -- ', full_image_path)
            continue

        # Get only the face
        train_image = extract_1face_and_preprocess(image, faces[0])

        output_file_name = face_file.split('.')[0] + '_face' + image_format
        output_file_path = os.path.join(os.getcwd(), output_images_dir, output_file_name)
        print('Ouput file -- ', output_file_path)

        cv2.imwrite(output_file_path, train_image)