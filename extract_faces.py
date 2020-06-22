## Assuming a csv file has:
## Image Name (ImageID) in column 1 (line[0])
## Full Resolution URL (OriginalURL) in column 3 (line[2])

import sys
import urllib.request
import csv
import os.path
import cv2
import numpy as np

# folder containing directories per instagram hashtag
# e.g 'blogger_girl'
SCRAPE_FOLDERS_PATH = "G:\My Drive\projects\\are_you_real\samples\instagram_scrape"
BASE_DIR = "G:\My Drive\projects\\are_you_real\data_processing"
JPGS_FOLDER_EXT = '_jpgs'
SCRAPE_FOLDER_EXT = '_scrape'
FACES_FOLDER_EXT = '_faces'
DISPLAY_URL_INDEX = 5
IS_RUN_FROM_CMD = len(sys.argv) > 1

prototxt_path = os.path.join(BASE_DIR, 'model_data', 'deploy.prototxt')
caffemodel_path = os.path.join(BASE_DIR, 'model_data', 'weights.caffemodel')


def prep_dirs(dir_name):
    dir_abs_path = os.path.join(SCRAPE_FOLDERS_PATH, dir_name)
    jpgs_folder = os.path.join(dir_abs_path, dir_name + JPGS_FOLDER_EXT)
    assert os.path.isdir(jpgs_folder)
    output_folder_path = os.path.join(dir_abs_path, dir_name + FACES_FOLDER_EXT)
    os.makedirs(output_folder_path, exist_ok=True)
    return jpgs_folder, output_folder_path


def create_box_around_face(model, img_abspath, output_abspath):
    im = cv2.imread(img_abspath)
    (h, w) = im.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(im, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    detections = model.forward()

    face_count = 0
    for i in range(0, 1):
        # for i in range(0, detections.shape[2]):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (start_x, start_y, end_x, end_y) = box.astype('int')
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            face_count += 1
            start_y = max(0, start_y)
            end_y= min(end_y, h - 1)
            start_x= max(0, start_x)
            end_x= min(end_x, w - 1)
            frame = im[start_y: end_y, start_x:end_x]
            # cv2.rectangle(im, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)

            output_path_base, putput_path_ext = output_abspath.split('.')
            face_out_path = f'{output_path_base}.{putput_path_ext}'
            # face_out_path = f'{output_path_base}_{face_count}.{putput_path_ext}'
            try:
                cv2.imwrite(face_out_path, frame)
            except:
                print(f'Error in image {face_out_path}')
            print('Done face')


def extract_faces():
    print('Extracting faces\n')

    model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

    for dir_name in os.listdir(SCRAPE_FOLDERS_PATH):
        print(f'Processing dir: {dir_name}...\n')
        jpgs_folder, output_folder_path = prep_dirs(dir_name)
        for im_num, jpg_fn in enumerate(os.listdir(jpgs_folder)):
            output_fn = f'face_{dir_name}_{im_num}.jpg'
            output_abspath = os.path.join(output_folder_path, output_fn)
            print(f'\nProcessing item {output_fn}')
            jpg_abspath = os.path.join(jpgs_folder, jpg_fn)
            assert os.path.isfile(jpg_abspath)
            if os.path.isfile(output_abspath):
                print(f"Face extraction skipped for {output_fn}")
            else:
                create_box_around_face(model, jpg_abspath, output_abspath)


def main():
    extract_faces()


main()
