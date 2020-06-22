import math
import os
import urllib
from copy import deepcopy

from data_processing.data_process_exception import DataProcessException
import matplotlib.pyplot as plt
import numpy as np
import cv2
import skimage.transform
from PIL import Image
from deepface import DeepFace

BB_ENDY = 3

BB_STARTY = 2

BB_STARTX = 0

BB_ENDX = 1

MODEL_PROCESSING_ABSPATH = 'G:\My Drive\projects\\are_you_real\src\data_processing'
POW_OF_TWO = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]


class SampleItem:
    prototxt_path = os.path.join(MODEL_PROCESSING_ABSPATH, 'model_data', 'deploy.prototxt')
    caffemodel_path = os.path.join(MODEL_PROCESSING_ABSPATH, 'model_data', 'weights.caffemodel')
    eye_classifier = os.path.join(MODEL_PROCESSING_ABSPATH, 'model_data', 'haarcascade_eye.xml')
    model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
    base_dir_abspath = 'G:\My Drive\projects\\are_you_real\samples\instagram_scrape'

    def __init__(self, sample_id, sample_class, csv_abspath, orig_image_url):
        self.sample_id = sample_id
        self.sample_class = sample_class
        self.csv_abspath = csv_abspath
        self.orig_image_url = orig_image_url
        self.orig_image_path = None
        self.orig_image_arr = None
        self.face_path = None
        self.face_arr = None

    def url_to_orig_image(self):
        if self.orig_image_path:
            return

        orig_images_dir_abspath = self.get_orig_images_dir_abspath()
        orig_image_basename = f'origim_{self.sample_id}.jpg'
        potential_orig_image_path = os.path.join(orig_images_dir_abspath, orig_image_basename)
        if os.path.isfile(potential_orig_image_path):
            self.orig_image_path = potential_orig_image_path
            self.orig_image_arr = cv2.imread(self.orig_image_path)
            return

        if self.sample_class in ['makeup', 'fashionista', 'blogger_girl', 'prettyface', 'portrait']:
            self.orig_image_path = -1
            return

        if self.orig_image_path == -1:
            return

        try:
            urllib.request.urlretrieve(self.orig_image_url, potential_orig_image_path)
            print(f"Found url for: {self.sample_id}")
            self.orig_image_path = potential_orig_image_path
            self.orig_image_arr = cv2.imread(self.orig_image_path)
        except Exception as e:
            self.orig_image_path = -1
            print(f"Couldn't find url for: {self.sample_id}, error:\n{e}")

    def orig_image_to_face(self):
        self.orig_image_to_face_bounding_box()
        if not self.bounding_box:
            raise DataProcessException('no bounding box')
        # self.bounding_box_to_aligned_face()
        self.bounding_box_to_face()

    def bounding_box_to_aligned_face(self):
        angle_to_rotate, direction = self.get_angle_to_rotate()
        if angle_to_rotate != -1:
            rotated_image = self.get_rotated_orig_image(angle_to_rotate)
            rotated_bounding_box = self.get_rotated_bounding_box(angle_to_rotate)
        else:
            print('Could not align face')
            rotated_image = self.orig_image_arr
            rotated_bounding_box = self.bounding_box

        self.face_arr = get_frame(rotated_image, rotated_bounding_box)
        face_output_basename = f'face_{self.sample_id}.jpg'
        face_dir_abspath = self.get_face_dir_abspath()
        self.face_path = os.path.join(face_dir_abspath, face_output_basename)
        cv2.imwrite(self.face_path, self.face_arr)

    def bounding_box_to_face(self):
        self.face_arr = get_frame(self.orig_image_arr, self.bounding_box)
        face_output_basename = f'face_{self.sample_id}.jpg'
        face_dir_abspath = self.get_face_dir_abspath()
        self.face_path = os.path.join(face_dir_abspath, face_output_basename)
        cv2.imwrite(self.face_path, self.face_arr)

    def orig_image_to_face_bounding_box(self):
        if not self.orig_image_path:
            raise DataProcessException('no image arr')
        self.orig_image_arr = cv2.imread(self.orig_image_path)
        (h, w) = self.orig_image_arr.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(self.orig_image_arr, (300, 300)), 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))
        SampleItem.model.setInput(blob)
        detections = SampleItem.model.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                break
        else:
            raise DataProcessException('No face detected')

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (start_x, start_y, end_x, end_y) = box.astype('int')

        start_x, end_x = max(0, start_x), min(end_x, w - 1)
        start_y, end_y = max(0, start_y), min(end_y, h - 1)

        self.bounding_box = [start_x, end_x, start_y, end_y]
        self.bounding_box = self.get_big_square_bb(self.bounding_box, w, h)

    def get_big_square_bb(self, bounding_box, w, h):
        bb = bounding_box
        bb_width = bounding_box[BB_ENDX] - bounding_box[BB_STARTX]
        bb_height = bounding_box[BB_ENDY] - bounding_box[BB_STARTY]
        for i, pow in enumerate(POW_OF_TWO):
            max_size = max(bb_width, bb_height)
            if max_size < pow:
                selected_size = pow
                break

        bb_mid_x = bb[BB_STARTX] + (bb_width / 2)
        bb_mid_y = bb[BB_STARTY] + (bb_height / 2)
        res_bb = deepcopy(bb)

        poten_new_bb_s_x = bb_mid_x - (selected_size / 2)
        poten_new_bb_s_y = bb_mid_y - (selected_size / 2)
        poten_new_bb_e_x = bb_mid_x + (selected_size / 2)
        poten_new_bb_e_y = bb_mid_y + (selected_size / 2)

        if poten_new_bb_s_x >= 0 and poten_new_bb_e_x < w:
            res_bb[BB_STARTX], res_bb[BB_ENDX] = poten_new_bb_s_x, poten_new_bb_e_x
        if poten_new_bb_s_x >= 0 and poten_new_bb_e_x >= w:
            diff = poten_new_bb_e_x - w
            res_bb[BB_STARTX], res_bb[BB_ENDX] = poten_new_bb_s_x - diff, w
        if poten_new_bb_s_x < 0 and poten_new_bb_e_x < w:
            diff = -poten_new_bb_s_x
            res_bb[BB_STARTX], res_bb[BB_ENDX] = 0, poten_new_bb_e_x + diff

        if poten_new_bb_s_y >= 0 and poten_new_bb_e_y < h:
            res_bb[BB_STARTY], res_bb[BB_ENDY] = poten_new_bb_s_y, poten_new_bb_e_y
        if poten_new_bb_s_y >= 0 and poten_new_bb_e_y >= h:
            diff = poten_new_bb_e_y - h
            res_bb[BB_STARTY], res_bb[BB_ENDY] = poten_new_bb_s_y - diff, h
        if poten_new_bb_s_y < 0 and poten_new_bb_e_y < h:
            diff = -poten_new_bb_s_y
            res_bb[BB_STARTY], res_bb[BB_ENDY] = 0, poten_new_bb_e_y + diff
        res_bb = [int(num) for num in res_bb]
        return res_bb

    def get_eyes_from_image(self):
        eye_detector = cv2.CascadeClassifier(SampleItem.eye_classifier)
        face_im = get_frame(self.orig_image_arr, self.bounding_box)
        img_gray = cv2.cvtColor(face_im, cv2.COLOR_BGR2GRAY)
        eyes = eye_detector.detectMultiScale(img_gray)
        if (eyes[0] is None or eyes[1] is None):
            print('Could not find eyes')
            return -1
        else:
            print('Found eyes')
            left_eye, right_eye = (eyes[0], eyes[1]) if (eyes[0][0] < eyes[1][0]) else (eyes[1], eyes[0])
            return left_eye, right_eye

    def get_angle_to_rotate(self):

        eyes = self.get_eyes_from_image()
        if eyes == -1:
            print('Could not find eyes')
            return -1
        else:
            left_eye, right_eye = eyes
        left_eye_center, right_eye_center, mid_point_center, direction = get_eye_centers(left_eye, right_eye)

        a = euclidean_distance(left_eye_center, mid_point_center)
        b = euclidean_distance(right_eye_center, left_eye_center)
        c = euclidean_distance(right_eye_center, mid_point_center)

        cos_a = (b * b + c * c - a * a) / (2 * b * c)
        angle = np.arccos(cos_a)
        angle = (angle * 180) / math.pi
        if direction == -1:
            angle = 90 - angle
        return angle, direction

    def get_rotated_bounding_box(self, rotation_angle):
        rotation_dummy_arr = np.zeros(self.orig_image_arr.shape)
        start_x, end_x, start_y, end_y = self.bounding_box
        rotation_dummy_arr[start_x:end_x, start_y:end_y] = 1
        rotation_dummy_arr = skimage.transform.rotate(rotation_dummy_arr, rotation_angle, True, cval=0)
        rect = np.where(rotation_dummy_arr == 1)
        start_rx, end_rx = rect[1].min(), rect[1].max()
        start_ry, end_ry = rect[0].min(), rect[0].max()
        return start_rx, end_rx, start_ry, end_ry

    def get_rotated_orig_image(self, angle):
        return skimage.transform.rotate(self.orig_image_arr, angle, True)

    """
    Stage snoopers
    """

    def has_orig_image(self):
        orig_images_dir_abspath = self.get_orig_images_dir_abspath()
        orig_image_basename = f'origim_{self.sample_id}.jpg'
        potential_orig_image_path = os.path.join(orig_images_dir_abspath, orig_image_basename)
        if os.path.isfile(potential_orig_image_path):
            self.orig_image_path = potential_orig_image_path
            return True
        else:
            return False

    def has_orig_image2(self):
        return self.orig_image_arr is not None

    def has_face(self):
        face_images_dir_abspath = self.get_face_dir_abspath()
        face_image_basename = f'face_{self.sample_id}.jpg'
        potential_face_image_path = os.path.join(face_images_dir_abspath, face_image_basename)
        if os.path.isfile(potential_face_image_path):
            self.orig_image_path = potential_face_image_path
            return True
        else:
            return False
    def could_not_load_image(self):
        return self.orig_image_path == -1

    """ 
    Get dirs
    """

    def get_orig_images_dir_abspath(self):
        return self.get_and_create_sample_dir('origs')

    def get_face_dir_abspath(self):
        return self.get_and_create_sample_dir('faces')

    def get_and_create_sample_dir(self, dir_extenstion):
        dir_basename = f'{self.sample_class}_{dir_extenstion}'
        dir_abspath = os.path.join(SampleItem.base_dir_abspath, self.sample_class, dir_basename)
        os.makedirs(dir_abspath, exist_ok=True)
        return dir_abspath


"""
Helper static functions
"""


def get_frame(image_arr, bounding_box):
    start_x, end_x, start_y, end_y = bounding_box
    return image_arr[start_y: end_y, start_x:end_x]


def euclidean_distance(a, b):
    x1 = a[0]
    y1 = a[1]

    x2 = b[0]
    y2 = b[1]
    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))


def get_eye_centers(left_eye, right_eye):
    left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
    left_eye_x = left_eye_center[0]
    left_eye_y = left_eye_center[1]

    right_eye_center = (int(right_eye[0] + (right_eye[2] / 2)), int(right_eye[1] + (right_eye[3] / 2)))
    right_eye_x = right_eye_center[0]
    right_eye_y = right_eye_center[1]

    if left_eye_y < right_eye_y:
        mid_point_center = (right_eye_x, left_eye_y)
        direction = -1  # rotate same direction to clock
    else:
        mid_point_center = (left_eye_x, right_eye_y)
        direction = 1  # rotate inverse direction of clock

    return left_eye_center, mid_point_center, right_eye_center, direction
