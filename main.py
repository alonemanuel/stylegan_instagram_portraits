import os
import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace

from data_processing.dataset_manager import DatasetManager

BASE_DIR_PATH = 'G:\My Drive\projects\\are_you_real\samples\instagram_scrape'


def main():
    dataset_builder = DatasetManager(BASE_DIR_PATH)
    # dataset_builder.init_dataset()
    # dataset_builder.url_to_images()
    # dataset_builder.extract_faces()
    # print("\n\n\n%%%%%%%%% CLEANING %%%%%%%")
    # dataset_builder.cleanup_dataset('ds2')
    # print("\n\n\n%%%%%%%%% RESIZING %%%%%%%")
    # dataset_builder.resize_up_dataset('ds2')
    # print("\n\n\n%%%%%%%%% FLIPPING %%%%%%%")
    # dataset_builder.mirror_dataset('ds2')
    print("\n\n\n%%%%%%%%% DELETING FLIPPED %%%%%%%")
    dataset_builder.delete_flipped('ds2')

def sandbox(img_n):
    img_abspath = os.path.join(BASE_DIR_PATH, 'prettyface', 'prettyface_jpgs', f'prettyface_{img_n}.jpg')
    detected_face = DeepFace.detectFace(img_abspath)
    plt.imshow(detected_face)
    plt.show()
    output_abspath = os.path.join(BASE_DIR_PATH, f'sandbox15.jpg')
    output_abspath_plt = os.path.join(BASE_DIR_PATH, f'sandbox{img_n}_plt.jpg')
    cv2.imwrite(output_abspath, detected_face)
    plt.imsave(output_abspath_plt, detected_face)


if __name__ == '__main__':
    main()
    # sandbox(34)
