import csv
import os
import pickle

import cv2
from tqdm import tqdm

from data_processing.data_process_exception import DataProcessException
from data_processing.sample_item import SampleItem

UPSAMPLE_SIZE = 1024

URL_TO_IM_PICKLING_FREQ = 50
IM_TO_FACE_PICKLING_FREQ = 50

PICKLE_NAME = 'ig_dataset.pickle'
SCRAPE_EXT = 'scrape'
ORIG_IM_DIR_EXT = 'origim'
IMAGE_URL_CSV_INDEX = 5
DATASETS_ABSPATH = 'G:\My Drive\projects\\are_you_real\samples\datasets'


class DatasetManager:

    def __init__(self, base_dir_abspath):
        self.base_dir_abspath = base_dir_abspath
        self.init_pickle()
        self.all_samples_dict = dict()
        # self.init_dict()

    def init_dict(self):

        with open(self.pickle_abspath, 'rb') as pickle_infile:
            self.all_samples_dict = pickle.load(pickle_infile)
        print(
            f"Unpickled dataset_dict of length: {len(self.all_samples_dict)}\n"
            f"\tWith first entry being: ")

    def init_pickle(self):
        self.pickle_abspath = os.path.join(self.base_dir_abspath, PICKLE_NAME)
        if not os.path.isfile(self.pickle_abspath):
            with open(self.pickle_abspath, 'w+b') as pickle_outfile:
                pickle.dump(dict(), pickle_outfile)

    def get_csv_file_abspath(self, dir_name):
        csv_folder_base_name = f'{dir_name}_{SCRAPE_EXT}'
        csv_folder_abspath = os.path.join(self.base_dir_abspath, dir_name, csv_folder_base_name)
        assert os.path.isdir(csv_folder_abspath)
        files_in_csv_folder = os.listdir(csv_folder_abspath)
        assert len(files_in_csv_folder) == 1
        csv_file_abspath = os.path.join(csv_folder_abspath, files_in_csv_folder[0])
        assert os.path.isfile(csv_file_abspath)
        return csv_file_abspath

    def get_orig_images_dir_abspath(self, dir_name):
        orig_images_dir_basename = f'{dir_name}_{ORIG_IM_DIR_EXT}'
        orig_images_dir_abspath = os.path.join(self.base_dir_abspath, orig_images_dir_basename)
        os.makedirs(orig_images_dir_abspath, exist_ok=True)
        return orig_images_dir_abspath

    def get_sample_id(self, dir_name, sample_n):
        sample_id = f'{dir_name}_{sample_n}'
        return sample_id

    def init_dataset(self):
        for dir_name in os.listdir(self.base_dir_abspath):
            if not os.path.isdir(os.path.join(self.base_dir_abspath, dir_name)):
                continue  # skip pickle

            print(f'\nProcessing dir: {dir_name}...')
            csv_file_abspath = self.get_csv_file_abspath(dir_name)

            with open(csv_file_abspath, 'r', encoding='utf8') as csv_file:
                reader = csv.reader(csv_file)
                next(reader)  # skip headers
                for item_n, line in enumerate(reader):
                    sample_id = self.get_sample_id(dir_name, item_n)
                    if sample_id in self.all_samples_dict.keys():
                        print(f'Passing on sample: {sample_id}')
                        pass
                    else:
                        print(f'Processing sample: {sample_id}')
                        orig_image_url = line[IMAGE_URL_CSV_INDEX]
                        sample_item = SampleItem(sample_id, dir_name, csv_file_abspath, orig_image_url)
                        self.all_samples_dict[sample_id] = sample_item

    def init_dataset2(self):
        """
        Assuming initial directory hierarchy of:

        |- base_dir_path (scrape folder)
        |--- tag_name_0
        |--- tag_name_1
        ...
        |--- tag_name_2
        |------- tag_name_2_scrape

        Creates dict of samples, where each sample contains
        """
        for dir_name in os.listdir(self.base_dir_abspath):
            if not os.path.isdir(os.path.join(self.base_dir_abspath, dir_name)):
                continue  # skip pickle

            print(f'\nProcessing dir: {dir_name}...')
            csv_file_abspath = self.get_csv_file_abspath(dir_name)

            with open(csv_file_abspath, 'r', encoding='utf8') as csv_file:
                reader = csv.reader(csv_file)
                next(reader)  # skip headers
                for item_n, line in enumerate(reader):
                    sample_id = self.get_sample_id(dir_name, item_n)
                    if sample_id in self.all_samples_dict.keys():
                        print(f'Passing on sample: {sample_id}')
                        pass
                    else:
                        print(f'Processing sample: {sample_id}')
                        orig_image_url = line[IMAGE_URL_CSV_INDEX]
                        sample_item = SampleItem(sample_id, dir_name, csv_file_abspath, orig_image_url)
                        self.all_samples_dict[sample_id] = sample_item
            self.pickle_dict()

    def url_to_images(self):
        print('In url_to_images')

        for i, (key, sample) in enumerate(self.all_samples_dict.items()):
            if sample.has_orig_image():
                print(f"Passing url-to-im: {sample.sample_id}")
                continue
            else:
                sample.url_to_orig_image()

    def url_to_images2(self):
        print('In url_to_images')
        passed_count = 0
        pickle_count = 1
        for i, (key, sample) in enumerate(self.all_samples_dict.items()):
            if sample.has_orig_image() or sample.could_not_load_image():
                passed_count += 1
                continue
            else:
                pickle_count += 1
                sample.url_to_orig_image()
            if pickle_count % URL_TO_IM_PICKLING_FREQ == 0:
                self.pickle_dict()
                pickle_count += 1
        print(f'Passed on: {passed_count}/{len(self.all_samples_dict)}')
        self.pickle_dict()

    def extract_faces(self):
        for i, (key, sample) in enumerate(self.all_samples_dict.items()):
            print(f'Extracting face for: {sample.sample_id}')
            if sample.has_orig_image():
                if not sample.has_face():
                    try:
                        sample.orig_image_to_face()
                    except DataProcessException as e:
                        print(f'Extract face error: {e.message}')
                    except Exception as e:
                        print(e)
                else:
                    print(f'Face exists')

    def extract_faces2(self):
        passed_count = 0
        pickle_count = 0
        for i, (key, sample) in enumerate(self.all_samples_dict.items()):
            if sample.has_face():
                passed_count += 1
                continue
            else:
                sample.orig_image_to_face()
                pickle_count += 1
            if pickle_count % IM_TO_FACE_PICKLING_FREQ == 0:
                self.pickle_dict()
        print(f'Passed on: {passed_count}/{len(self.all_samples_dict)}')
        self.pickle_dict()

    def pickle_dict(self):
        print('Pickling dir...')
        with open(self.pickle_abspath, 'w+b') as pickle_outfile:
            pickle.dump(self.all_samples_dict, pickle_outfile)

    def cleanup_dataset(self, dataset_name):
        print(f'Starting to clean')
        inputds = 'ds1'
        for sample_basename in os.listdir(os.path.join(DATASETS_ABSPATH, inputds)):
            sample_abspath = os.path.join(DATASETS_ABSPATH, inputds, sample_basename)
            sample_im = cv2.imread(sample_abspath)
            (h, w) = sample_im.shape[:2]
            if h != w:
                print(f'Cropping {sample_basename}')
                min_size = min(w, h)
                cropped_im = sample_im[0:min_size - 1, 0:min_size - 1]
                output_fn = os.path.join(DATASETS_ABSPATH, dataset_name, sample_basename)
                cv2.imwrite(output_fn, cropped_im)
                # os.remove(sample_abspath)

    def resize_up_dataset(self, dataset_name):
        for sample_basename in tqdm(os.listdir(os.path.join(DATASETS_ABSPATH, dataset_name))):
            sample_abspath = os.path.join(DATASETS_ABSPATH, dataset_name, sample_basename)
            sample_im = cv2.imread(sample_abspath)
            try:
                (h, w) = sample_im.shape[:2]
                if h != UPSAMPLE_SIZE:
                    print(f'Upsampling {sample_basename}')
                    resized = cv2.resize(sample_im, (UPSAMPLE_SIZE, UPSAMPLE_SIZE), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(sample_abspath, resized)
            except:
                print(f'Erred on {sample_basename}')

    def mirror_dataset(self, dataset_name):
        for sample_basename in tqdm(os.listdir(os.path.join(DATASETS_ABSPATH, dataset_name))):
            sample_abspath = os.path.join(DATASETS_ABSPATH, dataset_name, sample_basename)
            sample_im = cv2.imread(sample_abspath)
            sample_basename_no_ext = sample_basename.split('.')[0]
            flipped_basename = f'{sample_basename_no_ext}_flipped.jpg'
            flipped_abspath = os.path.join(DATASETS_ABSPATH, dataset_name, flipped_basename)
            if not os.path.isfile(flipped_abspath):
                # print(f'Flipping {sample_basename}')
                flipped = cv2.flip(sample_im, 1)
                cv2.imwrite(flipped_abspath, flipped)

    def delete_flipped(self, dataset_name):
        for sample_basename in tqdm(os.listdir(os.path.join(DATASETS_ABSPATH, dataset_name))):
            sample_basename_no_ext = sample_basename.split('.')[0]
            if sample_basename_no_ext.endswith('flipped'):
                sample_abspath = os.path.join(DATASETS_ABSPATH, dataset_name, sample_basename)
                # print(f'Removing {sample_basename}')
                os.remove(sample_abspath)
