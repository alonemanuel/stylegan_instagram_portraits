## Assuming a csv file has:
## Image Name (ImageID) in column 1 (line[0])
## Full Resolution URL (OriginalURL) in column 3 (line[2])

import sys
import urllib.request
import csv
import os.path

# folder containing directories per instagram hashtag
# e.g 'blogger_girl'
SCRAPE_FOLDERS_PATH = "G:\My Drive\projects\\are_you_real\samples\instagram_scrape"
JPG_FOLDER_EXT = '_jpgs'
SCRAPE_FOLDER_EXT = '_scrape'
DISPLAY_URL_INDEX = 5
IS_RUN_FROM_CMD = len(sys.argv) > 1


def prep_dirs(dir_name):
    dir_abs_path = os.path.join(SCRAPE_FOLDERS_PATH, dir_name)
    scrape_folder = os.path.join(dir_abs_path, dir_name + SCRAPE_FOLDER_EXT)
    assert os.path.isdir(scrape_folder)
    files_in_scrape_folder = os.listdir(scrape_folder)
    assert len(files_in_scrape_folder) == 1
    scrape_csv_file = os.path.join(scrape_folder, files_in_scrape_folder[0])
    assert os.path.isfile(scrape_csv_file)

    output_folder_path = os.path.join(dir_abs_path, dir_name + JPG_FOLDER_EXT)
    os.makedirs(output_folder_path, exist_ok=True)
    return scrape_csv_file, output_folder_path


def save_csvs_to_jpg(dirs_to_parse=None):
    for dir_name in os.listdir(SCRAPE_FOLDERS_PATH):
        if dirs_to_parse and dir_name not in dirs_to_parse:
            print(f'Skipping dir: {dir_name}...\n')
            continue
        else:
            print(f'Processing dir: {dir_name}...\n')

        scrape_csv_file, output_folder_path = prep_dirs(dir_name)
        with open(scrape_csv_file, 'r', encoding='utf8') as csv_file:
            reader = csv.reader(csv_file)
            next(reader)  # skip headers
            for item_n, line in enumerate(reader):
                output_fn = f'{dir_name}_{item_n}.jpg'
                print(f'\nProcessing item {output_fn}')
                output_absfn = os.path.join(output_folder_path, output_fn)
                image_url = line[DISPLAY_URL_INDEX]

                if os.path.isfile(output_absfn):
                    print(f"Image skipped for {output_fn}")
                else:
                    try:
                        urllib.request.urlretrieve(image_url, output_absfn)
                        print(f"Image saved for {output_fn}")
                    except Exception as e:
                        print(f"No result for {output_fn}, error:\n{e}")


save_csvs_to_jpg()
