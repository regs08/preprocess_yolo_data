import numpy as np
import os
import random
import shutil

from yolo_data.default_param_configs import label_to_id_map, image_exts
from yolo_data.LoadingData.load_utils import glob_text_files
from yolo_data.WritingRenamingFile.writing_to_file_utils import create_model_train_folder_structure
from yolo_data.SplittingFolders.utils import get_train_val_test_split_ratio


def split_folder_into_train_val_test(folder_path, output_folder, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Splits a folder into train, val, and test subdirectories.

    Arguments:
    folder_path -- the path to the folder to split
    output_folder -- the path to the output folder where the train, val, (test) subdirectories will be created
    train_ratio -- the ratio of files to include in the train subdirectory (default 0.8)
    val_ratio -- the ratio of files to include in the validation subdirectory (default 0.1)
    test_ratio -- the ratio of files to include in the test subdirectory (default 0.1)
    """
    # create output folders

    train_folder, val_folder, test_folder = create_model_train_folder_structure(output_folder)
    # get list of files
    files =  glob_text_files(folder_path)
    filenames = []
    for filename in files:
        if filename == 'classes.txt':
            continue
        else:
            filenames.append(os.path.basename(filename)[:-4]) # remove the extension
    print(filenames)
    # shuffle the files
    random.shuffle(filenames)

    #splitting our data into with the given ratios..
    train_files, val_files, test_files = get_train_val_test_split_ratio(filenames, train_ratio, val_ratio)

    # move files to output folders
    for file_list, folder_name in [(train_files, train_folder), (val_files, val_folder), (test_files, test_folder)]:
        for file in file_list:
            # move label file
            label_file = file + '.txt'
            src_path = os.path.join(folder_path, label_file)
            dst_path = os.path.join(folder_name, 'labels', label_file)
            shutil.move(src_path, dst_path)

            # move image file
            for ext in image_exts:
                image_file = file + ext
                src_path = os.path.join(folder_path, image_file)
                if os.path.exists(src_path):
                    dst_path = os.path.join(folder_name, 'images', image_file)
                    shutil.move(src_path, dst_path)
                    break  # break out of loop once the image is found and moved

    return train_folder, val_folder, test_folder


"""
splitting unsaved data into folders 
"""


def split_data_into_folders(data_list,
                            output_folder,
                            label_to_id_map=label_to_id_map,
                            train_ratio=0.8,
                            val_ratio=0.1,
                            test_ratio=0.1,
                            normilize=False,
                            ):
    """
    :param data_list: list of dicts containing a PIL image, bbox data, class_labels, filename, file_prefix
    :param output_folder: save folder
    :param label_to_id_map: label_id map to map the label
    :param train_ratio: train split ratio
    :param val_ratio: ..
    :param test_ratio: ..
    :param normilize: we assume the data is normilize e,g x_center/image_width set to True if not
    :return:
    """
    # Create the output folders
    train_folder, val_folder, test_folder = create_model_train_folder_structure(output_folder)

    # Shuffle the data list
    random.shuffle(data_list)

    # Split the data into train, val, and test sets
    train_data, val_data, test_data = get_train_val_test_split_ratio(data_list, train_ratio, val_ratio)


    # Save the data to the output folders
    for folder_name, folder_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        for i, data_dict in enumerate(folder_data):
            # Get the image, filename, and bounding box data
            image = data_dict['image']
            if isinstance(image, np.ndarray):
              continue
            else:
              filename = data_dict['file_name']

              bbox_data = data_dict['bboxes']
              class_labels = data_dict['class_labels']

              # Save the image to the images folder

              save_filename = data_dict['filename_prefix'] +'_'+ os.path.splitext(filename)[0]
              image_path = os.path.join(output_folder, folder_name, 'images', save_filename+ '.jpg')
              image.save(image_path)

              # Save the bounding box data to the labels folder
              label_path = os.path.join(output_folder, folder_name, 'labels', save_filename + '.txt')
              with open(label_path, 'w') as f:
                  for bbox, class_label in zip(bbox_data, class_labels):
                      class_id = label_to_id_map[class_label]
                      x, y, w, h = bbox
                      if normilize:
                          x = (x + w/2) / image.width
                          y = (y + h/2) / image.height
                          w = w / image.width
                          h = h / image.height

                      f.write(f'{int(class_id)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n')


"""
for splitting just images 
"""

def create_image_split_folder_structure(output_folder):
    """
    Creates the train and val subdirectories within the output folder,
    as well as the images subdirectory within each subdirectory.
    """
    folders = []
    folders.append(os.path.join(output_folder, 'train'))
    folders.append(os.path.join(output_folder, 'val'))

    for folder in folders:
        os.makedirs(os.path.join(folder, 'images'))

    return folders


def get_file_list(folder_path):
    """
    Returns a list of all the image files in the specified folder.
    """
    files = []
    for filename in os.listdir(folder_path):
        for ext in image_exts:
            if filename.endswith(ext):
                files.append(filename)
    return files


def split_file_list(file_list, train_ratio=0.8, val_ratio=0.2):
    """
    Splits the given file list into train and val sets based on the specified ratios.
    """
    random.shuffle(file_list)
    num_files = len(file_list)
    num_train = int(train_ratio * num_files)
    num_val = num_files - num_train
    train_files = file_list[:num_train]
    val_files = file_list[num_train:]
    return train_files, val_files


def move_files(file_list, folder_path, folder_name):
    """
    Moves the files in the given file list from the source folder to the destination folder.
    """
    for file in file_list:
        # move image file
        src_path = os.path.join(folder_path, file)
        dst_path = os.path.join(folder_name, 'images', file)
        shutil.move(src_path, dst_path)


def split_folder_into_train_val(image_folder, output_folder, train_ratio=0.8):
    """
    Splits an image folder into train and val subdirectories.

    Arguments:
    image_folder -- the path to the image folder to split
    output_folder -- the path to the output folder where the train and val subdirectories will be created
    train_ratio -- the ratio of files to include in the train subdirectory (default 0.8)
    """
    train_folder, val_folder = create_image_split_folder_structure(output_folder)
    file_list = get_file_list(image_folder)
    train_files, val_files = split_file_list(file_list, train_ratio=train_ratio)
    move_files(train_files, image_folder, train_folder)
    move_files(val_files, image_folder, val_folder)

    return train_folder, val_folder


"""
Splitting imageds into batches 
"""


def split_images_into_batches(source_folder, batch_size, destination_folder):
    # Create the destination folders
    num_images = len(os.listdir(source_folder))
    num_batches = (num_images + batch_size - 1) // batch_size
    #used to return the paths
    dest_folders = []
    for i in range(num_batches):
        folder_name = f"batch_{i + 1}"
        folder_path = os.path.join(destination_folder, folder_name)
        dest_folders.append(folder_path)

        os.makedirs(folder_path)



    # Loop through the source folder and move each image to a batch folder
    batch_index = 0
    num_images_in_batch = 0

    for i, file_name in enumerate(sorted(os.listdir(source_folder))):
        # Get the path of the source file
        source_path = os.path.join(source_folder, file_name)
        # Get the path of the destination folder for the current batch
        batch_folder = f"batch_{batch_index + 1}"
        folder_path = os.path.join(destination_folder, batch_folder)
        # Get the path of the destination file
        destination_path = os.path.join(folder_path, file_name)
        # Move the file to the destination folder
        shutil.move(source_path, destination_path)

        num_images_in_batch += 1

        # Check if we've reached the end of the current batch
        if num_images_in_batch == batch_size:
            batch_index += 1
            num_images_in_batch = 0
        elif i == num_images - 1:
            # Last batch may not be full, so check if there are leftover images and move them to the previous batch
            num_images_leftover = num_images % batch_size
            if num_images_leftover > 0:
                # Get the path of the destination folder for the previous batch
                previous_batch_folder = f"batch_{batch_index}"
                previous_folder_path = os.path.join(destination_folder, previous_batch_folder)
                # Move the leftover images to the previous batch folder
                for leftover_file_name in os.listdir(folder_path):
                    shutil.move(os.path.join(folder_path, leftover_file_name),
                                os.path.join(previous_folder_path, leftover_file_name))
                # Delete the current batch folder, since it's now empty
                dest_folders.pop()
                os.rmdir(folder_path)

    return dest_folders

folder = "/Users/cole/PycharmProjects/Forgit/yolo_data/Preprocessing/CropImage/test_crop/test_images/save"
split_folder_into_train_val_test(folder, folder)
