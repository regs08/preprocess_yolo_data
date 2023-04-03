import os
import shutil
import random
import numpy as np

from preprocess_yolo_data.default_param_configs import label_to_id_map


def split_folder_into_train_val_test(folder_path, output_folder, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Splits a folder into train, val, and Pinot-noir subdirectories.

    Arguments:
    folder_path -- the path to the folder to split
    output_folder -- the path to the output folder where the train, val, (test) subdirectories will be created
    train_ratio -- the ratio of files to include in the train subdirectory (default 0.8)
    val_ratio -- the ratio of files to include in the validation subdirectory (default 0.1)
    test_ratio -- the ratio of files to include in the test subdirectory (default 0.1)
    """
    # create output folders
    train_folder = os.path.join(output_folder, 'train')
    val_folder = os.path.join(output_folder, 'val')
    test_folder = os.path.join(output_folder, 'test')
    for folder in [train_folder, val_folder, test_folder]:
        os.makedirs(os.path.join(folder, 'images'))
        os.makedirs(os.path.join(folder, 'labels'))

    # get list of files
    files = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            if filename == 'classes.txt':
                continue
            else:
                files.append(filename[:-4]) # remove the extension

    # shuffle the files
    random.shuffle(files)

    # split into train, val, and Pinot-noir sets
    num_files = len(files)
    num_train = int(train_ratio * num_files)
    num_val = int(val_ratio * num_files)
    num_test = num_files - num_train - num_val
    train_files = files[:num_train]
    val_files = files[num_train:num_train+num_val]
    test_files = files[num_train+num_val:]

    # move files to output folders
    for file_list, folder_name in [(train_files, train_folder), (val_files, val_folder), (test_files, test_folder)]:
        for file in file_list:
            # move label file
            label_file = file + '.txt'
            src_path = os.path.join(folder_path, label_file)
            dst_path = os.path.join(folder_name, 'labels', label_file)
            shutil.move(src_path, dst_path)

            # move image file
            for ext in ['.jpg', '.png', '.PNG', '.JPG']:
                image_file = file + ext
                src_path = os.path.join(folder_path, image_file)
                if os.path.exists(src_path):
                    dst_path = os.path.join(folder_name, 'images', image_file)
                    shutil.move(src_path, dst_path)
                    break  # break out of loop once the image is found and moved
    return train_folder, val_folder, test_folder


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
    for folder_name in ['train', 'val', 'test']:
        images_folder = os.path.join(output_folder, folder_name, 'images')
        labels_folder = os.path.join(output_folder, folder_name, 'labels')
        os.makedirs(images_folder, exist_ok=True)
        os.makedirs(labels_folder, exist_ok=True)

    # Shuffle the data list
    random.shuffle(data_list)

    # Split the data into train, val, and test sets
    num_data = len(data_list)
    num_train = int(num_data * train_ratio)
    num_val = int(num_data * val_ratio)
    num_test = num_data - num_train - num_val

    train_data = data_list[:num_train]
    val_data = data_list[num_train:num_train+num_val]
    test_data = data_list[num_train+num_val:]

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

import os
import random
import shutil


def create_folder_structure(output_folder):
    """
    Creates the train and val subdirectories within the output folder,
    as well as the images subdirectory within each subdirectory.
    """
    train_folder = os.path.join(output_folder, 'train')
    val_folder = os.path.join(output_folder, 'val')
    for folder in [train_folder, val_folder]:
        os.makedirs(os.path.join(folder, 'images'))
    return train_folder, val_folder


def get_file_list(folder_path):
    """
    Returns a list of all the image files in the specified folder.
    """
    files = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.JPG') or filename.endswith('.PNG'):
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
    train_folder, val_folder = create_folder_structure(output_folder)
    file_list = get_file_list(image_folder)
    train_files, val_files = split_file_list(file_list, train_ratio=train_ratio)
    move_files(train_files, image_folder, train_folder)
    move_files(val_files, image_folder, val_folder)

    return train_folder, val_folder