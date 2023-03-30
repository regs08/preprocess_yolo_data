"""
Utility functions for reading yolo text files 

"""
import cv2
import numpy as np
import os
import random


def get_yolo_bboxes_from_txt_file(txt_path):
    """
    gets each line as a seperate bbox
    :param txt_file: the text file corresponding to the image
    :return:
    """
    lines = read_txt_file(txt_path)
    yolo_bboxes, class_ns = convert_text_lines_to_yolo_format(lines)

    return yolo_bboxes, class_ns


def read_txt_file(txt_path):
    txt_file = open(txt_path, "r")
    lines = txt_file.read().splitlines()
    return lines


def convert_text_lines_to_yolo_format(lines):
    bboxes = []
    class_ns = []
    for idx, line in enumerate(lines):
        value = line.split()
        x = y = w = h = cls = None
        cls = int(value[0])
        x = float(value[1])
        y = float(value[2])
        w = float(value[3])
        h = float(value[4])


        bboxes.append([x,y,w,h])
        class_ns.append(cls)

    return bboxes, class_ns

"""
#######
Loading data into pascal voc format 
"""
from preprocess_yolo_data.Conversions.convert_yolo_pascal_voc import convert_yolo_to_pascal_voc


def load_coords_in_pascal_voc_from_yolo_txt_file(txt_path, img_path):
    """
    takes in our yolo bboxes and converts them into pascal voc image by image. more easy to plot etc..
    :param txt_path: text path of the annotation
    :param img_path: image path of the corresponding image
    :return:
    """
    yolo_boxes, class_ns = get_yolo_bboxes_from_txt_file(txt_path)
    img_h, img_w = cv2.imread(img_path).shape[:2]

    #note the switching of height and width
    pascal_voc_boxes = [convert_yolo_to_pascal_voc((img_w, img_h), yolo_b) for yolo_b in yolo_boxes]

    return np.asarray(pascal_voc_boxes), class_ns


def load_image_filenames_from_folder(folder_path):
    """
    Load filenames of image files with extensions .jpg, .png, and .jpeg from a folder.

    Args:
        folder_path (str): The path to the folder containing the images.

    Returns:
        A list of image file names.
    """
    allowed_extensions = ['.jpg', '.png', '.jpeg']
    image_filenames = []
    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in allowed_extensions):
            image_path = os.path.join(folder_path, filename)
            image_filenames.append(image_path)
    return image_filenames


def get_annotation_paths(image_paths, ann_dir):
    """
    Given a list of image paths and an annotation directory, return a list of annotation file paths
    with the same name as the image files but with a .txt extension.

    Args:
        image_paths (list): A list of image file paths.
        ann_dir (str): The directory where the annotation files should be saved.

    Returns:
        A list of annotation file paths.
    """
    annotation_paths = []
    for image_path in image_paths:
        annotation_paths.append(get_annotation_path(image_path, ann_dir))
    return annotation_paths


def get_annotation_path(image_path, ann_dir):
    """
    Given an image path and an annotation directory, return the annotation file path
    with the same name as the image file but with a .txt extension.

    Args:
        image_path (str): The path to the image file.
        ann_dir (str): The directory where the annotation file should be saved.

    Returns:
        The annotation file path as a string.
    """
    basename = os.path.basename(image_path)
    annotation_name = os.path.splitext(basename)[0] + '.txt'
    annotation_path = os.path.join(ann_dir, annotation_name)
    return annotation_path


def select_random_files(text_folder, image_folder):
    # Get a list of all the text files in the text folder
    text_files = [f for f in os.listdir(text_folder) if f.endswith('.txt')]

    # Get a list of all the image files in the image folder with valid extensions
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Randomly select a text file and an image file
    selected_text_file = random.choice(text_files)
    selected_image_file = random.choice(image_files)

    # Return the selected text file and image file
    return os.path.join(text_folder, selected_text_file), os.path.join(image_folder, selected_image_file)

"""
reading from yolo file 
"""


def extract_bounding_boxes(yolo_file):
    with open(yolo_file, 'r') as f:
        lines = f.readlines()

    bounding_boxes = []
    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.split()[1:])
        bounding_boxes.append((x_center, y_center, width, height))

    return bounding_boxes
