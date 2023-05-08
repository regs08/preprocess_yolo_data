"""
Utility functions for reading from files

"""
import cv2
import numpy as np
import os
import random
import glob

from yolo_data.default_param_configs import image_exts
from yolo_data.Conversions.convert_yolo_pascal_voc import convert_yolo_to_pascal_voc


"""
globbing paths 
"""


def glob_text_files(ann_folder, ext='.txt'):
    return glob.glob(os.path.join(ann_folder, '*' + ext))


def glob_image_files(image_folder, exts=image_exts):
    image_paths = []
    for ext in exts:
        # Use glob to search for files with the current extension
        files = glob.glob(os.path.join(image_folder,'*' + ext))
        # Extend the matching_files list with the found file paths
        image_paths.extend(files)

    return image_paths


"""
Loading data into pascal voc format 
"""


def load_coords_in_pascal_voc_from_yolo_txt_file(txt_path, img_path):
    """
    takes in our yolo bboxes and converts them into pascal voc image by image. more easy to plot etc..
    :param txt_path: text path of the annotation
    :param img_path: image path of the corresponding image
    :return:
    """
    yolo_boxes, class_ns = get_yolo_bboxes_from_txt_file(txt_path)
    img = cv2.imread(img_path).shape[:2]

    #note the switching of height and width
    pascal_voc_boxes = [convert_yolo_to_pascal_voc(img, yolo_b) for yolo_b in yolo_boxes]

    return np.asarray(pascal_voc_boxes), class_ns

"""
loading anns from image paths 
"""


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
        ann_path = get_annotation_path(image_path, ann_dir)
        annotation_paths.append(ann_path)
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
    assert os.path.exists(annotation_path), f'PATH: {annotation_path} \nDOES NOT EXIST'
    return annotation_path


"""
reading from yolo file 
"""


def get_class_id_bbox_seg_from_yolo(txt_path):
    """
    gets each line as a seperate bbox
    :param txt_file: the text file corresponding to the image
    :return: class_id and bbox or class_id, boox, seg
    """
    lines = read_txt_file(txt_path)
    yolo_bboxes, class_ns, segs = convert_text_lines_to_yolo_format(lines)

    return yolo_bboxes, class_ns, segs


def read_txt_file(txt_path):
    txt_file = open(txt_path, "r")
    lines = txt_file.read().splitlines()
    return lines


def convert_text_lines_to_yolo_format(lines):
    bboxes = []
    class_ns = []
    segs = []
    for idx, line in enumerate(lines):
        value = line.split()
        cls = int(value[0])
        x = float(value[1])
        y = float(value[2])
        w = float(value[3])
        h = float(value[4])
        #if we have segmentation data append it
        if len(line) > 5:
            segs.append([float(i) for i in value[5:]])

        bboxes.append([x,y,w,h])
        class_ns.append(cls)

    return bboxes, class_ns, segs


def extract_bounding_boxes(yolo_file):
    with open(yolo_file, 'r') as f:
        lines = f.readlines()

    bounding_boxes = []
    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.split()[1:5])
        bounding_boxes.append((x_center, y_center, width, height))

    return bounding_boxes
"""
getting dict maps from text 
"""


def read_classes_from_file(filename):
    """
    returns id to label and label to id dict
    :param filename: path to file
    :return: id to label and label to id dict
    """
    id_to_cat_map = {}
    label_to_id_map = {}
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            class_name = line.strip()
            id_to_cat_map[i] = class_name
            label_to_id_map[class_name] = i
    return label_to_id_map, id_to_cat_map


"""
Selecting random files 
"""

def select_random_files(image_folder, text_folder):
    # Get a list of all the image files in the image folder with valid extensions
    image_files = glob_image_files(image_folder)

    # Randomly select a text file and an image file
    selected_image_file = random.choice(image_files)

    selected_text_filename = os.path.splitext(os.path.basename(selected_image_file))[0] + '.txt'
    selected_text_file = os.path.join(text_folder, selected_text_filename)
    # Return the selected text file and image file
    return os.path.join(image_folder, selected_image_file), os.path.join(text_folder, selected_text_file)
