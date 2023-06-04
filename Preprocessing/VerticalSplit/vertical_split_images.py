from yolo_data.WritingRenamingFile.writing_to_file_utils import insert_string_before_extension
from yolo_data.LoadingData.load_utils import glob_image_files, get_annotation_path
from yolo_data.yolo2pvoc import convert_yolo_to_pascal_voc
from yolo_data.Preprocessing.CropImage.crop_utils import get_bbox_extreme_with_min_pixel_value
from yolo_data.LoadingData.load_utils import get_class_id_bbox_seg_from_yolo
from yolo_data.Preprocessing.VerticalSplit.utils import get_split_points, save_images_from__list_of_A_dict

import cv2
import albumentations as A
from PIL import Image
import os
from typing import Sequence
import numpy as np

"""
Splitting single images 
"""

def vertical_split_with_A(img, x_min, x_max,y_min, y_max, bboxes, class_labels, format):
    """
    takes in an image and returns a
    :param img: img as array
    :param x_min: min val dimensions for our split
    :param x_max: max val
    :param bboxes: bboxes from the image
    :param class_labels: class_labels as strs
    :return: a dict containing , image, bboxes, category_ids
    """
    aug = A.Compose([
        A.Crop(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max),
    ], bbox_params=A.BboxParams(format=format, min_visibility=0.3))

    # def check_bbox(bbox: Sequence) -> None:
    #     """Check if bbox boundaries are in range 0, 1 and minimums are lesser then maximums"""
    #     for name, value in zip(["x_min", "y_min", "x_max", "y_max"], bbox[:4]):
    #         if not 0 <= value <= 1 and not np.isclose(value, 0) and not np.isclose(value, 1):
    #             raise ValueError(
    #                 "Expected {name} !!!!!!!!!!!!!!!!{bbox} "
    #                 "to be in the range [0.0, 1.0], got {value}.".format(bbox=bbox, name=name, value=value)
    #             )
    #     x_min, y_min, x_max, y_max = bbox[:4]
    #     if x_max <= x_min:
    #         raise ValueError("x_max is less than or equal to x_min for bbox {bbox}.".format(bbox=bbox))
    #     if y_max <= y_min:
    #         raise ValueError("y_max is less than or equal to y_min for bbox {bbox}.".format(bbox=bbox))
    #
    # def check_bboxes(bboxes: Sequence[Sequence]) -> None:
    #     """Check if bboxes boundaries are in range 0, 1 and minimums are lesser then maximums"""
    #     for bbox in bboxes:
    #         check_bbox(bbox)
    # check_bboxes(bboxes)

    vertical_split_image = aug(image=img, bboxes=bboxes, category_ids=class_labels)
    #to take up less memory we put as PIL object
    # look here abnns aregetting saved as p_voc
    image_h, image_w, _ = vertical_split_image['image'].shape
    vertical_split_image['bboxes'] = [pascal_voc_to_yolo(box, image_w, image_h) for box in vertical_split_image['bboxes'] ]

    vertical_split_image['image'] = Image.fromarray(cv2.cvtColor(vertical_split_image['image'], cv2.COLOR_BGR2RGB))
    # prinplit_image['bboxes'])
    return vertical_split_image


def vertical_split_with_intervals(img, intervals, bboxes, class_labels, **args):
    """
    splits the image using the albumentations library the x values are gotten from intervals. the y values can be given
    their default is 0, hieght of the image
    :param img: path or arr, if path we add a filename to our dict
    :param intervals: the poitns where we will crop
    :param bboxes: the bboxes, default is yolo format, just coords we will add the class label onto the 5th index.
    :param class_labels: list of the class ids, ints
    :return: a list of dictionaries containing image (PIL) bboxes and cat_ids
    """

    add_filename = False
    if isinstance(img, str):
        img_path = img
        img = cv2.imread(img)
        add_filename = True

    format = args.get('format', 'pascal_voc')
    y_min = args.get('y_min', 0)
    y_max = img.shape[0]
    # getting the "ValueError: Your 'label_fields' are not valid - them must have same names as params in dict" so getting rid
    # adding the label on to the end of the box

    for i, box in enumerate(bboxes):
        box.append(class_labels[i])

    split_images = []
    for i in range(len(intervals) -1):
        x_min = intervals[i]
        x_max =intervals[i+1]
        if y_min < 0 : y_min = 0
        split_image = vertical_split_with_A(img=img,
                                            x_min=x_min,
                                            x_max=x_max,
                                            y_min=y_min,
                                            y_max=y_max,
                                            bboxes=bboxes, class_labels=class_labels, format=format)

        if add_filename:
            img_filename = os.path.basename(img_path)
            file_no = f'_{i}'
            split_image['filename'] = insert_string_before_extension(img_filename, file_no)

        split_images.append(split_image)
    return split_images

"""
Splitting images in folder 
"""

#note need to update the function for get_split_points function here so that it works
def split_images_in_folder(image_folder, interval, save_folder, ann_folder='',
                          save=True, min_boxes=1, bbox_extremes=False, min_pixel_value=1280):
    """
    takes in an image folder (and ann_folder default is image folder) splits the images vertically using the points provided
    by the interval. note the default values for the y coords is 0 and height of the image
    :param image_folder: location of images
    :param interval: interval for the split
    :param save_folder: save folder for the cropped images
    :param ann_folder: if theres a different folder for annotations..
    :param save: save the images
    :param min_boxes: check so that all images have an instance present
    :return:
    """

    #if anns are in image folder
    if not ann_folder:
        ann_folder=image_folder
    image_paths = glob_image_files(image_folder)
    print(f'Splitting {len(image_paths)} image(s) in .../{os.path.basename(image_folder)}')
    for img_path in image_paths:
        img = cv2.imread(img_path)
        #getting the coords for our vertical split for each image 
        h,w, _ = img.shape
        #returns our annotations in a yolo foramtted.txt
        ann_path = get_annotation_path(img_path, ann_folder)
        #reading from text file
        class_ns, bboxes, _ = get_class_id_bbox_seg_from_yolo(ann_path)

        # instead of loading in the text file we can just load in bbox extremes here
        if bbox_extremes:
            p_voc_boxes = [convert_yolo_to_pascal_voc(yolo_box=box, image_width=w, image_height=h) for box in bboxes]
            #note ymax is producing wei
            xmin, y_min, xmax, y_max = get_bbox_extreme_with_min_pixel_value(p_voc_boxes, min_pixel_value)
            split_intervals = get_split_points(xmin, xmax, interval)
            split_images = vertical_split_with_intervals(img=img_path,
                                                            bboxes=p_voc_boxes,
                                                            class_labels=class_ns,
                                                            intervals=split_intervals,
                                                            y_min=y_min,
                                                            y_max=y_max)

        else:
            split_intervals = get_split_points(0,w, interval)
            split_images = vertical_split_with_intervals(img=img_path,
                                                         bboxes=bboxes,
                                                         class_labels=class_ns,
                                                         intervals=split_intervals)

        if save:
            #save images from the dict key filename and save folder for each image in the list
            save_images_from__list_of_A_dict(split_images, save_folder, min_boxes)



# Convert Pascal_Voc bb to Yolo
def pascal_voc_to_yolo(box, image_w, image_h):
    x1, y1, x2,y2 = box[:4]
    return [((x2 + x1)/(2*image_w)), ((y2 + y1)/(2*image_h)), (x2 - x1)/image_w, (y2 - y1)/image_h]


