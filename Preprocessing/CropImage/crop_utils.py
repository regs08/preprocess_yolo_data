import numpy as np
from preprocess_yolo_data.LoadingData.load_utils import get_yolo_bboxes_from_txt_file
from preprocess_yolo_data.default_param_configs import min_x_crop, min_y_crop #1280, 1280


def get_min_max_bbox_coords_from_bbox_arr(bbox_arr, min_x_crop=min_x_crop, min_y_crop=min_y_crop):
    """
    takes in an array of bounding boxes. returns the min from each of  the columns: xmin, ymin, xmax, ymax
    :param arr: array of bounding boxes of a given image
    :return: a list of [xmin, ymin, xmax, ymax] of the bbox
    """

    xmin = np.min(bbox_arr[:, 0:1].astype(int))
    ymin = np.min(bbox_arr[:, 1:2].astype(int))


    xmax = np.max(bbox_arr[:, 2:3].astype(int))
    ymax = np.max(bbox_arr[:, 3:4].astype(int))

    #adjusting for appropiate pixel value
    min_width = min_x_crop + xmin
    min_height = min_y_crop + ymin

    #setting our dims to the shape expected by our network
    if xmax <min_width:
        xmax = min_width
    if ymax < min_height:
        ymax=min_height

    coords = [xmin, ymin, xmax, ymax]
    return coords


def get_bbox_center(bbox):
    """
    Given a bounding box in Pascal VOC format, returns its center point.

    Args:
        bbox: A tuple of floats in the format (xmin, ymin, xmax, ymax), representing
              the coordinates of the top-left and bottom-right corners of the bounding box.

    Returns:
        A tuple of floats in the format (x_center, y_center), representing the coordinates
        of the center point of the bounding box.
    """
    xmin, ymin, xmax, ymax = bbox
    x_center = int((xmin + xmax) / 2)
    y_center = int((ymin + ymax) / 2)
    return (x_center, y_center)



