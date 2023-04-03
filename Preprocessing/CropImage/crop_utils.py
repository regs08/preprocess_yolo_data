import numpy as np
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

"""
Bounding box business 
"""


def check_cropped_image_for_min_pixel_value(min_x, min_y, max_x, max_y, min_pixel_value):

    # Calculate the width and height of the cropped image
    width = max_x - min_x
    height = max_y - min_y

    # Check if the width and height of the cropped image meet the minimum pixel value
    if width < min_pixel_value:
        min_x -= (min_pixel_value - width) // 2
        max_x += (min_pixel_value - width + 1) // 2
    if height < min_pixel_value:
        min_y -= (min_pixel_value - height) // 2
        max_y += (min_pixel_value - height + 1) // 2

    return min_x, min_y, max_x, max_y


def get_bbox_extremes(bboxes):
    """
    Given a list of bounding boxes (xmin, ymin, xmax, ymax), return the lowest xmin and ymin and highest xmax and ymax
    across all the bounding boxes.

    Args:
    - bboxes: list of tuples representing bounding boxes in the format (xmin, ymin, xmax, ymax)

    Returns:
    - Tuple containing (lowest xmin, lowest ymin, highest xmax, highest ymax)
    """
    min_x = min([bbox[0] for bbox in bboxes])
    min_y = min([bbox[1] for bbox in bboxes])
    max_x = max([bbox[2] for bbox in bboxes])
    max_y = max([bbox[3] for bbox in bboxes])

    return (min_x, min_y, max_x, max_y)


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



