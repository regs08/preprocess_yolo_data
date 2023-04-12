"""
Bounding box business 
"""


def check_bbox_extremes_for_min_pixel_value(bbox, min_pixel_value):

    # Calculate the width and height of the cropped image
    min_x, min_y, max_x, max_y = bbox
    width = max_x - min_x
    height = max_y - min_y

    # Check if the width and height of the cropped image meet the minimum pixel value
    if width < min_pixel_value:
        min_x -= (min_pixel_value - width) // 2
        max_x += (min_pixel_value - width + 1) // 2
    if height < min_pixel_value:
        min_y -= round(min_pixel_value - height) // 2
        max_y += round(min_pixel_value - height + 1) // 2
    return (int(min_x), int(min_y), int(max_x), int(max_y))


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


def get_bbox_extreme_with_min_pixel_value(bboxes, min_pixel_value):
    bbox_extremes = get_bbox_extremes(bboxes)
    bbox_extremes_with_min_pixel_value = check_bbox_extremes_for_min_pixel_value(bbox_extremes, min_pixel_value)
    return bbox_extremes_with_min_pixel_value


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



