"""
python file utilizing the crop utils to crop on a single image
"""

from PIL import Image
from preprocess_yolo_data.Preprocessing.CropImage.crop_utils import get_bbox_extremes, \
    check_bbox_extremes_for_min_pixel_value


def crop_image_via_bbox_extremes(image_path, bboxes, min_pixel_value):
    """
    Crop an image using bounding box extremes and return a PIL Image object.

    Args:
    - image_path: path to the input image
    - bboxes: list of bounding boxes in the format (xmin, ymin, xmax, ymax)
    - min_pixel_value: minimum pixel value for the width and height of the cropped image

    Returns:
    - PIL Image object containing the cropped image
    """
    # Open the image using PIL
    image = Image.open(image_path)

    # Get the lowest xmin, ymin and highest xmax, ymax from the bounding boxes
    min_x, min_y, max_x, max_y = get_bbox_extremes(bboxes)
    min_x, min_y, max_x, max_y = check_bbox_extremes_for_min_pixel_value(min_x, min_y, max_x, max_y, min_pixel_value)
    # Crop the image using the bbox extremes
    cropped_image = image.crop((min_x, min_y, max_x, max_y))

    return cropped_image


def split_image_vertically(image, **args):
    """
    loads an img from a path as a PIL image. splits the image into n splits based on the split width.
    if there is a remainder it as added on to the last image.
    :param image_path:
    :param split_width:
    :return: split images
    """

    #Setting default value
    split_width = args.get('split_width', 1280)
    # Load the image
    if isinstance(image, str):
        # Load the image if a path is provided
        image = Image.open(image)
    else:
        # Convert the array to a PIL Image object
        image = Image.fromarray(image)

    # Get the image size
    width, height = image.size
    # Calculate the number of splits
    split = width // split_width
    # Determine if there is a remainder and adjust the last split width accordingly
    remainder = width % split_width

    # Create a list to store the split images
    split_images = []
    # Iterate over the splits
    for i in range(split):
        # Determine the left and right coordinates of the split
        left = i * split_width
        if i+1 == split and remainder!=0:
            right=left+split_width+remainder
        else:
            right = left + split_width
        # Crop the split image
        split_image = image.crop((left, 0, right, height))
        # Add the split image to the list
        split_images.append(split_image)

    # Return the split images
    return split_images

