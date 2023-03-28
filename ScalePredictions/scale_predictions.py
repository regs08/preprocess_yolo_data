import numpy as np

def scale_bboxes(original_image, pred_image, bboxes):
    """
    Scales the bounding boxes of the predicted image to the original image and returns the bounding boxes
    in normalized YOLO format.

    Parameters:
    original_image (numpy array): The original image.
    pred_image (numpy array): The predicted image.
    bboxes (numpy array): The bounding boxes of the predicted image, in the format [xmin, ymin, xmax, ymax].

    Returns:
    numpy array: The scaled bounding boxes, in normalized YOLO format [x_center, y_center, width, height].
    """

    # Get the dimensions of the original and predicted images
    original_h, original_w, _ = original_image.shape
    pred_h, pred_w, _ = pred_image.shape

    # Calculate the scaling factors for the height and width
    scale_h = original_h / pred_h
    scale_w = original_w / pred_w

    # Scale the bounding boxes
    scaled_bboxes = np.zeros_like(bboxes)
    scaled_bboxes[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2 * scale_w / original_w
    scaled_bboxes[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / 2 * scale_h / original_h
    scaled_bboxes[:, 2] = (bboxes[:, 2] - bboxes[:, 0]) * scale_w / original_w
    scaled_bboxes[:, 3] = (bboxes[:, 3] - bboxes[:, 1]) * scale_h / original_h

    return scaled_bboxes
