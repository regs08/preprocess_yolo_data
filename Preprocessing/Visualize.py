import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np


from yolo_data.LoadingData.load_utils import get_class_id_bbox_seg_from_yolo
from yolo_data.Preprocessing.VerticalSplit.vertical_split_images import get_split_points

"""
########
Cropped images with bounding boxes
########
"""


def plot_transformed_images(transformed, format):
    """
    takes in a transformed object from the albumentations and plots it given a format. the supported ones now
    are yolo, pascal voc, albumentations
    Note this is assuming the transformed object has the keys: image, bboxes and class_labels

    :param transformed:
    :param format: how the bboxes are stored;
    :return:
    """
    for t in transformed:
        if 'class_labels' in t.keys():
            class_labels = t['class_labels']
        else:
            class_labels = t['category_ids']

        image, bboxes = t['image'], t['bboxes']
        if format=='yolo':
            plot_boxes_yolo_format(image, bboxes, class_labels)
        if format=='pascal_voc':
            plot_bounding_boxes_pascal_voc_format(image, bboxes, class_labels)
        if format=='albumentations':
            plot_image_with_bboxes_albumentations_format(image, bboxes, class_labels)


def plot_boxes_yolo_format(image, boxes, labels):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    h, w, _ = image.shape

    for box, label in zip(boxes, labels):
        center_x, center_y, width, height = box
        xmin = int((center_x - width/2) * w)
        ymin = int((center_y - height/2) * h)
        xmax = int((center_x + width/2) * w)
        ymax = int((center_y + height/2) * h)

        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)

        label_text = f"{label}"
        ax.text(xmin, ymin - 10, label_text, fontsize=12, color='g', backgroundcolor='w')

    plt.axis('off')
    plt.show()


def plot_bounding_boxes_pascal_voc_format(image_array, bboxes, labels):
    """
    from chat gpt
    Plot image with bounding boxes and labels.
    :param image_array: Numpy array representing the image.
    :param bounding_boxes: List of bounding boxes in Pascal VOC format [(xmin, ymin, xmax, ymax), ...].
    :param labels: List of corresponding labels for each bounding box ['label1', 'label2', ...].
    :return: None.
    """
    # Create figure and axis objects
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image_array)

    # Plot each bounding box
    for i, bbox in enumerate(bboxes):
        # Get the coordinates of the bounding box
        xmin, ymin, xmax, ymax = bbox

        # Create a Rectangle patch
        rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin),
                                 linewidth=2, edgecolor='r', facecolor='none')

        # Add the patch to the plot
        ax.add_patch(rect)

        # Add the label to the plot
        ax.text(xmin, ymin, labels[i], fontsize=12,
                bbox=dict(facecolor='red', alpha=0.5))

    # Show the plot
    plt.show()


def plot_image_with_bboxes_albumentations_format(image_array, bboxes, labels):
    """
    "" from chapt GPT
    Plot an image array with its bounding boxes and corresponding labels.

    Args:
        image_array (numpy.ndarray): The image array.
        bboxes (List[Tuple[float]]): A list of bounding boxes in the normalized Pascal VOC format (xmin, ymin, xmax, ymax).
        labels (List[str]): A list of labels corresponding to each bounding box.
    """
    # Create a figure and axes
    fig, ax = plt.subplots(1)

    # Plot the image
    ax.imshow(image_array)

    # Add the bounding boxes and labels
    for bbox, label in zip(bboxes, labels):
        # Convert the normalized bounding box coordinates to pixel coordinates
        xmin = int(bbox[0] * image_array.shape[1])
        ymin = int(bbox[1] * image_array.shape[0])
        xmax = int(bbox[2] * image_array.shape[1])
        ymax = int(bbox[3] * image_array.shape[0])

        # Create a Rectangle patch
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')

        # Add the patch to the axes
        ax.add_patch(rect)

        # Add the label to the patch
        ax.text(xmin, ymin, label, fontsize=12, color='r')

    # Show the plot
    plt.show()

"""
########
Plotting random example 
########
"""


def plot_random_image_with_yolo_annotations(image_folder, annotation_folder, format='yolo'):
    #format is assumed to be in yolo..
    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(image_folder) if
                   f.endswith(('.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG'))]

    # Randomly select an image file and its corresponding annotation file
    selected_image_file = random.choice(image_files)
    selected_annotation_file = os.path.splitext(selected_image_file)[0] + '.txt'

    # Read in the image and its corresponding annotation file
    image_path = os.path.join(image_folder, selected_image_file)
    ann_path = os.path.join(annotation_folder, selected_annotation_file)
    plot_image_with_boxes(image_path, ann_path)

"""
#######
Functions for colab 
#######
"""


def plot_boxes_yolo_format_in_colab(image, boxes, labels):
    from google.colab.patches import cv2_imshow

    h, w, _ = image.shape
    for box, label in zip(boxes, labels):
        center_x, center_y, width, height = box
        left = int((center_x - width/2) * w)
        top = int((center_y - height/2) * h)
        right = int((center_x + width/2) * w)
        bottom = int((center_y + height/2) * h)
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        label_text = f"{label}"
        label_size = cv2.getTextSize(label_text, font, 0.5, 2)[0]
        cv2.rectangle(image, (left, top - label_size[1]), (left + label_size[0], top), (0, 255, 0), -1)
        cv2.putText(image, label_text, (left, top), font, 0.5, (0, 0, 0), 2)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    cv2_imshow(image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plot_image_with_boxes(image_path, label_path):
    # Load the image
    image = cv2.imread(image_path)
    # Convert the image from BGR to RGB for Matplotlib
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    class_ids, bboxes, _  = get_class_id_bbox_seg_from_yolo(label_path)

    # Parse the label data and draw the boxes
    for box in bboxes:
        x_center, y_center, width, height = box
        x_min = int((x_center - width / 2) * image.shape[1])
        y_min = int((y_center - height / 2) * image.shape[0])
        x_max = int((x_center + width / 2) * image.shape[1])
        y_max = int((y_center + height / 2) * image.shape[0])
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Show the image with boxes using Matplotlib in Google Colab
    plt.imshow(image)
    plt.axis('off')
    plt.show()

"""
drawing vertical lines on image
"""


def draw_vertical_lines(image, points, color=(0, 255, 0), thickness=25):
    """
    Draw vertical lines on an image at given point(s).

    :param image: Image array or file path.
    :type image: numpy.ndarray or str
    :param points: List of points where vertical lines will be drawn. Each point is represented as (x, y) tuple.
    :type points: list of tuples
    :param color: Color of the lines in BGR format. Default is (0, 255, 0) (green).
    :type color: tuple, optional
    :param thickness: Thickness of the lines. Default is 1.
    :type thickness: int, optional
    :return: Image with vertical lines drawn.
    :rtype: numpy.ndarray
    """
    if isinstance(image, str):
        image = cv2.imread(image)

    for point in points:
        x, y = point
        cv2.line(image, (x, 0), (x, image.shape[0]), color, thickness)
    return image


def draw_vertical_lines_at_interval(image, interval, color=(0, 255, 0), thickness=25):
    """
    Draw vertical lines on an image at every n number of pixels.

    :param image: Image array or file path.
    :type image: numpy.ndarray or str
    :param interval: Interval in pixels at which lines will be drawn.
    :type interval: int
    :param color: Color of the lines in BGR format. Default is (0, 255, 0) (green).
    :type color: tuple, optional
    :param thickness: Thickness of the lines. Default is 1.
    :type thickness: int, optional
    :return: Image with vertical lines drawn at every n number of pixels.
    :rtype: numpy.ndarray
    """
    if isinstance(image, str):
        image = cv2.imread(image)

    height, width, _ = image.shape
    intervals = get_split_points(image, interval)

    for x in intervals:

        cv2.line(image, (x, 0), (x, height), color, thickness)
    return image

"""
plotting yolo instance segmentation 
"""


def plot_image_with_annotations(image_path, annotation_path):
    """
    Load an image and its annotations (bounding box and segmentation) in YOLO format,
    and plot the image with the annotations overlaid on top.

    Args:
        image_path (str): The path to the input image file.
        annotation_path (str): The path to the annotation file in YOLO format.

    Returns:
        None.
    """
    # Load the image
    image = cv2.imread(image_path)

    # Load the annotations
    with open(annotation_path, 'r') as f:
        annotation = f.readline().strip().split(' ')
        class_label = int(annotation[0])
        bbox_norm = np.array([float(x) for x in annotation[1:5]])
        segmentation_norm = np.array([float(x) for x in annotation[5:]])

    # Convert normalized bbox to pixel coordinates
    h, w, _ = image.shape
    bbox = bbox_norm * np.array([w, h, w, h])
    bbox = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])

    # Convert normalized segmentation points to pixel coordinates
    segmentation = segmentation_norm.reshape(-1, 2) * np.array([w, h])
    segmentation = segmentation.astype(np.int32)

    # Create a binary mask from the segmentation points
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [segmentation], 255)

    # Set the color and transparency of the mask
    mask_color = (255, 0, 0)
    mask_alpha = 0.5

    # Apply the color and transparency to the mask
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask = mask.astype(np.float32) / 255.0
    mask = mask * mask_color
    mask = (mask_alpha * mask).astype(np.uint8)

    # Apply the mask to the image
    masked_image = cv2.addWeighted(mask, 0.5, image, 0.5, 0)

    # Extract the x, y, w, h values from the bounding box
    x, y, w, h = bbox
    x_min, y_min, x_max, y_max = int(x), int(y), int(x + w), int(y + h)

    # Draw the bounding box and segmentation mask on the image
    cv2.rectangle(masked_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Display the image
    plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    plt.show()
