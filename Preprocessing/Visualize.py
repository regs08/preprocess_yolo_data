import cv2
import os
import random
from PIL import Image, ImageDraw


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

        image, bboxes, class_labels = t['image'], t['bboxes'], t['class_labels']
        if format=='yolo':
            plot_boxes_yolo_format(image, bboxes, class_labels)
        if format=='pascal_voc':
            plot_bounding_boxes_pascal_voc_format(image, bboxes, class_labels)
        if format=='albumentations':
            plot_image_with_bboxes_albumentations_format(image, bboxes, class_labels)


def plot_boxes_yolo_format(image, boxes, labels):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


import matplotlib.pyplot as plt
import matplotlib.patches as patches


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


def plot_random_image_with_yolo_annotations(image_folder, annotation_folder):
    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(image_folder) if
                   f.endswith(('.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG'))]

    # Randomly select an image file and its corresponding annotation file
    selected_image_file = random.choice(image_files)
    selected_annotation_file = os.path.splitext(selected_image_file)[0] + '.txt'

    # Read in the image and its corresponding annotation file
    image_path = os.path.join(image_folder, selected_image_file)
    image = Image.open(image_path)
    annotation_path = os.path.join(annotation_folder, selected_annotation_file)
    with open(annotation_path, 'r') as f:
        annotations = f.readlines()

    # Create a drawing object to draw the bounding boxes on the image
    draw = ImageDraw.Draw(image)

    # Loop through each annotation and draw its bounding box on the image
    for annotation in annotations:
        # Parse the annotation to get the label and bounding box coordinates
        label, x_center, y_center, width, height = annotation.strip().split()
        x_min = int(float(x_center) * image.width - float(width) * image.width / 2)
        y_min = int(float(y_center) * image.height - float(height) * image.height / 2)
        x_max = int(float(x_center) * image.width + float(width) * image.width / 2)
        y_max = int(float(y_center) * image.height + float(height) * image.height / 2)

        # Draw the bounding box on the image
        draw.rectangle([x_min, y_min, x_max, y_max], outline='red')

    # Show the image with bounding boxes drawn
    image.show()

"""
Functions for colab 
#######
"""
import cv2
import matplotlib.pyplot as plt


def plot_image_with_boxes(image_path, label_path):
    # Load the image
    image = cv2.imread(image_path)
    # Convert the image from BGR to RGB for Matplotlib
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load the labels from the YOLO-formatted text file
    with open(label_path, 'r') as f:
        labels = f.readlines()

    # Parse the label data and draw the boxes
    for label in labels:
        class_id, x_center, y_center, width, height = [float(x) for x in label.strip().split()]
        x_min = int((x_center - width / 2) * image.shape[1])
        y_min = int((y_center - height / 2) * image.shape[0])
        x_max = int((x_center + width / 2) * image.shape[1])
        y_max = int((y_center + height / 2) * image.shape[0])
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Show the image with boxes using Matplotlib in Google Colab
    plt.imshow(image)
    plt.axis('off')
    plt.show()

