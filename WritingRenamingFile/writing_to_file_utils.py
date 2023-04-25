import os


def insert_string_before_extension(file_basename, string_to_insert):
    # Get the file extension
    file_extension = os.path.splitext(file_basename)[1]
    # Get the file name without extension
    file_name = os.path.splitext(file_basename)[0]
    # Create new file name with string inserted before extension
    new_file_name = os.path.basename(file_name + string_to_insert + file_extension)
    # Rename the file
    return new_file_name


def save_image(image, save_dir, filename):
    # Create the full file path
    filepath = os.path.join(save_dir, filename)

    # Save the image to disk
    image.save(filepath)


def save_yolo_annotations(bboxes, labels, file_name, save_dir):

    # Create a new file for YOLO annotations
    yolo_filename = os.path.splitext(file_name)[0] + '.txt'
    yolo_filepath = os.path.join(save_dir, yolo_filename)
    yolo_file = open(yolo_filepath, 'w')

    # Loop through each bounding box and its corresponding label
    for bbox, label in zip(bboxes, labels):
        # Write the YOLO annotation to the file
        yolo_line = str(label) + ' ' + str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2]) + ' ' + str(
            bbox[3]) + '\n'
        yolo_file.write(yolo_line)

    # Close the YOLO annotation file
    yolo_file.close()


def write_lines_to_file(filepath, data):
    """
    writes data to filepath
    data is a list of lists. where the inner list elements represent a line for the file
    """
    with open(filepath, 'w') as f:
        for lines in data:
          for line in lines:
            f.write(line + '\n')

"""
writing as yolo with segmentation 
"""


def yolo_format_line(class_label, bbox, segmentation):
    # Extract normalized bounding box coordinates in YOLO format
    #assumes the segmentation is a flattend list [x,y,x1,y1...xn,yn]
    x_center, y_center, width, height = bbox

    # Create YOLO-formatted line
    line = f"{class_label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

    # Add normalized segmentation data
    segmentation_str = " ".join(str(x) for x in segmentation)
    line += " " + segmentation_str

    return line


def segmentation_to_yolo_line(class_id, bbox, segmentation):
  """
  prepares our segmentation, flattens it, and creates our for a binary  mask
  """
  flat_seg = [float(coord) for tup in segmentation for coord in tup]
  lines = []
  lines.append(yolo_format_line(class_id, bbox, flat_seg))
  return lines

"""
folder structures
"""


def create_model_train_folder_structure(output_folder):
    train_folder = os.path.join(output_folder, 'train')
    val_folder = os.path.join(output_folder, 'val')
    test_folder = os.path.join(output_folder, 'test')
    for folder in [train_folder, val_folder, test_folder]:
        os.makedirs(os.path.join(folder, 'images'))
        os.makedirs(os.path.join(folder, 'labels'))
    return train_folder, val_folder, test_folder