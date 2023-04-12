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

