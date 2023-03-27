import os


def insert_string_before_extension(file_path, string_to_insert):
    # Get the file extension
    file_extension = os.path.splitext(file_path)[1]
    # Get the file name without extension
    file_name = os.path.splitext(file_path)[0]
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


def save_results_yolo_format(results, save_dir):
    """
    Save results in YOLO format
    Args:
        results (ultralytics.yolo.engine.results.Results): Results object from YOLO
        save_dir (str): save_folder
    """
    num_predictions = len(results.boxes.xywhn)
    filename = os.path.splitext(os.path.basename(results.path))[0] + '.txt'
    path = os.path.join(save_dir, filename)
    with open(path, "w") as f:
        for i in range(num_predictions):
            class_id = str(int(results.boxes.cls[i]))
            bbox_yolo= results.boxes.xywhn[i].tolist()
            bbox_yolo = [str(x) for x in bbox_yolo]
            x,y,w,h = bbox_yolo
            line = f'{class_id} {x} {y} {w} {h} \n'
            f.write(line)
