import os
import yaml
from preprocess_yolo_data.default_param_configs import cat_id_map

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


def save_results_xywhn(results, save_dir, for_roboflow=False):
    """
    Save results in YOLO format
    note that the results will not have a class_id but rather the label. this is for uploading
    to roboflow.
    Args:
        results (ultralytics.yolo.engine.results.Results): Results object from YOLO
        save_dir (str): save_folder
    """
    num_predictions = len(results.boxes.xywhn)
    filename = os.path.splitext(os.path.basename(results.path))[0] + '.txt'
    path = os.path.join(save_dir, filename)
    with open(path, "w") as f:
        for i in range(num_predictions):
            #get label map from config
            if for_roboflow:
                class_id = cat_id_map[int(results.boxes.cls[i])]
            else:
                class_id = int(results.boxes.cls[i])
            bbox = results.boxes.xyxy[i].tolist()
            bbox = [str(x) for x in bbox]
            xmin,ymin,xmax,ymax = bbox
            line = f'{class_id} {xmin} {ymin} {xmax} {ymax} \n'
            f.write(line)


def save_results_yolo_format(results, save_dir):
    """
    Save results in YOLO format
    note that the results will not have a class_id but rather the label. this is for uploading
    to roboflow.
    Args:
        results (ultralytics.yolo.engine.results.Results): Results object from YOLO
        save_dir (str): save_folder
    """
    num_predictions = len(results.boxes.xywhn)
    filename = os.path.splitext(os.path.basename(results.path))[0] + '.txt'
    path = os.path.join(save_dir, filename)
    with open(path, "w") as f:
        for i in range(num_predictions):
            #get label map from config

            class_id = int(results.boxes.cls[i])
            bbox = results.boxes.xywhn[i].tolist()
            bbox = [str(x) for x in bbox]
            x, y, w, h = bbox
            line = f'{class_id} {x} {y} {w} {h} \n'
            f.write(line)


"""
yaml
"""
def write_data_yaml_file(train_dir, val_dir, test_dir, class_labels, outdir):
    """
    :param DIRS: dict containing our  train, val, test, paths
    :param class_labels: class labels used for training
    :param outdir: save dir for the yaml file
    :return:
    """

    yaml_dict = {'train': train_dir,
                 'val': val_dir,
                 'test': test_dir,
                 'nc': len(class_labels),
                 'names': class_labels}
    yaml_path = os.path.join(outdir, 'data.yaml')
    with open(yaml_path, 'w') as file:
        documents = yaml.dump(yaml_dict, file)
    return yaml_path


def update_yaml_paths(yaml_file, DIRS):
    """

    :param yaml_file: location of the yaml file
    :param DIRS: dictionary containing keys: tran, val, test
    :return:
    """

    # read YAML file
    with open(yaml_file, 'r') as file:
        yaml_dict = yaml.load(file, Loader=yaml.FullLoader)

    # update paths
    if 'train' in yaml_dict:
        yaml_dict['train'] = DIRS['TRAIN']
    else:
        print('train line not found in yaml')

    if 'val' in yaml_dict:
        yaml_dict['val'] = DIRS['VAL']
    else:
        print('val line not found in yaml')

    if 'test' in yaml_dict:
        yaml_dict['test'] = DIRS['TEST']
    else:
        print('test line not found in yaml')

    # write updated YAML file
    with open(yaml_file, 'w') as file:
        documents = yaml.dump(yaml_dict, file)