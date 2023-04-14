"""
yaml
"""
import os
import yaml


def write_data_yaml_file(dataset_folder, class_labels, outdir):
    """
    :param DIRS: dict containing our  train, val, test, paths
    :param class_labels: class labels used for training
    :param outdir: save dir for the yaml file
    :return:
    """
    train_dir = os.path.join(dataset_folder, 'train')
    val_dir = os.path.join(dataset_folder,  'val')
    test_dir = os.path.join(dataset_folder, 'test')
    yaml_dict = {'train': train_dir,
                 'val': val_dir,
                 'test': test_dir,
                 'nc': len(class_labels),
                 'names': class_labels}
    yaml_path = os.path.join(outdir, 'data.yaml')
    with open(yaml_path, 'w') as file:
        documents = yaml.dump(yaml_dict, file)
    return yaml_path


def update_yaml_paths(dataset_folder_path, yaml_file=''):
    """

    :param dataset_folder_path: path to the dataset containing the folders train, val, test
    :param yaml_file: path to yaml file or defalut is os.path.join(dataset_folder_path,'data.yaml')
    :return:
    """
    if not yaml_file:
        yaml_file = os.path.join(dataset_folder_path, 'data.yaml')
    folders = os.listdir(dataset_folder_path)

    # read YAML file
    with open(yaml_file, 'r') as file:
        yaml_dict = yaml.load(file, Loader=yaml.FullLoader)

    #renaming the following keys in the yaml dict
    data_to_rename = ['train', 'val', 'test', 'valid'] #adding in valid here for roboflow data

    # update paths
    for folder in folders:
        if folder in data_to_rename:
            new_path = os.path.join(dataset_folder_path, folder)
            yaml_dict[folder] = new_path

    # write updated YAML file
    with open(yaml_file, 'w') as file:
        documents = yaml.dump(yaml_dict, file)

    return yaml_file