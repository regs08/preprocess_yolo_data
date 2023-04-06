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