import os
import random
import glob


def upload_images_anns_to_project_split(image_dir, ann_dir, upload_project, train_pct=1.0, val_pct=0.0, test_pct=0.0, ext='.jpg'):
    # create image glob
    image_glob = glob.glob(os.path.join(image_dir, f'*{ext}'))
    n_images = len(image_glob)
    image_paths = [os.path.abspath(path) for path in image_glob]
    random.shuffle(image_paths)

    # create train/val/test splits
    train_end = int(train_pct * n_images)
    val_end = int((train_pct + val_pct) * n_images)
    train_paths = image_paths[:train_end]
    val_paths = image_paths[train_end:val_end]
    test_paths = image_paths[val_end:]

    def upload_to_dataset(paths, dataset):
        for image_path in paths:
            image_filename = os.path.basename(image_path)
            ann_file_path = os.path.join(ann_dir, os.path.splitext(image_filename)[0]+ '.txt')
            upload_project.upload(image_path, ann_file_path, batch_name='my_batch', split=dataset)

    # copy image files to their respective split directories and upload to Roboflow
    upload_to_dataset(train_paths, 'train')
    upload_to_dataset(val_paths, 'val')
    upload_to_dataset(test_paths, 'test')