from yolo_formats.LoadingData.load_utils import load_image_filenames_from_folder,\
    get_annotation_path, \
    get_yolo_bboxes_from_txt_file
from yolo_formats.default_param_configs import cat_id_map
from yolo_formats.WritingRenamingFile.writing_to_file_utils import save_yolo_annotations, \
    save_image, insert_string_before_extension


import cv2
from PIL import Image


class Preprocess():
    def __init__(self,
                 image_dir,
                 ann_dir,
                 **args,
                 ):
        self.image_dir = image_dir
        self.ann_dir = ann_dir
        self.args = args

        self.image_paths = load_image_filenames_from_folder(image_dir)
        self.cropped_images = []

    def set_crop_function(self, crop_function):
        self.args['crop_function'] = crop_function

    def perform_crop_function(self):
        """
        here we load in our necessary arguments to pass a crop function that will crop the orginal images and save them
        to a specified folder.
        we assume that the crop function returns a transform object containing a new key, filename prefix which will be
        used to give the filename a new name
        :return:
        """
        cropped_images = []
        for img_path in self.image_paths:
            ann_path = get_annotation_path(img_path, self.ann_dir)
            yolo_boxes, class_ns = get_yolo_bboxes_from_txt_file(ann_path)
            class_labels = [cat_id_map[label] for label in class_ns]

            img_arr = cv2.imread(img_path)
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)

            cropped_transform = self.args['crop_function'](img_arr=img_arr, bboxes=yolo_boxes, class_labels=class_labels)
            # to save on memory using PILs Image
            cropped_transform['image'] = Image.fromarray(cropped_transform['image'])

            cropped_transform['file_name'] = \
                insert_string_before_extension(img_path, '_'+cropped_transform['filename_prefix'])
            cropped_images.append(cropped_transform)

        self.cropped_images.extend(cropped_images)
        return cropped_images

    def save_cropped_images_and_anns(self):
        for img_ann in self.cropped_images:
            save_yolo_annotations(bboxes=img_ann['bboxes'],
                                  labels=img_ann['class_labels'],
                                  file_name=img_ann['file_name'],
                                  save_dir=self.args['save_dir'])
            save_image(img_ann['image'],
                       save_dir=self.args['save_dir'],
                       filename=img_ann['file_name'])
