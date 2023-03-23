"""
Example of our preprocess class
"""

from yolo_formats.Preprocessing.CropImage.crop_images_from_dir import left_crop_from_center_and_bbox_dat, right_crop_from_center_and_bbox_dat
from yolo_formats.Preprocessing.Preprocess.preprocess import Preprocess

image_dir = "/Users/cole/PycharmProjects/Forgit/yolo_formats/Preprocessing/SplitFolderFromLabelImg/Pinot-noir/val/images"
ann_dir = "/Users/cole/PycharmProjects/Forgit/yolo_formats/Preprocessing/SplitFolderFromLabelImg/Pinot-noir/val/labels"
save_dir = "/Users/cole/PycharmProjects/Forgit/yolo_formats/Preprocessing/CropImage/test_crop_save_dir"

pp = Preprocess(image_dir=image_dir, ann_dir=ann_dir, crop_function=left_crop_from_center_and_bbox_dat)
#
pp.args['save_dir'] = save_dir

cropped_images = pp.perform_crop_function()

pp.set_crop_function(right_crop_from_center_and_bbox_dat)

pp.perform_crop_function()
pp.save_cropped_images_and_anns()