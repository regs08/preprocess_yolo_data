from preprocess_yolo_data.WritingRenamingFile.writing_to_file_utils import insert_string_before_extension,\
    save_yolo_annotations
from preprocess_yolo_data.LoadingData.load_utils import glob_image_files, get_annotation_path
from preprocess_yolo_data.Conversions.convert_yolo_pascal_voc import convert_yolo_to_pascal_voc
from preprocess_yolo_data.Preprocessing.CropImage.crop_utils import get_bbox_extreme_with_min_pixel_value
from preprocess_yolo_data.LoadingData.load_utils import get_yolo_bboxes_from_txt_file


import cv2
import albumentations as A
from PIL import Image
import os


def get_split_points(image, interval):
    if isinstance(image, str):
        image = cv2.imread(image)

    height, width, _ = image.shape

    # if theres a remainder we subtract the last line
    num_lines = (width // interval) -1  if (width // interval) > 0 else 0
    #for the first value of xmin for our splits
    intervals = [0]

    # Draw vertical lines at every n number of pixels until step > width
    if num_lines > 0:
        x = interval
        for i in range(num_lines):
            intervals.append(x)
            x += interval

    # appending image with for the last split
    intervals.append(width)
    return intervals


def vertical_split_with_A(img, x_min, x_max,y_min, y_max, bboxes, class_labels, format):
    """
    takes in an image and returns a
    :param img: img as array
    :param x_min: min val dimensions for our split
    :param x_max: max val
    :param bboxes: bboxes from the image
    :param class_labels: class_labels as strs
    :return: a dict containing , image, bboxes, category_ids
    """
    aug = A.Compose([
        A.Crop(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max),
    ], bbox_params=A.BboxParams(format=format, min_visibility=0.3))
    vertical_split_image = aug(image=img, bboxes=bboxes, category_ids=class_labels)
    #to take up less memory we put as PIL object

    vertical_split_image['image'] = Image.fromarray(cv2.cvtColor(vertical_split_image['image'], cv2.COLOR_BGR2RGB))

    return vertical_split_image


def vertical_split_with_intervals(img, intervals, bboxes, class_labels, **args):
    """
    splits the image using the albumentations library the x values are gotten from intervals. the y values can be given
    their default is 0, hieght of the image
    :param img: path or arr, if path we add a filename to our dict
    :param intervals: the poitns where we will crop
    :param bboxes: the bboxes, default is yolo format, just coords we will add the class label onto the 5th index.
    :param class_labels: list of the class ids, ints
    :return: a list of dictionaries containing image (PIL) bboxes and cat_ids
    """

    add_filename = False
    if isinstance(img, str):
        img_path = img
        img = cv2.imread(img)
        add_filename = True

    format = args.get('format', 'yolo')
    y_min = args.get('y_min', 0)
    y_max = args.get('y_max', img.shape[0])

    # getting the "ValueError: Your 'label_fields' are not valid - them must have same names as params in dict" so getting rid
    # adding the label on to the end of the box

    for i, box in enumerate(bboxes):
        box.append(class_labels[i])

    split_images = []
    for i in range(len(intervals) -1):
        x_min = intervals[i]
        x_max =intervals[i+1]

        split_image = vertical_split_with_A(img=img,
                                            x_min=x_min,
                                            x_max=x_max,
                                            y_min=y_min,
                                            y_max=y_max,
                                            bboxes=bboxes, class_labels=class_labels, format=format)

        if add_filename:
            img_filename = os.path.basename(img_path)
            file_no = f'_{i}'
            split_image['filename'] = insert_string_before_extension(img_filename, file_no)

        split_images.append(split_image)
    return split_images


def save_images_from__list_of_A_dict(images, save_folder, min_boxes):
    """
    iterates throught the splits of one image, checks to see if our image has a bounding and filename if it does
    it gets saved
    :param images:
    :param save_folder:
    :return:
    """
    print(f'Saving {len(images)} to ../{os.path.basename(save_folder)}')
    for img_dict in images:
        if len(img_dict['bboxes'])>= min_boxes:
            assert 'filename' in img_dict.keys(), 'save parameter is True. dict must contain filename'
            save_path = os.path.join(save_folder, img_dict['filename'])
            img_dict['image'].save(save_path)
            save_yolo_annotations(img_dict['bboxes'], img_dict['category_ids'], img_dict['filename'], save_folder)


def split_images_in_folder(image_folder, interval, save_folder, ann_folder='',
                           save=True, min_boxes=1, bbox_extremes=False, min_pixel_value=1280):
    """
    takes in an image folder (and ann_folder default is image folder) splits the images vertically using the points provided
    by the interval. note the default values for the y coords is 0 and height of the image
    :param image_folder: location of images
    :param interval: interval for the split
    :param save_folder: save folder for the cropped images
    :param ann_folder: if theres a different folder for annotations..
    :param save: save the images
    :param min_boxes: check so that all images have an instance present
    :return:
    """

    #if anns are in image folder
    if not ann_folder:
        ann_folder=image_folder

    image_paths = glob_image_files(image_folder)
    print(f'Splitting {len(image_paths)} image(s) in .../{os.path.basename(img_folder)}')
    for img_path in image_paths:
        img = cv2.imread(img_path)

        #getting the coords for our vertical split for each image 
        split_intervals = get_split_points(img_path, interval)

        #returns our annotations in a yolo foramtted.txt
        ann_path = get_annotation_path(img_path, ann_folder)
        #reading from text file
        bboxes, class_ns = get_yolo_bboxes_from_txt_file(ann_path)

        # instead of loading in the text file we can just load in bbox extremes here
        if bbox_extremes:
            p_voc_boxes = [convert_yolo_to_pascal_voc(img.shape, box) for box in bboxes]
            _,y_min,_,y_max = get_bbox_extreme_with_min_pixel_value(p_voc_boxes, min_pixel_value)
            split_images = vertical_split_with_intervals(img=img_path,
                                                            bboxes=bboxes,
                                                            class_labels=class_ns,
                                                            intervals=split_intervals,
                                                            y_min=y_min,
                                                            y_max=y_max)

        else:
            split_images = vertical_split_with_intervals(img=img_path,
                                                         bboxes=bboxes,
                                                         class_labels=class_ns,
                                                         intervals=split_intervals)

        if save:
            #save images from the dict key filename and save folder for each image in the list
            save_images_from__list_of_A_dict(split_images, save_folder, min_boxes)





