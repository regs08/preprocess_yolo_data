import albumentations as A
import os
import cv2
from my_SAM.seg2mask import seg2mask

from yolo_data.LoadingData.load_utils import get_class_id_bbox_seg_from_yolo, glob_image_files, get_annotation_path
from yolo_data.WritingRenamingFile.writing_to_file_utils import insert_string_before_extension
from yolo_data.Preprocessing.VerticalSplit.utils import get_split_points
from yolo_data.yolo2pvoc import convert_yolo_to_pascal_voc
from yolo_data.Preprocessing.CropImage.crop_utils import get_bbox_extreme_with_min_pixel_value


def vertical_split_with_A(img, xmin, xmax, ymin, ymax, bboxes, class_labels, format, masks=None):
    """
    takes in an image and returns a
    :param img: img as array
    :param x_min: min val dimensions for our split
    :param x_max: max val
    :param bboxes: bboxes from the image
    :param class_labels: class_labels as strs
    :return: a dict containing , image, bboxes, category_ids
    """
    # Note we should be using , min_visibility=0.3) but it is not getting rid of masks
    # as well so were getting weird numbers and errors. much better results without however there are small masks
    # check for use of class albumentations.augmentations.geometric.resize.SmallestMaxSize
    aug = A.Compose([
        A.Crop(x_min=xmin, x_max=xmax, y_min=ymin, y_max=ymax),
    ], bbox_params=A.BboxParams(format=format))
    for i, box in enumerate(bboxes):
        bboxes[i] = [0 if coord < 0 else coord for coord in box]
        print(box)
    if masks:
        vertical_split_image = aug(image=img, bboxes=bboxes, category_ids=class_labels, masks=masks)
    else:
        vertical_split_image = aug(image=img, bboxes=bboxes, category_ids=class_labels)

    # vertical_split_image['image'] = Image.fromarray(cv2.cvtColor(vertical_split_image['image'], cv2.COLOR_BGR2RGB))
    return vertical_split_image


def vertical_split_with_intervals(img, intervals, bboxes, class_ids, **args):
    """
    splits the image using the albumentations library the x values are gotten from intervals. the y values can be given
    their default is 0, hieght of the image
    :param img: path or arr, if path we add a filename to our dict
    :param intervals: the poitns where we will crop
    :param bboxes: the bboxes, default is yolo format, just coords we will add the class label onto the 5th index.
    :param class_labels: list of the class ids, ints
    :return: a list of dictionaries. the id number is the key that has the split info
    e.g out[0][2]['image'] will be the first image that was split and the second split of that image
    """

    format = args.get('format', 'pascal_voc')
    ymin = args.get('ymin', 0)
    ymax = args.get('ymax', img.shape[0])
    masks = args.get('masks', None)
    filename = args.get('filename', False)
    # getting the "ValueError: Your 'label_fields' are not valid - them must have same names as params in dict" so getting rid
    # adding the label on to the end of the box

    for i, box in enumerate(bboxes):
        box.append(class_ids[i])

    split_images = []
    # getting our xmin and xmax from our intervals
    for i in range(len(intervals) - 1):
        split_image = vertical_split_with_A(img=img,
                                            # x values fromm interval list
                                            xmin=intervals[i],
                                            xmax=intervals[i + 1],
                                            # y value from bbox extreme
                                            ymin=ymin,
                                            ymax=ymax,
                                            bboxes=bboxes,
                                            class_labels=class_ids,
                                            masks=masks,
                                            format=format)

        if filename:
            file_no = f'_{i}'
            split_image['filename'] = insert_string_before_extension(filename, file_no)
        num_instances = len(split_image['bboxes'])
        if num_instances < 1:
            print('num_instances for', split_image['filename'], ': ', num_instances)
            continue
        else:
            # getting rid of small masks
            # we have a list of empty and full masks
            # mask_count = 0
            # for m in vertical_split_image['masks']:
            #   unique = np.unique(m)
            #   if len(unique) > 1:
            #     mask_count+=1
            # mask_bbox_dif = mask_count - num_instances
            # if mask_bbox
            split_images.append(split_image)

    return split_images


def split_and_crop_images_with_bbox_extremes(image_folder, ann_folder, format='yolo', split_value=640):
    image_paths = glob_image_files(image_folder)
    out = []
    print(f'Splitting {len(image_paths)} image(s) in .../{os.path.basename(image_folder)}')
    for path in image_paths:
        ####
        # extracting data from path
        ####
        filename = os.path.basename(path)
        ann_path = get_annotation_path(path, ann_folder)

        class_ids, bboxes, segs = get_class_id_bbox_seg_from_yolo(ann_path)

        img_arr = cv2.imread(path)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        h, w, _ = img_arr.shape
        masks = [seg2mask(seg, h, w) for seg in segs]

        ####
        ####

        ####
        # bbox business; converting to pascal voc and getting extremes
        ####
        p_voc_boxes = [convert_yolo_to_pascal_voc(box, h, w) for box in bboxes]
        bbox_extremes = get_bbox_extreme_with_min_pixel_value(p_voc_boxes, split_value)
        print(bbox_extremes)
        xmin, ymin, xmax, ymax = bbox_extremes
        ####
        ####

        ####
        # setting our intervals: start is the first bbox and end is the last bbox moving horizontally
        ####
        intervals = get_split_points(start=xmin, end=xmax, interval=split_value)
        ####
        ####
        split_image = vertical_split_with_intervals(img=img_arr,
                                                    intervals=intervals,
                                                    bboxes=p_voc_boxes,
                                                    class_ids=class_ids,
                                                    ymin=ymin,
                                                    ymax=ymax,
                                                    masks=masks,
                                                    filename=filename,
                                                    format=format)

        out.append(split_image)
    return out



