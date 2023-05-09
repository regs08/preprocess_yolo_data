import os
from yolo_data.WritingRenamingFile.writing_to_file_utils import save_yolo_annotations


def get_split_points(start, end, interval):
    """
    getting split points for an image, usually start will be equal to 0 and end
    will be the images width.

    In our case we are starting where the bboxes start and endning where they end.
    """
    #start is the minimum value
    #end is the width of the image gotten from our bbox extreme

    # if theres a remainder we subtract the last line
    num_lines = (end // interval) -1  if (end // interval) > 0 else 0
    #for the first value of start for our splits
    intervals = [start]

    # Draw vertical lines at every n number of pixels until step > end
    if num_lines > 0:
        x = interval + start
        for i in range(num_lines):
            intervals.append(x)
            x += interval
            if x + interval > end:
              break
    # appending image with for the last split
    intervals.append(end)
    return intervals


def save_images_from__list_of_A_dict(images, save_folder, min_boxes):
    """
    iterates throught the splits of one image, checks to see if our image has a bounding and filename if it does
    it gets saved
    :param images:
    :param save_folder:
    :return:
    """
    print(f'Saving {len(images)} images(s) to ../{os.path.basename(save_folder)}')
    for img_dict in images:
        if len(img_dict['bboxes'])>= min_boxes:
            assert 'filename' in img_dict.keys(), 'save parameter is True. dict must contain filename'
            save_path = os.path.join(save_folder, img_dict['filename'])
            img_dict['image'].save(save_path)
            save_yolo_annotations(img_dict['bboxes'], img_dict['category_ids'], img_dict['filename'], save_folder)

