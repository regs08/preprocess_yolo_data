"""
dealing with saving from yolo.results object
"""

from yolo_data.default_param_configs import cat_id_map
import os


def save_results_xyxy(results, save_dir, for_roboflow=False):
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
            #roboflow wants a label not id
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
            class_id = int(results.boxes.cls[i])
            bbox = results.boxes.xywhn[i].tolist()
            bbox = [str(x) for x in bbox]
            x, y, w, h = bbox
            line = f'{class_id} {x} {y} {w} {h} \n'
            f.write(line)