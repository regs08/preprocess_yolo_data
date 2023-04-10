import numpy as np

def convert_yolo_to_pascal_voc(img_size, yolo_box):
    box = np.zeros(4)

    dh = 1. / img_size[0]
    dw = 1. / img_size[1]

    x,y,w,h = yolo_box

    x = x / dw
    w = w / dw
    y = y / dh
    h = h / dh

    box[0] = x - (w / 2.0)
    box[1] = y - (h / 2.0)
    box[2] = x + (w / 2.0)
    box[3] = y + (h / 2.0)
    for b in box:
        if b < 0:
            b=0
    return (box)


