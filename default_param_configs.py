min_x_crop = 1280
min_y_crop = 1280
cat_id_map = {0: 'grape'}
label_to_id_map = {v: k for k, v in cat_id_map.items()}
class_labels = list(label_to_id_map.keys())
image_exts = ['.jpg', '.png', '.jpeg']
upper = [e.upper() for e in image_exts]
image_exts.extend(upper)