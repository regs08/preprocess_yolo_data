def convert_yolo_to_pascal_voc(yolo_box, image_height, image_width, normilized=True):
  """
  yolo box not normalized outputs a p_voc box
  """
  center_x, center_y, width, height = yolo_box
  xmin = int((center_x - width/2) * image_width)
  ymin = int((center_y - height/2) * image_height)
  xmax = int((center_x + width/2) * image_width)
  ymax = int((center_y + height/2) * image_height)

  box = [xmin, ymin, xmax, ymax]
  box = [0 if coord < 0 else coord for coord in box]

  return box