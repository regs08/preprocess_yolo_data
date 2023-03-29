import numpy as np
from preprocess_yolo_data.Model.Eval.calculate_mAP import get_val_test_mAP_from_model


def compare_mAP_models(old_model, new_model, data_yaml):
    old_model_avg_mAP = np.average(get_val_test_mAP_from_model(old_model, data_yaml))
    new_model_avg_mAP = np.average(get_val_test_mAP_from_model(new_model, data_yaml))

    if new_model_avg_mAP > old_model_avg_mAP:
        return True
    else:
        return False