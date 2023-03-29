import numpy as np
"""
functions that calculate the mAP over test and val sets
"""


def get_val_test_mAP_from_model(model, data_yaml):
    val_set_metrics = model.val(data=data_yaml)
    test_set_metrics = model.val(data=data_yaml, split='test')

    return [val_set_metrics.box.map, test_set_metrics.box.map]


def get_highest_mAP_from_models(models: list, data_yaml):
    """
    takes in a list of models, compares the val, and test mAP from the data_yaml,
    returns the model with the highest, e.g best performing.
    """

    avg_mAPs = [np.average(get_val_test_mAP_from_model(m, data_yaml) for m in models)]
    idx_max = np.argmax(avg_mAPs)

    return models[idx_max]


