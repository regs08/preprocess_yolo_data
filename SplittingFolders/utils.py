"""
split utils
"""


def get_train_val_test_split_ratio(data, train_ratio, val_ratio):
    num_files = len(data)
    num_train = int(train_ratio * num_files)
    num_val = int(val_ratio * num_files)

    train_data= data[:num_train]
    val_data = data[num_train:num_train + num_val]
    test_data = data[num_train + num_val:]

    return train_data, val_data, test_data