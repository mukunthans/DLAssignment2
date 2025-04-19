
h_params = {
    "epochs": 10,
    "learning_rate": 0.0001,
    "batch_size": 32,
    "num_of_filter": 64,
    "filter_size": [3,3,3,3,3],
    "actv_func": "gelu",
    "filter_multiplier": 2,
    "data_augumentation": False,
    "batch_normalization": True,
    "dropout": 0.4,
    "conv_layers": 5,
    "dense_layer_size": 256,
    "image_size": 224,
    "num_classes": 10
}
TRAIN_DATA_DIR = "/kaggle/input/assignment2dataset/inaturalist_12K/train"
VAL_DATA_DIR = "/kaggle/input/assignment2dataset/inaturalist_12K/val"
WANDB_PROJECT = "DL Assignment 2"
WANDB_KEY = ""  # Place your wandb key here
