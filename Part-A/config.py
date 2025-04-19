
BEST_CONFIG = {
    "epochs": 15,
    "learning_rate": 0.0005,
    "batch_size": 32,
    "last_unfreeze_layers": 3,
    "image_size": 224,
    "num_classes": 10,
    "num_workers": 2,
    "seed": 42,
    "dropout": 0.4
}
TRAIN_DATA_DIR = "/kaggle/input/assignment2dataset/inaturalist_12K/train"
VAL_DATA_DIR = "/kaggle/input/assignment2dataset/inaturalist_12K/val"
WANDB_PROJECT = "DL Assignment 2B"
WANDB_KEY = ""  # <-- Place your wandb key here
