import torch
import wandb
from config import h_params, TRAIN_DATA_DIR, VAL_DATA_DIR, WANDB_PROJECT, WANDB_KEY
from data import DataPreparer
from model import FlexibleCNN
from trainer import Trainer


def main():
    wandb.login(key=WANDB_KEY)
    data_preparer = DataPreparer(h_params, h_params["image_size"], TRAIN_DATA_DIR, VAL_DATA_DIR)
    training_data = data_preparer.get_loaders()
    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"{h_params['actv_func']}_ep_{h_params['epochs']}_lr_{h_params['learning_rate']}_init_fltr_cnt_{h_params['num_of_filter']}_fltr_sz_{h_params['filter_size']}_fltr_mult_{h_params['filter_multiplier']}_data_aug_{h_params['data_augumentation']}_batch_norm_{h_params['batch_normalization']}_dropout_{h_params['dropout']}_dense_size_{h_params['dense_layer_size']}",
        config=h_params
    )

    trainer = Trainer(FlexibleCNN, h_params, training_data)
    trainer.fit()


if __name__ == "__main__":
    main()
