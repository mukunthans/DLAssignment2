# EfficientNetV2 Transfer Learning Classifier for iNaturalist 12K

This project provides an image classification pipeline for the iNaturalist 12K dataset using transfer learning with the EfficientNetV2-S model. The approach is designed for flexibility, reproducibility, and robust experiment tracking.

## Overview

- **Transfer Learning:** Leverages EfficientNetV2-S pretrained on ImageNet, adapting it for the iNaturalist 12K classification task.
- **Configurable Fine-Tuning:** Allows selective unfreezing of the last layers of the classifier for fine-tuning, enabling control over the balance between transfer learning and task-specific adaptation.
- **Stratified Data Splitting:** Ensures that training and validation sets maintain balanced class distributions for fair evaluation.
- **Data Augmentation:** Incorporates standard image augmentation techniques to improve model generalization and robustness.
- **Experiment Tracking:** Integrates with Weights & Biases (wandb) for comprehensive logging of metrics, hyperparameters, and experiment results.
- **Hyperparameter Optimization:** Supports automated sweeps for key hyperparameters, such as learning rate, batch size, dropout, and the number of unfrozen classifier layers.

## Workflow

1. **Hyperparameter Configuration:** Experiment parameters such as epochs, learning rate, batch size, and model fine-tuning depth are set in a configuration dictionary.
2. **Data Preparation:** Images are resized, normalized, and augmented. The dataset is split into training, validation, and test sets using a stratified approach to preserve class balance.
3. **Model Setup:** EfficientNetV2-S is loaded with pretrained weights. The classifier head is modified for the number of target classes, and selected layers are unfrozen for fine-tuning.
4. **Training and Validation:** The model is trained using cross-entropy loss and the Adam optimizer, with a learning rate scheduler to enhance convergence. Training and validation performance is monitored and logged.
5. **Testing and Saving:** After training, the model is evaluated on the test set and the best-performing model is saved for future use.
6. **Hyperparameter Sweep:** Automated sweeps can be launched to optimize key parameters and maximize validation accuracy.

## Features

- Pretrained EfficientNetV2-S backbone for efficient and accurate transfer learning.
- Configurable model fine-tuning for optimal performance on the target dataset.
- Advanced data augmentation and normalization for robust training.
- Stratified sampling for balanced and fair dataset splits.
- Learning rate scheduler to improve training stability.
- Comprehensive experiment tracking and visualization with wandb.
- Automated hyperparameter optimization for efficient experimentation.

## Customization

- Dataset paths and class numbers can be easily adjusted to fit different datasets or classification tasks.
- Model architecture and training parameters are fully configurable for flexibility in experimentation.

## Requirements

- Python 3.7 or higher
- PyTorch and torchvision
- numpy
- wandb

## License

This project is intended for educational and research purposes.

