
# Flexible CNN Classifier for iNaturalist 12K

This project provides a flexible and configurable Convolutional Neural Network (CNN) for classifying images from the iNaturalist 12K dataset. The main goal is to accurately assign images to one of ten biological classes using deep learning.

## Overview

- **Customizable Model:** You can easily adjust the number of convolutional layers, filter sizes, activation functions, and other settings to experiment with different architectures.
- **Data Handling:** The dataset is split into training, validation, and test sets, ensuring balanced representation from each class.
- **Training and Validation:** The model is trained and validated using standard deep learning practices, with support for regularization techniques like batch normalization and dropout to improve performance.
- **Experiment Tracking:** Integration with Weights & Biases (wandb) allows you to track experiments, log metrics, and visualize results.
- **Prediction Visualization:** After training, the model's predictions on test images are displayed in a grid, making it easy to see which images were classified correctly or incorrectly.
- **Hyperparameter Tuning:** The setup supports automated hyperparameter search to find the best model configuration.

## How It Works

1. **Set Hyperparameters:** Adjust settings such as learning rate, batch size, number of filters, and more to define your experiment.
2. **Prepare Data:** Images are resized and optionally augmented before being split into training, validation, and test sets.
3. **Train the Model:** The CNN learns to classify images by minimizing prediction errors on the training data, while validation data helps prevent overfitting.
4. **Evaluate Performance:** After training, the model's accuracy is measured on unseen test data.
5. **Visualize Results:** A grid of test images shows the true and predicted classes, highlighting correct and incorrect predictions.
6. **Track Experiments:** All metrics and visualizations are logged for easy comparison and analysis.

## Requirements

- Python 3.7 or higher
- Common deep learning libraries (PyTorch, torchvision, numpy, matplotlib, opencv-python)
- Weights & Biases (wandb) for experiment tracking

## Customization

- You can easily swap out dataset paths, adjust the number of classes, and modify model architecture settings.
- The code is designed for flexibility, making it suitable for a variety of image classification tasks beyond iNaturalist.

## License

This project is for educational and research purposes.
