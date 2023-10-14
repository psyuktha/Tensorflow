# Fashion MNIST Classification Project

This project uses TensorFlow and Keras to classify fashion clothing items from the Fashion MNIST dataset. It includes training the model, making predictions, analyzing the results using a confusion matrix, and visualizing random predictions. The accuracy of each prediction is indicated by text color (green for correct, red for incorrect).

## Colab Notebook

The project is implemented in a Google Colab notebook. You can access the notebook and run the code by clicking on the following link:

[Google Colab Notebook](https://colab.research.google.com/drive/1wmNO6cYG9d8YGOTwTncUdl6_iyHkshxK?usp=sharing)

## Overview

### Dataset

The [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) consists of 60,000 training images and 10,000 test images across 10 different fashion item classes. Each image is a grayscale 28x28 pixel image.

### Project Structure

- `fashion_mnist.ipynb`: The main Colab notebook containing the code for data loading, model training, prediction, and result analysis.

## Requirements

- Tensorflow package
- numpy 
- matplotlib
- itertools

## Getting Started

1. Open the provided Colab notebook using the link above.

2. Run the code cells in the notebook sequentially to train the model and make predictions.

3. Analyze the results, including the confusion matrix and visualized random predictions.

## Visualization

Random predictions from the test data will be displayed as images with accompanying information:

- The image of the clothing item.
- The predicted class and its probability.
- Accuracy of the prediction indicated by text color (green for correct, red for incorrect).

## Confusion Matrix

A confusion matrix is used to provide an overview of the model's performance. It shows how well the model predicts each class in the dataset.

## Dependencies

To run the project, make sure you have the required Python packages installed. You can install them using `pip`:



