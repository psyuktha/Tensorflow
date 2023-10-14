# Fashion MNIST Classification Project

This project uses TensorFlow and Keras to classify fashion clothing items from the Fashion MNIST dataset. It includes training the model, making predictions, analyzing the results using a confusion matrix, and visualizing random predictions. The accuracy of each prediction is indicated by text color (green for correct, red for incorrect).

## Colab Notebook

The project is implemented in a Google Colab notebook. You can access the notebook and run the code by clicking on the following link:

[Google Colab Notebook](https://colab.research.google.com/drive/1wmNO6cYG9d8YGOTwTncUdl6_iyHkshxK?usp=sharing)

## Overview
The Fashion MNIST dataset is a collection of grayscale images of fashion items, including clothing, shoes, and accessories. It consists of 10 classes, each representing a specific type of fashion item:

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

The goal of this project is to train a machine learning model to classify these items into their respective categories.

### Dataset

The [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) consists of 60,000 training images and 10,000 test images across 10 different fashion item classes. Each image is a grayscale 28x28 pixel image.

### Project Structure

- `fashion_mnist.ipynb`: The main Colab notebook containing the code for data loading, model training, prediction, and result analysis.

### Requirements

- Python
- Tensorflow 
- Numpy 
- Matplotlib
- Scikit-Learn

## Getting Started

1. Open the provided Colab notebook using the link above.

2. Run the code cells in the notebook sequentially to train the model and make predictions.

3. Analyze the results, including the confusion matrix and visualized random predictions.

### Data Loading and Preprocessing

Load the Fashion MNIST dataset and preprocess it for training and testing.

### Model Training

Train a deep learning model on the preprocessed data. You can experiment with different architectures and hyperparameters.

### Prediction

Make predictions on the test dataset and visualize some random predictions along with their confidence scores.

### Visualization

Random predictions from the test data will be displayed as images with accompanying information:

- The image of the clothing item.
- The predicted class and its probability.
- Accuracy of the prediction indicated by text color (green for correct, red for incorrect).

### Confusion Matrix

A confusion matrix is used to provide an overview of the model's performance. It shows how well the model predicts each class in the dataset.

## Results

The project will generate the following outcomes:

- A trained model that can classify fashion clothing items.
- Visualization of random predictions showing images, prediction probabilities, and whether the prediction was correct (green) or incorrect (red).
- A confusion matrix to evaluate the model's classification performance.

## Dependencies

To run the project, make sure you have the required Python packages installed. You can install them using `pip`

## Author

Yuktha PS
Feel free to reach out if you have any questions or suggestions!

This project is for educational purposes and meant to demonstrate the classification of fashion items using the Fashion MNIST dataset with TensorFlow and Keras. The actual dataset and model performance may vary depending on the project's configuration.



