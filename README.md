# Fashion MNIST Classification with Keras and TensorFlow

## Objective
This project aims to build a multiclass classification model using Keras with a TensorFlow backend to classify fashion items in the Fashion MNIST dataset. The dataset includes grayscale images of clothing and accessory items in 10 categories, and this model uses a dense neural network with regularization to achieve a target accuracy of at least 90%.

## Dataset
The **Fashion MNIST** dataset consists of 70,000 grayscale images, each 28x28 pixels in size, distributed across 10 categories:
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

- **Training Set**: 60,000 images
- **Test Set**: 10,000 images

## Project Structure
This project includes a Jupyter Notebook that:
1. Loads and preprocesses the Fashion MNIST dataset.
2. Visualizes a subset of images for reference.
3. Defines and trains an Artificial Neural Network (ANN) model with two hidden layers.
4. Evaluates the model's performance on the test set.
5. Generates a classification report, confusion matrix, and visualizations of precision, recall, and F1-score for each class.
6. Plots the training and validation loss and accuracy curves.

## Getting Started

### Prerequisites
Ensure you have Python 3.6 or higher installed. You'll also need to install the required libraries listed in the `requirements.txt` file.

### Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/your-username/fashion-mnist-classification.git
cd fashion-mnist-classification
pip install -r requirements.txt
