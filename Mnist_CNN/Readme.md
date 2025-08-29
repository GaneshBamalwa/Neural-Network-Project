![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)
![MNIST](https://img.shields.io/badge/Dataset-MNIST-yellow.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-99%25-green.svg)



# Convolutional Neural Network (CNN) on MNIST

## Abstract

This project explores the implementation of a Convolutional Neural Network (CNN) in PyTorch for handwritten digit recognition on the MNIST dataset. The model is trained from scratch without transfer learning or pretrained weights. By leveraging convolutional layers, pooling, and fully connected layers, the network achieves ~99% accuracy on the test set. The project emphasizes both the theoretical foundation and the practical execution of deep learning pipelines, including dataset preparation, model design, training, evaluation, and visualization.

## Introduction

The MNIST dataset is a benchmark in the field of machine learning and computer vision, consisting of 70,000 grayscale images of handwritten digits (0–9), each of size 28×28 pixels. Although considered a "toy dataset" today, it is still widely used for:

- Understanding neural network fundamentals

- Benchmarking architectures and optimizers

- Rapid prototyping of deep learning models

This project implements a CNN to classify MNIST digits with high accuracy, while also demonstrating how to track metrics and visualize the training process.


## Methodology

### 1. Dataset
- Training set: 60,000 images
- Test set: 10,000 images
- Preprocessing: Normalization to [0, 1]

### 2. Model Architecture

The CNN architecture follows a classic design:

      Convolution Layer 1: Conv2D (1 → 32 filters, 3×3 kernel), ReLU, MaxPooling (2×2)
      
      Convolution Layer 2: Conv2D (32 → 64 filters, 3×3 kernel), ReLU, MaxPooling (2×2)
      
      Fully Connected Layer 1: 128 units, ReLU
      
      Fully Connected Layer 2: 10 units (softmax for digit classification)

### 3. Training Setup
- Optimizer: Adam
- Loss Function: CrossEntropyLoss
- Batch Size: 64
- Epochs: 5 (extendable)
- Framework: PyTorch


## Results
### Training & Validation Loss
<img width="700" height="547" alt="losss" src="https://github.com/user-attachments/assets/bb6ff42a-d86b-46f7-9414-11e69dd11281" /><br><br>
### Training & Validation Accuracy
<br>
<img width="700" height="547" alt="accuracy" src="https://github.com/user-attachments/assets/f2433af4-c992-4c84-8e53-f2829ffc46ad" /><br>

### Inference:
From the above graph it is visible that there is a big gap between the training and validation accuracy and loss, with even validation loss peaking at times. Classic sign of overfitting.
- **Final Training Accuracy: ~99.7%**
- **Final Validation Accuracy: ~99.1%**
- **Test Accuracy: ~98.9%**
- The model demonstrates excellent generalization with minimal overfitting.

## Moving Forward
 The current model achieves strong performance but shows signs of overfitting, as indicated by the gap between training and validation accuracy. To address this and further improve performance, the following steps are planned:
- **Batch Normalization**: Adding normalization layers after convolutional layers to stabilize learning and reduce overfitting.
- **Optimizer Improvements**: Experimenting with momentum-based optimizers (e.g., SGD with momentum) to improve convergence and potentially generalization.
- **Data Augmentation**: Applying techniques such as random rotations, shifts, and flips to artificially increase dataset variability and make the model more robust.
- **Regularization**: Introducing dropout or weight decay to mitigate overfitting while retaining accuracy.
- **Deeper Architectures**:
            -Exploring more complex CNN architectures beyond two convolutional layers for higher representation capacity.## Moving forward : 
            - Since we have classic signs of overfitting as an issue , i think batch normalization would be the next step.
            - To improve the accuracy further I am planning to integrate the momentum concept in the updation of wieghts and biases.
            - Data augmentation would also be a fine technique that'll help me expand this further and would be helpful when there's less availability of training data.
