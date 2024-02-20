# PyTorch Tutorial

## Introduction

This repository contains my sources of [PyTorch Tutorials - Complete Beginner Course](https://www.youtube.com/playlist?list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4)
by [Patrick Loeber](https://patloeber.com/) ([Github](https://github.com/patrickloeber), [Youtube](https://www.youtube.com/@patloeber)) 
on [Youtube](https://www.youtube.com).

There is on overlap with [Patrick's repo](https://github.com/patrickloeber/pytorchTutorial), but I want to have my own version.

## Lessons

### [Lesson 1 - Installation](https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=1)

#### Virtual environment

Create you virtual environment.

Local user

    python -m venv .venv

If virtual environment already exist activate it.

Linux

    . .venv/Scripts/activate

Win

    .venv/Scripts/activate.bat  // CMD
    .venv/Scripts/Activate.ps1  // Powershell

#### Install libraries

Install PyTorch, Numpy, and Matplot without CUDA support.
This works on _all_ computers.

Admin

    pip3 install numpy matplotlib scikit-learn tensorboard torch torchvision torchaudio

Local user

    python -m pip install numpy matplotlib scikit-learn tensorboard torch torchvision torchaudio


Install PyTorch, NumPy, and Matplot for CUDA.
This works only if you have installed a graphic card with Nvidia Chip.
Then you can use GPU for tensor calculation which is way faster than CPU.

Admin

    pip3 install numpy matplotlib scikit-learn torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Local user

    python -m pip install numpy matplotlib scikit-learn torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


Create `requirements.txt`

Admin

    pip freeze > requirements.txt

Local user

    python -m pip freeze > requirements.txt


### [Lesson 2 - Tensor Basics](https://www.youtube.com/watch?v=exaWOE8jvy8&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=2)

Content

* Create tensors
* Calculations with tensors
* Transform tensors
* Translate tensors to Numpy arrays forth and back


### [Lesson 3 - Gradient Calculation With Autograd](https://www.youtube.com/watch?v=DbeIqrwb_dE&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=3)

Content

* Create tensors with gradient
* Calculate the gradient
* Remove gradient from tensor


### [Lesson 4 - Backpropagation-Theory With Example](https://www.youtube.com/watch?v=3Kb0QS6z7WA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=4)

Content

* Theory
* Calculate a backpropagation


### [Lesson 5 - Gradient Descent With Autograd And Backpropagation](https://www.youtube.com/watch?v=E-I2DNVzQLg&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=5)

Content

* Train a manually implemented neuron


### [Lesson 6 - Training Pipeline: Model, Loss, And Optimizer](https://www.youtube.com/watch?v=VVDHU_TWwUg&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=6)

Content

* Train a PyTorch neural network (either an instance of nn.Linear or a subclass of nn.Module)
* Minimizes the error of a tensor to another tensor


### [Lesson 7 - Linear Regression](https://www.youtube.com/watch?v=YAJ5XBwlN4o&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=7)

Content

* Implement a linear regression by training of a neural network
* Minimizes the error of a tensor to [Scikit-learn.](https://scikit-learn.org/stable/index.html) data set
* (Lesson 6 and 7 are almost the same)


### [Lesson 8 - Logistic Regression](https://www.youtube.com/watch?v=OGpQxIkR4ao&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=8)

Content

* Implement a logistic regression by training of a neural network
* Minimizes the error of a tensor to a [Scikit-learn.](https://scikit-learn.org/stable/index.html) data set
* (again very similar to lesson 7, but usage of another activation function and loss)


### [Lesson 9 - Dataset And Dataloader](https://www.youtube.com/watch?v=PXOzkkB5eH0&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=9)

Content

* Implement a dataset (subclass of Dataset)
* Use Dataloader to iterate over data in batches


### [Lesson 10 - Dataset Transforms](https://www.youtube.com/watch?v=X_QOZEko5uE&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=10)

Content

* Quick Introduction to existing Transforms
* Implement different Transforms
* Combine those Transforms and execute it on Dataset


### [Lesson 11 - Softmax And Cross Entropy](https://www.youtube.com/watch?v=7q7E91pHoW4&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=11)

Content

* Softmax
  * Softmax function takes a vector of input values and transforms them to values (probability distribution) between 0 and 1.
  Constraint is that the sum of all values will be 1.
  * It is used for multi-class classification. (Usually non-binary problems there sigmoid is more common.)
  * Softmax implementation in numpy and pytorch.
* Cross-entropy
  * Cross-entropy is a loss function for classification.
  * Cross-entropy is a metric to quantify the difference between two probability distributions (e.g. predicted and true distribution).


### [Lesson 12 - Activation Functions](https://www.youtube.com/watch?v=3t9lZM7SS7k&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=12)

Content

* Activation Functions
  * Step function
  * Sigmoid
  * TanH
  * ReLU
  * Leaky ReLU
  * Softmax
* ReLU is the most commonly used action function between hidden layers.
* Softmax is mostly the last activation function in classification output.
* Sigmoid is mostly the last activation function in binary output.

### [Lesson 13 - Feed-Forward Neural Network](https://www.youtube.com/watch?v=oPhxf2fXHkQ&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=13)

Content

* Implementation of a Neural Network with one hidden layer using the ReLU activation function
* Uses a DataLoader to load data from MNIST using SciKit-Learn
* Uses a cross entropy loss and an Adam optimizer 
* Defines a training loop with a forward and a backward pass
* Usage of GPU if available


### [Lesson 14 - Convolutional Neural Network](https://www.youtube.com/watch?v=pDdP0TFzsoQ&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=14)

Content

* Implementation of a Convolutional Neural Network with multiple convolutional, maxpool, and linear layers
* The convolutional layer uses a 5x5 filter with zero padding and a stride of one.
* The formula to calculate the new image size is `(original-size - filter + 2 * padding) / stride + 1`
  * Examples
    * 1st convolutional layer: (32 - 5 + 2 * 0) / 1 + 1 = 28
    * 1st maxpool layer: (28 - 2 + 2 * 0) / 2 + 1 = 14
* Uses a DataLoader to load data from MNIST using SciKit-Learn
* Uses a cross entropy loss and a stochastic gradient descent (SGD) optimizer
* Defines a training loop with a forward and a backward pass
* Usage of GPU if available

Open points

* What kind of filter also known as kernel is used by the convolutional layer?


### [Lesson 15 - Transfer Learning](https://www.youtube.com/watch?v=K0lWSB2QoIQ&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=15)

Content

* Train a pre-trained CNN for you distinct purpose (to save time)
* Two cases
  1. Continue to train the complete CNN
  2. Train the last layer only


### [Lesson 16 - Tensorboard](https://www.youtube.com/watch?v=VJW9wU-1n18&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=16)

Content

* Generate statistics to analyze the efficiency of the neural net 

Run Tensorboard

    .venv/Scripts/tensorboard.exe --logdir=runs


### [Lesson 17 - Saving And Loading Models](https://www.youtube.com/watch?v=9L9jEOwRrCg&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=17)

Content

* Save and load model only,
* Save and load model and optimizers, called checkpoint during training.
