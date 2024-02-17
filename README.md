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

    pip3 install numpy matplotlib scikit-learn torch torchvision torchaudio

Local user

    python -m pip install numpy matplotlib scikit-learn torch torchvision torchaudio


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

