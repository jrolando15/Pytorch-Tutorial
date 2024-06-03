# Pytorch-Tutorial

# Project Description
This tutorial provides a comprehensive introduction to PyTorch, a popular deep learning framework.

# Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Working with Tensors](#working-with-tensors)
- [Building Neural Networks](#building-neural-networks)
- [Training the Model](#training-the-model)
- [Advanced Topics](#advanced-topics)
- [License](#license)

# Introduction
PyTorch is a deep learning framework that provides an n-dimensional Tensor, similar to NumPy, but with GPU acceleration. It also includes automatic differentiation for building and training neural networks. This tutorial will guide you through the basics of PyTorch and demonstrate how to use it for a simple NLP task.

# Installation
To get started with this tutorial, you need to install PyTorch and TorchVision. You can install them using pip:
```bash
pip install torch torchvision
```

# Working with Tensors
Tensors are the fundamental building blocks of PyTorch. They are multi-dimensional arrays similar to NumPy arrays but can run on GPUs. Here are some basic operations with tensors:
```python
# Creating a tensor
data = torch.tensor([[1][2], [3][4], [5][6]])
print(data)

# Initializing a tensor with an explicit data type
data = torch.tensor([[0.11111111, 1], [2][3], [4][5]], dtype=torch.float32)
print(data)

# Creating tensors with specific values
zeros = torch.zeros(2, 5)
print(zeros)

ones = torch.ones(3, 4)
print(ones)

# Tensor operations
rr = torch.arange(1, 10)
print(rr + 2)
print(rr * 2)
```

# Building Neural Networks 
To define a neural network in PyTorch, create a class that inherits from torch.nn.Module. Define the layers of the network in the __init__ function and specify how data will pass through the network in the forward function.
Example model definition:
```bash
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
print(model)
```

# Training the model
```bash
# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

# Training loop
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Example usage
train(training_data, model, loss_fn, optimizer)
```

# Advanced Topics

1. AutoGrad
PyTorch's automatic differentiation feature allows you to compute gradients automatically. Here's an example:
```bash
x = torch.tensor([2.], requires_grad=True)
y = x * x * 3  # 3x^2
y.backward()
print(x.grad)  # d(y)/d(x) = d(3x^2)/d(x) = 6x = 12
```

2. Custom Modules:
```bash
class MultilayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MultilayerPerceptron, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x

model = MultilayerPerceptron(5, 3)
print(model)
```

# License
This README template includes all the pertinent information about your project, such as installation instructions, usage, project structure, data processing, model training, model evaluation, and details about the web application. It also includes sections for contributing and licensing, which are important for open-source projects.
