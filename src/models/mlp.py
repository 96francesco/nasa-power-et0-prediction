import torch.nn as nn

class MLP(nn.Module):
    """
    A multilayer perceptron (MLP) model designed for regression tasks, 
    implemented using PyTorch's neural network module.

    This class defines a simple feedforward neural network with three linear layers, 
    ReLU activations, and dropout for regularization. It's suitable for tasks like 
    predicting continuous variables from a given set of input features.

    Attributes:
        input_size (int): The number of input features the model expects.
        dropout_rate (float): The dropout rate used in the dropout layers for regularization.

    Methods:
        forward(x):
            Defines the forward pass of the model. Takes an input tensor `x` and returns the model's output tensor.

    Example:
        >>> model = MLPModel(input_size=10, dropout_rate=0.5)
        >>> print(model)
        MLPModel(
          (fc1): Linear(in_features=10, out_features=256, bias=True)
          (fc2): Linear(in_features=256, out_features=128, bias=True)
          (fc3): Linear(in_features=128, out_features=1, bias=True)
          (relu): ReLU()
          (dropout): Dropout(p=0.5, inplace=False)
        )
    """
    def __init__(self, input_size=11):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x