# Model implementation
# MLP = Multi-Layer Perceptron
# This code builds a configurable fully connected neural network with tanh activations and careful weight initialization, designed for stable PINN/PDE training. Each layer does a weighted mix of signals (Linear), bends them (Tanh), and passes them deeper so the network can build a complex smooth function. Initialized carefully (Xavier) so training PDE gradients does not explode or vanish.

import torch.nn as nn


class MLP(nn.Module):
    # Constructor for MLP
    # inputs: in_dim (int): input dimension; defaults to 2. Corresponds to (x,t) inputs.
    #         out_dim (int): output dimension; defaults to 2. Corresponds to (y,z) outputs.
    #         hidden (int): number of hidden units per layer; defaults to 64.
    #         depth (int): number of hidden layers; defaults to 5.
    # outputs: nn.Module: MLP model
    def __init__(self, in_dim=2, out_dim=2, hidden=64, depth=5):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, out_dim))

        # Linear Layer means y = Wx + b, where x is the input vector (a x 1), W is weights matrix (b x a) and b is bias vector (b x 1), y is output vector (b x 1). So it mixes the inputs, scales them, and adds a bias. Note this is still a straight line in high dimensions no curves yet. To add non-linearity we add activation functions like Tanh.

        # Each layer is Linear followed by Tanh activation, except the last layer which is just Linear. Tanh applies non-linearity to every number: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)), squashing inputs to (-1, 1). Each hidden layer does h_{k+1}​=tanh(W_{k}h_{k}​+b_{k}​) where h_{k}​ is the output of layer k, W_{k} is weights matrix, b_{k} is bias vector.

        # Wrap all layers into a single PyTorch module
        self.net = nn.Sequential(*layers)

        # Weight Initialization. If the layer is Linear it:
        # 1. Initializes weights using Xavier uniform (good for tanh networks). 
        # 2. Initializes biases to zero.
        # Helps stabilize/smooth PDE training.
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # Forward pass
    def forward(self, x):
        return self.net(x)
