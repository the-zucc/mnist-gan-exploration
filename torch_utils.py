import torch

class MNISTFlatten(torch.nn.Conv2d):
    # 1. Reshapes the MNIST dataset to add a dimension
    # 2. Performs the Conv2d calculations
    def forward(self, X: torch.tensor) -> torch.tensor:
        return super().forward(X.view(X.shape[0],1,X.shape[1], X.shape[2]))

class Passthrough(torch.nn.Module):
    def __init__(self, verbose=True):
        super(Passthrough, self).__init__()
        self.verbose = verbose
    def forward(self, x):
        if self.verbose:
            print(x.shape)
        return x

class RandomizedLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, random_features):
        super().__init__(in_features+random_features, out_features)
        self.random_features = random_features
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(size=(input.shape[0],1,self.random_features))
        return super().forward(torch.cat([input,noise],2))

class MNISTReshape(torch.nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.view(input.shape[0],28,28)*torch.Tensor([255]) 