import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors


def knn(x, k):                                  # Input = [n, 9]
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(x)                                  
    _ , indices = knn.kneighbors(x)             # distances = [n, k], indices = [n, k]
    graph_features = x[indices]                 # graph_features = [n, k, 9]
    return torch.Tensor(graph_features)
