import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from utils.initiate import initiate_dataset, initiate_model
import os
import math
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from condensation.sort_distance import (
    compute_class_centroids,
    compute_class_geometric_medians,
    compute_distance_to_centroids,
    normalized_distance_sample,
    compute_normalized_distance_scores,
)

if __name__ == "__main__":
    # Load dataset and model
    # dataset, testset = initiate_dataset("CIFAR10", "resnet18")
    dataset, testset = initiate_dataset("MNIST", "resnet18")
    # 1. Compute arithmetic centroids
    print("Computing arithmetic centroids...")
    arith_centroids = compute_class_centroids(dataset, batch_size=128)
    print("Arithmetic centroid for class 0 (first 5 values):", arith_centroids[0][:5])

    # 2. Compute Euclidean distances to arithmetic centroids
    print("Computing Euclidean distances to arithmetic centroids...")
    distances = compute_distance_to_centroids(dataset, arith_centroids, batch_size=128)
    print("Euclidean Distances (first 10):", distances[:10])

    # 3. Compute geometric medians (robust centers)
    print("Computing geometric medians (this may take a while)...")
    geo_medians = compute_class_geometric_medians(
        dataset, batch_size=128, eps=1e-5, max_iter=500
    )
    print("Geometric median for class 0 (first 5 values):", geo_medians[0][:5])

    # 4. Compute normalized distance scores using geometric medians
    print("Computing normalized distance scores (geometric median as center)...")
    norm_scores = compute_normalized_distance_scores(
        dataset, geo_medians, batch_size=128, threshold=0.9, mode="max"
    )
    print("Normalized Distance Scores (first 10):", norm_scores[:10])
