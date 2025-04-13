import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from utils.initiate import initiate_dataset, initiate_model

####################################
# 1. Compute Class Centroids (Arithmetic Mean)
####################################
def compute_class_centroids(dataset, batch_size=128) -> dict:
    """
    Computes the arithmetic mean (centroid) for each class using batched operations.
    
    Args:
        dataset (torch.utils.data.Dataset): A dataset yielding (feature, label) tuples.
        batch_size (int): Batch size for processing.
        
    Returns:
        dict: Dictionary mapping each class label (int) to its centroid tensor (shape: [D]).
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    class_features = defaultdict(list)
    
    # Process in batches and group features by label.
    with torch.no_grad():
        for features, labels in loader:
            for feat, label in zip(features, labels):
                # Group feature vectors into a list per class.
                class_features[int(label.item())].append(feat.cpu())
    
    # Compute arithmetic mean for each class.
    centroids = {}
    for label, feats in class_features.items():
        feats_tensor = torch.stack(feats, dim=0)  # shape: [N_class, D]
        centroids[label] = torch.mean(feats_tensor, dim=0)
    
    return centroids

####################################
# 2. Compute Class Robust Centers (Geometric Medians)
####################################
def geometric_median_tensor(features: torch.Tensor, eps: float = 1e-5, max_iter: int = 500) -> torch.Tensor:
    """
    Computes the geometric median for a set of feature vectors using Weiszfeld's algorithm.
    
    Args:
        features (torch.Tensor): Tensor of shape [N, D].
        eps (float): Convergence tolerance.
        max_iter (int): Maximum number of iterations.
    
    Returns:
        torch.Tensor: The geometric median (shape: [D]).
    """
    median = torch.mean(features, dim=0)  # Initialize with arithmetic mean.
    for _ in range(max_iter):
        diff = features - median              # [N, D]
        distances = torch.norm(diff, p=2, dim=1)  # [N]
        inv_distances = 1.0 / (distances + eps)   # Avoid division by zero.
        new_median = torch.sum(features * inv_distances.unsqueeze(1), dim=0) / torch.sum(inv_distances)
        if torch.norm(new_median - median, p=2) < eps:
            break
        median = new_median
    return median

def compute_class_geometric_medians(dataset, batch_size=128, eps: float = 1e-5, max_iter: int = 500) -> dict:
    """
    Computes the geometric median (robust center) for each class in the dataset.
    
    Args:
        dataset (torch.utils.data.Dataset): A dataset yielding (feature, label) tuples.
        batch_size (int): Batch size for processing.
        eps (float): Convergence tolerance.
        max_iter (int): Maximum number of iterations.
    
    Returns:
        dict: Dictionary mapping each class label to its geometric median tensor.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    class_features = defaultdict(list)
    
    with torch.no_grad():
        for features, labels in loader:
            for feat, label in zip(features, labels):
                class_features[int(label.item())].append(feat.cpu())
    
    medians = {}
    for label, feats in class_features.items():
        feats_tensor = torch.stack(feats, dim=0)
        medians[label] = geometric_median_tensor(feats_tensor, eps, max_iter)
    
    return medians

############################################
# 3. Compute Euclidean Distance to Class Centroids
############################################
def compute_distance_to_centroids(dataset, centroids: dict, batch_size=128) -> list:
    """
    Computes the Euclidean distance of each datapoint's feature vector to its class centroid.
    
    Args:
        dataset (torch.utils.data.Dataset): Dataset yielding (feature, label) tuples.
        centroids (dict): Dictionary mapping class labels to centroid tensors.
        batch_size (int): Batch size for processing.
    
    Returns:
        list of tuples: (distance, sample_index) sorted in descending order.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    distances = []
    sample_index = 0
    
    with torch.no_grad():
        for features, labels in loader:
            for feat, label in zip(features, labels):
                label_int = int(label.item())
                center = centroids[label_int]
                dist = torch.norm(feat.cpu() - center, p=2)
                distances.append((dist.item(), sample_index))
                sample_index += 1
                
    # Sort distances in descending order.
    return sorted(distances, key=lambda x: x[0], reverse=True)

##########################################################################
# 4. Normalized Distance Score: Distance Divided by Scale from Similar Directional Samples
##########################################################################
def normalized_distance_sample(
    sample: torch.Tensor, 
    center: torch.Tensor, 
    class_features: torch.Tensor, 
    threshold: float = 0.9, 
    mode: str = 'max',
    eps: float = 1e-6
) -> float:
    """
    Computes a normalized Euclidean distance for a sample's feature vector from its class center.
    The raw L2 distance is divided by a scale factor computed from samples in the class with similar alignment.
    
    Args:
        sample (torch.Tensor): 1D tensor representing the sample's feature vector (shape: [D]).
        center (torch.Tensor): The class center (either arithmetic or geometric median).
        class_features (torch.Tensor): Tensor of all feature vectors for this class (shape: [N_class, D]).
        threshold (float): Cosine similarity threshold for selecting similarly aligned samples.
        mode (str): 'max' to use the maximum or 'avg' to use the average distance from selected samples as the scale factor.
        eps (float): A small constant to prevent division by zero.
    
    Returns:
        float: Normalized distance score.
    """
    diff_sample = sample - center
    norm_sample = torch.norm(diff_sample, p=2)
    if norm_sample.item() == 0:
        return 0.0

    direction = diff_sample / norm_sample  # Unit vector for sample's direction.
    diff_all = class_features - center      # [N_class, D]
    norms_all = torch.norm(diff_all, p=2, dim=1)  # [N_class]
    
    # Compute unit vectors for all samples and cosine similarities with the sample's direction.
    unit_vectors = diff_all / (norms_all.unsqueeze(1) + eps)
    cos_sim = torch.matmul(unit_vectors, direction)  # [N_class]
    mask = cos_sim >= threshold
    selected_norms = norms_all[mask]
    
    if selected_norms.numel() == 0:
        scale = torch.mean(norms_all) if mode == 'avg' else torch.max(norms_all)
    else:
        scale = torch.mean(selected_norms) if mode == 'avg' else torch.max(selected_norms)
    
    return norm_sample / (scale + eps)

def compute_normalized_distance_scores(dataset, centroids: dict, batch_size=128, threshold=0.9, mode='max') -> list:
    """
    Computes normalized distance scores for each datapoint in the dataset.
    The normalized score is computed as the raw Euclidean distance from its feature vector to its class center,
    divided by a scale factor derived from features in the class with similar directional alignment.
    
    Args:
        dataset (torch.utils.data.Dataset): Dataset yielding (feature, label) tuples.
        centroids (dict): Dictionary mapping each class label to its center tensor.
        batch_size (int): Batch size for processing.
        threshold (float): Cosine similarity threshold.
        mode (str): 'max' for maximum or 'avg' for average as the scale.
    
    Returns:
        list of tuples: (normalized_score, sample_index) sorted in descending order.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    class_features = defaultdict(list)
    ordered_features = []
    ordered_labels = []
    sample_index = 0
    scores = []
    
    # First, group features by label.
    with torch.no_grad():
        for features, labels in loader:
            for feat, label in zip(features, labels):
                label_int = int(label.item())
                feat_cpu = feat.cpu()
                ordered_features.append(feat_cpu)
                ordered_labels.append(label_int)
                class_features[label_int].append(feat_cpu)
                sample_index += 1
    
    # Stack features for each class.
    for label in class_features:
        class_features[label] = torch.stack(class_features[label], dim=0)
    
    # Compute normalized distance for each sample.
    for idx, (feat, label) in enumerate(zip(ordered_features, ordered_labels)):
        center = centroids[label]
        class_feats = class_features[label]
        norm_score = normalized_distance_sample(feat, center, class_feats, threshold=threshold, mode=mode)
        scores.append((norm_score, idx))
    
    # Sort scores in descending order.
    return sorted(scores, key=lambda x: x[0], reverse=True)
