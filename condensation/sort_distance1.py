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

####################################
# Helper: Remove the FC Layer from the Model
####################################
def remove_fc_layer(model: nn.Module) -> nn.Module:
    """
    Modifies the given model by replacing its final fully connected (fc) layer with the identity mapping.
    This ensures that the model's output is the penultimate representation.
    
    Args:
        model (torch.nn.Module): Trained model with an fc layer.
        
    Returns:
        torch.nn.Module: The modified model that outputs penultimate features.
    """
    # Check and modify if the model has an attribute 'fc'
    if hasattr(model, 'fc'):
        model.fc = nn.Identity()
    return model

####################################
# 1. Compute Class Centroids (Arithmetic Mean) Using Model Features
####################################
def compute_class_centroids(dataset, model: nn.Module, batch_size=128, device='cpu') -> dict:
    """
    Computes class centroids using feature vectors extracted via the given model. The model's fc layer is removed
    so that the features are taken from the penultimate layer.
    
    Args:
        dataset (torch.utils.data.Dataset): Dataset yielding (image, label) tuples.
        model (torch.nn.Module): A trained model with an fc layer (will be modified to remove fc).
        batch_size (int): Batch size for processing.
        device (str): Device for computation.
        
    Returns:
        dict: Dictionary mapping each class label to its centroid tensor.
    """
    # Remove the FC layer.
    model = remove_fc_layer(model)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    class_features = defaultdict(list)
    
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            features = model(images)  # features now come from the penultimate layer
            features = features.cpu()
            for feat, label in zip(features, labels):
                class_features[int(label.item())].append(feat)
    
    centroids = {}
    for label, feats in class_features.items():
        feats_tensor = torch.stack(feats, dim=0)  # shape: [N_class, D]
        centroids[label] = torch.mean(feats_tensor, dim=0)
    
    return centroids

####################################
# 2. Compute Class Robust Centers (Geometric Medians) Using Model Features
####################################
def geometric_median_tensor(features: torch.Tensor, eps: float = 1e-5, max_iter: int = 500) -> torch.Tensor:
    """
    Computes the geometric median for a set of feature vectors using Weiszfeld's algorithm.
    
    Args:
        features (torch.Tensor): Tensor of shape [N, D].
        eps (float): Convergence tolerance.
        max_iter (int): Maximum iterations.
        
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

def compute_class_geometric_medians(dataset, model: nn.Module, batch_size=128, device='cpu', eps: float = 1e-5, max_iter: int = 500) -> dict:
    """
    Computes the geometric median (robust center) for each class using features from the model.
    The model's fc layer is removed so that the features are taken from the penultimate layer.
    
    Args:
        dataset (torch.utils.data.Dataset): Dataset yielding (image, label) tuples.
        model (torch.nn.Module): A trained model with an fc layer (will be modified to remove fc).
        batch_size (int): Batch size for processing.
        device (str): Device for computation.
        eps (float): Convergence tolerance.
        max_iter (int): Maximum iterations for geometric median computation.
        
    Returns:
        dict: Dictionary mapping each class label to its geometric median tensor.
    """
    # Remove the FC layer.
    model = remove_fc_layer(model)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    class_features = defaultdict(list)
    
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            features = model(images)
            features = features.cpu()
            for feat, label in zip(features, labels):
                class_features[int(label.item())].append(feat)
    
    medians = {}
    for label, feats in class_features.items():
        feats_tensor = torch.stack(feats, dim=0)
        medians[label] = geometric_median_tensor(feats_tensor, eps, max_iter)
    
    return medians

############################################
# 3. Compute Euclidean Distance to Class Centroids Using Model Features
############################################
def compute_distance_to_centroids(model: nn.Module,dataset, batch_size=128, device='cpu') -> list:
    """
    Computes the Euclidean distance of each datapoint's feature vector (extracted via the model) to its
    corresponding class centroid. The model's fc layer is removed so that features are from the penultimate layer.
    
    Args:
        dataset (torch.utils.data.Dataset): Dataset yielding (image, label) tuples.
        model (torch.nn.Module): A trained model with an fc layer (will be modified to remove fc).
        centroids (dict): Mapping from class label to centroid tensor.
        batch_size (int): Batch size for processing.
        device (str): Device for computation.
    
    Returns:
        list of tuples: (distance, sample_index) sorted in descending order.
    """
    centroids = compute_class_centroids(dataset, model)
    # Remove the FC layer.
    model = remove_fc_layer(model)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    distances = []
    sample_index = 0
    
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            features = model(images)
            features = features.cpu()
            for feat, label in zip(features, labels):
                label_int = int(label.item())
                center = centroids[label_int]
                dist = torch.norm(feat - center, p=2)
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
        center (torch.Tensor): The class center (arithmetic or geometric median).
        class_features (torch.Tensor): All feature vectors for this class (shape: [N_class, D]).
        threshold (float): Cosine similarity threshold for selecting similarly aligned samples.
        mode (str): 'max' or 'avg' to use maximum or average distance from selected samples as the scale.
        eps (float): Small constant to avoid division by zero.
    
    Returns:
        float: Normalized distance score.
    """
    diff_sample = sample - center
    norm_sample = torch.norm(diff_sample, p=2)
    if norm_sample.item() == 0:
        return 0.0

    direction = diff_sample / norm_sample  # Unit vector
    diff_all = class_features - center      # [N_class, D]
    norms_all = torch.norm(diff_all, p=2, dim=1)  # [N_class]
    
    # Compute cosine similarities for all features.
    unit_vectors = diff_all / (norms_all.unsqueeze(1) + eps)
    cos_sim = torch.matmul(unit_vectors, direction)  # [N_class]
    mask = cos_sim >= threshold
    selected_norms = norms_all[mask]
    
    if selected_norms.numel() == 0:
        scale = torch.mean(norms_all) if mode == 'avg' else torch.max(norms_all)
    else:
        scale = torch.mean(selected_norms) if mode == 'avg' else torch.max(selected_norms)
    
    return norm_sample / (scale + eps)

def compute_normalized_distance_scores(dataset, model: nn.Module, centroids: dict, batch_size=128, device='cpu', threshold=0.9, mode='max') -> list:
    """
    Computes normalized distance scores for each datapoint using its feature vector (from the penultimate layer)
    extracted via the model. The model's fc layer is removed so that the features come from the penultimate representation.
    
    Args:
        dataset (torch.utils.data.Dataset): Dataset yielding (image, label) tuples.
        model (torch.nn.Module): A trained model with an fc layer (will be modified to remove fc).
        centroids (dict): Mapping from class label to center tensor.
        batch_size (int): Batch size for processing.
        device (str): Device for computation.
        threshold (float): Cosine similarity threshold.
        mode (str): 'max' or 'avg' to use for computing the scale factor.
    
    Returns:
        list of tuples: (normalized_score, sample_index) sorted in descending order.
    """
    # Remove the FC layer.
    model = remove_fc_layer(model)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    class_features = defaultdict(list)
    ordered_features = []
    ordered_labels = []
    sample_index = 0
    scores = []
    
    model.to(device)
    model.eval()
    
    # Extract features and group by label.
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            features = model(images)
            features = features.cpu()
            for feat, label in zip(features, labels):
                label_int = int(label.item())
                ordered_features.append(feat)
                ordered_labels.append(label_int)
                class_features[label_int].append(feat)
                sample_index += 1
    
    # Stack features for each class.
    for label in class_features:
        class_features[label] = torch.stack(class_features[label], dim=0)
    
    # Compute normalized distance
