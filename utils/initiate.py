from typing import Optional, Tuple, Union
import os
import math
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# Saved properties for models and datasets.
MODEL_DIMS = {
    "resnet18": {"mul": 32, "pref": 32},
    "resnet50": {"mul": 64, "pref": 244},
    "lenet":   {"pref": 28}
    # Add others as needed.
}

DATASET_PROPERTIES = {
    "mnist":   {"dim": 28,  "channels": 1, "num_classes": 10},
    "cifar10": {"dim": 32,  "channels": 3, "num_classes": 10},
    "imagenet": {"dim": 224, "channels": 3, "num_classes": 1000},
}


##############################################
# 1. Get Dimensions
##############################################
def get_dimensions(model_name: str, dataset_name: str) -> Tuple[Optional[int], Optional[float]]:
    """
    Compute the appropriate resize value for dataset images and the scaling ratio
    for adjusting the model based on the saved properties.
    
    Parameters:
      model_name (str): e.g., "resnet18" or "resnet50"
      dataset_name (str): e.g., "mnist" or "cifar10"
    
    Returns:
      A tuple (resize_val, ratio) where:
        - resize_val: An integer size to which dataset images should be resized,
                      or None if no resizing is necessary.
        - ratio: A scaling ratio for adjusting the model's first convolution layer,
                 or None if no adjustment is needed.
    
    Raises:
      ValueError if the dataset or model properties cannot be found or are incompatible.
    """
    model_key = model_name.lower()
    dataset_key = dataset_name.lower()
    
    ds_props = DATASET_PROPERTIES.get(dataset_key)
    if ds_props is None:
        raise ValueError(f"Dataset properties for '{dataset_name}' not found.")
    ds_dim = ds_props["dim"]
    
    model_props = MODEL_DIMS.get(model_key)
    if model_props is None:
        raise ValueError(f"Model dimensions for '{model_name}' not found.")
    
    # Use the minimum dimension if available; otherwise use the preferred dimension.
    minimum_dim = model_props.get("mul")
    preferred_dim = model_props.get("pref")
    
    if minimum_dim:
        R = minimum_dim / ds_dim
        r_val = round(R)
        chosen_dim = minimum_dim
    else:
        R = preferred_dim / ds_dim
        r_val = 1
        chosen_dim = preferred_dim
    
    diff = r_val - R
    if diff == 0:
        resize_val = None
        ratio = None
    elif 0 < diff < 0.25:
        chosen_dim = math.ceil(R) * chosen_dim
        resize_val = chosen_dim
        ratio = r_val
    elif -0.25 < diff < 0:
        chosen_dim = math.floor(R) * chosen_dim
        resize_val = chosen_dim
        ratio = r_val
    else:
        raise ValueError(f"Dataset {dataset_name} dimensions not compatible with {model_name}.")
    
    return resize_val, ratio


##############################################
# 2. Initiate Model
##############################################
def initiate_model(model_name: str, 
                   dataset_name: str,
                   weights: Optional[str] = None) -> nn.Module:
    """
    Build and return a model based on the model name and dataset properties.
    
    If pretrained weights are provided, load them and adjust any mismatched layers:
      - The first conv layer (conv1) for input channel differences.
      - The final fully-connected layer (fc) for class count differences.
      
    Parameters:
      model_name (str): e.g., "resnet18" or "resnet50"
      dataset_name (str): e.g., "mnist" or "cifar10"
      weights (Optional[str]): If provided, either "default" for default pretrained weights
                               or a filepath to a state_dict.
    
    Returns:
      nn.Module: The constructed model with appropriate modifications.
    """
    # --- Get dataset properties ---
    dataset_key = dataset_name.lower()
    ds_props = DATASET_PROPERTIES.get(dataset_key)
    if ds_props is None:
        raise ValueError(f"Dataset properties for '{dataset_name}' not found.")
    
    ds_channels = ds_props["channels"]
    ds_num_classes = ds_props["num_classes"]
    _, ratio = get_dimensions(model_name, dataset_name)

    # --- Construct the base model ---
    model_name_lower = model_name.lower()
    model_constructor = getattr(models, model_name_lower, None)
    if model_constructor is None:
        raise ValueError(f"Invalid model name: {model_name}")

    # For flexibility, always build a base model with weights=None
    # so we have control over the architecture.
    model = model_constructor(weights=None)

    # --- Adjust the first convolutional layer (conv1) ---
    if ratio and hasattr(model, "conv1"):
        conv1 = model.conv1
        new_kernel = max(3, round(ratio * conv1.kernel_size[0]))
        new_stride = max(3, round(ratio * conv1.stride[0]))
        new_padding = max(3, round(ratio * conv1.padding[0]))
        # Only change if dataset channels differ from current conv1 input channels.
        if conv1.in_channels != ds_channels:
            model.conv1 = nn.Conv2d(
                in_channels=ds_channels,  # match the dataset (e.g., 1 for grayscale)
                out_channels=conv1.out_channels,
                kernel_size=new_kernel,
                stride=new_stride,
                padding=new_padding,
                bias=(conv1.bias is not None)
            )
            print(f"Modified conv1: updated in_channels to {ds_channels}.")
    elif ratio and not hasattr(model, "conv1"):
        raise ValueError("The model does not have a conv1 attribute.")

    # --- Adjust the final fully-connected layer (fc) ---
    if ds_num_classes and hasattr(model, "fc"):
        num_ftrs = model.fc.in_features
        if model.fc.out_features != ds_num_classes:
            model.fc = nn.Linear(num_ftrs, ds_num_classes)
            print(f"Modified fc layer: updated out_features to {ds_num_classes}.")

    # --- Load pretrained weights (if provided) ---
    # There are two cases:
    # a) A file path is given -> load from file.
    # b) "default" is given -> load standard pretrained weights.
    if weights:
        if os.path.isfile(weights):
            # Load checkpoint; using strict=False allows mismatched layers to be skipped.
            state_dict = torch.load(weights, map_location="cpu",weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded weights from file: {weights}")
        else:
            # If weights is "default", then we load the default pretrained weights.
            # However, this branch means the pretrained model was built with its default architecture.
            # You may then need to adjust mismatched layers as done above.
            model = model_constructor(weights=weights)
            print("Loaded default pretrained weights.")
    else:
        print("Initialized untrained model.")

    return model



##############################################
# 3. Initiate Dataset
##############################################
def initiate_dataset(dataset_name: str,
                     model_name: str) -> Tuple[torchvision.datasets.VisionDataset,
                                               torchvision.datasets.VisionDataset]:
    """
    Build and return the train and test datasets for a given dataset name.
    The model name is used to compute the appropriate image size (resize value)
    via get_dimensions so that the dataset matches the expected input size.
    
    Parameters:
      dataset_name (str): e.g., "mnist" or "cifar10"
      model_name (str): e.g., "resnet18" or "resnet50"
    
    Returns:
      A tuple (trainset, testset)
    
    Raises:
      ValueError if the dataset cannot be found.
    """
    resize_val, _ = get_dimensions(model_name, dataset_name)
    
    dataset_name_upper = dataset_name.upper()
    dataset_constructor = getattr(torchvision.datasets, dataset_name_upper, None)
    if dataset_constructor is None:
        raise ValueError(f"Dataset '{dataset_name_upper}' not found.")
    
    if resize_val:
        transform = transforms.Compose([
            transforms.Resize(resize_val),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.ToTensor()
    
    trainset = dataset_constructor(root='./data', train=True, download=True, transform=transform)
    testset = dataset_constructor(root='./data', train=False, download=True, transform=transform)
    
    return trainset, testset
