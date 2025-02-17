from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
import math

#########################
# 1. Initiate Model
#########################
def initiate_model(model_name,num_classes=None,in_channels=None,ratio=None,weights=None):
    """
    Build a model based on its name and desired output.
    
    Parameters:
      model_name (str): e.g. "resnet18" or "resnet50" (case-insensitive).
      num_classes (int): Number of output classes.
      input_channels (int): Number of channels in the input.
      conv_params (dict, optional): Dictionary with optional keys:
          'kernel_size', 'stride', 'padding', 'bias'.  
          Defaults: kernel_size=7, stride=2, padding=3, bias=False.
      weights (None or str): 
          - None: do not load any weights.
          - "default": load default pretrained weights (ImageNet).
          - Otherwise, interpreted as a filename from which to load state_dict.
    
    Returns:
      model (nn.Module)
    """

    # Normalize model name (we use lower-case keys in MODEL_DIMS later).
    model_name = model_name.lower()
    model_constructor = getattr(models, model_name, None)
    if model_constructor is None:
        raise ValueError(f"Invalid model name: {model_name}")
    
    # Instantiate model.
    if weights:
        if os.path.isfile(weights):
            model = model_constructor(weights=None)
            model.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
            print(f"Loaded weights from stored model{weights}.")
        else:
            model = model_constructor(weights=weights)
            print(f"Loaded weights from pretrained.")
    else:
        model = model_constructor(weights=None)
        print(f"Untrained Model.")
    
        # Use defaults if conv_params not provided.
    if ratio:
        if hasattr(model, "conv1"):
            model.conv1 = nn.Conv2d(
                in_channels=in_channels or model.conv1.in_channels,
                out_channels=model.conv1.out_channels,
                kernel_size= max(3, round(ratio*model.conv1.kernel_size[0])),
                stride=max(3, round(ratio*model.conv1.stride[0])),
                padding=max(3, round(ratio*model.conv1.padding[0])),
                bias=model.conv1.bias
            )
        else:
            raise ValueError("The model does not have a conv1 attribute.")
    
    # Replace the final fully-connected layer to match num_classes.
    if num_classes and hasattr(model, "fc"):
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model


#########################
# 2. Initiate Dataset
#########################
def initiate_dataset(dataset_name:str, resize: Optional[Union[int, Tuple[int, int]]] = None):
    """
    Build the dataset (and DataLoaders) given its name.
    
    Parameters:
      dataset_name (str): e.g. "mnist" or "cifar10" (case-insensitive).
      batch_size (int): Batch size for DataLoaders.
      resize (int or tuple, optional): If provided, images will be resized.
          Otherwise, the dataset’s native dimension is used.
    
    Returns:
      trainset, testset, train_loader, test_loader
    """
    # Normalize dataset name.
    dataset_name = dataset_name.lower()
    dataset_constructor = getattr(torchvision.datasets, dataset_name.upper(), None)
    if dataset_constructor is None:
        raise ValueError(f"Dataset '{dataset_name.upper()}' not found.")
    
    # If no explicit resize is provided, use the dataset’s native dimension.
    if resize:
        transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),  
        ])
    
    trainset = dataset_constructor(root='./data', train=True, download=True, transform=transform)
    testset = dataset_constructor(root='./data', train=False, download=True, transform=transform)
    
    return trainset, testset



# Saved properties for models and datasets.
MODEL_DIMS = {
    "resnet18": {"mul": 32, "pref": 32},
    "resnet50": {"mul": 64, "pref": 244},
    "lenet":{"pref":28}
    # Add others as needed.
}

DATASET_PROPERTIES = {
    "mnist":   {"dim": 28,  "channels": 1, "num_classes": 10},
    "cifar10": {"dim": 32,  "channels": 3, "num_classes": 10},
    "imagenet": {"dim": 224, "channels": 3, "num_classes": 1000},
}
#########################
# 3. Initiate Model & Dataset (Matcher)
#########################
def initiate_model_and_dataset(model_name, dataset_name,
                               weights=None,
                               batch_size=64):
    # Normalize keys.
    model_key = model_name.lower()
    dataset_key = dataset_name.lower()
    
    ds_props = DATASET_PROPERTIES.get(dataset_key)
    if ds_props is None:
        raise ValueError(f"Dataset properties for '{dataset_name}' not found.")
    ds_dim = ds_props["dim"]
    ds_channels = ds_props["channels"]
    ds_num_classes = ds_props["num_classes"]
    
    model_dims = MODEL_DIMS.get(model_key)
    if model_dims is None:
        raise ValueError(f"Model dimensions for '{model_name}' not found.")
    
    resize_val = None
    ratio=None

    minimum_dim = model_dims["mul"]
    preferred_dim= model_dims["pref"]  

    if minimum_dim:
        R = minimum_dim/ds_dim 
        r= round(R)
        chosen_dim=minimum_dim
    else:
        R = preferred_dim/ds_dim 
        r=1
        chosen_dim=preferred_dim

    diff = r - R
    if diff==0:
        resize_val=None
        ratio= None
    elif 0.25> diff > 0:
        chosen_dim = math.ceil(R) * chosen_dim
        resize_val = chosen_dim
        ratio=r
    elif -0.25< diff < 0:
        chosen_dim = math.floor(R) * chosen_dim
        resize_val = chosen_dim
        ratio=r
    else:
        raise ValueError(f"dataset{dataset_name} dimensions not compatible with minimum dimensions of {model_name}. diff:{diff, minimum_dim,ds_dim,chosen_dim,math.ceil(R)}")
    
    # Initiate dataset with chosen resize.
    trainset, testset = initiate_dataset(
        dataset_name, resize=resize_val
    )
    
    # Initiate model using dataset properties.
    model = initiate_model(model_name, ds_num_classes, ds_channels,
                           ratio=ratio, weights=weights)
    return model, (trainset, testset)
