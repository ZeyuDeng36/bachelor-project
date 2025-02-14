import torch
import torch.nn as nn
import torchvision.transforms as transforms


def transform_model_and_dataset(model: nn.Module, dataset, 
                                model_type: str, dataset_name: str):
    """
    Transforms both the model and dataset so that they are compatible.
    
    Steps:
    1. Look up the dataset's native dimension, channel count, and number of classes.
    2. Look up the model's preferred input dimension.
    3. If the dataset's dimension is below the preferred size, modify its transform to resize to that dimension.
    4. Adjust the model's first convolutional layer:
         - Ensure its in_channels match the dataset.
         - If the dataset was resized, compute a scaling ratio and adjust kernel_size, stride, and padding proportionally.
    5. Replace the final (fully connected) layer of the model so that it outputs the correct number of classes.
    
    Parameters:
      model: a PyTorch nn.Module (assumed to have attributes 'conv1' and 'fc')
      dataset: a torchvision dataset with a 'transform' attribute
      model_type: a key for the model in MODEL_PREFERRED_DIM (e.g., "resnet")
      dataset_name: a key for the dataset in DATASET_PROPERTIES (e.g., "mnist")
    
    Returns:
      (model, dataset) after modifications.
    """
    
    # Retrieve dataset properties and model preferred dimension.
    ds_props = DATASET_PROPERTIES.get(dataset_name.lower())
    if ds_props is None:
        raise ValueError(f"Dataset properties for '{dataset_name}' not found.")
    ds_dim = ds_props["dim"]
    ds_channels = ds_props["channels"]
    ds_num_classes = ds_props["num_classes"]
    
    preferred_dim = MODEL_PREFERRED_DIM.get(model_type.lower())
    if preferred_dim is None:
        raise ValueError(f"Model preferred dimension for '{model_type}' not found.")
    
    # Determine scaling ratio if dataset is smaller than preferred dimensions.
    ratio = preferred_dim / ds_dim if ds_dim < preferred_dim else 1.0
    
    # --- Transform the Dataset ---
    # If the dataset's native dimension is below the preferred, add a resize transform.
    if ratio > 1.0:
        new_transform = transforms.Compose([
            transforms.Resize((preferred_dim, preferred_dim)),
            transforms.ToTensor(),
        ])
        dataset.transform = new_transform  # Override existing transform.
    
    # --- Transform the Model ---
    # 1. Adjust the first convolution layer if present.
    if hasattr(model, "conv1"):
        orig_conv = model.conv1
        
        # If the current conv1 doesn't accept the dataset's channel count, or if scaling is needed:
        if orig_conv.in_channels != ds_channels or ratio > 1.0:
            # Save original conv parameters (assuming square kernel, stride, and padding)
            orig_kernel = orig_conv.kernel_size[0]
            orig_stride = orig_conv.stride[0]
            orig_padding = orig_conv.padding[0]
            
            # Compute new parameters using the ratio.
            # The idea: maintain the same relative receptive field with respect to the input size.
            new_kernel = max(3, round(orig_kernel * ratio))
            new_stride = max(1, round(orig_stride * ratio))
            new_padding = max(0, round(orig_padding * ratio))
            
            # Create a new convolutional layer with adjusted parameters.
            new_conv = nn.Conv2d(
                in_channels=ds_channels,
                out_channels=orig_conv.out_channels,
                kernel_size=new_kernel,
                stride=new_stride,
                padding=new_padding,
                bias=(orig_conv.bias is not None)
            )
            model.conv1 = new_conv
        # If in_channels already match and no scaling is needed, leave conv1 as is.
    
    # 2. Adjust the final fully connected layer (assumes attribute 'fc').
    if hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, ds_num_classes)
    
    return model, dataset
