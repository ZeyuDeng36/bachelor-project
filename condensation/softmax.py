import random
import torch
import torch.nn.functional as F
import math

def predictive_entropy(logits):
    """
    Computes the predictive entropy of a batch of logits.
    
    Args:
        logits (torch.Tensor): Raw outputs from the model (shape: [batch_size, num_classes])
    
    Returns:
        entropy (torch.Tensor): Predictive entropy for each example in the batch.
    """
    # Compute softmax probabilities
    probs = F.softmax(logits, dim=1)
    # Compute entropy: -sum(p * log(p))
    entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)
    return entropy

def log_percentage_entropy(logits):
    """
    Computes the log of the percentage of maximum possible entropy.
    The maximum possible entropy for num_classes is log(num_classes).
    
    Args:
        logits (torch.Tensor): Raw outputs from the model (shape: [batch_size, num_classes])
    
    Returns:
        log_pct_entropy (torch.Tensor): Log percentage of max entropy for each example.
    """
    num_classes = logits.size(1)
    entropy = predictive_entropy(logits)
    max_entropy = torch.log(torch.tensor(num_classes, dtype=entropy.dtype, device=entropy.device))
    # Calculate percentage (between 0 and 1)
    pct_entropy = entropy / max_entropy
    # Take the log (you might add a small constant to avoid log(0))
    log_pct_entropy = torch.log(pct_entropy + 1e-12)
    return log_pct_entropy

# Example usage:
if __name__ == "__main__":
    # Suppose you have a batch of logits from your model's forward pass
    # For example, with batch_size=4 and 10 classes:
    logits = torch.tensor([[2.0, 1.0, 0.1, 0.5, 0.3, 0.2, 0.1, 0.05, 0.0, -0.5],
                           [1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
                           [0.1, 0.2, 0.3, 0.4, 0.5, 1.5, 0.6, 0.7, 0.8, 0.9],
                           [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]])
    
    # Compute raw predictive entropy
    entropy = predictive_entropy(logits)
    print("Predictive Entropy:", entropy)
    
    # Compute log percentage entropy
    log_pct_ent = log_percentage_entropy(logits)
    print("Log Percentage Entropy:", log_pct_ent)

def condense_dataset(model, dataset, entropy_func='predictive_entropy',rate=1):
    """
    Condenses the dataset by selecting the top N samples with the highest uncertainty (entropy).
    
    Args:
        model (torch.nn.Module): Pre-trained model to make predictions.
        dataset (torch.utils.data.Dataset): Dataset from which to select samples.
        entropy_func (str): Either 'predictive_entropy' or 'log_percentage_entropy' to use for uncertainty estimation.
        top_n (int): The number of most uncertain samples to select for condensation.
    
    Returns:
        condensed_samples (list): List of the top N most uncertain samples.
    """
    if (1-rate)==1:
        return dataset
    else:
        # Move model to the appropriate device (GPU or CPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()  # Set the model to evaluation mode
        all_entropies = []  # To store entropy values
        all_indices = []  # To store indices of the samples

        # Iterate through the dataset
        with torch.no_grad():  # No need to compute gradients for this operation
            for idx, (inputs, labels) in enumerate(dataset):
                inputs, labels = inputs.to(device), labels
                inputs = inputs.unsqueeze(0)  # Add batch dimension (since dataset is typically single-sample)
                
                # Get the logits from the model (no softmax here, we're dealing with logits)
                logits = model(inputs)
                
                # Compute entropy based on the chosen function
                if entropy_func == 'predictive_entropy':
                    entropy = predictive_entropy(logits)
                elif entropy_func == 'log_percentage_entropy':
                    entropy = log_percentage_entropy(logits)
                else:
                    raise ValueError(f"Invalid entropy function: {entropy_func}")
                
                all_entropies.append(entropy.item())  # Store entropy value
                all_indices.append(idx)  # Store the index of the sample

        # Sort the samples by entropy in descending order (most uncertain samples first)
        sorted_indices = sorted(zip(all_entropies, all_indices), reverse=True, key=lambda x: x[0])

        LEN = int(len(dataset)*(1-rate))
        # Select top N uncertain samples
        top_n_indices = [idx for _, idx in sorted_indices[:LEN]]
        condensed_samples = [dataset[i] for i in top_n_indices]  # Get the actual samples
        
        return condensed_samples

def condense_dataset1(model, dataset, entropy_func='predictive_entropy',rate=1):
    """
    Condenses the dataset by selecting the top N samples with the highest uncertainty (entropy).
    
    Args:
        model (torch.nn.Module): Pre-trained model to make predictions.
        dataset (torch.utils.data.Dataset): Dataset from which to select samples.
        entropy_func (str): Either 'predictive_entropy' or 'log_percentage_entropy' to use for uncertainty estimation.
        top_n (int): The number of most uncertain samples to select for condensation.
    
    Returns:
        condensed_samples (list): List of the top N most uncertain samples.
    """
    if (1-rate)==1:
        return dataset
    else:
        # Move model to the appropriate device (GPU or CPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()  # Set the model to evaluation mode
        all_entropies = []  # To store entropy values
        all_indices = []  # To store indices of the samples

        # Iterate through the dataset
        with torch.no_grad():  # No need to compute gradients for this operation
            for idx, (inputs, labels) in enumerate(dataset):
                inputs, labels = inputs.to(device), labels
                inputs = inputs.unsqueeze(0)  # Add batch dimension (since dataset is typically single-sample)
                
                # Get the logits from the model (no softmax here, we're dealing with logits)
                logits = model(inputs)
                
                # Compute entropy based on the chosen function
                if entropy_func == 'predictive_entropy':
                    entropy = predictive_entropy(logits)
                elif entropy_func == 'log_percentage_entropy':
                    entropy = log_percentage_entropy(logits)
                else:
                    raise ValueError(f"Invalid entropy function: {entropy_func}")
                
                all_entropies.append(entropy.item())  # Store entropy value
                all_indices.append(idx)  # Store the index of the sample

        # Sort the samples by entropy in descending order (most uncertain samples first)
        sorted_indices = sorted(zip(all_entropies, all_indices), reverse=True, key=lambda x: x[0])

        LEN = int(len(dataset)*(1-rate))
        # Select top N uncertain samples
        top_half_indices = [idx for _, idx in sorted_indices[:math.floor(LEN/2)]]
        bottom_half_indices = [idx for _, idx in sorted_indices[-math.ceil(LEN/2):]]
        condensed_samples = [dataset[i] for i in bottom_half_indices+top_half_indices]  # Get the actual samples
        
        return condensed_samples

def condense_dataset2(model, dataset, entropy_func='predictive_entropy', rate=1):
    """
    Condenses the dataset by selecting samples in a balanced manner over the different labels.
    For each label, roughly the same number of samples are chosen based on their uncertainty (entropy)
    with higher entropy indicating samples closer to the decision boundary.
    
    Args:
        model (torch.nn.Module): Pre-trained model to make predictions.
        dataset (torch.utils.data.Dataset): Dataset from which to select samples.
        entropy_func (str): Either 'predictive_entropy' or 'log_percentage_entropy' to use for uncertainty estimation.
        rate (float): Fraction of the dataset to drop (e.g., rate=0.2 keeps 80%).
    
    Returns:
        list: Selected samples, balanced over the different labels.
    """
    # If no condensation is required, return the whole dataset.
    if (1 - rate) == 1:
        return dataset
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        all_entropies = []  # store entropy for each sample
        all_indices   = []  # store dataset indices
        all_labels    = []  # store corresponding labels
        
        # Compute uncertainty for each sample.
        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(dataset):
                inputs = inputs.to(device).unsqueeze(0)  # add batch dimension
                logits = model(inputs)
                
                if entropy_func == 'predictive_entropy':
                    entropy = predictive_entropy(logits)
                elif entropy_func == 'log_percentage_entropy':
                    entropy = log_percentage_entropy(logits)
                else:
                    raise ValueError(f"Invalid entropy function: {entropy_func}")
                    
                all_entropies.append(entropy.item())
                all_indices.append(idx)
                # Ensure labels is a standard number.
                all_labels.append(labels.item() if hasattr(labels, 'item') else labels)
        
        # Determine the overall number of samples to select.
        total_to_select = int(len(dataset) * (1 - rate))
        # Find out which unique labels exist.
        unique_labels = list(set(all_labels))
        num_classes = len(unique_labels)
        # Calculate the number of samples to select per class.
        per_class_select = total_to_select // num_classes
        
        # Group the samples by label.
        label_groups = {label: [] for label in unique_labels}
        for entropy, idx, label in zip(all_entropies, all_indices, all_labels):
            label_groups[label].append((entropy, idx))
        
        # For each label, sort samples by entropy (descending) and pick the top ones.
        selected_indices = []
        for label, group in label_groups.items():
            sorted_group = sorted(group, key=lambda x: x[0], reverse=True)
            # If a group has fewer than 'per_class_select' samples, take all available.
            selected_indices.extend([idx for (_, idx) in sorted_group[:per_class_select]])
        
        # If due to rounding the total is less than required, fill the remainder from the overall ranking.
        if len(selected_indices) < total_to_select:
            overall_sorted = sorted(zip(all_entropies, all_indices), key=lambda x: x[0], reverse=True)
            overall_indices = [idx for (_, idx) in overall_sorted]
            for idx in overall_indices:
                if idx not in selected_indices:
                    selected_indices.append(idx)
                    if len(selected_indices) == total_to_select:
                        break
        
        condensed_samples = [dataset[i] for i in selected_indices]
        return condensed_samples

def condense_dataset3(model, dataset, entropy_func='predictive_entropy', rate=1):
    """
    Condenses the dataset by selecting samples in a balanced manner over the different labels.
    For each label, roughly the same number of samples are chosen based on their uncertainty (entropy)
    with higher entropy indicating samples closer to the decision boundary.
    
    Args:
        model (torch.nn.Module): Pre-trained model to make predictions.
        dataset (torch.utils.data.Dataset): Dataset from which to select samples.
        entropy_func (str): Either 'predictive_entropy' or 'log_percentage_entropy' to use for uncertainty estimation.
        rate (float): Fraction of the dataset to drop (e.g., rate=0.2 keeps 80%).
    
    Returns:
        list: Selected samples, balanced over the different labels.
    """
    # If no condensation is required, return the whole dataset.
    if (1 - rate) == 1:
        return dataset
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        all_entropies = []  # store entropy for each sample
        all_indices   = []  # store dataset indices
        all_labels    = []  # store corresponding labels
        
        # Compute uncertainty for each sample.
        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(dataset):
                inputs = inputs.to(device).unsqueeze(0)  # add batch dimension
                logits = model(inputs)
                
                if entropy_func == 'predictive_entropy':
                    entropy = predictive_entropy(logits)
                elif entropy_func == 'log_percentage_entropy':
                    entropy = log_percentage_entropy(logits)
                else:
                    raise ValueError(f"Invalid entropy function: {entropy_func}")
                    
                all_entropies.append(entropy.item())
                all_indices.append(idx)
                # Ensure labels is a standard number.
                all_labels.append(labels.item() if hasattr(labels, 'item') else labels)
        
        # Determine the overall number of samples to select.
        total_to_select = int(len(dataset) * (1 - rate))
        # Find out which unique labels exist.
        unique_labels = list(set(all_labels))
        num_classes = len(unique_labels)
        # Calculate the number of samples to select per class.
        per_class_select = total_to_select // num_classes
        
        # Group the samples by label.
        label_groups = {label: [] for label in unique_labels}
        for entropy, idx, label in zip(all_entropies, all_indices, all_labels):
            label_groups[label].append((entropy, idx))
        
        # For each label, sort samples by entropy (descending) and pick the top ones.
        selected_indices = []
        for label, group in label_groups.items():
            sorted_group = sorted(group, key=lambda x: x[0], reverse=True)
            # If a group has fewer than 'per_class_select' samples, take all available.
            selected_indices.extend([idx for (_, idx) in sorted_group[:math.floor(per_class_select/2)]])
            selected_indices.extend([idx for (_, idx) in sorted_group[-math.ceil(per_class_select/2):]])
        
        # If due to rounding the total is less than required, fill the remainder from the overall ranking.
        if len(selected_indices) < total_to_select:
            overall_sorted = sorted(zip(all_entropies, all_indices), key=lambda x: x[0], reverse=True)
            overall_indices = [idx for (_, idx) in overall_sorted]
            for idx in overall_indices:
                if idx not in selected_indices:
                    selected_indices.append(idx)
                    if len(selected_indices) == total_to_select:
                        break
        
        condensed_samples = [dataset[i] for i in selected_indices]
        return condensed_samples
    

def condense_dataset4(model, dataset, entropy_func='predictive_entropy',rate=1):
    """
    Condenses the dataset by selecting the top N samples with the highest uncertainty (entropy).
    
    Args:
        model (torch.nn.Module): Pre-trained model to make predictions.
        dataset (torch.utils.data.Dataset): Dataset from which to select samples.
        entropy_func (str): Either 'predictive_entropy' or 'log_percentage_entropy' to use for uncertainty estimation.
        top_n (int): The number of most uncertain samples to select for condensation.
    
    Returns:
        condensed_samples (list): List of the top N most uncertain samples.
    """
    if (1-rate)==1:
        return dataset
    else:
        # Move model to the appropriate device (GPU or CPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()  # Set the model to evaluation mode
        all_entropies = []  # To store entropy values
        all_indices = []  # To store indices of the samples

        # Iterate through the dataset
        with torch.no_grad():  # No need to compute gradients for this operation
            for idx, (inputs, labels) in enumerate(dataset):
                inputs, labels = inputs.to(device), labels
                inputs = inputs.unsqueeze(0)  # Add batch dimension (since dataset is typically single-sample)
                
                # Get the logits from the model (no softmax here, we're dealing with logits)
                logits = model(inputs)
                
                # Compute entropy based on the chosen function
                if entropy_func == 'predictive_entropy':
                    entropy = predictive_entropy(logits)
                elif entropy_func == 'log_percentage_entropy':
                    entropy = log_percentage_entropy(logits)
                else:
                    raise ValueError(f"Invalid entropy function: {entropy_func}")
                
                all_entropies.append(entropy.item())  # Store entropy value
                all_indices.append(idx)  # Store the index of the sample

        # Sort the samples by entropy in descending order (most uncertain samples first)
        sorted_indices = sorted(zip(all_entropies, all_indices), reverse=True, key=lambda x: x[0])
        # Step 3: Divide dataset into 100 groups while keeping order
        num_groups = 100
        group_size = len(dataset) // num_groups
        grouped_indices = [sorted_indices[i * group_size: (i + 1) * group_size] for i in range(num_groups)]

        # Step 4: Randomly remove `rate` fraction from each group
        final_selected_indices = []
        for group in grouped_indices:
            num_to_keep = int(len(group) * (1 - rate))
            sampled_indices = random.sample(group, num_to_keep) if num_to_keep < len(group) else group
            final_selected_indices.extend([idx for (_, idx) in sampled_indices])

        # Retrieve the actual samples from the dataset
        condensed_samples = [dataset[i] for i in final_selected_indices]
        return condensed_samples

def condense_dataset_boundary(model, dataset, rate=1, alpha=0.01, max_steps=10, criterion=torch.nn.CrossEntropyLoss()):
    """
    Boundary-aware coreset selection.
    
    For each sample (x, y) in the dataset, the distance-to-boundary is estimated by iteratively
    perturbing x (using FGSM-style updates) until the model's prediction changes.
    
    Then, samples are grouped by their computed distance. A coverage-centric sampling is then 
    performed over the groups so that the final coreset covers a range of distances.
    
    Args:
        model (torch.nn.Module): A trained classification model with a method f(x) that returns logits.
        dataset (torch.utils.data.Dataset): Dataset providing (x, y) pairs.
        rate (float): Fraction of the dataset to drop (e.g., rate=0.2 keeps 80%).
        alpha (float): Step size for the perturbation.
        max_steps (int): Maximum number of perturbation steps per sample.
        criterion (torch.nn.Module): Loss function (e.g., CrossEntropyLoss).
    
    Returns:
        condensed_samples (list): A coreset of the dataset.
    """
    # If no condensation is required, return the full dataset.
    if (1 - rate) == 1:
        return dataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    distances = []    # Will hold the estimated distance for each sample.
    indices_list = [] # Corresponding dataset indices.

    # Estimate the distance-to-boundary for each sample.
    for idx, (inputs, label) in enumerate(dataset):
        # Move inputs and label to device.
        inputs = inputs.to(device)
        # Ensure label is a tensor of shape (1,) for loss computation.
        label_tensor = torch.tensor([label]).to(device)
        
        # Add batch dimension if needed.
        if inputs.dim() == 3:
            x_k = inputs.unsqueeze(0)
        else:
            x_k = inputs
        
        # Initialize distance counter.
        distance = 0

        # Check if the sample is already misclassified.
        logits = model(x_k)
        pred = torch.argmax(logits, dim=1)
        if pred.item() != label:
            distances.append(0)
            indices_list.append(idx)
            continue

        # Iteratively perturb the sample.
        for k in range(max_steps):
            # Enable gradient computation for the current input.
            x_k.requires_grad = True
            logits = model(x_k)
            loss = criterion(logits, label_tensor)
            model.zero_grad()
            loss.backward()
            grad = x_k.grad

            # FGSM-style update: move in the direction that increases the loss.
            x_k = x_k + alpha * torch.sign(grad)
            x_k = x_k.detach()  # Detach from the current computation graph.

            # Check the model prediction after the update.
            logits = model(x_k)
            pred = torch.argmax(logits, dim=1)
            distance = k + 1
            if pred.item() != label:
                # Stop when misclassification is reached.
                break

        distances.append(distance)
        indices_list.append(idx)
    
    # Group sample indices by their distance values.
    groups = {}
    for d, idx in zip(distances, indices_list):
        groups.setdefault(d, []).append(idx)
    
    # Determine total number of samples to select.
    total_to_select = int(len(dataset) * (1 - rate))
    remaining_budget = total_to_select
    selected_indices = []

    # Coverage-centric sampling: allocate a portion of the budget for each distance group.
    # We iteratively select from the group with the fewest samples.
    while groups and remaining_budget > 0:
        num_groups = len(groups)
        # Find the group with the fewest examples.
        group_key = min(groups, key=lambda k: len(groups[k]))
        group_samples = groups[group_key]
        # Allocate budget for this group: at least one, but no more than group size.
        m_D = min(len(group_samples), remaining_budget // num_groups)
        if m_D == 0:
            m_D = 1  # Ensure we pick at least one sample if budget remains.
        # Randomly sample m_D examples from this group.
        sampled = random.sample(group_samples, m_D) if m_D < len(group_samples) else group_samples
        selected_indices.extend(sampled)
        # Remove the group from further consideration.
        del groups[group_key]
        remaining_budget -= m_D

    # Retrieve the actual samples from the dataset.
    condensed_samples = [dataset[i] for i in selected_indices]
    return condensed_samples
