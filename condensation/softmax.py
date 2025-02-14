import torch
import torch.nn.functional as F

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

        # Select top N uncertain samples
        top_n_indices = [idx for _, idx in sorted_indices[:int(len(dataset)*(1-rate))]]
        condensed_samples = [dataset[i] for i in top_n_indices]  # Get the actual samples
        
        return condensed_samples

