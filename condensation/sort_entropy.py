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
def max_softmax(logits):
    """
    Computes the maximum softmax value (confidence score) for each example in the batch.
    
    Args:
        logits (torch.Tensor): Raw outputs from the model (shape: [batch_size, num_classes])
    
    Returns:
        max_softmax (torch.Tensor): Maximum softmax value (confidence score) for each example.
    """
    # Compute softmax probabilities
    probs = F.softmax(logits, dim=1)
    # Get the maximum probability (confidence score) for each example
    max_softmax_values = torch.max(probs, dim=1)[0]
    return max_softmax_values
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

def sort_by_entropy(model, dataset, entropy_func):
    """
    Computes and sorts samples by an entropy measure (descending).

    Args:
        model (torch.nn.Module): Model used for predictions.
        dataset (torch.utils.data.Dataset): Dataset to evaluate.
        entropy_func (callable): Function that takes logits and returns an entropy tensor.
        
    Returns:
        list of tuples: Each tuple is (entropy_value, sample_index) sorted descending.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    scores = []
    
    with torch.no_grad():
        for idx, (inputs, _) in enumerate(dataset):
            inputs = inputs.to(device)
            if inputs.dim() == 3:  # if sample is (C, H, W), add batch dimension.
                inputs = inputs.unsqueeze(0)
            logits = model(inputs)
            ent = entropy_func(logits).item()
            scores.append((ent, idx))
    
    sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)
    return sorted_scores

def sort_by_predictive_entropy(model, dataset):
    return sort_by_entropy(model, dataset, predictive_entropy)

def sort_by_log_percentage_entropy(model, dataset):
    return sort_by_entropy(model, dataset, log_percentage_entropy)

def sort_by_max_softmax(model, dataset):
    return sort_by_entropy(model, dataset, max_softmax)
