# file2.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def compute_total_evidence_batched(evidence: torch.Tensor) -> torch.Tensor:
    """
    Computes total evidence for each sample in the batch.
    
    Args:
        evidence (torch.Tensor): Raw model output evidence (shape: [batch_size, num_classes])
    
    Returns:
        torch.Tensor: Total evidence per sample (shape: [batch_size])
    """
    return torch.sum(evidence, dim=1)

def compute_total_uncertainty_batched(evidence: torch.Tensor) -> torch.Tensor:
    """
    Computes total epistemic uncertainty for each sample using the Dirichlet formulation.
    
    Args:
        evidence (torch.Tensor): Raw model output evidence (shape: [batch_size, num_classes])
    
    Returns:
        torch.Tensor: Uncertainty per sample (shape: [batch_size])
    """
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)  # shape: [batch_size, 1]
    uncertainty = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1)
    return uncertainty

def compute_label_evidence_batched(evidence: torch.Tensor) -> torch.Tensor:
    """
    Computes the evidence for the predicted label for each sample.
    
    Args:
        evidence (torch.Tensor): Raw model output evidence (shape: [batch_size, num_classes])
    
    Returns:
        torch.Tensor: Evidence for the predicted label (shape: [batch_size])
    """
    preds = torch.argmax(evidence, dim=1)
    batch_indices = torch.arange(evidence.size(0), device=evidence.device)
    return evidence[batch_indices, preds]

def compute_label_uncertainty_batched(evidence: torch.Tensor) -> torch.Tensor:
    """
    Computes the epistemic uncertainty for the predicted label for each sample.
    
    Args:
        evidence (torch.Tensor): Raw model output evidence (shape: [batch_size, num_classes])
    
    Returns:
        torch.Tensor: Label uncertainty per sample (shape: [batch_size])
    """
    alpha = evidence + 1
    preds = torch.argmax(alpha, dim=1)
    batch_indices = torch.arange(alpha.size(0), device=alpha.device)
    alpha_hat = alpha[batch_indices, preds]
    S = torch.sum(alpha, dim=1)
    uncertainty = alpha_hat * (S - alpha_hat) / (S * S * (S + 1))
    return uncertainty

def get_scores_batched(model, dataset, func, batch_size=128):
    """
    Computes and sorts scores for each sample based on a batched function.
    
    Args:
        model (torch.nn.Module): Trained evidential model.
        dataset (torch.utils.data.Dataset): Dataset to evaluate.
        func (callable): Function that takes batched evidence and returns a tensor of scores.
        batch_size (int): Batch size for processing.
    
    Returns:
        list of tuples: (score, sample_index) sorted in descending order.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    scores = []
    index_offset = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            evidence = model(inputs)
            # Apply softplus to ensure positive evidence
            evidence1 = F.softplus(evidence)
            # func now processes the full batch at once
            batch_scores = func(evidence1).tolist()
            batch_indices = list(range(index_offset, index_offset + len(batch_scores)))
            scores.extend(zip(batch_scores, batch_indices))
            index_offset += len(batch_scores)
    return sorted(scores, key=lambda x: x[0], reverse=True)

def sort_by_total_evidence(model, dataset, batch_size=128):
    return get_scores_batched(model, dataset, compute_total_evidence_batched, batch_size)

def sort_by_total_uncertainty(model, dataset, batch_size=128):
    return get_scores_batched(model, dataset, compute_total_uncertainty_batched, batch_size)

def sort_by_label_evidence(model, dataset, batch_size=128):
    return get_scores_batched(model, dataset, compute_label_evidence_batched, batch_size)

def sort_by_label_uncertainty(model, dataset, batch_size=128):
    return get_scores_batched(model, dataset, compute_label_uncertainty_batched, batch_size)
