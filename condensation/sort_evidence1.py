import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
            labels = labels.to(device)
            evidence = model(inputs)
            # Apply softplus to ensure positive evidence
            evidence1 = F.softplus(evidence)
            # func now processes the full batch at once
            batch_scores = func(evidence1, labels).tolist()
            batch_indices = list(range(index_offset, index_offset + len(batch_scores)))
            scores.extend(zip(batch_scores, batch_indices))
            index_offset += len(batch_scores)
    return sorted(scores, key=lambda x: x[0], reverse=True)

def compute_label_belief_batched(evidence: torch.Tensor, labels) -> torch.Tensor:
    # Convert raw evidence to Dirichlet parameters
    alpha = evidence + 1.0
    # Total evidence per sample: S = sum(alpha) over classes
    S = torch.sum(alpha, dim=1, keepdim=True)  # shape: [batch_size, 1]
    # Determine the predicted label based on the highest alpha value.
    preds = torch.argmax(alpha, dim=1)
    batch_indices = torch.arange(alpha.size(0), device=alpha.device)
    # Extract the alpha corresponding to the predicted class
    alpha_hat = alpha[batch_indices, preds]
    # Compute the belief mass for the predicted label: (alpha_hat - 1) / S
    belief = (alpha_hat - 1) / S.squeeze(1)
    return belief

def sort_by_label_belief(model, dataset, batch_size=128):
    return get_scores_batched(model, dataset, compute_label_belief_batched, batch_size)
def compute_total_evidence_dirichlet(evidence: torch.Tensor, labels) -> torch.Tensor:
    # Convert raw evidence to Dirichlet parameters.
    alpha = evidence + 1.0
    # Total evidence S = sum_j alpha_j for each sample.
    num_classes = evidence.size(1)
    S = num_classes/torch.sum(alpha, dim=1)
    return S

def sort_by_total_evidence_dirichlet(model, dataset, batch_size=128):
    return get_scores_batched(model, dataset, compute_total_evidence_dirichlet, batch_size)

def compute_input_gradient_norm(model, dataset, batch_size=128):
    """
    Computes gradient norm (sensitivity) for each sample in the dataset.
    
    Args:
        model (torch.nn.Module): Trained model.
        dataset (torch.utils.data.Dataset): Dataset to evaluate.
        batch_size (int): Batch size for processing.
    
    Returns:
        list of tuples: (gradient_norm, sample_index) sorted in descending order.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    scores = []
    index_offset = 0
    
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs.requires_grad = True  # Enable gradient computation for inputs

        model.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()

        # Compute gradient norm for each sample in the batch
        gradient = inputs.grad  # shape: [batch_size, C, H, W] or similar
        gradient_norm = gradient.view(gradient.size(0), -1).norm(p=2, dim=1)  # L2 norm per sample

        batch_scores = gradient_norm.detach().cpu().tolist()
        batch_indices = list(range(index_offset, index_offset + len(batch_scores)))
        scores.extend(zip(batch_scores, batch_indices))
        index_offset += len(batch_scores)

        inputs.requires_grad = False  # Clean up

    return sorted(scores, key=lambda x: x[0], reverse=True)