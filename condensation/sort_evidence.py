import torch
import torch.nn.functional as F
def compute_total_evidence(evidence: torch.Tensor) -> float:
    """
    Compute total evidence as the sum of evidence across all classes.
    
    Args:
        evidence (torch.Tensor): Raw model output evidence, shape [1, num_classes].
    
    Returns:
        float: Total evidence.
    """
    return torch.sum(evidence)


def compute_total_uncertainty(evidence: torch.Tensor) -> float:
    """
    Compute total epistemic uncertainty using the Dirichlet formulation.
    
    Converts evidence to Dirichlet parameters alpha (alpha = evidence + 1),
    then computes:
    
        uncertainty = sum[ alpha * (S - alpha) / (S^2 * (S + 1)) ]
    
    where S = sum(alpha).
    
    Args:
        evidence (torch.Tensor): Raw model output evidence, shape [1, num_classes].
    
    Returns:
        float: Total epistemic uncertainty.
    """
    alpha = evidence + 1
    S = torch.sum(alpha)  # Shape: [1, 1]
    uncertainty = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)))
    return uncertainty


def compute_label_evidence(evidence: torch.Tensor) -> float:
    """
    Compute the evidence for the predicted label.
    
    Args:
        evidence (torch.Tensor): Raw model output evidence, shape [1, num_classes].
    
    Returns:
        float: Evidence corresponding to the predicted label.
    """
    # Identify predicted label from raw evidence.
    predicted_label = torch.argmax(evidence, dim=1)
    return evidence[0, predicted_label].item()


def compute_label_uncertainty(evidence: torch.Tensor) -> float:
    """
    Compute the epistemic uncertainty for the predicted label.
    
    Converts evidence to Dirichlet parameters alpha (alpha = evidence + 1),
    then for the predicted label (alpha_hat), computes:
    
        label_uncertainty = alpha_hat * (S - alpha_hat) / (S^2 * (S + 1))
    
    where S = sum(alpha).
    
    Args:
        evidence (torch.Tensor): Raw model output evidence, shape [1, num_classes].
    
    Returns:
        float: Uncertainty corresponding to the predicted label.
    """
    alpha = evidence + 1
    # Identify predicted label using alpha.
    predicted_label = torch.argmax(alpha)
    alpha_hat = alpha[0, predicted_label]
    S = torch.sum(alpha)
    return alpha_hat * (S - alpha_hat) / (S * S * (S + 1))


def get_scores(model, dataset, func):
    """
    Computes and sorts scores for each sample in the dataset based on the given mode.
    
    Supported modes:
        "total_evidence"   : Total evidence across classes.
        "total_uncertainty": Dirichlet epistemic uncertainty over all classes.
        "label_evidence"   : Evidence for the predicted label.
        "label_uncertainty": Dirichlet uncertainty for the predicted label.
    
    Args:
        model (torch.nn.Module): Trained evidential model.
        dataset (torch.utils.data.Dataset): Dataset to evaluate.
        mode (str): Sorting criterion.
    
    Returns:
        list of tuples: (score, index) for each sample, sorted in descending order.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    scores = []
    
    with torch.no_grad():
        for i, (inputs,labels) in enumerate(dataset):
            inputs = inputs.to(device).unsqueeze(0)  # Add batch dimension
            evidence = model(inputs)
            evidence1 = F.softplus(evidence)  # Raw evidence output from model¨
            #print(f"Name: {evidence}, Age: {evidence1}")
            score = func(evidence1)        
            scores.append((score, i))
    
    return sorted(scores, key=lambda x: x[0], reverse=True)

def sort_by_total_evidence(model, dataset):
    return get_scores(model, dataset, compute_total_evidence)

def sort_by_total_uncertainty(model, dataset):
    return get_scores(model, dataset, compute_total_uncertainty)

def sort_by_label_evidence(model, dataset):
    return get_scores(model, dataset, compute_label_evidence)

def sort_by_label_uncertainty(model, dataset):
    return get_scores(model, dataset, compute_label_uncertainty)

# Example usage:
# sorted_scores = get_scores(model, dataset, mode="total_uncertainty")
