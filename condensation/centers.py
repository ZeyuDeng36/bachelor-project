
def compute_class_centers(representations, labels, num_classes):
    """
    Compute the class center for each class as the mean of its representations.
    
    Args:
        representations (torch.Tensor): Tensor of shape (N, rep_dim).
        labels (torch.Tensor): Tensor of shape (N,).
        num_classes (int): Total number of classes.
        
    Returns:
        class_centers (list of torch.Tensor): List of length num_classes with center for each class.
    """
    class_centers = []
    for c in range(num_classes):
        idxs = (labels == c).nonzero(as_tuple=True)[0]
        center = representations[idxs].mean(dim=0)
        class_centers.append(center)
    return class_centers

def compute_distances(representations, labels, class_centers):
    """
    Compute the Euclidean distance from each sample's representation to its class center.
    
    Args:
        representations (torch.Tensor): Tensor of shape (N, rep_dim).
        labels (torch.Tensor): Tensor of shape (N,).
        class_centers (list of torch.Tensor): Precomputed centers for each class.
        
    Returns:
        distances (list of float): Euclidean distances for each sample.
    """
    distances = []
    for i, rep in enumerate(representations):
        label = labels[i].item()
        center = class_centers[label]
        distance = torch.norm(rep - center, p=2).item()
        distances.append(distance)
    return distances