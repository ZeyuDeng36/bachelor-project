import random

def random_selection(dataset, rate, scores=None):
    """
    Randomly selects a fraction of the dataset.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to sample from.
        rate (float): Fraction of samples to keep (1.0 means keep all).
        scores (None): Unused, but included for consistent function calls.

    Returns:
        list: A subset of the dataset.
    """
    if rate >= 1.0:
        return dataset
    num_to_keep = int(len(dataset) * rate)
    selected_indices = random.sample(range(len(dataset)), num_to_keep)
    return [dataset[i] for i in selected_indices]

def balanced_by_score(dataset, rate, scores, num_groups=100):
    """
    Selects a fraction of the dataset while ensuring balance across score groups.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to sample from.
        rate (float): Fraction of samples to keep.
        scores (list of tuples): List of (score, index) tuples.
        num_groups (int): Number of groups to divide the dataset into.

    Returns:
        list: A subset of the dataset.
    """
    if rate >= 1.0:
        return dataset
    total = len(scores)
    group_size = max(1, total // num_groups)
    groups = [scores[i * group_size: (i + 1) * group_size] for i in range(num_groups)]
    
    selected_indices = []
    for group in groups:
        num_to_keep = int(len(group) * rate)
        sampled = random.sample(group, num_to_keep) if num_to_keep < len(group) else group
        selected_indices.extend(idx for _, idx in sampled)
    
    return [dataset[i] for i in selected_indices]

def balanced_by_label(dataset, rate, scores):
    """
    Selects a fraction of the dataset while ensuring balance across labels.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to sample from.
        rate (float): Fraction of samples to keep.
        scores (list of tuples): List of (score, index) tuples.

    Returns:
        list: A subset of the dataset.
    """
    if rate >= 1.0:
        return dataset
    total_to_select = int(len(dataset) * rate)
    
    # Group indices by label
    label_groups = {}
    for score, idx in scores:
        _, label = dataset[idx]
        label = label.item() if hasattr(label, 'item') else label
        label_groups.setdefault(label, []).append((score, idx))
    
    num_classes = len(label_groups)
    per_class = total_to_select // num_classes
    selected_indices = []

    for label, group in label_groups.items():
        group_sorted = sorted(group, key=lambda x: x[0], reverse=True)
        selected_indices.extend(idx for _, idx in group_sorted[:per_class])

    # Fill remaining slots if rounding left out some samples
    if len(selected_indices) < total_to_select:
        overall = [idx for _, idx in scores if idx not in selected_indices]
        selected_indices.extend(overall[:(total_to_select - len(selected_indices))])

    return [dataset[i] for i in selected_indices]

def select_top(dataset, rate, scores):
    """
    Selects the top fraction of samples based on their scores.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to sample from.
        rate (float): Fraction of samples to keep.
        scores (list of tuples): List of (score, index) tuples.

    Returns:
        list: A subset of the dataset.
    """
    if rate >= 1.0:
        return dataset
    num_to_keep = int(len(scores) * rate)
    selected_indices = [idx for _, idx in scores[:num_to_keep]]
    return [dataset[i] for i in selected_indices]
