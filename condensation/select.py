from collections import defaultdict
import math
import random
import time
import numpy as np
def random_selection(dataset, rate, scores=None):
    """
    Randomly selects a fraction of the dataset.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to sample from.
        rate (float): Fraction of samples to keep (1.0 means keep all).
        scores (None): Unused, but included for consistent function calls.

    Returns:
        list: selected indices of a subset of the dataset.
    """
    if rate >= 1.0:
        return [idx for _, idx in scores]
    num_to_keep = int(len(dataset) * rate)
    selected_indices = random.sample(range(len(dataset)), num_to_keep)
    return selected_indices

def balanced_by_score(dataset, rate, scores, num_groups=100):
    """
    Selects a fraction of the dataset while ensuring balance across score groups.
    This version prints the execution time of the function.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to sample from.
        rate (float): Fraction of samples to keep.
        scores (list of tuples): List of (score, index) tuples.
        num_groups (int): Number of groups to divide the dataset into.

    Returns:
        list: selected indices of a subset of the dataset.
    """

    if rate >= 1.0:
        print("Execution time: 0.0 seconds")
        return [idx for _, idx in scores]  # If no selection, return immediately with 0 time

    total = len(scores)
    group_size = max(1, total // num_groups)
    groups = [scores[i * group_size: (i + 1) * group_size] for i in range(num_groups)]
    
    selected_indices = []
    for group in groups:
        num_to_keep = int(len(group) * rate)
        sampled = random.sample(group, num_to_keep) if num_to_keep < len(group) else group
        selected_indices.extend(idx for _, idx in sampled)
    
    return selected_indices


def balanced_by_score1(dataset, rate, scores, num_groups=100):
    """
    Selects a fraction of the dataset while ensuring balance across score groups
    using quantile binning.
    
    Args:
        dataset (torch.utils.data.Dataset): Dataset to sample from.
        rate (float): Fraction of samples to keep.
        scores (list of tuples): List of (score, index) tuples.
        num_groups (int): Number of groups to divide the dataset into.
    
    Returns:
        list: selected indices of a subset of the dataset.
    """
    if rate >= 1.0:
        #print("Execution time: 0.0 seconds")
        return [idx for _, idx in scores]

    #start_time = time.time()
    #print(scores[:5]) 
    # Convert scores to NumPy array: shape (N, 2)
    scores_array = np.array(scores)
    total = scores_array.shape[0]

    # Split the array into `num_groups` groups (each group is roughly equal-sized)
    groups = np.array_split(scores_array, num_groups)

    selected_indices = []
    # Loop over groups (only ~100 iterations)
    for group in groups:
        group_len = group.shape[0]
        num_to_keep = max(1, int(group_len * rate))
        if group_len > num_to_keep:
            # Randomly select indices from the group
            chosen = np.random.choice(group_len, num_to_keep, replace=False)
            selected_indices.extend(group[chosen, 1].astype(int).tolist())
        else:
            selected_indices.extend(group[:, 1].astype(int).tolist())

    #elapsed_time = time.time() - start_time
    #print(f"Execution time: {elapsed_time:.4f} seconds")
    return selected_indices

def balanced_by_range(dataset, rate, scores, num_bins=100):
    """
    Selects a fraction of the dataset while ensuring balance across uncertainty intervals.
    The method computes the minimum and maximum uncertainty scores, divides the range into
    `num_bins` equal-width intervals, and then samples n = (total/num_bins * rate) datapoints 
    from each interval.
    
    Args:
        dataset (torch.utils.data.Dataset): Dataset to sample from.
        rate (float): Fraction of samples to keep overall.
        scores (list of tuples): List of (score, index) tuples, where score is the uncertainty.
        num_bins (int): Number of intervals to divide the uncertainty range into.
    
    Returns:
        list: selected indices of a subset of the dataset.
    """
    if rate >= 1.0:
        print("Execution time: 0.0 seconds")
        return [idx for _, idx in scores]

    start_time = time.time()

    # Convert scores to a NumPy array of shape (N, 2)
    scores_array = np.array(scores)
    total = scores_array.shape[0]
    
    # Determine the number of samples to select per bin (ensuring overall rate)
    n_per_bin = max(1, int((total / num_bins) * rate))
    
    # Extract uncertainty values and indices
    score_values = scores_array[:, 0]
    indices_all = scores_array[:, 1].astype(int)

    # Compute min, max, and interval width of the uncertainty scores
    min_score = score_values.min()
    max_score = score_values.max()
    interval_width = (max_score - min_score) / num_bins if num_bins > 0 else 0

    # Vectorized assignment of bin indices
    if interval_width > 0:
        bin_indices = np.floor((score_values - min_score) / interval_width).astype(int)
        # Ensure scores equal to max_score fall into the last bin
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)
    else:
        bin_indices = np.zeros_like(score_values, dtype=int)

    selected_indices = []
    # Loop over each bin (only 100 iterations)
    for b in range(num_bins):
        # Get indices of datapoints in bin b using vectorized filtering
        bin_mask = (bin_indices == b)
        bin_group = indices_all[bin_mask]
        bin_count = bin_group.shape[0]
        if bin_count > 0:
            if bin_count > n_per_bin:
                chosen = np.random.choice(bin_group, n_per_bin, replace=False)
            else:
                chosen = bin_group
            selected_indices.extend(chosen.tolist())

    elapsed_time = time.time() - start_time
    print(f"Execution time: {elapsed_time:.4f} seconds")
    return selected_indices

def balanced_by_label(dataset, rate, scores, num_groups=100):
    """
    Selects a fraction of the dataset while ensuring balance across labels,
    using quantile‐based (score1) sampling within each label.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to sample from.
        rate (float): Fraction of samples to keep.
        scores (list of tuples): List of (score, index) tuples.
        num_groups (int): Number of quantile groups per label.

    Returns:
        list: selected indices of a subset of the dataset.
    """
    if rate >= 1.0:
        return [idx for _, idx in scores]

    total = len(scores)
    total_to_select = int(total * rate)

    # 1) Group indices by label
    label_groups = {}
    for score, idx in scores:
        _, label = dataset[idx]
        label = label.item() if hasattr(label, 'item') else label
        label_groups.setdefault(label, []).append((score, idx))

    num_classes = len(label_groups)
    per_class = total_to_select // num_classes

    selected_indices = []
    for label, group in label_groups.items():
        # Convert group to numpy array for quantile splitting
        arr = np.array(group, dtype=float)  # shape (N,2): [score, idx]
        N = arr.shape[0]

        # Compute the per-label keep-rate so that
        # len(sel) ≈ per_class
        class_rate = per_class / N

        # Split into quantile bins
        subgroups = np.array_split(arr, min(num_groups, N))
        class_sel = []

        # Within each bin, pick roughly class_rate fraction
        for sub in subgroups:
            k = max(1, int(len(sub) * class_rate))
            if len(sub) > k:
                # sample without replacement
                chosen = np.random.choice(sub[:,1].astype(int), k, replace=False)
            else:
                chosen = sub[:,1].astype(int)
            class_sel.extend(chosen.tolist())

        # If we under‐shot (due to rounding), fill from top scores
        if len(class_sel) < per_class:
            remaining = sorted(
                [(s, i) for s, i in group if i not in class_sel],
                key=lambda x: x[0], reverse=True
            )
            needed = per_class - len(class_sel)
            class_sel.extend(i for _, i in remaining[:needed])

        # If we over‐shot (rare), truncate
        selected_indices.extend(class_sel[:per_class])

    # If we still have slots left (due to rounding down), fill from overall top
    if len(selected_indices) < total_to_select:
        used = set(selected_indices)
        remaining = [idx for _, idx in scores if idx not in used]
        needed = total_to_select - len(selected_indices)
        selected_indices.extend(remaining[:needed])

    return selected_indices

def select_top(dataset, rate, scores):
    """
    Selects the top fraction of samples based on their scores.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to sample from.
        rate (float): Fraction of samples to keep.
        scores (list of tuples): List of (score, index) tuples.

    Returns:
        list: selected indices of a subset of the dataset.
    """
    if rate >= 1.0:
        return [idx for _, idx in scores]
    num_to_keep = int(len(scores) * rate)
    selected_indices = [idx for _, idx in scores[:num_to_keep]]
    return selected_indices

def select_bottom(dataset, rate, scores):
    """
    Selects the bottom fraction of samples based on their scores.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to sample from.
        rate (float): Fraction of samples to keep.
        scores (list of tuples): List of (score, index) tuples, assumed sorted in descending order.

    Returns:
        list: A subset of the dataset containing the bottom fraction of samples.
    """
    if rate >= 1.0:
        return [idx for _, idx in scores]
    num_to_keep = int(len(scores) * rate)
    # Select the last num_to_keep items from the scores list
    selected_indices = [idx for _, idx in scores[-num_to_keep:]]
    return selected_indices

def select_median_centered(dataset, rate, scores):
    """
    Selects a fraction of the dataset by taking a contiguous block of samples 
    centered around the median score. This method simply divides the number 
    of samples to take by two and takes that many from each side of the median.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to sample from.
        rate (float): Fraction of samples to keep.
        scores (list of tuples): List of (score, index) tuples, assumed to be sorted 
                                 (either ascending or descending).

    Returns:
        list: A subset of the dataset containing the samples around the median score.
    """
    if rate >= 1.0:
        return [idx for _, idx in scores]

    total = len(scores)
    num_to_keep = int(total * rate)
    if num_to_keep < 1:
        num_to_keep = 1

    # Find the median index
    median_index = total // 2

    # Compute how many to take from each side.
    left_count = num_to_keep // 2
    right_count = num_to_keep - left_count

    # Make sure we don't exceed the boundaries of the scores list.
    start_index = max(0, median_index - left_count)
    end_index = min(total, median_index + right_count)

    # The selected block of indices (from the sorted list)
    selected_indices = [scores[i][1] for i in range(start_index, end_index)]
    return selected_indices

def select_least_bin(scores, eta, seed=None):
    """
    Implements the "Coverage‐Centric Sampling" (least‐bin) procedure:
      1) Partition all examples into bins by their integer distance.
      2) Repeat:
         a) Find the bin with smallest cardinality.
         b) From that bin, select min( bin_size, floor(q / (#bins_remain)) ) examples.
         c) Remove that bin, subtract from q, and add chosen indices to S.
      3) Stop when either q ≤ 0 or no bins remain.

    Args:
        scores (List[ (int distance, int idx) ]): 
            Each tuple holds (distance_i, index_i). Distances should be 
            nonnegative integers (0,1,…,k).
        eta (float): Fraction ∈ (0,1] specifying how large the coreset should be
                     relative to the full dataset size m = len(scores).
        seed (int, optional): Random seed for reproducibility.

    Returns:
        List[int]: The list of selected indices (the η‐coreset).
    """
    if seed is not None:
        random.seed(seed)
    
    # 1) Total number of examples:
    m = len(scores)
    if not (0 < eta <= 1.0):
        raise ValueError("eta must be in (0,1].")
    
    # 2) Compute how many we need in the end (q):
    q = int(math.floor(m * eta))
    if q <= 0:
        return []
    
    # 3) Build a dictionary mapping distance → list of indices
    D = defaultdict(list)
    for dist, idx in scores:
        D[dist].append(idx)
    
    # 4) Initialize the coreset list
    S_co = []
    
    # 5) Iterate until either no bins remain or q ≤ 0
    while D and q > 0:
        # 5a) Find the key 'dist_min' whose bin is smallest
        dist_min = min(D.keys(), key=lambda d: len(D[d]))
        bin_min = D[dist_min]
        size_min = len(bin_min)
        
        # 5b) Number of bins currently in D
        B = len(D)
        
        # 5c) How many to sample from this bin:
        avg_allotment = math.floor(q / B)
        m_D = min(size_min, avg_allotment)
        
        # 5d) Sample m_D indices from bin_min
        if m_D >= size_min:
            chosen = bin_min[:]  # take all if bin is small enough
        else:
            chosen = random.sample(bin_min, m_D)
        
        # 5e) Add them to S_co
        S_co.extend(chosen)
        
        # 5f) Remove this bin from D
        del D[dist_min]
        
        # 5g) Update remaining budget
        q -= m_D
    
    # 6) Return the indices we selected
    return S_co
