import random
def random_condense_dataset(dataset, rate=1):
    """
    Condenses the dataset by randomly selecting top N samples.
    
    Args:
        dataset (torch.utils.data.Dataset): Dataset from which to select samples.
        top_n (int): The number of random samples to select for condensation.
    
    Returns:
        condensed_samples (list): List of randomly selected samples.
    """
    # Randomly select top_n indices from the dataset
    length=len(dataset)
    random_indices = random.sample(range(length), int(length*(1-rate)))
    
    # Get the actual samples corresponding to the selected indices
    condensed_samples = [dataset[i] for i in random_indices]
    
    return condensed_samples

