import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from condensation.select import random_selection,balanced_by_label,balanced_by_score
from condensation.sort_entropy import sort_by_log_percentage_entropy,sort_by_predictive_entropy
from condensation.sort_evidence import sort_by_total_evidence,sort_by_label_evidence, sort_by_label_uncertainty,sort_by_total_uncertainty
from utils.trainer_standard import Trainer
from utils.trainer_Evidence import Trainer as TrainerE
from typing import List, Optional
from utils.initiate import initiate_dataset,initiate_model
from typing import List, Optional

# Assume these dictionaries and functions are defined elsewhere:
scoring_methods = {
    "log_percentage_entropy": sort_by_log_percentage_entropy,
    "predictive_entropy": sort_by_predictive_entropy,
    "evidence_total": sort_by_total_evidence,
    "evidence_label": sort_by_label_evidence,
    "uncertainty_total": sort_by_total_uncertainty,
    "uncertainty_label": sort_by_label_uncertainty,
}
selection_methods = {
    "random": random_selection,  # should be a function: selection_func(sorted_scores, keep_fraction)
    "balanced_by_label": balanced_by_label,
    "balanced_by_score": balanced_by_score
}

def repeat(
    dataset_name: str,
    model_name: str,
    rates: List[float],
    scoring_method: Optional[str],
    selection_method: str,
    pretrained: Optional[str],
):
    # Pick the scoring and selection functions from the dictionaries.
    score_func = scoring_methods[scoring_method] if scoring_method else None
    selection_func = selection_methods[selection_method]
    
    # Load pretrained model if provided (for scoring purposes).
    model1 = initiate_model(model_name, dataset_name, pretrained) if pretrained else None

    # Load the dataset (and test set) using your initiation function.
    trainset, testset = initiate_dataset(dataset_name, model_name)
 
    # For each condensation rate, create a new training model and condense the training set.
    for rate in rates:
        model = initiate_model(model_name, dataset_name)  # new untrained model for training
        sorted_scores=None
        if rate <= 0:
            trainset_condensed = trainset
        else:
            if score_func and model1:
                # Score the training set using the pretrained model.
                sorted_scores = score_func(model1, trainset)
            trainset_condensed = selection_func(trainset, 1 - rate,sorted_scores)
        
        parts = [model_name, dataset_name, scoring_method, selection_method, rate]
        filename = "-".join(str(p) for p in parts if p is not None)
        print(filename)
        trainer = Trainer(
            model,
            trainset_condensed,
            testset,
            save=filename
        )
        trainer.train(verbose=True)

def repeat1(
    dataset_name: str,
    model_name: str,
    rates: List[float],
    scoring_method: Optional[str],
    selection_method: str,
    pretrained: Optional[str],
):
    # Pick the scoring and selection functions from the dictionaries.
    score_func = scoring_methods[scoring_method] if scoring_method else None
    selection_func = selection_methods[selection_method]
    
    # Load pretrained model if provided (for scoring purposes).
    model1 = initiate_model(model_name, dataset_name, pretrained) if pretrained else None

    # Load the dataset (and test set) using your initiation function.
    trainset, testset = initiate_dataset(dataset_name, model_name)
 
    # For each condensation rate, create a new training model and condense the training set.
    for rate in rates:
        model = initiate_model(model_name, dataset_name)  # new untrained model for training
        sorted_scores=None
        if rate <= 0:
            trainset_condensed = trainset
        else:
            if score_func and model1:
                # Score the training set using the pretrained model.
                sorted_scores = score_func(model1, trainset)
            trainset_condensed = selection_func(trainset,1 - rate,sorted_scores)
        modelStr = model_name+"-EvidenceLoss"
        parts = [modelStr, dataset_name, scoring_method, selection_method, rate]
        filename = "-".join(str(p) for p in parts if p is not None)
        trainer = TrainerE(
            model,
            trainset_condensed,
            testset,
            save=filename
        )
        trainer.train(verbose=True)