import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from utils.run_experiments import run_experiments,run_experiments1,run_experiments2,run_experiments3
from utils.trainer_Evidence import Trainer as TrainerE
from utils.trainer_Evidence_kl import Trainer as TrainerE1
from utils.initiate import initiate_dataset, initiate_model
from utils.trainer_standard import Trainer as TrainerS
def run_experiment1():
    models = ["resnet18"]
    datasets = ["MNIST"]
    rates = [0.95, 0.5, 0.7, 0.8, 0.9, 0.95]
    uncertainty_metrics = ["predictive_entropy"]
    selection_methods = ["top","bottom","balanced_by_label","median","balanced_by_score1"]
    for model_name in models:
        for dataset_name in datasets:
            for suffix in [1,2,3]:
                # Second training with different save path
                model = initiate_model(model_name, dataset_name)
                trainset, testset = initiate_dataset(dataset_name, model_name)
                trainer = TrainerS(
                    model,
                    trainset, 
                    testset,   
                    save=f"{model_name}_{dataset_name}_{suffix}"
                )
                trainer.train(verbose=True)

                run_experiments(dataset_name, model_name, rates, None, "random", f"models/{model_name}_{dataset_name}_{suffix}",suffix)
                for uncertainty_metric in uncertainty_metrics:
                    for selection_method in selection_methods:
                        run_experiments(dataset_name, model_name, rates, uncertainty_metric, selection_method, f"models/{model_name}_{dataset_name}_{suffix}",suffix)
def run_experiment2():
    models = ["resnet18"]
    datasets = ["MNIST"]
    rates = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
    uncertainty_metrics = ["EVIDENCE"]
    selection_methods = ["top","bottom","balanced_by_label","median","balanced_by_score1"]
    for model_name in models:
        for dataset_name in datasets:
            for suffix in [1,2,3]:
                # Second training with different save path
                model = initiate_model(model_name, dataset_name)
                trainset, testset = initiate_dataset(dataset_name, model_name)
                trainer = TrainerE1(
                    model,
                    trainset, 
                    testset,   
                    save=f"{model_name}_{dataset_name}_E1_{suffix}"
                )
                trainer.train(verbose=True)

                run_experiments(dataset_name, model_name, rates, None, "random", f"models/{model_name}_{dataset_name}_E1_{suffix}",suffix)
                for uncertainty_metric in uncertainty_metrics:
                    for selection_method in selection_methods:
                        run_experiments(dataset_name, model_name, rates, uncertainty_metric, selection_method, f"models/{model_name}_{dataset_name}_E1_{suffix}",suffix)
def run_experiment3():
    models = ["resnet18"]
    datasets = ["CIFAR10"]
    rates = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
    uncertainty_metrics = ["predictive_entropy"]
    selection_methods = ["top","bottom","balanced_by_label","median","balanced_by_score1"]
    for model_name in models:
        for dataset_name in datasets:
            for suffix in [1,2,3]:
                # Second training with different save path
                model = initiate_model(model_name, dataset_name)
                trainset, testset = initiate_dataset(dataset_name, model_name)
                trainer = TrainerS(
                    model,
                    trainset, 
                    testset,   
                    save=f"{model_name}_{dataset_name}_{suffix}"
                )
                trainer.train(verbose=True)

                run_experiments(dataset_name, model_name, rates, None, "random", f"models/{model_name}_{dataset_name}_{suffix}",suffix)
                for uncertainty_metric in uncertainty_metrics:
                    for selection_method in selection_methods:
                        run_experiments(dataset_name, model_name, rates, uncertainty_metric, selection_method, f"models/{model_name}_{dataset_name}_{suffix}",suffix)
def run_experiment4():
    models = ["resnet18"]
    datasets = ["CIFAR10"]
    rates = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
    uncertainty_metrics = ["EVIDENCE"]
    selection_methods = ["top","bottom","balanced_by_label","median","balanced_by_score1"]
    for model_name in models:
        for dataset_name in datasets:
            for suffix in [1,2,3]:
                # Second training with different save path
                model = initiate_model(model_name, dataset_name)
                trainset, testset = initiate_dataset(dataset_name, model_name)
                trainer = TrainerE1(
                    model,
                    trainset, 
                    testset,   
                    save=f"{model_name}_{dataset_name}_E1_{suffix}"
                )
                trainer.train(verbose=True)

                run_experiments(dataset_name, model_name, rates, None, "random", f"models/{model_name}_{dataset_name}_E1_{suffix}",suffix)
                for uncertainty_metric in uncertainty_metrics:
                    for selection_method in selection_methods:
                        run_experiments(dataset_name, model_name, rates, uncertainty_metric, selection_method, f"models/{model_name}_{dataset_name}_E1_{suffix}",suffix)

if __name__ == '__main__':    
    #model,(trainset, testset) = initiate_model_and_dataset("resnet18","cifar10")
    #run_experiment3()
    #run_experiment4()
    run_experiment2()
    run_experiment1()