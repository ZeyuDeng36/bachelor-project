import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from utils.repeater import repeat,repeat1,repeat2,repeat3
from utils.trainer_Evidence import Trainer as TrainerE
from utils.trainer_Evidence1 import Trainer as TrainerE1
from utils.initiate import initiate_dataset, initiate_model
from utils.trainer_standard import Trainer as TrainerS
def run_experiment1():
    models = ["resnet18"]
    datasets = ["MNIST"]
    rates = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
    uncertainty_metrics = ["predictive_entropy"]
    selection_methods = ["balanced_by_score","top","bottom"]
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

                repeat(dataset_name, model_name, rates, None, "random", f"models/{model_name}_{dataset_name}_{suffix}",suffix)
                for uncertainty_metric in uncertainty_metrics:
                    for selection_method in selection_methods:
                        repeat(dataset_name, model_name, rates, uncertainty_metric, selection_method, f"models/{model_name}_{dataset_name}_{suffix}",suffix)

def run_experiment2():
    models = ["resnet18"]
    datasets = ["CIFAR10"]
    rates = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
    uncertainty_metrics = ["EVIDENCE","BELIEF"]
    selection_methods = ["balanced_by_score"]
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

                repeat2(dataset_name, model_name, rates, None, "random", f"models/{model_name}_{dataset_name}_E1_{suffix}",suffix)
                for uncertainty_metric in uncertainty_metrics:
                    for selection_method in selection_methods:
                        repeat2(dataset_name, model_name, rates, uncertainty_metric, selection_method, f"models/{model_name}_{dataset_name}_E1_{suffix}",suffix)
def run_experiment3():
    models = ["resnet18"]
    datasets = ["CIFAR10"]
    rates = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
    uncertainty_metrics = ["GRADIENT"]
    selection_methods = ["balanced_by_score"]
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
                    save=f"{model_name}_{dataset_name}_E1_{suffix}"
                )
                trainer.train(verbose=True)

                repeat2(dataset_name, model_name, rates, None, "random", f"models/{model_name}_{dataset_name}_E1_{suffix}",suffix)
                for uncertainty_metric in uncertainty_metrics:
                    for selection_method in selection_methods:
                        repeat2(dataset_name, model_name, rates, uncertainty_metric, selection_method, f"models/{model_name}_{dataset_name}_E1_{suffix}",suffix)
def run_experiment4():
    models = ["resnet18"]
    datasets = ["CIFAR10"]
    rates = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
    uncertainty_metrics = ["DISTANCE"]
    selection_methods = ["median"]
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
                repeat3(dataset_name, model_name, rates, "random", f"models/{model_name}_{dataset_name}_E1_{suffix}", suffix)
                for selection_method in selection_methods:
                    repeat3(dataset_name, model_name, rates, selection_method, f"models/{model_name}_{dataset_name}_E1_{suffix}", suffix)

if __name__ == '__main__':    
    #model,(trainset, testset) = initiate_model_and_dataset("resnet18","cifar10")
    #run_experiment1()
    #run_experiment2()
    #run_experiment3()
    run_experiment4()
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # Assuming `train_loader` and `val_loader` are DataLoader objects for training and validation datasets
    
    # Train the model with optional parameters
    trainer = Trainer(
        model,
        trainset, 
        testset ,   
        save="resnet18_MNIST"
    )
    #trainer.train(verbose=True)
    trainset_randCond = random_condense_dataset(trainset,5000)
    trainer1 = Trainer(
        model,
        trainset_randCond, 
        testset ,   
        save="resnet18_MNIST_randCond_5000"
    )
    #trainer1.train(verbose=True)
    model1 = resnet18(weights=None) #"IMAGENET1K_V1")
    model1.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model1.fc.in_features
    model1.fc = nn.Linear(num_ftrs, 10)
    model1.load_state_dict(torch.load("models/resnet18_MNIST", weights_only=True))

    #trainset_softmaxCond = condense_dataset(model1,trainset, top_n=5000)

    #trainer2 = Trainer(
        #model,
        #trainset_softmaxCond, 
       # testset ,   
       # save="resnet18_MNIST_softmaxCond_5000"
   # )
    #trainer2.train(verbose=True)"
    """
    """
    model = initiate_model("resnet18", "MNIST")
    trainset, testset = initiate_dataset("MNIST", "resnet18")
    trainer = TrainerE(
        model,
        trainset, 
        testset ,   
        save="resnet18_MNIST_Evidence",
    )
    trainer.train(verbose=True)

    model1 = initiate_model("resnet18", "MNIST")
    trainset1, testset1 = initiate_dataset("MNIST", "resnet18")
    trainer1 = Trainer(
        model1,
        trainset1, 
        testset1 ,   
        save="resnet18_MNIST"
    )
    trainer1.train(verbose=True)
    repeat("MNIST","resnet18",[0,0.3,0.5,0.7,0.8,0.9,0.95],None,"random",None)
    repeat1("MNIST","resnet18",[0,0.3,0.5,0.7,0.8,0.9,0.95],None,"random",None)
    repeat("MNIST","resnet18",[0,0.3,0.5,0.7,0.8,0.9,0.95],"predictive_entropy","balanced_by_score","models/resnet18_MNIST")
    repeat1("MNIST","resnet18",[0,0.3,0.5,0.7,0.8,0.9,0.95],"evidence_label","balanced_by_score","models/resnet18_MNIST_Evidence")
    repeat1("MNIST","resnet18",[0,0.3,0.5,0.7,0.8,0.9,0.95],"evidence_total","balanced_by_score","models/resnet18_MNIST_Evidence")
    repeat1("MNIST","resnet18",[0,0.3,0.5,0.7,0.8,0.9,0.95],"uncertainty_label","balanced_by_score","models/resnet18_MNIST_Evidence")
    repeat1("MNIST","resnet18",[0,0.3,0.5,0.7,0.8,0.9,0.95],"uncertainty_total","balanced_by_score","models/resnet18_MNIST_Evidence")
    repeat("MNIST","resnet18",[0,0.3,0.5,0.7,0.8,0.9,0.95],"predictive_entropy","balanced_by_score","models/resnet18_MNIST")
    """

    
    """    
    repeat1("MNIST","resnet18",[0,0.3,0.5,0.7,0.8,0.9,0.95],"uncertainty_total","balanced_by_score","models/resnet18_MNIST_Evidence","")
    repeat1("MNIST","resnet18",[0,0.3,0.5,0.7,0.8,0.9,0.95],"uncertainty_total","balanced_by_label","models/resnet18_MNIST_Evidence","")
    repeat("MNIST","resnet18",[0,0.3,0.5,0.7,0.8,0.9,0.95],"uncertainty_total","balanced_by_score","models/resnet18_MNIST_Evidence","")
    repeat("MNIST","resnet18",[0,0.3,0.5,0.7,0.8,0.9,0.95],"uncertainty_total","balanced_by_score1","models/resnet18_MNIST_Evidence","")"

    """
