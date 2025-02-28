from utils.initiate import initiate_dataset, initiate_model
from utils.trainer_Evidence import Trainer


if __name__ == '__main__':    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    from torchvision.models import resnet18
    from utils.repeater import repeat,repeat1
    from utils.trainer_Evidence import Trainer as TrainerE
    #model,(trainset, testset) = initiate_model_and_dataset("resnet18","cifar10")
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
    #trainer2.train(verbose=True)
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
    """
    repeat1("MNIST","resnet18",[0,0.3,0.5,0.7,0.8,0.9,0.95],"evidence_label","balanced_by_score","models/resnet18_MNIST_Evidence")
    repeat1("MNIST","resnet18",[0,0.3,0.5,0.7,0.8,0.9,0.95],"evidence_total","balanced_by_score","models/resnet18_MNIST_Evidence")
    repeat1("MNIST","resnet18",[0,0.3,0.5,0.7,0.8,0.9,0.95],"uncertainty_label","balanced_by_score","models/resnet18_MNIST_Evidence")
    repeat1("MNIST","resnet18",[0,0.3,0.5,0.7,0.8,0.9,0.95],"uncertainty_total","balanced_by_score","models/resnet18_MNIST_Evidence")