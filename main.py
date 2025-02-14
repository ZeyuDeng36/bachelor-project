if __name__ == '__main__':    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    from torchvision.models import resnet18
    from condensation.random import random_condense_dataset
    from condensation.softmax import condense_dataset
    from utils.trainer_standard import Trainer
    from utils.repeater import repeat
    from utils.repeater import repeat1

    # Define transforms: resize MNIST images to 32x32 (to match ResNet input size) 
    # and normalize using MNIST statistics.
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    # Load MNIST training and test datasets.
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Load a ResNet-18 model and modify it for 1-channel input and 10 output classes.
    model = resnet18(weights=None) #"IMAGENET1K_V1")
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)

    """
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
    repeat1([0,0.3,0.5,0.7,0.8,0.9,0.95])
