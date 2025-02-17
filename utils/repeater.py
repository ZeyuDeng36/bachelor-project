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
from condensation.random import random_condense_dataset
from condensation.softmax import condense_dataset, condense_dataset1,condense_dataset1,condense_dataset2,condense_dataset3,condense_dataset4
from utils.trainer_standard import Trainer

def repeat(rates):

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    for rate in rates:
        model = resnet18(weights=None) #"IMAGENET1K_V1")
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)
        trainset_softmaxCond = random_condense_dataset(trainset, rate)
        trainer2 = Trainer(
            model,
            trainset_softmaxCond, 
            testset ,   
            save=f"resnet18_MNIST_randCond_{rate}"
        )
        trainer2.train(verbose=True)
    return


def repeat1(rates):
    model1 = resnet18(weights=None) #"IMAGENET1K_V1")
    model1.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model1.fc.in_features
    model1.fc = nn.Linear(num_ftrs, 10)
    model1.load_state_dict(torch.load("models/resnet18_MNIST", weights_only=True))

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    for rate in rates:
        model = resnet18(weights=None) #"IMAGENET1K_V1")
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)
        trainset_softmaxCond = condense_dataset(model1,trainset,'predictive_entropy', rate)
        trainer2 = Trainer(
            model,
            trainset_softmaxCond, 
            testset ,   
            save=f"resnet18_MNIST_softMaxCondPE_{rate}"
        )
        trainer2.train(verbose=True)
    return


def repeat2(rates):
    model1 = resnet18(weights=None) #"IMAGENET1K_V1")
    model1.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model1.fc.in_features
    model1.fc = nn.Linear(num_ftrs, 10)
    model1.load_state_dict(torch.load("models/resnet18_MNIST", weights_only=True))

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    for rate in rates:
        model = resnet18(weights=None) #"IMAGENET1K_V1")
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)
        trainset_softmaxCond = condense_dataset1(model1,trainset,'predictive_entropy', rate)
        trainer2 = Trainer(
            model,
            trainset_softmaxCond, 
            testset ,   
            save=f"resnet18_MNIST_softMaxCondPE__mostANDleast_{rate}"
        )
        trainer2.train(verbose=True)
    return


def repeat3(rates):
    model1 = resnet18(weights=None) #"IMAGENET1K_V1")
    model1.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model1.fc.in_features
    model1.fc = nn.Linear(num_ftrs, 10)
    model1.load_state_dict(torch.load("models/resnet18_MNIST", weights_only=True))

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    for rate in rates:
        model = resnet18(weights=None) #"IMAGENET1K_V1")
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)
        trainset_softmaxCond = condense_dataset2(model1,trainset,'predictive_entropy', rate)
        trainer2 = Trainer(
            model,
            trainset_softmaxCond, 
            testset ,   
            save=f"resnet18_MNIST_softMaxCondPE_equal_{rate}"
        )
        trainer2.train(verbose=True)
    return

def repeat4(rates):
    model1 = resnet18(weights=None) #"IMAGENET1K_V1")
    model1.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model1.fc.in_features
    model1.fc = nn.Linear(num_ftrs, 10)
    model1.load_state_dict(torch.load("models/resnet18_MNIST", weights_only=True))

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    for rate in rates:
        model = resnet18(weights=None) #"IMAGENET1K_V1")
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)
        trainset_softmaxCond = condense_dataset3(model1,trainset,'predictive_entropy', rate)
        trainer2 = Trainer(
            model,
            trainset_softmaxCond, 
            testset ,   
            save=f"resnet18_MNIST_softMaxCondPE_equal_mostANDleast{rate}"
        )
        trainer2.train(verbose=True)
    return

def repeat5(rates):
    model1 = resnet18(weights=None) #"IMAGENET1K_V1")
    model1.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model1.fc.in_features
    model1.fc = nn.Linear(num_ftrs, 10)
    model1.load_state_dict(torch.load("models/resnet18_MNIST", weights_only=True))

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    for rate in rates:
        model = resnet18(weights=None) #"IMAGENET1K_V1")
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)
        trainset_softmaxCond = condense_dataset4(model1,trainset,'predictive_entropy', rate)
        trainer2 = Trainer(
            model,
            trainset_softmaxCond, 
            testset ,   
            save=f"resnet18_MNIST_softMaxCondPE_CCS{rate}"
        )
        trainer2.train(verbose=True)
    return