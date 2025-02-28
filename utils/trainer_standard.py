import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, batch_size=64, learning_rate=1e-3, num_epochs=1, criterion = "crossEntropy", optimizer_type = "adam", save=""):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.save=save
        # Initialize the data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Optimizer
        if optimizer_type == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        elif optimizer_type == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_type == "rmsprop":
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Invalid optimizer type: {optimizer_type}. Valid optimizers: 'adam', 'sgd', 'rmsprop'")

        # Loss function
        if criterion == "crossEntropy":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Invalid criterion: {criterion}. Valid criteria: 'crossEntropy'")
        
        # Move the model to the appropriate device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def train_one_epoch(self, verbose=False,epoch=0):
        self.model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(self.train_loader, 0):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Compute the loss
            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 100 == 99 & verbose:  # print every 100 mini-batches
                print('[Epoch %d, Batch %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
        
        epoch_loss = running_loss / len(self.train_loader)
        accuracy = correct / total * 100
        return epoch_loss, accuracy
    
    def evaluate(self):
        self.model.eval()  # Set the model to evaluation mode
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():  # Don't compute gradients during evaluation
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute the loss
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = running_loss / len(self.val_loader)
        accuracy = correct / total * 100
        return val_loss, accuracy
    
    def save_model(self, fileName, train_loss, train_accuracy,val_loss, val_accuracy):
        # Save the trained model
        path = os.path.join("models", fileName)
        print(path)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
        with open("models/modelStats.txt", 'a') as file:
            file.write(f"MODEL:{fileName} , Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%" + "\n")


    def train(self, verbose = False):
        for epoch in range(self.num_epochs):
            train_loss, train_accuracy = self.train_one_epoch(verbose,epoch)
            val_loss, val_accuracy = self.evaluate()
            if verbose:
                print(f"Epoch {epoch+1}/{self.num_epochs}:")
                print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
                print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        if self.save != "":
            self.save_model(self.save,train_loss, train_accuracy,val_loss, val_accuracy)
    

    