import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F

# -------------------------
# Define the evidential classification loss
# -------------------------
import torch
import torch.nn.functional as F


def evidential_classification_loss(y_true, evidence):
    """
    Computes the evidential classification loss.
    
    Args:
        y_true: Tensor containing class indices, shape [batch,] where each element is an integer.
        evidence: Predicted evidence for each class (nonnegative), shape [batch, num_classes]. 
                  The model's output should be raw evidence (not probabilities), which will be transformed into Dirichlet parameters.
    
    Returns:
        Scalar loss that models the aleatoric uncertainty.
    """
    # Apply Softplus to ensure non-negative evidence
    evidence1 = F.softplus(evidence)

    # Convert evidence to Dirichlet parameters: alpha = evidence + 1
    alpha = evidence1 + 1.0
    S = torch.sum(alpha, dim=1, keepdim=True)  # total evidence per sample
    # Expected probability per class: p = alpha / S
    p = alpha / S

    # If y_true is not one-hot encoded (i.e., it contains class indices), convert to one-hot encoding
    if y_true.dim() == 1:  # If y_true is a tensor of class indices
        y_true = F.one_hot(y_true, num_classes=p.size(1)).float()

    # Data-fit term: squared error between one-hot labels and expected probability
    error = torch.sum((y_true - p) ** 2, dim=1)
    
    # Uncertainty term: derived from the Dirichlet variance.
    # For Dirichlet, Var[p_c] = (alpha_c*(S - alpha_c))/(S^2*(S+1)).
    uncertainty = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1)
    
    loss = error + uncertainty
    #print(f"TOTUNCERTAINTY: {uncertainty}")
    return torch.mean(loss)


# -------------------------
# Trainer class modified to support evidential loss
# -------------------------
class Trainer:
    def __init__(self, model, train_dataset, val_dataset, batch_size=64, learning_rate=1e-3, num_epochs=1,
                 criterion="evidential", optimizer_type="adam", save=""):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.save = save
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
        elif criterion == "evidential":
            # Use the custom evidential loss function.
            self.criterion = evidential_classification_loss
        else:
            raise ValueError(f"Invalid criterion: {criterion}. Valid criteria: 'crossEntropy', 'evidential'")

        # Move the model to the appropriate device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def train_one_epoch(self, verbose=False, epoch=0):
        self.model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(self.train_loader, 0):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass: if using evidential loss, model should output evidence.
            outputs = self.model(inputs)
            
            # For evidential loss, assume model returns evidence (for all classes).
            # If using evidential loss, we assume that labels are one-hot encoded.
            loss = self.criterion(labels, outputs) if self.criterion == evidential_classification_loss else self.criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            # For evaluation of accuracy (if applicable), use argmax over the expected probabilities.
            # For evidential loss, convert evidence to probabilities: p = (evidence + 1) / sum(evidence + 1)
            if self.criterion == evidential_classification_loss:
                evidence = outputs
                evidence = F.softplus(evidence)  # Apply softplus here as well
                alpha = evidence + 1.0
                S = torch.sum(alpha, dim=1, keepdim=True)
                probs = alpha / S
                _, predicted = torch.max(probs, 1)
            else:
                _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            # If labels are one-hot, convert to indices:
            if labels.dim() > 1:
                labels_indices = labels.argmax(dim=1)
            else:
                labels_indices = labels
            correct += (predicted == labels_indices).sum().item()

            if i % 100 == 99 and verbose:  # print every 100 mini-batches
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
                
                outputs = self.model(inputs)
                
                loss = self.criterion(labels, outputs) if self.criterion == evidential_classification_loss else self.criterion(outputs, labels)
                
                running_loss += loss.item()
                
                if self.criterion == evidential_classification_loss:
                    evidence = outputs
                    evidence = F.softplus(evidence)  # Apply softplus here as well
                    alpha = evidence + 1.0
                    S = torch.sum(alpha, dim=1, keepdim=True)
                    probs = alpha / S
                    _, predicted = torch.max(probs, 1)
                else:
                    _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                if labels.dim() > 1:
                    labels_indices = labels.argmax(dim=1)
                else:
                    labels_indices = labels
                correct += (predicted == labels_indices).sum().item()
        
        val_loss = running_loss / len(self.val_loader)
        accuracy = correct / total * 100
        return val_loss, accuracy
    
    def save_model(self, fileName, train_loss, train_accuracy, val_loss, val_accuracy):
        path = os.path.join("models", fileName)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
        with open("models/modelStats.txt", 'a') as file:
            file.write(f"MODEL:{fileName} , Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%" + "\n")

    def train(self, verbose=False):
        for epoch in range(self.num_epochs):
            train_loss, train_accuracy = self.train_one_epoch(verbose, epoch)
            val_loss, val_accuracy = self.evaluate()
            if verbose:
                print(f"Epoch {epoch+1}/{self.num_epochs}:")
                print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
                print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        if self.save != "":
            self.save_model(self.save, train_loss, train_accuracy, val_loss, val_accuracy)
