import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
# -------------------------
# Helper functions for evidential loss components
# -------------------------
def calc_error(y_true, p):
    """
    Computes the squared error between the one-hot true labels and the expected probabilities.
    
    Args:
        y_true: One-hot encoded tensor of true labels, shape [batch, num_classes].
        p: Expected probabilities, shape [batch, num_classes].
    
    Returns:
        Tensor of shape [batch] containing the error per sample.
    """
    return torch.sum((y_true - p) ** 2, dim=1)


def calc_variance(alpha, S):
    """
    Computes the uncertainty (variance) term derived from the Dirichlet distribution.
    For Dirichlet, Var[p_c] = (alpha_c * (S - alpha_c)) / (S^2 * (S+1)).
    
    Args:
        alpha: Dirichlet parameters, shape [batch, num_classes].
        S: Total evidence per sample, shape [batch, 1].
    
    Returns:
        Tensor of shape [batch] containing the variance per sample.
    """
    return torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1)


def calc_kl_divergence(alpha):
    """
    Computes the KL divergence between the predicted Dirichlet distribution 
    (with parameters alpha) and a uniform Dirichlet prior (with parameters 1).
    
    The formula used is:
    KL(Dir(alpha) || Dir(1)) = log(Γ(S)) - log(Γ(K)) - sum(log(Γ(alpha_i)))
                                 + sum((alpha_i - 1) * (ψ(alpha_i) - ψ(S)))
    where S = sum(alpha) and K is the number of classes.
    
    Args:
        alpha: Dirichlet parameters, shape [batch, num_classes].
    
    Returns:
        Tensor of shape [batch] containing the KL divergence per sample.
    """
    # Sum of Dirichlet parameters for each sample: shape [batch]

    S = torch.sum(alpha, dim=1)
    K = alpha.size(1)
    
    ones = torch.ones([1, K], dtype=torch.float32, device=S.device)
    first_term = (
        torch.lgamma(S)
        - torch.lgamma(alpha).sum(dim=1)
        + torch.lgamma(ones).sum(dim=1)
        - torch.lgamma(ones.sum(dim=1))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(S).unsqueeze(1))
        .sum(dim=1)
    )
    kl = first_term + second_term
    return kl


# -------------------------
# Define the evidential classification loss with KL divergence
# -------------------------
def evidential_classification_loss(y_true, evidence, epoch):
    """
    Computes the evidential classification loss, which includes the data-fit term, 
    the uncertainty (variance) term, and the KL divergence regularization term.
    
    Args:
        y_true: Tensor containing class indices or one-hot encoded labels.
        evidence: Raw model outputs representing evidence for each class, shape [batch, num_classes].
        kl_weight: Weighting factor for the KL divergence term.
    
    Returns:
        Scalar loss value.
    """
    # Ensure non-negative evidence using softplus
    evidence_pos = F.softplus(evidence)
    
    # Convert evidence to Dirichlet parameters: alpha = evidence + 1
    alpha = evidence_pos + 1.0
    S = torch.sum(alpha, dim=1, keepdim=True)  # shape [batch, 1]
    
    # Expected probability per class: p = alpha / S
    p = alpha / S
    
    # Convert y_true to one-hot encoding 
    y_true = F.one_hot(y_true, num_classes=p.size(1)).float()
    
    # Calculate each component of the loss using helper functions.
    error = calc_error(y_true, p)          # Data-fit term
    variance = calc_variance(alpha, S)       # Uncertainty term (Dirichlet variance)
    kl = calc_kl_divergence(alpha)           # KL divergence regularization term
    
    # Calculate annealing_coef:
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch / 10, dtype=torch.float32),
    )
    # Combine the terms. Note: error and variance are per sample, as is KL.
    loss = error + variance + annealing_coef * kl
    return torch.mean(loss)


# -------------------------
# Trainer class modified to support evidential loss with KL divergence
# -------------------------
class Trainer:
    def __init__(self, model, train_dataset, val_dataset, batch_size=64, learning_rate=5e-4, num_epochs=20,
                 criterion="evidential", optimizer_type="adam", save=""):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.save = save
        # Initialize the data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,  drop_last=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,  drop_last=True)

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
            # Use the custom evidential loss function that includes KL divergence.
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
            
            # Compute loss
            if self.criterion == evidential_classification_loss:
                loss = self.criterion(labels, outputs,epoch)
            else:
                loss = self.criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            # For evaluation of accuracy (if applicable), use argmax over the expected probabilities.
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
            # Convert one-hot labels to indices if needed.
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
        
        with torch.no_grad():  # No gradients during evaluation
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                
                if self.criterion == evidential_classification_loss:
                    loss = self.criterion(labels, outputs,0) #0, because annealing step should be 1
                else:
                    loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                
                if self.criterion == evidential_classification_loss:
                    evidence = outputs
                    evidence = F.softplus(evidence)
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
            file.write(f"MODEL:{fileName} , Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}, "
                       f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%\n")

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
