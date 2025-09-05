import config
import dataloader
from dataloader import trainloader, testloader
import model

import torch
from torch import optim
from torch import nn
import tqdm

num_epochs = config.num_epochs
device = config.device

def main():

    modelo = model.AdvancedCNN(num_classes=config.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(modelo.parameters(), lr=config.learning_rate)

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        for batch_index, (data, targets) in enumerate(tqdm.tqdm(trainloader)):
            # Move data and targets to the device (GPU/CPU)
            data = data.to(device)
            targets = targets.to(device)

            # Forward pass: compute the model output
            scores = modelo(data)
            loss = criterion(scores, targets)

            # Backward pass: compute the gradients
            optimizer.zero_grad()
            loss.backward()

            # Optimization step: update the model parameters
            optimizer.step()

    def check_accuracy(loader, model):
        """
        Checks the accuracy of the model on the given dataset loader.

        Parameters:
            loader: DataLoader
                The DataLoader for the dataset to check accuracy on.
            model: nn.Module
                The neural network model.
        """
        if loader.dataset.train:
            print("Checking accuracy on training data")
        else:
            print("Checking accuracy on test data")

        num_correct = 0
        num_samples = 0
        model.eval()  # Set the model to evaluation mode

        with torch.no_grad():  # Disable gradient calculation
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)

                # Forward pass: compute the model output
                scores = model(x)
                _, predictions = scores.max(1)  # Get the index of the max log-probability
                num_correct += (predictions == y).sum()  # Count correct predictions
                num_samples += predictions.size(0)  # Count total samples

            # Calculate accuracy
            accuracy = float(num_correct) / float(num_samples) * 100
            print(f"Got {num_correct}/{num_samples} with accuracy {accuracy:.2f}%")
        
        model.train()  # Set the model back to training mode

    # Final accuracy check on training and test sets
    check_accuracy(trainloader, modelo)
    check_accuracy(testloader, modelo)

main()