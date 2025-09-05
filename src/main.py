import config
import dataloader
from dataloader import trainloader, testloader
import model
import matplotlib.pyplot as plt

import torch
from torch import optim
from torch import nn
import tqdm

num_epochs = config.num_epochs
device = config.device

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
        return accuracy

def main():

    best_loss = float("inf")

    train_losses = []
    train_accuracies = []
    test_accuracies = []

    modelo = model.CNN(in_channels=3, num_classes=config.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(modelo.parameters(), lr=config.learning_rate)

    for epoch in range(num_epochs):
        running_loss = 0
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

            running_loss += loss.item()
            
            # Mostrar pérdida cada 100 batches
            if batch_index % 100 == 0:
                print(f"\nBatch {batch_index}/{len(trainloader)} - Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(trainloader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss:{avg_loss:.4f}")

        train_acc = check_accuracy(trainloader, modelo)
        test_acc = check_accuracy(testloader, modelo)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        if avg_loss < best_loss:
          best_loss = avg_loss
          torch.save(modelo.state_dict(), "best_model.pth")
          print(f"✅ Nuevo mejor modelo guardado (loss={best_loss:.4f})")
        # Final accuracy check on training and test sets
    print("Training complete.")

    # Plot training loss and accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy over Epochs')
    plt.legend()
    plt.tight_layout()
    plt.show()

    
    
if __name__ == "__main__":  
    main()