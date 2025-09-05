from model import CNN
from dataloader import testloader
from config import device
import torch

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

model = CNN(in_channels=3, num_classes=10)
model.load_state_dict(torch.load("outputs/best_model.pth"))
model.eval()

# Ejemplo para la matriz de confusión
all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in testloader:
        x, y = x.to(device), y.to(device)
        scores = model(x)
        _, preds = scores.max(1)
        all_preds.extend(preds.numpy())
        all_labels.extend(y.numpy())

for i in range(10):
    print(f"Prediccion {all_preds[i]}, Real {all_labels[i]}")

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=testloader.dataset.classes)
disp.plot(cmap='Blues')
disp.ax_.set_title("Matriz de Confusión")
plt.show()
plt.savefig("outputs/matriz_confusion.png")