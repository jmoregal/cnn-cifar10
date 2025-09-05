import torch
import torchvision
import torchvision.transforms as transforms

# 1. Definir transformaciones (normalización + ToTensor)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),   # media (RGB)
                         (0.5, 0.5, 0.5))   # std (RGB)
])

# 2. Descargar dataset CIFAR-10 (train y test)
trainset = torchvision.datasets.CIFAR10(
    root='../data',          # carpeta donde se guardará
    train=True,             # conjunto de entrenamiento
    download=True,          # lo descarga si no está ya
    transform=transform     # aplica transformaciones
)
testset = torchvision.datasets.CIFAR10(
    root='../data',
    train=False,
    download=True,
    transform=transform
)

# 3. DataLoaders (para batch training)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=64,          # número de imágenes por batch
    shuffle=True,           # mezcla los datos
    num_workers=0           # procesos en paralelo
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)

# 4. Clases disponibles en CIFAR-10
classes = trainset.classes
print("Clases:", classes)