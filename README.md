# cnn-cifar10

Project that implements a CNN created from scratch to clssify the CIFAR-10 dataset. It has been developed using pytorch in the vscode environment. Training and evaluation has been performed on Google Colab (using GPU)

## Project structure
```bash
root/
├─ src/
│  ├─ config.py        # Hyperparameters
│  ├─ dataloader.py    # Preprocessed data and DataLoaders
│  ├─ model.py         # Model definition
│  └─ main.py          # Training and evaluation
├─ data/               # Dataset folder (ignored in Git)
├─ outputs/            # Trained models and results (ignored in Git)
├─ requirements.txt
└─ README.md
```

## Usage
### Training
Execute the main script by using: 
```bash
python src/main.py
```
Tt is recomended the usage of GPU to train fasly (i.e. Google Colab)
It will show the loss during the training and testing and the final loss.
The best model trained will be saved as best_model.pth

### Evaluation
The trained model can be loaded loaclly and infere predictions to check the accuracy and funtioning of the model

## Dataset
CIFAR-10 has been used, with 60k images and 10 classes. It is automatically downloaded thank to dataloader.py
