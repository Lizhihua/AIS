import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), r'D:\code\pytorch-CycleGAN-and-pix2pix\VGG_gradient\checkpoint.pth')
        self.val_loss_min = val_loss

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root=r'D:\code\pytorch-CycleGAN-and-pix2pix\datasets\dwi\trainB', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

val_dataset = datasets.ImageFolder(root=r'D:\code\pytorch-CycleGAN-and-pix2pix\datasets\dwi\trainB', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
vgg = models.vgg16(pretrained=True)
# Modifying classifiers for binary classification tasks
vgg.classifier[6] = nn.Linear(vgg.classifier[6].in_features, 2)
# Freeze feature extraction layer (optional)
for param in vgg.features[-1:].parameters():
    param.requires_grad = True
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, vgg.parameters()), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device)
num_epochs=100
early_stopping = EarlyStopping(patience=5, verbose=True)
for epoch in range(num_epochs):
    vgg.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = vgg(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {train_loss}")

    vgg.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = vgg(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    print(f"Validation Loss: {val_loss}")
    print(f"Accuracy: {100 * correct / total}%")

    early_stopping(val_loss, vgg)

    if early_stopping.early_stop:
        print("Early stopping")
        break


