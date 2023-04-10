import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
from loaders.pano_loader import *

def delta_vecteur(i, n):
    vecteur = np.zeros(n)
    vecteur[i] = 1
    return vecteur

train_data, test_data = EUD_loader_grey()

# Convert numpy arrays to PyTorch tensors
train_data = torch.tensor(train_data, dtype=torch.float32)
test_data = torch.tensor(test_data, dtype=torch.float32)

# Create a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data):
        self.labels = data[:, 0, 0, 0, 0].clone().detach().long()
        self.images = data[:, :, :, 0, 0].clone().detach().float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


train_dataset = CustomDataset(train_data)
test_dataset = CustomDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=15, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=15, shuffle=False)

# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=0, stride=1)
        self.MP1 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(14 * 14 * 32, 512)
        self.fc2 = nn.Linear(512, 65)

    def forward(self, x):
        x = F.sigmoid(self.conv1(x))
        x = self.MP1(x)
        x = x.view(-1, 14*14*32)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x

model = CNN()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.5)

def eval(i, t):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.unsqueeze(1)  # Add a channel dimension
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Epoch {i + 1}/{t}, Accuracy: {accuracy:.3f}% ({correct}/{total})")

# Create the training loop
num_epochs = 50
print('hey')
eval(0, num_epochs)
for epoch in range(num_epochs):
    model.train()
    i = 0
    for images, labels in train_loader:
        i += 1
        if i % 1000 == 0:
            print(i)
        images = images.unsqueeze(1)  # Add a channel dimension
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    eval(epoch, num_epochs)
