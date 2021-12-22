import glob
# import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from database import CustomDataset
import torch.nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Transformations we can perform on our dataset for augmentation
from torch import nn  # All neural network modules
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # For nice progress bar!
from cnn import CNN


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 3

dataset = CustomDataset(csv_file=r'database.csv', root_dir=r'processed', transform=transforms.ToTensor())
train_set, test_set = torch.utils.data.random_split(dataset, [329, 100])

train_Loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_Loader=DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)


# Initialize network
model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_Loader)):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)


    model.train()
    return num_correct/num_samples


print(f"Accuracy on training set: {check_accuracy(train_Loader, model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_Loader, model)*100:.2f}")