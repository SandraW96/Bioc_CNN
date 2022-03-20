import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np


from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import os

from torchvision.datasets.folder import pil_loader

os.getcwd()
print(os.getcwd())

labels = pd.read_csv(r'data_v4.csv')
submission = pd.read_csv(r'test-2.csv')

train_path = r'processed_2/'
test_path = r'test-files/'

labels.head()
print(labels.head())
labels.tail()
print(labels.tail())
labels['has_contrast'].value_counts()

label = 'Hasn\'t Contrast', 'Has Contrast'
plt.figure(figsize = (8,8))
plt.pie(labels.groupby('has_contrast').size(), labels = label, autopct='%1.1f%%', shadow=True, startangle=90)
plt.show()


import matplotlib.image as img
fig,ax = plt.subplots(1,5,figsize = (15,3))

for i,idx in enumerate(labels[labels['has_contrast'] == 1]['id'][-5:]):
    path = os.path.join(train_path,idx)
    ax[i].imshow(img.imread(path))



fig,ax = plt.subplots(1,5,figsize = (15,3))
for i,idx in enumerate(labels[labels['has_contrast'] == 0]['id'][:5]):
    path = os.path.join(train_path,idx)
    ax[i].imshow(img.imread(path))

def imshow(image, ax=None, title=None, normalize=True):
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))
    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax



class CactiDataset(Dataset):
    def __init__(self, data, path , transform = None):
        super().__init__()
        self.data = data.values
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name, label = self.data[index]
        img_path = os.path.join(self.path, img_name)
        image = img.imread(img_path)[:,:,3]
        if self.transform is not None:
            image = self.transform(np.uint8(image))
        return image, label




train_transform = transforms.Compose([transforms.ToPILImage(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5), (0.5))])

test_transform = transforms.Compose([transforms.ToPILImage(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5), (0.5))])

valid_transform = transforms.Compose([transforms.ToPILImage(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5), (0.5))])

train, valid_data = train_test_split(labels, stratify=labels.has_contrast, test_size=0.2)

train_data = CactiDataset(train, train_path, train_transform )
valid_data = CactiDataset(valid_data, train_path, valid_transform )
test_data = CactiDataset(submission, test_path, test_transform )


num_epochs = 35
num_classes = 2
batch_size = 7
learning_rate = 0.001

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device(type = 'cuda', index=0)

train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(dataset = valid_data, batch_size = batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(dataset = test_data, batch_size = batch_size, shuffle=False, num_workers=0)

import numpy as np
import matplotlib.pyplot as plt

def imshow(image, ax=None, title=None, normalize=True):
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.5])
        std = np.array([0.5])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax


trainimages, trainlabels = next(iter(train_loader))

fig, axes = plt.subplots(figsize=(12, 12), ncols=5)
print('training images')
for i in range(5):
    axe1 = axes[i]
    imshow(trainimages[i], ax=axe1, normalize=False)

print(trainimages[0].size())


epochs = 35
batch_size = 25
learning_rate = 0.001

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv3 = nn.Conv2d(20, 40, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(153760, 21)
        self.fc2 = nn.Linear(21, 2)
        # self.fc3=nn.Linear

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(x.shape[0],-1)
        # x = x.view(-1, 20*10*10)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


model = CNN()
print(model)

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

train_losses = []
valid_losses = []

for epoch in range(1, num_epochs + 1):
    # keep-track-of-training-and-validation-loss
    train_loss = 0.0
    valid_loss = 0.0

    # training-the-model
    model.train()
    for data, target in train_loader:
        # move-tensors-to-GPU
        data = data.to(device)
        target = target.to(device)

        # clear-the-gradients-of-all-optimized-variables
        optimizer.zero_grad()
        # forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
        output = model(data)
        # calculate-the-batch-loss
        loss = criterion(output, target)
        # backward-pass: compute-gradient-of-the-loss-wrt-model-parameters
        loss.backward()
        # perform-a-ingle-optimization-step (parameter-update)
        optimizer.step()
        # update-training-loss
        train_loss += loss.item() * data.size(0)

    # validate-the-model
    model.eval()
    for data, target in valid_loader:

        data = data.to(device)
        target = target.to(device)

        output = model(data)

        loss = criterion(output, target)

        # update-average-validation-loss
        valid_loss += loss.item() * data.size(0)

    # calculate-average-losses
    train_loss = train_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(valid_loader.sampler)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    # print-training/validation-statistics
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))


print("Train succeded")
# model test
model.eval()  # it-disables-dropout
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in valid_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print('Test Accuracy of the model: {} %'.format(100 * correct / total))

    # print('Test Accuracy of the model: {} %'.format(100 * correct / total))


from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

y_pred = []
y_true = []

nb_samples = 123;
nb_classes = 2
output = torch.randn(nb_samples, nb_classes)
pred = torch.argmax(output, 1)
target = torch.randint(0, nb_classes, (nb_samples,))

conf_matrix = torch.zeros(nb_classes, nb_classes)
for t, p in zip(target, pred):
    conf_matrix[t, p] += 1

print('Confusion matrix\n', conf_matrix)

TP = conf_matrix.diag()
for c in range(nb_classes):
    idx = torch.ones(nb_classes).byte()
    idx[c] = 0
    # all non-class samples classified as non-class
    TN = conf_matrix[
        idx.nonzero()[:, None], idx.nonzero()].sum()  # conf_matrix[idx[:, None], idx].sum() - conf_matrix[idx, c].sum()
    # all non-class samples classified as class
    FP = conf_matrix[idx, c].sum()
    # all class samples not classified as class
    FN = conf_matrix[c, idx].sum()

    print('Class {}\nTP {}, TN {}, FP {}, FN {}'.format(
        c, TP[c], TN, FP, FN))
# Save
torch.save(model.state_dict(), 'model.ckpt')

plt.plot(train_losses, label='Training loss')
plt.plot(valid_losses, label='Validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
# valid = [0.6146951521315226, 0.4751594058726346, 0.5198560458857838, 0.3159975449486477, 0.28194280041427144, 0.31895555555820465, 0.28347309351694294, 0.28618278463439245, 0.28649014089165664, 0.2938157363635738, 0.28285124807096107, 0.29739245299885914, 0.29230193757429357, 0.28438691794872284, 0.2869170135477694, 0.29097609258279566, 0.2868037138406823, 0.2889465191742269, 0.282860905658908, 0.2972363393481185, 0.2820142320379978, 0.2845259930302457, 0.30311773208583276, 0.2841210287155175, 0.28407029498641084, 0.28205748138631265, 0.2847745658420935, 0.29199733385225624, 0.2826470905324308, 0.290903867017932, 0.2857700132015275, 0.2846133910664698, 0.28250592201948166, 0.2880768746864505, 0.28197951796578197, 0.2957821448401707, 0.28444665579534156, 0.28349449521884684, 0.2870352838824435, 0.28391717556046275, 0.28305982125968465, 0.28410014237572506, 0.2820437108961547, 0.2837054531385259, 0.2937307804822922, 0.28288037747871586, 0.2852954248466143, 0.2874840119989907, 0.2849189319261691, 0.2822044524477749, 0.2899837065033796, 0.2822583608511018, 0.28294147278477505, 0.2838928047113302, 0.2829405145674217, 0.2890304625034332, 0.28632061365174083, 0.2823006247238415, 0.28269251472339396, 0.2828491846235787, 0.2819408140167957, 0.2850038525534839, 0.28376997044173685, 0.2884627364394141, 0.285430697951375, 0.2822323739528656, 0.28208544595939355, 0.2830320943782969, 0.283628543884289, 0.2819666638970375]
# train_l = [1.6266530618932604, 0.4162809174598717, 0.5951233388083738, 0.5049989043575961, 0.376290132541482, 0.4063031040132046, 0.3504280977678008, 0.3476616590306526, 0.37131452278756516, 0.36945918011592654, 0.33461914034333173, 0.3525035660259607, 0.35296939622338225, 0.3505308173324277, 0.3216858866011224, 0.3252540620089304, 0.33082364090695615, 0.3258423361382106, 0.33582784580748254, 0.346193377928036, 0.33232690480242416, 0.33261071072845927, 0.3368943335897312, 0.32807538526632435, 0.320512797138313, 0.3162798948767709, 0.312574896279995, 0.33338748963504306, 0.3257722397584741, 0.324532320041482, 0.32216458267918446, 0.32122408262476687, 0.32409811792213744, 0.3229506733577426, 0.31145869595248526, 0.3203806316071167, 0.3111763523391834, 0.3132237616199546, 0.3167743939542916, 0.3080630336047673, 0.3106725215911865, 0.3044579405701015, 0.29840474394036504, 0.31612563728377585, 0.3203063916596698, 0.30504427972908427, 0.3048746369143085, 0.3061789693083705, 0.31363255657800815, 0.29279840628548365, 0.30582751910679223, 0.3057980366596362, 0.3031967376336092, 0.29987276677133107, 0.29210172343726565, 0.3158292438289741, 0.31280219482212535, 0.29811349411199733, 0.2983246454741897, 0.2971858390856807, 0.2843228626269393, 0.30312393383100267, 0.2983457371138218, 0.3086614005903645, 0.29744040911517494, 0.3001060363177846, 0.2944004499785057, 0.2980278582289452, 0.29912490715704315, 0.29344593374649197]
