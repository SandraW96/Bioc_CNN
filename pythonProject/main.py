import numpy as np
from os import listdir
import numpy
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from pydicom.data import get_testdata_file
from tqdm import tqdm
import os
from PIL import Image
import glob
import sys
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


test_path=r'test'
train_path=r'train'

test_files = os.listdir(test_path)          #130 files
test_list = [pydicom.dcmread(os.path.join(test_path, filename), force=True) for filename in test_files]

train_files=os.listdir(train_path)          #1464 files
train_list=[pydicom.dcmread(os.path.join(train_path, filename), force=True) for filename in train_files]

#dicom to numpy
dicom_list = []
test_array = np.zeros((130, 512, 512))
numpy.set_printoptions(threshold=sys.maxsize)

#dicomTrain to array
dicom_list_train = []
train_array = np.zeros((430, 512, 512))

for i in range(len(test_list)):
    dicom_list.append(test_list[i].pixel_array)


for i in range(len(train_list)):
    dicom_list_train.append(train_list[i].pixel_array)



for i in range(len(test_list)):
    for j in range(512):
        for k in range(512):
            test_array[i][j][k] = dicom_list[i][j][k]


for i in range(len(train_list)):
    for j in range(512):
        for k in range(512):
            train_array[i][j][k] = dicom_list_train[i][j][k]

def muteValues(map):
    #every value lower than 1000 will be replaced by 0,
    # every higher than 1850 replaced by 0
    map[map <= 950] = 0
    map[map >= 1850] = 0
    return map

#exectue first threshold - test array
for i in range(len(test_array)):
    test_array[i]=muteValues(test_array[i])
    # plt.imshow(dicom_array[i], cmap='gray')
    # plt.show()

#exectue first threshold - train array
for i in range(len(train_array)):
    train_array[i]=muteValues(train_array[i])
    # plt.imshow(dicom_array[i], cmap='gray')
    # plt.show()

def bandpassLikeFilter(map):
    output_img = np.clip(map, 1000, 1400)   # everything < 1000 will by replaced with 1000,
    # plt.imshow(output_img, cmap='gray')     # val > 1400 will be replaced by 1400
    # plt.title("pierwszy clip, 1000-1300")
    # plt.show()
    kernel = np.ones((15,15))
    output_erode = cv2.erode(output_img, kernel)
    # plt.imshow(output_erode, cmap='gray')
    # plt.title("erozja")
    # plt.show()
    output_dilate = cv2.dilate(output_erode, kernel)
    # plt.imshow(output_dilate, cmap='gray')
    # plt.title("dylacja, obraz wyjsciowy ")
    # plt.show()
    return output_dilate

output_test = np.zeros((130, 512, 512))       # array for output images. This will be processed in next steps
output_train = np.zeros((430, 512, 512))

#get the output test
for i in range(len(test_array)):
    output_test[i]=bandpassLikeFilter(test_array[i])

for i in range(len(train_array)):
    output_train[i]=bandpassLikeFilter(train_array[i])



processed_path=r'processed'
for i in range(len(output_train)):
    plt.imsave(r'processed/'+str(i)+'.png', output_train[i])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 5
batch_size = 4
learning_rate = 0.001


