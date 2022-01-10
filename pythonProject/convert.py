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

images = [None]*427
img_path = os.listdir(r'processed')

for i in range(len(img_path)-1):
    images[i]=Image.open('processed/'+str(i)+'.png')
    plt.imsave('processedJPG/'+str(i)+'.jpg', images[i])

