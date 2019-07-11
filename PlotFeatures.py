import csv
import numpy as np
import pandas as pd
from Model import LLCDataset
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
import torch.nn.functional as F
from logger import Logger
import cv2


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean, std = std),
    ])

testData = LLCDataset(csvPath='TestList.csv', 
                        rootDir='../_datasets/_ExDark/', 
                        transform=transform)

testLoader = DataLoader(dataset=testData, batch_size=64)

model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 12)
model = model.to(device)

model.load_state_dict(torch.load('model20.ckpt'))
model.eval()

logger = Logger('./logs')


with torch.no_grad():
    for i, (imgs, labels) in enumerate(testLoader):
        imgs = imgs.to(device)
        step = 1
        for name, layer in model._modules.items():
            if name != 'fc':
                img = imgs[0:3, :, :, :]
                print(img.size())
                img = torch.mean(img, 1, keepdim=False).cpu()       
                feature = make_grid(img, normalize=True, scale_each=True, padding=0)
                logger.image_summary('features' + str(i), feature, step)
                imgs = layer(imgs)
                step += 1

        if i == 4:
            break