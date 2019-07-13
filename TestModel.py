import csv
import numpy as np
import pandas as pd
from Model import LLCDataset
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader


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
                        rootDir='../_datasets/ExDark/', 
                        transform=transform)

testLoader = DataLoader(dataset=testData, batch_size=64)

model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 12)
model = model.to(device)

model.load_state_dict(torch.load('./trainedModels/model20.ckpt'))
model.eval()

predList = [] 

with torch.no_grad():
    correct = 0
    total = 0
    for i, (imgs, labels) in enumerate(testLoader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = model(imgs)
        _, pred = torch.max(outputs.data, 1)

        predList.extend(pred.tolist())
        isCorrect = (pred == labels)
        print(isCorrect.tolist())

        total += labels.size(0)
        correct += isCorrect.sum().item()

    print('Test Accuracy: {:.3f} %'.format(100 * correct / total))

print(total, total - correct)

with open('./results/mdPred.csv', 'w') as f:
    w = csv.writer(f)    
    for l in predList:
        w.writerow([l])