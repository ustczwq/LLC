import torch
import torch.nn as nn
from Model import LLCDataset
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epochs = 20
lr = 1e-4

mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean, std = std),
    ])

trainData = LLCDataset(csvPath='TrainList.csv', 
                        rootDir='../_datasets/_ExDark/', 
                        transform=transform)

testData = LLCDataset(csvPath='TestList.csv',  
                        rootDir='../_datasets/_ExDark/', 
                        transform=transform)

trainLoader = DataLoader(dataset=trainData, batch_size=64, shuffle=True)
testLoader = DataLoader(dataset=testData, batch_size=64)

model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 12)
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for ep in range(epochs):
    if ep == 9:
        lr = 1e-5
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for i, (imgs, labels) in enumerate(trainLoader):
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)
        # _, pred = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print("Epoch [{}/{}], Step [{}], Loss: {:.6f}".format(ep + 1, epochs, i + 1, loss.item())) 

    if (ep + 1) % 5 == 0:
        torch.save(model.state_dict(), './trainedModels/model' + str(ep + 1) + '.ckpt')


model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for i, (imgs, labels) in enumerate(testLoader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = model(imgs)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    print('Test Accuracy: {:.3f} %'.format(100 * correct / total))