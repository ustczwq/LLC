import numpy as np
from Model import LLCDataset
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from logger import Logger
from skimage import io
import matplotlib.pyplot as plt


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

model.load_state_dict(torch.load('./trainedModels/model20.ckpt'))
model.eval()

logger = Logger('./logs')

# fig, axs = plt.subplots(1, 1)

with torch.no_grad():
    for i, (imgs, labels) in enumerate(testLoader):
        imgs = imgs.to(device)
        for d, (name, layer) in enumerate(model._modules.items()):
            if d < 7:
                imgs = layer(imgs)
                img = imgs[0]
                img = torch.mean(img, 0, keepdim=False).cpu()

                fea = make_grid(img, normalize=True, scale_each=True, padding=0).numpy()[0]
                fea = np.fliplr(fea)
                fea = np.flipud(fea)

                plt.clf()
                plt.pcolormesh(fea, cmap = plt.cm.jet)
                plt.title(str(i) + ' - ' + name)
                plt.colorbar(shrink=0.6)
                plt.savefig('./results/' + str(i) + '-' + str(d) + '.jpg')

                # logger.image_summary('features' + str(i), feature, step)
                
        if i == 1:
            break

