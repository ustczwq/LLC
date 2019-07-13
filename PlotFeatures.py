import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.utils import make_grid
from logger import Logger
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image

def modelLoad(path):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 12)
    model = model.to(device)
    model.load_state_dict(torch.load(path))
    return model

def imgLoad(path, transform):
    img = Image.open(path).convert('RGB')
    if transform is not None:
        img = transform(img)

    img = [img.numpy()]
    return torch.tensor(img)

def getFeatureMap(img):
    with torch.no_grad():
        fea = img[0]
        fea = torch.mean(fea, 0, keepdim=False).cpu()
        fea = make_grid(fea, normalize=True, scale_each=True, padding=0).numpy()[0]
        fea = np.flipud(fea)
    return fea

def plotFeatures(imgPath, model, trans, depth, ax):
    ax[0].imshow(io.imread(imgPath))
    img = imgLoad(imgPath, trans).to(device)
    for d, (name, layer) in enumerate(model._modules.items()):
        if d < depth:
            img = layer(img)
            ax[d + 1].pcolormesh(getFeatureMap(img), cmap=plt.cm.jet)
            ax[d + 1].set(aspect=1, adjustable='box')
    return ax


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

items = ['Bik', 'Boa', 'Bot', 'Bus', 'Car', 'Cat', 'Cha', 'Cup', 'Dog', 'Mot', 'Peo', 'Tab']

mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean, std = std),
    ])

testPath = 'TestList.csv'
dataRoot  = '../_datasets/ExDark/'
_dataRoot = '../_datasets/_ExDark/'
modelPath  = './trainedModels/model20.ckpt'
_modelPath = './trainedModels/_model20.ckpt'
predsPath  = './results/mdPred.csv'
_predsPath = './results/_m_dPred.csv'

preds = pd.read_csv(predsPath, header=None).values.T[0]
_preds = pd.read_csv(_predsPath, header=None).values.T[0]
images = pd.read_csv(testPath, usecols=[0], header=None).values.T[0]
labels = pd.read_csv(testPath, usecols=[1], header=None).values.T[0]

model = modelLoad('./trainedModels/model20.ckpt')
_model = modelLoad('./trainedModels/_model20.ckpt')

depth = 6

for i in range(len(preds)):
    if preds[i] != labels[i] and _preds[i] == labels[i]:

        fig, axs = plt.subplots(2, depth + 1, figsize=(100, 40))

        plotFeatures(_dataRoot + images[i], 
                    _model, 
                    transform, 
                    depth, 
                    axs[0])

        plotFeatures(dataRoot + images[i], 
                    model, 
                    transform, 
                    depth, 
                    axs[1])

        fig.subplots_adjust(wspace =0, hspace =0)
        fig.tight_layout()
        fig.savefig('./results/' + items[_preds[i]] + '_' + items[preds[i]] + '_' + str(i) + '.jpg')
        plt.rcParams.update({'figure.max_open_warning': 0})

