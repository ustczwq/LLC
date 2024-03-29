import csv
import numpy as np
import pandas as pd
from skimage import io
import numpy as np
import matplotlib.pyplot as plt


class IndexTracker(object):
    def __init__(self, ax, imgs, msgs):
        self.ax = ax
        self.imgs = imgs
        self.msgs = msgs
        self.slices = len(self.imgs)
        self.idx = 0

        self.update()

    def onscroll(self, event):
        # print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.idx = (self.idx - 1) % self.slices
        else:
            self.idx = (self.idx + 1) % self.slices
        self.update()

    def update(self):
        img = io.imread(self.imgs[self.idx])
        self.im = ax.imshow(img)
        ax.set_title(self.msgs[self.idx])
        ax.set_ylabel('slice %s' % self.idx)
        self.im.axes.figure.canvas.draw()


items = ['Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table']

dataRoot = '../_datasets/_ExDark/'
testPath = 'TestList.csv'
predPath = './results/_m_dPred.csv'

preds = pd.read_csv(predPath, header=None).values.T[0]
names = pd.read_csv(testPath, usecols=[0], header=None).values.T[0]
labels = pd.read_csv(testPath, usecols=[1], header=None).values.T[0]

wrongImgs = []
errorMsgs = []
for i in range(len(preds)):
    if preds[i] != labels[i]:
        img = dataRoot + names[i]
        # img = io.imread(dataRoot + names[i])
        wrongImgs.extend([img])
        errorMsgs.extend(['Prediction: ' + items[preds[i]] + '  (Label: ' + items[labels[i]] + ')'])

err = len(wrongImgs)
acc = 1 - err/len(preds)
print('Total errors:', err, '  Accuracy: {:.3f} %'.format(100 * acc))

fig, ax = plt.subplots(1, 1)

tracker = IndexTracker(ax, wrongImgs, errorMsgs)

fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
plt.show()

# for i in range(err):
#     img = io.imread(wrongImgs[i])
#     ax.imshow(img)
#     ax.set_title(errorMsgs[i])
#     fig.savefig('../_result/_ExDark/Errors/' + str(i) + '.jpg')