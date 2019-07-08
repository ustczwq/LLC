import os
import csv

dataRoot = "../_datasets/ExDark/"

dirs = os.listdir(dataRoot)
dirs.sort()

trainRate = 0.9
train = []
test = []

l = 0
for d in dirs:
    imgs = os.listdir(dataRoot + d)
    totalNum = len(imgs)
    trainNum = int(trainRate * totalNum)
    label = str(l)
    for i in range(totalNum):
        img = d + '/' + imgs[i]
        if i < trainNum:
            train.append([img, label])
        else:
            test.append([img, label])
    l += 1

with open('train.csv', 'w') as f:
    w = csv.writer(f)
    w.writerows(train)

with open('test.csv', 'w') as f:
    w = csv.writer(f)
    w.writerows(test)