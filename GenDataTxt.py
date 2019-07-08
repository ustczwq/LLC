import os

dataRoot = "../_datasets/ExDark/"

dirs = os.listdir(dataRoot)
dirs.sort()

trainRate = 0.9

label = 0
for d in dirs:
    imgs = os.listdir(dataRoot + d)
    totalNum = len(imgs)
    trainNum = int(trainRate * totalNum)
    
    with open('train.txt', 'w' if label == 0 else 'a') as ft:
        for i in range(0, trainNum):
            ft.write(imgs[i] + ' ' + str(label) + '\n')

    with open('test.txt', 'w' if label == 0 else 'a') as fv:
        for i in range(trainNum, totalNum):
            fv.write(imgs[i] + ' ' + str(label) + '\n')

    label += 1
