import os
import csv
import pandas
from random import shuffle

def txt2csv(name):
    data = []

    with open(name + '.txt', 'r') as fr:
        ls = fr.readlines()
        for l in ls:
            img, label = l.split()
            data.append([img, label])
        # shuffle(data)


    with open(name + '.csv', 'w') as fw:
        w = csv.writer(fw)
        w.writerows(data)

txt2csv('TrainList')
txt2csv('TestList')