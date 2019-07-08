import os
import numpy as np
from skimage import io, exposure

dataRoot = "../_datasets/ExDark/"
newRoot = "../_datasets/_ExDark/"

dirs = os.listdir(dataRoot)
dirs.sort()

for d in dirs:
    if not os.path.exists(newRoot + d):
        os.makedirs(newRoot + d)

    names = os.listdir(dataRoot + d)
    names.sort()
    for name in names:
        img = io.imread(dataRoot + d + '/' + name)
        img = exposure.adjust_gamma(img/255, 0.6)
        img = np.uint8(img*255)
        print(d + '/' + name)
        io.imsave(newRoot + d + '/' + name, img, check_contrast=False)

