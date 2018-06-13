import os
import numpy as np
import pandas as pd
import settings

from utils import read_train_data, read_test_data, enhance_images
from utils import ReadData, Plot, RunLengthEncoder, ImageSegment
from keras.callbacks import EarlyStopping, ModelCheckpoint
from skimage.transform import resize
from skimage import io
from skimage.util import random_noise
from matplotlib import pyplot as plt
import sys

model_name = 'model-dsbowl-2018.h5'
antialias_flag = False 

settings.init()

#my_data = ReadData()
#my_data.train_data()
#my_data.test_data()

# get train data
X_train, Y_train = enhance_images()

# get test train data
X_test, test_sizes = read_test_data()

myImgSeg = ImageSegment(model_name, X_train, Y_train, X_test, test_sizes)
myImgSeg.run()

#test_ids, rles = allmasks_to_rles(preds_test_resized)
rle = RunLengthEncoder(myImgSeg.preds_test_resized, settings.test_ids)
rle.run()

preds_train_thr = myImgSeg.preds_train
preds_val_thr = myImgSeg.preds_val
preds_test_thr = myImgSeg.preds_test

preds    = [preds_train_thr, preds_val_thr, preds_test_thr]
data_all = [X_train, Y_train, X_test]

my_plot = Plot(data_all, preds)
my_plot.plot()
