import os
import numpy as np
import pandas as pd
import settings

from utils import ReadData, Plot, Enhance, RunLengthEncoder, ImageSegment
from keras.callbacks import EarlyStopping, ModelCheckpoint
from skimage.transform import resize
from skimage.util import random_noise
import sys

model_name = 'model-dsbowl-2018.h5'
antialias_flag = False 

settings.init()

data = ReadData()
data.train_data()
data.test_data()

# get train and test data
X_train = data.X_train
Y_train = data.Y_train
X_test     = data.X_test
test_sizes = data.test_sizes 

# enhance train data
enhanced = Enhance(X_train, Y_train)
enhanced.enhance()

X_train = enhanced.X_out
Y_train = enhanced.Y_out

# run segmentaion algorithm
myImgSeg = ImageSegment(model_name, X_train, Y_train, X_test, test_sizes)
myImgSeg.run()

# run-length encoder
rle = RunLengthEncoder(myImgSeg.preds_test_resized, settings.test_ids)
rle.run()

# predict labels
preds_train_thr = myImgSeg.preds_train
preds_val_thr = myImgSeg.preds_val
preds_test_thr = myImgSeg.preds_test

# plot results
preds    = [preds_train_thr, preds_val_thr, preds_test_thr]
data_all = [X_train, Y_train, X_test]

my_plot = Plot(data_all, preds)
my_plot.plot()
