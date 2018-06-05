import os
import numpy as np
import pandas as pd
import settings

from utils import read_train_data, read_test_data, enhance_images
from utils import RunLengthEncoder, ImageSegment
from keras.callbacks import EarlyStopping, ModelCheckpoint
from skimage.transform import resize
from skimage import io
from skimage.util import random_noise
from matplotlib import pyplot as plt
import random
import sys

model_name = 'model-dsbowl-2018.h5'
antialias_flag = False 

settings.init()

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

# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_thr) - 1)
plt.figure(figsize=(10,10))
plt.subplot(131)
plt.title('image')
io.imshow(X_train[ix])
plt.subplot(132)
plt.title('mask')
io.imshow(np.squeeze(Y_train[ix]))
plt.subplot(133)
plt.title('prediction')
io.imshow(np.squeeze(preds_train_thr[ix]))
plt.savefig('train_{:d}'.format(ix)+'.jpg', bbox_inches='tight')

ix = random.randint(0, len(preds_train_thr) - 1) 
plt.figure(figsize=(10,10))
plt.subplot(131)
plt.title('image')
io.imshow(X_train[ix])
plt.subplot(132)
plt.title('mask')
io.imshow(np.squeeze(Y_train[ix]))
plt.subplot(133)
plt.title('prediction')
io.imshow(np.squeeze(preds_train_thr[ix]))
plt.savefig('train_{:d}'.format(ix)+'.jpg', bbox_inches='tight')

# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_val_thr) - 1)
plt.figure(figsize=(10,10))
plt.subplot(131)
plt.title('image')
io.imshow(X_train[ix])
plt.subplot(132)
plt.title('mask')
io.imshow(np.squeeze(Y_train[ix]))
plt.subplot(133)
plt.title('prediction')
io.imshow(np.squeeze(preds_train_thr[ix]))
plt.savefig('valid_{:d}'.format(ix)+'.jpg', bbox_inches='tight')

ix = random.randint(0, len(preds_val_thr) - 1)
plt.figure(figsize=(10,10))
plt.subplot(131)
plt.title('image')
io.imshow(X_train[ix])
plt.subplot(132)
plt.title('mask')
io.imshow(np.squeeze(Y_train[ix]))
plt.subplot(133)
plt.title('prediction')
io.imshow(np.squeeze(preds_train_thr[ix]))
plt.savefig('valid_{:d}'.format(ix)+'.jpg', bbox_inches='tight')

# Perform a sanity check on some random test samples
ix = random.randint(0, len(preds_test_thr) - 1)
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.title('image')
io.imshow(X_test[ix])
plt.subplot(122)
plt.title('prediction')
io.imshow(np.squeeze(preds_test_thr[ix]))
plt.savefig('test_{:d}'.format(ix)+'.jpg', bbox_inches='tight')

ix = random.randint(0, len(preds_test_thr) - 1)
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.title('image')
io.imshow(X_test[ix])
plt.subplot(122)
plt.title('prediction')
io.imshow(np.squeeze(preds_test_thr[ix]))
plt.savefig('test_{:d}'.format(ix)+'.jpg', bbox_inches='tight')

#plt.show()
