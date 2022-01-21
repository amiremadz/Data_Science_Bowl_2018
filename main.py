# Autor: Amir

import os
import numpy as np
import pandas as pd

from utils import read_train_data, read_test_data, allmasks_to_rles, train_masks_to_rles, enhance_images
from model import build_unet, dice_coef, mean_iou
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from skimage.transform import resize
from skimage import io
from skimage.util import random_noise
from matplotlib import pyplot as plt
import random
import sys

model_name = 'model-dsbowl-2018.h5'
antialias_flag = False 

# get train data
X_train, Y_train = enhance_images()

# get test train data
X_test, test_sizes = read_test_data()

if os.path.isfile(model_name):
    model = load_model(model_name, custom_objects={'mean_iou': mean_iou})
    #model = load_model(model_name, custom_objects={'dice_coef': dice_coef})
else:
    # get u-net model
    model = build_unet()
    # train model
    print("\nTraining ...")
    earlystopper = EarlyStopping(patience=5, verbose=1)
    checkpointer = ModelCheckpoint(model_name, verbose=1, save_best_only=True)
    results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=4, epochs=50,
            callbacks=[earlystopper, checkpointer])

# Predict using test data
print("\nPredicitng ...")
if os.path.isfile('preds_train.npy') : 
    print("Train prediction data loaded from memory ...")
    preds_train = np.load('preds_train.npy')
else:
    preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
    np.save('preds_train.npy', preds_train)

if os.path.isfile('preds_val.npy'): 
   print("Validation prediction data loaded from memory ...")
   preds_val = np.load('preds_val.npy')
else:
   preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
   np.save('preds_val.npy', preds_val)
                
if os.path.isfile('preds_test.npy'): 
   print("Test prediction data loaded from memory ...")
   preds_test = np.load('preds_test.npy')
else:
   preds_test = model.predict(X_test, verbose=1)
   np.save('preds_test.npy', preds_test)

# Threshold predictions
preds_train_thr = (preds_train > 0.5).astype(np.bool)
preds_val_thr   = (preds_val > 0.5).astype(np.bool)
preds_test_thr  = (preds_test > 0.5).astype(np.bool)

# Create list of resized test masks
print("\nResizing test data ...")
preds_test_resized = []
for idx in range(len(X_test)):
    this_mask = preds_test_thr[idx]
    this_size = test_sizes[idx]
    preds_test_resized.append(resize(np.squeeze(this_mask), (this_size[0], this_size[1]), mode='constant', preserve_range=True, anti_aliasing=antialias_flag))
test_ids, rles = allmasks_to_rles(preds_test_resized)

# Create submission data frame
print("\nCreating submit file ...")
submit = pd.DataFrame()
submit['ImageId'] = test_ids
submit['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
submit.sort_values(by=['ImageId'], inplace=True)
submit.to_csv('submit-dsbowol-2018.csv', index=False)

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
