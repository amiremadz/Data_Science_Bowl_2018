import os
import numpy as np
import pandas as pd

from utils import read_train_data, read_test_data, flip_images, eltransform_images, allmasks_to_rles
from model import build_unet, dice_coef
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from skimage.transform import resize
from skimage import io
from matplotlib import pyplot as plt
import random

epochs = 50
model_name = 'model-dsbowl-2018.h5'

# get train train data
X_train, Y_train = read_train_data()
X_train, Y_train = flip_images(X_train, Y_train)
X_train, Y_train = eltransform_images(X_train, Y_train)

# get test train data
X_test, test_sizes = read_test_data()

if os.path.isfile(model_name):
	#model = load_model(model_name, custom_objects={'mean_iou': mean_iou})
	model = load_model(model_name, custom_objects={'dice_coef': dice_coef})
else:
	# get u-net model
	model = build_unet()

	# train model
	print("\nTraining ...")
	earlystopper = EarlyStopping(patience=5, verbose=1)
	checkpointer = ModelCheckpoint('model-dsbowl-2018.h5', verbose=1, save_best_only=True)
	results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=epochs,
 		callbacks=[earlystopper, checkpointer])

# Predict using test data
print("\nPredicitng ...")
if os.path.isfile('preds_train.npy') and os.path.isfile('preds_val.npy') and os.path.isfile('preds_test.npy'): 
	print("Prediction data loaded from memory ...")
	preds_train = np.load('preds_train.npy')
	preds_val = np.load('preds_val.npy')
	preds_test = np.load('preds_test.npy')
else:
	preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
	preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
	preds_test = model.predict(X_test, verandbose=1)
	np.save('preds_train.npy', preds_train)
	np.save('preds_val.npy', preds_val)
	np.save('preds_test.npy', preds_test)

# Threshold predictions
preds_train_thr = (preds_train > 0.5).astype(np.uint8)
preds_val_thr   = (preds_val > 0.5).astype(np.uint8)
preds_test_thr  = (preds_test > 0.5).astype(np.uint8)

# Create list of resized test masks
X_test_resized = []
for idx in range(len(X_test)):
	this_mask = preds_test[idx]
	this_size = test_sizes[idx]
	print(idx, this_size)
	X_test_resized.append(resize(np.squeeze(this_mask), (this_size[0], this_size[1]), 
		mode='constant', preserve_range=True))
test_ids, rles = allmasks_to_rles(X_test_resized)

# Create submission data frame
submit = pd.DataFrame()
submit['ImageId'] = test_ids
submit['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
submit.to_csv('submit-dsbowol-2018.csv', index=False)

# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_thr))
plt.figure(figsize=(10,10))
plt.subplot(131)
io.imshow(X_train[ix])
plt.subplot(132)
io.imshow(np.squeeze(Y_train[ix]))
plt.subplot(133)
io.imshow(np.squeeze(preds_train_thr[ix]))
plt.show()

# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_val_thr))
plt.figure(figsize=(10,10))
plt.subplot(131)
io.imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.subplot(132)
io.imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.subplot(133)
io.imshow(np.squeeze(preds_val_thr[ix]))
plt.show()