import os
import numpy as np
import pandas as pd

from utils import read_train_data, read_test_data, augment_imgs, allmasks_to_rles
from model import build_unet, dice_coef
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from skimage.transform import resize
from skimage.io import imshow
from matplotlib import pyplot as plt

epochs = 50
model_name = 'model-dsbowl-2018.h5'

# get train train data
X_train, Y_train = read_train_data()

# get test train data
X_test, test_sizes = read_test_data()

if os.path.isfile(model_name):
	model = load_model(model_name, custom_objects={'mean_iou': mean_iou})
else:
	# get u-net model
	model = build_unet()

	# train model
	print("\nTraining ...")
	earlystopper = EarlyStopping(patience=5, verbose=1)
	checkpointer = ModelCheckpoint('model-dsbowl-2018.h5', verbose=1, save_best_only=True)
	results = u_net.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=epochs,
 		callbacks=[earlystopper, checkpointer])

# Predict using test data
print("\nPredicitng ...")
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

# Threshold predictions
preds_train_thr = (preds_train > 0.5).astype(np.uint8)
preds_val_thr   = (preds_val > 0.5).astype(np.uint8)
preds_test_thr  = (preds_test > 0.5).astype(np.uint8)

# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.show()

# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_val_t))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.show()

# Create list of resized test masks
test_masks_resized = []
for idx in range(len(test_masks)):
	this_mask = preds_test[i]
	this_size = test_sizes[i]
	test_masks_resized.append(resize(np.squeeze(this_mask), (this_size[0], this_size[1]), 
		mode='constant', preserve_range=True))
test_ids, rles = allmasks_to_rles(test_masks_resized)

# Create submission data frame
submit = pd.DataFrame()
submit['ImageId'] = test_ids
submit['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(idx) for idx in x)
submit.to_csv('submit-dsbowol-2018.csv', index=False)