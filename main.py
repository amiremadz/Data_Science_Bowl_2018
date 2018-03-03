from utils import read_train_data, read_test_data, allmasks_to_rles
from model import build_unet
from skimage.transform import resize
import numpy as np
import pandas as pd

epochs = 50

# get train train data
train_img, train_mask = read_train_data()

# get test train data
test_img, test_mask = read_test_data()

# get u-net model
u_net = build_unet()

# train model
print("\nTraining ...")
u_net.fit(train_img, train_mask, batch_size=16, epochs=epochs)

# Predict using test data
print("\nPredicitng ...")
test_masks = u_net.predict(test_img, verbose=1)

# Create list of resized test masks
test_masks_resized = []
for idx in range(len(test_masks)):
	this_mask = test_masks[i]
	test_masks_resized.append(resize(np.squeeze(this_mask), (this_mask[0], this_mask[1]), 
		mode='constant', preserve_range=True))
test_ids, rles = allmasks_to_rles(test_masks_resized)

# Create submission data frame
submit = pd.DataFrame()
submit['ImageId'] = test_ids
submit['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(idx) for idx in x)
submit.to_csv('submit-dsbowol-2018.csv', index=False)