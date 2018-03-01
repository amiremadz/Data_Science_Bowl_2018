import warnings
import numpy as np
import random
import os
import sys


from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from keras.utils import Progbar


warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

# Setting seed for reproducability
seed = 42
random.seed = seed
np.random.seed = seed

# Data Path
TRAIN_PATH = 'data/stage1_train/'
TEST_PATH = 'data/stage1_test/'

# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

# Read train images and mask return as nump array
def read_train_data(IMG_WIDTH=256, IMG_HEIGHT=256, IMG_CHANNELS=3):
	X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
	Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
	print('Getting and resizing train images and masks ... ')
	sys.stdout.flush()

	if os.path.isfile('train_img.npy') and os.path.isfile('train_mask.npy'):
		print("Training data loaded from memory")
		X_train = np.load('train_img')
		Y_train = np.load('train_mask')
		return X_train, Y_train

	pbar = Progbar(len(train_ids))
	for img_num, id_ in enumerate(train_ids):
		path = os.path.join(TRAIN_PATH, id_)
		img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
		# 3 channels	
		img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
		# 3 channels resized
		X_train[img_num] = img

		mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
		for mask_file in next(os.walk(path + '/masks/'))[2]:
			mask_ = imread(os.path.join(path + '/masks/', mask_file))
			# shape: (IMG_HEIGHT, IMG_WIDTH)
			mask_ = resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
			# shape: (IMG_HEIGHT, IMG_WIDTH, 1)
			mask_ = np.expand_dims(mask_, axis=-1)
			mask = np.maximum(mask, mask_)
		Y_train[img_num] = mask
		pbar.update(img_num)
	np.save("train_img", X_train)
	np.save("train_mask", Y_train)
	return X_train, Y_train

read_train_data()