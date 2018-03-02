import warnings
import numpy as np
import random
import os
import sys

from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from keras.utils import Progbar
from skimage.morphology import label

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

# Read train images and mask and return as nump array
def read_train_data(IMG_WIDTH=256, IMG_HEIGHT=256, IMG_CHANNELS=3):
	X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
	Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
	print('\nGetting and resizing train images and masks ... ')
	sys.stdout.flush()

	if os.path.isfile('train_img.npy') and os.path.isfile('train_mask.npy'):
		print("Training data loaded from memory")
		X_train = np.load('train_img')
		Y_train = np.load('train_mask')
		return X_train, Y_train

	pbar = Progbar(len(train_ids))
	for img_num, id_ in enumerate(train_ids):
		path = os.path.join(TRAIN_PATH, id_)
		# pick color channels, ignore alpha
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

# Read test images and return as numpy array
def read_test_data(IMG_WIDTH=256, IMG_HEIGHT=256, IMG_CHANNELS=3):
	X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.int8)
	test_sizes = []
	print('\nGetting and resizing test images ... ')
	sys.stdout.flush()

	if os.path.isfile('test_img.npy') and os.path.isfile('test_sizes.npy'):
		print('Test data loaded from memory')
		X_test = np.load('test_img.npy')
		test_sizes = np.load('test_sizes')
		return X_test, test_sizes

	pbar = Progbar(len(test_ids))
	for img_num, id_ in enumerate(test_ids):
		path = os.path.join(TEST_PATH, id_)
		img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
		test_sizes.append([img.shape[0], img.shape[1]])
		img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
		X_test[img_num] = img
		pbar.update(img_num)
	np.save('test_img', X_test)
	np.save('test_sizes', test_sizes)
	return X_test, test_sizes

# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rl_encoder(x):
	'''
	x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length encodings as list
	'''
	dots = np.where(x.T.flatten() == 1)[0]	#.T sets Fortran order down-then-right 
	run_length = []
	prev = -2
	print(dots)
	for b in dots:
		if b > (prev + 1):
			run_length.extend([b, 0])
			print(run_length)
		run_length[-1] += 1
		prev = b

	return run_length

def mask_to_rles(x, cutoff=0.5):
	'''
	x: numpy array of shape (height, width), 1 - mask, 0 - background
    Yields run length for all segments in image
	'''	
	labeled_img = label(x > cutoff)
	for seg_id in range(1, labeled_img.max() + 1):
		yield rl_encoder(labeled_img == seg_id)

# Iterate over the test IDs and generate run-length encodings for each 
# seperate mask identified by skimage
def allmasks_to_rles(test_masks):
	test_ids_new = []
	rles = []
	for num, id_ in enumerate(test_ids):
		rle = list(mask_to_rles(test_masks[num]))
		rles.extend(rle)
		test_ids_new.extend([id_] * len(rle))
	return test_ids, rles