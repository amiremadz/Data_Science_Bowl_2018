import warnings
import numpy as np
import random
import os
import sys
import cv2

from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from keras.utils import Progbar
from skimage.morphology import label
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

antialias_flag = False 

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
        X_train = np.load('train_img.npy')
        Y_train = np.load('train_mask.npy')
        return X_train, Y_train

    pbar = Progbar(len(train_ids))
    for img_num, id_ in enumerate(train_ids):
        path = os.path.join(TRAIN_PATH, id_)
        # pick color channels, ignore alpha
        img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
        # 3 channels	
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True, anti_aliasing=antialias_flag)
        # 3 channels resized
        X_train[img_num] = img

        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(os.path.join(path + '/masks/', mask_file))
            # shape: (IMG_HEIGHT, IMG_WIDTH)
            mask_ = resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True, anti_aliasing=antialias_flag)
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
    X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    test_sizes = []
    print('\nGetting and resizing test images ... ')
    sys.stdout.flush()

    if os.path.isfile('test_img.npy') and os.path.isfile('test_sizes.npy'):
        print('Test data loaded from memory')
        X_test = np.load('test_img.npy')
        test_sizes = np.load('test_sizes.npy')
        return X_test, test_sizes

    pbar = Progbar(len(test_ids))
    for img_num, id_ in enumerate(test_ids):
        path = os.path.join(TEST_PATH, id_)
        img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
        test_sizes.append([img.shape[0], img.shape[1]])
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True, anti_aliasing=antialias_flag)
        X_test[img_num] = img
        pbar.update(img_num)
    np.save('test_img', X_test)
    np.save('test_sizes', test_sizes)
    return X_test, test_sizes

def read_images(tot=3, IMG_WIDTH=256, IMG_HEIGHT=256, IMG_CHANNELS=3):
    X = np.zeros((tot, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y = np.zeros((tot, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for img_num, id_ in enumerate(train_ids[:tot]):
        id_ = train_ids[0]
        path = os.path.join(TRAIN_PATH, id_)
        # pick color channels, ignore alpha
        img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True, ant_aliasing=antialias_flag)
        X[img_num] = img

        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(os.path.join(path + '/masks/', mask_file))
            # shape: (IMG_HEIGHT, IMG_WIDTH)
            mask_ = resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True, anti_aliasing=antialias_flag)
            # shape: (IMG_HEIGHT, IMG_WIDTH, 1)
            mask_ = np.expand_dims(mask_, axis=-1)
            mask = np.maximum(mask, mask_)
        Y[img_num] = mask
    return X, Y

# Define function to draw a grid
def draw_grid(image, grid_size):
    """
    image: numpy array of shape (height, width, channels)
    grid_size: int
    """
    # Draw grid lines
    for i in range(0, image.shape[1], grid_size):
        cv2.line(image, (i, 0), (i, image.shape[0]), color=(255, 255, 255))
    for j in range(0, image.shape[0], grid_size):
        cv2.line(image, (0, j), (image.shape[1], j), color=(255, 255, 255))

def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
       image: numpy array of shape(height, width, cannels)
       alpha: float
       sigma: float
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape[:2]

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
    
    image_elastic = np.zeros(image.shape, dtype=np.uint8)
    for idx_ch in range(image.shape[2]):
    	image_elastic[:, :, idx_ch] = map_coordinates(image[:, :, idx_ch], indices, order=1, mode='reflect').reshape(shape)
    return image_elastic

def eltransform_images(imgs, labels):
    num_imgs = imgs.shape[0]
    labels.dtype = np.uint8
    
    elt_imgs = []
    elt_labels = []

    print('\nPerforming elastic transform on train data ... ')
    sys.stdout.flush()

    if os.path.isfile('train_img_elt.npy') and os.path.isfile('train_mask_elt.npy'):
        print('Train data loaded from memory')
        imgs_elt = np.load('train_img_elt.npy')
        labels_elt = np.load('train_mask_elt.npy')
        return imgs_elt, labels_elt

    pbar = Progbar(imgs.shape[0])
    for idx in range(num_imgs):
        img = imgs[idx]
        label = labels[idx]

        alpha = img.shape[1] * 2
        sigma = img.shape[1] * 0.08
        img_elt = elastic_transform(img, alpha, sigma)
        label_elt = elastic_transform(label, alpha, sigma)

        elt_imgs.append(img_elt)
        elt_labels.append(label_elt)

        pbar.update(idx)

    elt_imgs = np.array(elt_imgs)
    elt_labels = np.array(elt_labels)

    imgs_elt = np.concatenate((imgs, elt_imgs))
    labels_elt = np.concatenate((labels, elt_labels))
    labels_elt.dtype = np.bool

    np.save("train_img_elt", imgs_elt)
    np.save("train_mask_elt", labels_elt)
    return imgs_elt, labels_elt

def flip_images(imgs, labels):
    num_imgs = imgs.shape[0]
    labels = np.squeeze(labels)
    labels.dtype = np.uint8
    
    vrt_imgs = []
    hrz_imgs = []
    vrt_labels = []
    hrz_labels = []

    print('\nPerforming horizental and vertical flipping on train data ... ')
    sys.stdout.flush()

    if os.path.isfile('flp_vrt.npz') and os.path.isfile('flp_hrz.npz'):
        print('Flipped data loaded from memory')
        flp_vrt = np.load('flp_vrt.npz')
        vrt_imgs   = flp_vrt['vrt_imgs']
        vrt_labels = flp_vrt['vrt_labels']

        flp_hrz = np.load('flp_hrz.npz')
        hrz_imgs   = flp_hrz['hrz_imgs']
        hrz_labels = flp_hrz['hrz_labels'] 
        
        return [hrz_imgs, hrz_labels], [vrt_imgs, vrt_labels]

    pbar = Progbar(num_imgs)
    for idx in range(num_imgs):
        img = imgs[idx]
        label = labels[idx]

        img_hf = cv2.flip(img, 0)
        img_vf = cv2.flip(img, 1)

        label_hf = cv2.flip(label, 0)
        label_vf = cv2.flip(label, 1)

        hrz_imgs.append(img_hf)
        vrt_imgs.append(img_vf)

        vrt_labels.append(label_vf)
        hrz_labels.append(label_hf)

        pbar.update(idx)
    
    vrt_imgs   = np.array(vrt_imgs)
    hrz_imgs   = np.array(hrz_imgs)

    vrt_labels = np.array(vrt_labels)
    hrz_labels = np.array(hrz_labels)
    
    labels     = np.expand_dims(labels, axis=-1)
    vrt_labels = np.expand_dims(vrt_labels, axis=-1)
    hrz_labels = np.expand_dims(hrz_labels, axis=-1)

    labels.dtype     = np.bool
    vrt_labels.dtype = np.bool
    hrz_labels.dtype = np.bool

    np.savez("flp_vrt", vrt_imgs=vrt_imgs, vrt_labels=vrt_labels)
    np.savez("flp_hrz", hrz_imgs=hrz_imgs, hrz_labels=hrz_labels)
    
    return [hrz_imgs, hrz_labels], [vrt_imgs, vrt_labels]

# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rl_encoder(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length encodings as list
    '''
    dots = np.where(x.T.flatten() == 1)[0]	#.T sets Fortran order down-then-right 
    run_length = []
    prev = -2
    for b in dots:
        if b > (prev + 1):
                run_length.extend((b + 1, 0))
                #print(b + 1)
        run_length[-1] += 1
        prev = b
    return run_length

def mask_to_rles(x, cutoff=0.5):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Yields run length for all segments in image
    '''	
    # Label connected regions of an integer array.
    # background: 0
    labeled_img = label(x > cutoff)
    #if labeled_img.max() < 1:
    #    labeled_img[0, 0] = 1 # ensure at least one prediction per image
    
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
    return test_ids_new, rles

def train_masks_to_rles(train_masks):
    train_masks = train_masks.astype('int')
    train_ids_new = []
    rles = []
    for num, id_ in enumerate(train_ids):
        rle = list(mask_to_rles(train_masks[num]))
        rles.extend(rle)
        train_ids_new.extend([id_] * len(rle))
    return train_ids_new, rles
