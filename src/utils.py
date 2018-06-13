import warnings
import numpy as np
import pandas as pd
import random
import os
import sys
import cv2
import random

import settings

from model import build_unet, dice_coef, mean_iou
from keras.models import load_model

from skimage import img_as_ubyte
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize, warp, AffineTransform, rotate
from skimage.util import random_noise, invert
from keras.utils import Progbar
from skimage.morphology import label
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

from matplotlib import pyplot as plt
from skimage import io

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

antialias_flag = False 

# Setting seed for reproducability
seed = 42
random.seed = seed
np.random.seed = seed

settings.init()
train_ids = settings.train_ids
test_ids  = settings.test_ids 

class ReadData(object):
    def __init__(self, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=3):
        self.img_height   = IMG_HEIGHT
        self.img_width    = IMG_WIDTH
        self.img_channels = IMG_CHANNELS
        self.X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
        self.Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        self.X_test  = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
        self.test_sizes = []

    # Read train images and mask and return as nump array
    def train_data(self):
        print('\nGetting and resizing train images and masks ... ')
        sys.stdout.flush()

        if os.path.isfile('train_img.npy') and os.path.isfile('train_mask.npy'):
            print("Training data loaded from memory")
            self.X_train = np.load('train_img.npy')
            self.Y_train = np.load('train_mask.npy')
            return

        pbar = Progbar(len(train_ids))
        for img_num, id_ in enumerate(train_ids):
            path = os.path.join(settings.TRAIN_PATH, id_)
            # pick color channels, ignore alpha
            img = imread(path + '/images/' + id_ + '.png')[:, :, :self.img_channels]
            # 3 channels	
            img = resize(img, (self.img_height, self.img_width), mode='constant', preserve_range=True, anti_aliasing=antialias_flag)
            # 3 channels resized
            self.X_train[img_num] = img

            mask = np.zeros((self.img_height, self.img_width, 1), dtype=np.bool)
            for mask_file in next(os.walk(path + '/masks/'))[2]:
                mask_ = imread(os.path.join(path + '/masks/', mask_file))
                # shape: (IMG_HEIGHT, IMG_WIDTH)
                mask_ = resize(mask_, (self.img_height, self.img_width), mode='constant', preserve_range=True, anti_aliasing=antialias_flag)
                # shape: (IMG_HEIGHT, IMG_WIDTH, 1)
                mask_ = np.expand_dims(mask_, axis=-1)
                mask = np.maximum(mask, mask_)
            self.Y_train[img_num] = mask
            pbar.update(img_num)

        np.save("train_img",  self.X_train)
        np.save("train_mask", self.Y_train)

    # Read test images and return as numpy array
    def test_data(self):
        print('\nGetting and resizing test images ... ')
        sys.stdout.flush()

        if os.path.isfile('test_img.npy') and os.path.isfile('test_sizes.npy'):
            print('Test data loaded from memory')
            self.X_test = np.load('test_img.npy')
            self.test_sizes = np.load('test_sizes.npy')
            return

        pbar = Progbar(len(test_ids))
        for img_num, id_ in enumerate(test_ids):
            path = os.path.join(settings.TEST_PATH, id_)
            img = imread(path + '/images/' + id_ + '.png')
            if len(img.shape) > 2:
                img = img[:, :, :self.img_channels]

            else:
                img = np.stack((img,) * 3, -1)
            self.test_sizes.append([img.shape[0], img.shape[1]])
            img = resize(img, (self.img_height, self.img_width), mode='constant', preserve_range=True, anti_aliasing=antialias_flag)
            self.X_test[img_num] = img
            pbar.update(img_num)
        np.save('test_img',   self.X_test)
        np.save('test_sizes', self.test_sizes)

class Plot(object):
    def __init__(self, data, preds):
        self.X_train = data[0]
        self.Y_train = data[1]
        self.X_test  = data[2]
        self.preds_train = preds[0]
        self.preds_val   = preds[1]
        self.preds_test  = preds[2]
    
    def plot(self, show=False):
        # Perform a sanity check on some random training samples
        ix = random.randint(0, len(self.preds_train) - 1)
        plt.figure(figsize=(10,10))
        plt.subplot(131)
        plt.title('image')
        io.imshow(self.X_train[ix])
        plt.subplot(132)
        plt.title('mask')
        io.imshow(np.squeeze(self.Y_train[ix]))
        plt.subplot(133)
        plt.title('prediction')
        io.imshow(np.squeeze(self.preds_train[ix]))
        plt.savefig('train_{:d}'.format(ix)+'.jpg', bbox_inches='tight')

        ix = random.randint(0, len(self.preds_train) - 1) 
        plt.figure(figsize=(10,10))
        plt.subplot(131)
        plt.title('image')
        io.imshow(self.X_train[ix])
        plt.subplot(132)
        plt.title('mask')
        io.imshow(np.squeeze(self.Y_train[ix]))
        plt.subplot(133)
        plt.title('prediction')
        io.imshow(np.squeeze(self.preds_train[ix]))
        plt.savefig('train_{:d}'.format(ix)+'.jpg', bbox_inches='tight')

        # Perform a sanity check on some random validation samples
        ix = random.randint(0, len(self.preds_val) - 1)
        plt.figure(figsize=(10,10))
        plt.subplot(131)
        plt.title('image')
        io.imshow(self.X_train[ix])
        plt.subplot(132)
        plt.title('mask')
        io.imshow(np.squeeze(self.Y_train[ix]))
        plt.subplot(133)
        plt.title('prediction')
        io.imshow(np.squeeze(self.preds_train[ix]))
        plt.savefig('valid_{:d}'.format(ix)+'.jpg', bbox_inches='tight')

        ix = random.randint(0, len(self.preds_val) - 1)
        plt.figure(figsize=(10,10))
        plt.subplot(131)
        plt.title('image')
        io.imshow(self.X_train[ix])
        plt.subplot(132)
        plt.title('mask')
        io.imshow(np.squeeze(self.Y_train[ix]))
        plt.subplot(133)
        plt.title('prediction')
        io.imshow(np.squeeze(self.preds_train[ix]))
        plt.savefig('valid_{:d}'.format(ix)+'.jpg', bbox_inches='tight')

        # Perform a sanity check on some random test samples
        ix = random.randint(0, len(self.preds_test) - 1)
        plt.figure(figsize=(10,10))
        plt.subplot(121)
        plt.title('image')
        io.imshow(self.X_test[ix])
        plt.subplot(122)
        plt.title('prediction')
        io.imshow(np.squeeze(self.preds_test[ix]))
        plt.savefig('test_{:d}'.format(ix)+'.jpg', bbox_inches='tight')

        ix = random.randint(0, len(self.preds_test) - 1)
        plt.figure(figsize=(10,10))
        plt.subplot(121)
        plt.title('image')
        io.imshow(self.X_test[ix])
        plt.subplot(122)
        plt.title('prediction')
        io.imshow(np.squeeze(self.preds_test[ix]))
        plt.savefig('test_{:d}'.format(ix)+'.jpg', bbox_inches='tight')

        if show:
            plt.show()

class Enhance(object):
    def __init__(self, X_in, Y_in):
        self.X_in = X_in
        self.Y_in = Y_in
        self.X_out = None
        self.Y_out = None

    def enhance(self):
        if os.path.isfile('enhanced_img.npy') and os.path.isfile('enhanced_mask.npy'):
            print('Enhanced data loaded from memory')
            self.X_out = np.load('enhanced_img.npy')
            self.Y_out = np.load('enhanced_mask.npy')

        X_orig = self.X_in
        Y_orig = self.Y_in

        #elt_imgs, elt_labels = eltransform_images(X_orig, Y_orig)
        #X_blr, Y_blr = blur_images(X_orig, Y_orig)
        X_crop, Y_crop     = self.__crop()
        hrz_flp, vrt_flp   = self.__flip()
        X_aft, Y_aft       = self.__affine()
        X_rot90, Y_rot90   = self.__rotate(90)
        X_rot180, Y_rot180 = self.__rotate(180)
        X_rot270, Y_rot270 = self.__rotate(270)
        X_inv              = self.__invert() 

        self.X_out = np.concatenate((X_orig, vrt_flp[0], hrz_flp[0], X_aft, X_rot90, X_rot180, X_rot270, X_inv,  X_crop))
        self.Y_out = np.concatenate((Y_orig, vrt_flp[1], hrz_flp[1], Y_aft, Y_rot90, Y_rot180, Y_rot270, Y_orig, Y_crop))
        
        np.save('enhanced_img',  self.X_out)
        np.save('enhanced_mask', self.Y_out)

    def __invert(self):
        imgs = self.X_in

        num_imgs = imgs.shape[0]
        inverted_imgs = []

        print('\nInverting train data ... ')
        sys.stdout.flush()

        if os.path.isfile('inverted_imgs.npy'):
            print('Inverted data loaded from memory')
            inverted_imgs = np.load('inverted_imgs.npy')
            return inverted_imgs

        pbar = Progbar(num_imgs)
        for idx in range(num_imgs):
            img = imgs[idx]
            img = invert(img)
            inverted_imgs.append(img)
            pbar.update(idx)
        
        inverted_imgs = np.array(inverted_imgs)
        np.save("inverted_imgs",inverted_imgs)
       
        return inverted_imgs

    def __blur(self):
        imgs   = self.X_in
        labels = self.Y_in

        num_imgs = imgs.shape[0]
        blur_imgs = []
        blur_labels = []
        labels.dtype = np.uint8

        print('\nBluring train data ... ')
        sys.stdout.flush()
        
        save_name = 'blur_imgs.npz'
        if os.path.isfile(save_name):
            print('blured data loaded from memory')
            blur = np.load(save_name)
            blur_imgs   = blur['blur_imgs']
            blur_labels = blur['blur_labels']
            return [blur_imgs, blur_labels]

        pbar = Progbar(num_imgs)
        for idx in range(num_imgs):
            img   = imgs[idx]
            label = labels[idx]

            sigma = img.shape[1] * 0.05/4
            blur_size = int(2 * sigma) | 1 

            img   = cv2.GaussianBlur(img, ksize=(blur_size, blur_size), sigmaX=sigma)
            label = cv2.GaussianBlur(label, ksize=(blur_size, blur_size), sigmaX=sigma)

            blur_imgs.append(img)
            blur_labels.append(label)

            pbar.update(idx)
       
        blur_imgs = np.array(blur_imgs)
        blur_imgs = img_as_ubyte(blur_imgs)

        blur_labels = np.array(blur_labels) 
        blur_labels = np.expand_dims(blur_labels, axis=-1)
        blur_labels.dtype = np.bool

        labels.dtype = np.bool

        np.savez(save_name, blur_imgs=blur_imgs, blur_labels=blur_labels)
     
        return [blur_imgs, blur_labels]

     
    def __crop(self, crop_rate=0.7, IMG_WIDTH=256, IMG_HEIGHT=256, IMG_CHANNELS=3):
        imgs   = self.X_in
        labels = self.Y_in
        num_imgs = imgs.shape[0]
        crp_imgs = []
        crp_labels = []

        print('\nCropping train data with rate {:.2f} ...'.format(crop_rate))
        sys.stdout.flush()
        
        save_name = 'crop_{:d}'.format(int(crop_rate*100))+'.npz'
        if os.path.isfile(save_name):
            print('{:.2f} rate cropped data loaded from memory'.format(crop_rate))
            crp = np.load(save_name)
            crp_imgs   = crp['crp_imgs']
            crp_labels = crp['crp_labels']
            return [crp_imgs, crp_labels]

        pbar = Progbar(num_imgs)
        for idx in range(num_imgs):
            img   = imgs[idx]
            label = labels[idx]

            size = img.shape[0]
            csize = random.randint(np.floor(crop_rate * size), size)
            w_c = random.randint(0, size - csize)
            h_c = random.randint(0, size - csize)

            img   = img[w_c:w_c + size, h_c:h_c + size, :]
            label = label[w_c:w_c + size, h_c:h_c + size, :]

            img   = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=False, anti_aliasing=antialias_flag)
            label = resize(label, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=False, anti_aliasing=antialias_flag)

            img   = img_as_ubyte(img)
            label = img_as_ubyte(label)
        
            label.dtype = np.bool

            crp_imgs.append(img)
            crp_labels.append(label)

            pbar.update(idx)
       
        crp_imgs   = np.array(crp_imgs)
        crp_labels = np.array(crp_labels) 

        np.savez(save_name, crp_imgs=crp_imgs, crp_labels=crp_labels)
     
        return [crp_imgs, crp_labels]
        
    def __rotate(self, angle):
        imgs   = self.X_in
        labels = self.Y_in

        num_imgs = imgs.shape[0]
        rot_imgs = []
        rot_labels = []

        print('\nRotating train data for {:d} degrees  ...'.format(angle))
        sys.stdout.flush()
        
        save_name = 'rotate_{:d}'.format(angle)+'.npz'
        if os.path.isfile(save_name):
            print('{:d} degree rotated data loaded from memory'.format(angle))
            rot = np.load(save_name)
            rot_imgs   = rot['rot_imgs']
            rot_labels = rot['rot_labels']
            return [rot_imgs, rot_labels]

        pbar = Progbar(num_imgs)
        for idx in range(num_imgs):
            img   = imgs[idx]
            label = labels[idx]

            img   = rotate(img, angle)
            label = rotate(label, angle)

            rot_imgs.append(img)
            rot_labels.append(label)

            pbar.update(idx)
       
        rot_imgs = np.array(rot_imgs)
        rot_imgs = img_as_ubyte(rot_imgs)

        rot_labels = np.array(rot_labels) 
        rot_labels = img_as_ubyte(rot_labels)
        rot_labels.dtype = np.bool

        np.savez(save_name, rot_imgs=rot_imgs, rot_labels=rot_labels)
     
        return [rot_imgs, rot_labels]

    def __flip(self):
        imgs   = self.X_in
        labels = self.Y_in
        
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

    def __affine(self):
        imgs   = self.X_in
        labels = self.Y_in        
        
        num_imgs = imgs.shape[0]
        aft_imgs = []
        aft_labels = []

        print('\nAffine trsaforming train data ... ')
        sys.stdout.flush()

        if os.path.isfile('aft.npz'):
            print('Affine transformed data loaded from memory')
            aft = np.load('aft.npz')
            aft_imgs   = aft['aft_imgs']
            aft_labels = aft['aft_labels']
            return [aft_imgs, aft_labels]

        affine_tf = AffineTransform(shear=0.15, rotation=(0./180)*np.pi)
        
        pbar = Progbar(num_imgs)
        for idx in range(num_imgs):
            img   = imgs[idx]
            label = labels[idx]

            img   = warp(img, inverse_map=affine_tf, mode='edge')
            label = warp(label, inverse_map=affine_tf, mode='edge')

            aft_imgs.append(img)
            aft_labels.append(label)

            pbar.update(idx)
       
        aft_imgs = np.array(aft_imgs)
        aft_imgs = img_as_ubyte(aft_imgs)

        aft_labels = np.array(aft_labels) 
        aft_labels = img_as_ubyte(aft_labels)
        aft_labels.dtype = np.bool

        np.savez("aft", aft_imgs=aft_imgs, aft_labels=aft_labels)
     
        return [aft_imgs, aft_labels]

    def __elastic_transform(self, image, alpha, sigma, random_state=None):
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

    def __eltransform(self):
        imgs   = self.X_in
        labels = self.Y_in

        num_imgs = imgs.shape[0]
        labels.dtype = np.uint8
        
        elt_imgs = []
        elt_labels = []

        print('\nPerforming elastic transform on train data ... ')
        sys.stdout.flush()

        if os.path.isfile('elstc.npz'):
            print('Elastic trsformed data loaded from memory')
            elstc = np.load('elstc.npz')
            elt_imgs   = elstc['elt_imgs'] 
            elt_labels = elstc['elt_labels']
            return elt_imgs, elt_labels

        pbar = Progbar(imgs.shape[0])
        for idx in range(num_imgs):
            img = imgs[idx]
            label = labels[idx]

            alpha = img.shape[1] * 1
            sigma = img.shape[1] * 0.05
            
            img_elt   = self.__elastic_transform(img, alpha, sigma)
            label_elt = self.__elastic_transform(label, alpha, sigma)

            elt_imgs.append(img_elt)
            elt_labels.append(label_elt)

            pbar.update(idx)

        elt_imgs = np.array(elt_imgs)
        elt_labels = np.array(elt_labels)

        elt_labels.dtype = np.bool
        labels.dtype = np.bool

        np.savez("elstc", elt_imgs=elt_imgs, elt_labels=elt_labels)
        
        return elt_imgs, elt_labels

    def __add_noise(self):
        imgs = self.X_in

        num_imgs = imgs.shape[0]
        noisy_imgs = []

        print('\nAdding noise on train data ... ')
        sys.stdout.flush()

        if os.path.isfile('noisy_imgs.npy'):
            print('Noisy data loaded from memory')
            noisy_imgs = np.load('noisy_imgs.npy')
            return noisy_imgs

        pbar = Progbar(num_imgs)
        for idx in range(num_imgs):
            img = imgs[idx]
            img = random_noise(img)
            noisy_imgs.append(img)
            pbar.update(idx)
        
        noisy_imgs = np.array(noisy_imgs)
        np.save("noisy_imgs",noisy_imgs)
       
        return noisy_imgs

class ImageSegment(object):
    def __init__(self, model_name, X_train, Y_train, X_test, test_sizes):
        self.model_name = model_name
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test  = X_test
        self.test_sizes = test_sizes
        self.model = None
        self.train_results = None
        self.preds_train = None
        self.preds_val = None
        self.preds_test = None
        self.preds_test_resized = []

    def run(self):
        self.__train()
        self.__predict()

    def __train(self):
        if os.path.isfile(self.model_name):
            self.model = load_model(self.model_name, custom_objects={'mean_iou': mean_iou})
            #model = load_model(model_name, custom_objects={'dice_coef': dice_coef})
        else:
            # get u-net model
            self.model = build_unet()
            # train model
            print("\nTraining ...")
            earlystopper = EarlyStopping(patience=5, verbose=1)
            checkpointer = ModelCheckpoint(self.model_name, verbose=1, save_best_only=True)
            self.train_results = self.model.fit(self.X_train, self.Y_train, validation_split=0.1, batch_size=4, epochs=50,
                    callbacks=[earlystopper, checkpointer])

    def __predict(self):
        # Predict using test data
        print("\nPredicitng ...")
        if os.path.isfile('preds_train.npy') : 
            print("Train prediction data loaded from memory ...")
            self.preds_train = np.load('preds_train.npy')
        else:
            self.preds_train = self.model.predict(self.X_train[:int(self.X_train.shape[0]*0.9)], verbose=1)
            # Threshold predictions
            self.preds_train = (self.preds_train > 0.5).astype(np.bool)
            np.save('preds_train.npy', self.preds_train)

        if os.path.isfile('preds_val.npy'): 
           print("Validation prediction data loaded from memory ...")
           self.preds_val = np.load('preds_val.npy')
        else:
           self.preds_val = self.model.predict(self.X_train[int(self.X_train.shape[0]*0.9):], verbose=1)
           # Threshold predictions
           self.preds_val = (self.preds_val > 0.5).astype(np.bool)
           np.save('preds_val.npy', self.preds_val)
                        
        if os.path.isfile('preds_test.npy'): 
           print("Test prediction data loaded from memory ...")
           self.preds_test = np.load('preds_test.npy')
        else:
           self.preds_test = self.model.predict(self.X_test, verbose=1)
           # Threshold predictions
           self.preds_test = (self.preds_test > 0.5).astype(np.bool)
           np.save('preds_test.npy', self.preds_test)

        # Create list of resized test masks
        print("\nResizing test data ...")
        for idx in range(len(self.X_test)):
            this_mask = self.preds_test[idx]
            this_size = self.test_sizes[idx]
            self.preds_test_resized.append(resize(np.squeeze(this_mask), (this_size[0], this_size[1]), mode='constant', preserve_range=True, anti_aliasing=antialias_flag))

class RunLengthEncoder(object):
    def __init__(self, masks, ids):
        '''
        masks: list of masks
        ids:   list of ids
        '''
        self.masks   = masks
        self.ids     = ids
        self.ids_per_mask = []
        self.rles    = []
        self.masks_resized = []
        self.submit  = pd.DataFrame()

    def run(self):
        self.__allmasks_to_rles()
        print("\nCreating submit file ...")
        self.__to_csv()

    # Create submission data frame
    def __to_csv(self):
        self.submit['ImageId']       = self.ids_per_mask
        self.submit['EncodedPixels'] = pd.Series(self.rles).apply(lambda x: ' '.join(str(y) for y in x))
        self.submit.sort_values(by=['ImageId'], inplace=True)
        self.submit.to_csv('submit-dsbowol-2018.csv', index=False)

    # Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
    def __rl_encoder(self, x):
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
            run_length[-1] += 1
            prev = b
        return run_length

    def __mask_to_rles(self, x, cutoff=0.5):
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
            yield self.__rl_encoder(labeled_img == seg_id)

    # Iterate over the test IDs and generate run-length encodings for each 
    # seperate mask identified by skimage
    def __allmasks_to_rles(self):
        for num, id_ in enumerate(self.ids):
            rle = list(self.__mask_to_rles(self.masks[num]))
            self.rles.extend(rle)
            self.ids_per_mask.extend([id_] * len(rle))

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
