import os
import numpy as np
import pandas as pd

from utils import read_train_data, read_test_data, flip_images, eltransform_images, allmasks_to_rles, train_masks_to_rles, draw_grid, elastic_transform, add_noise, affine_transform, rotate_images, invert_images, blur_images
from model import build_unet, dice_coef, mean_iou
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from skimage.transform import resize, warp, AffineTransform, rotate
from skimage import io, img_as_ubyte
from skimage.util import invert
from matplotlib import pyplot as plt
import random
from cv2 import GaussianBlur

# get train train data
X_train, Y_train = read_train_data()

if 1:
    X_b, Y_b = blur_images(X_train, Y_train)
    ix = 2
    img = X_train[ix]
    label = np.squeeze(Y_train[ix])
    img_tf   = X_b[ix] 
    label_tf = Y_b[ix]

    plt.figure(figsize=(8, 8))
    plt.subplot(221)
    plt.title('image')
    io.imshow(img)
    plt.subplot(222)
    plt.title('label')
    io.imshow(label)
    plt.subplot(223)
    plt.title('blur')
    io.imshow(img_tf)
    plt.subplot(224)
    plt.title('blur')
    io.imshow(label_tf)

if 0:
    ix = 2
    img = X_train[ix]
    label = np.squeeze(Y_train[ix])
    sigma = img.shape[1] * 0.05/4
    blur_size = int(2 * sigma) | 1 
    img_tf   = GaussianBlur(img, ksize=(blur_size, blur_size), sigmaX=sigma)
    label.dtype = np.uint8
    label_tf = GaussianBlur(label, ksize=(blur_size, blur_size), sigmaX=sigma)
    label.dtype = np.bool
    label_tf.dtype= np.bool

    plt.figure(figsize=(8, 8))
    plt.subplot(221)
    plt.title('image')
    io.imshow(img)
    plt.subplot(222)
    plt.title('label')
    io.imshow(label)
    plt.subplot(223)
    plt.title('blur')
    io.imshow(img_tf)
    plt.subplot(224)
    plt.title('blur')
    io.imshow(label_tf)

if 0:
    X_inv = invert_images(X_train)
    ix = 2
    img = X_train[ix]
    label = np.squeeze(Y_train[ix])
    img_tf   = X_inv[ix]
    label_tf = label 
    plt.figure(figsize=(8, 8))
    plt.subplot(221)
    plt.title('image')
    io.imshow(img)
    plt.subplot(222)
    plt.title('label')
    io.imshow(label)
    plt.subplot(223)
    plt.title('invert')
    io.imshow(img_tf)
    plt.subplot(224)
    plt.title('invert')
    io.imshow(label_tf)

if 0:
    ix = 2
    img = X_train[ix]
    label = np.squeeze(Y_train[ix])
    img_tf   = invert(img)
    label_tf = label 
    plt.figure(figsize=(8, 8))
    plt.subplot(221)
    plt.title('image')
    io.imshow(img)
    plt.subplot(222)
    plt.title('label')
    io.imshow(label)
    plt.subplot(223)
    plt.title('invert')
    io.imshow(img_tf)
    plt.subplot(224)
    plt.title('invert')
    io.imshow(label_tf)

if 0:
    ix = 367
    angle = 270 
    img = X_train[ix]
    label = np.squeeze(Y_train[ix])
    X_rot, Y_rot = rotate_images(X_train, Y_train, angle)
    img_tf   = X_rot[ix]
    label_tf = np.squeeze(Y_rot[ix])
    plt.figure(figsize=(8, 8))
    plt.subplot(221)
    plt.title('image')
    io.imshow(img)
    plt.subplot(222)
    plt.title('label')
    io.imshow(label)
    plt.subplot(223)
    plt.title('rotate')
    io.imshow(img_tf)
    plt.subplot(224)
    plt.title('rotate')
    io.imshow(label_tf)

if 0:
    ix = 367
    angle = 90
    img = X_train[ix]
    label = np.squeeze(Y_train[ix])
    label.dtype = np.uint8
    img_tf   = rotate(img, angle)
    img_tf = img_as_ubyte(img_tf)
    label_tf = rotate(label, angle) 
    label_tf = img_as_ubyte(label_tf)
    label_tf.dtype = np.bool
    plt.figure(figsize=(8, 8))
    plt.subplot(221)
    plt.title('image')
    io.imshow(img)
    plt.subplot(222)
    plt.title('label')
    io.imshow(label)
    plt.subplot(223)
    plt.title('affine')
    io.imshow(img_tf)
    plt.subplot(224)
    plt.title('affine')
    io.imshow(label_tf)

if 0:
    X_aft, Y_aft = affine_transform(X_train, Y_train)
    ix = 436
    img   = X_train[ix]
    label = np.squeeze(Y_train[ix])
    img_tf   = X_aft[ix] 
    label_tf = np.squeeze(Y_aft[ix])
    plt.figure(figsize=(8, 8))
    plt.subplot(221)
    plt.title('image')
    io.imshow(img)
    plt.subplot(222)
    plt.title('label')
    io.imshow(label)
    plt.subplot(223)
    plt.title('affine')
    io.imshow(img_tf)
    plt.subplot(224)
    plt.title('affine')
    io.imshow(label_tf)

if 0:
    ix = 324
    img = X_train[ix]
    label = np.squeeze(Y_train[ix])
    afine_tf = AffineTransform(shear=0, rotation=(90./180)*np.pi)
    img_tf = warp(img, inverse_map=afine_tf, mode='edge')
    label_tf = warp(label, inverse_map=afine_tf, mode='edge')
    plt.figure(figsize=(8, 8))
    plt.subplot(221)
    plt.title('image')
    io.imshow(img)
    plt.subplot(222)
    plt.title('label')
    io.imshow(label)
    plt.subplot(223)
    plt.title('affine')
    io.imshow(img_tf)
    plt.subplot(224)
    plt.title('affine')
    io.imshow(label_tf)

if 0:
    Y_train = np.squeeze(Y_train)
    noisy_imgs = add_noise(X_train)
    ix = 0
    plt.figure(figsize=(8, 8))
    plt.subplot(221)
    plt.title('image')
    io.imshow(X_train[ix])
    plt.subplot(222)
    plt.title('noisy')
    io.imshow(noisy_imgs[ix])
    plt.subplot(223)
    io.imshow(Y_train[ix])
    plt.subplot(224)
    io.imshow(Y_train[ix])

if 0:
    hrz_flp, vrt_flp = flip_images(X_train, Y_train)
    #ix = random.randint(0, len(Y_train))
    #ix = 249
    #ix = 556
    ix = 557
    #plt.figure(figsize=(10,6))
    plt.figure()
    plt.subplot(231)
    plt.title('image')
    io.imshow(X_train[ix])
    plt.subplot(232)
    plt.title('hrz flipped')
    io.imshow(hrz_flp[0][ix])
    plt.subplot(233)
    plt.title('vrt flipped')
    io.imshow(vrt_flp[0][ix])
    plt.subplot(234)
    io.imshow(Y_train[ix])
    plt.subplot(235)
    io.imshow(np.squeeze(hrz_flp[1][ix]))
    plt.subplot(236)
    io.imshow(np.squeeze(vrt_flp[1][ix]))

if 0:
    elt_imgs, elt_labels = eltransform_images(X_train, Y_train)
    ix = 212
    elt_img = elt_imgs[ix]
    elt_label = elt_labels[ix]
    img = X_train[ix]
    label = Y_train[ix]
    #draw_grid(img, 50)
    alpha = img.shape[1] * 1
    sigma = img.shape[1] * 0.04
    elt_img   = elt_imgs[ix] 
    elt_label = elt_labels[ix]

    Y_train = np.squeeze(Y_train)
    plt.figure(figsize=(8, 8))
    plt.subplot(221)
    plt.title('image')
    io.imshow(X_train[ix])
    plt.subplot(222)
    plt.title('elastic')
    io.imshow(elt_img)
    plt.subplot(223)
    io.imshow(Y_train[ix])
    plt.subplot(224)
    io.imshow(np.squeeze(elt_label))

if 0:
    ix = 267 
    img = X_train[ix]
    label = Y_train[ix]
    label.dtype = np.uint8
    draw_grid(img, 50)
    alpha = img.shape[1] * 1
    sigma = img.shape[1] * 0.05
    elt_img   = elastic_transform(img, alpha, sigma)
    elt_label = elastic_transform(label, alpha, sigma)
    elt_label.dtype = np.bool

    Y_train = np.squeeze(Y_train)
    plt.figure(figsize=(8, 8))
    plt.subplot(221)
    plt.title('image')
    io.imshow(X_train[ix])
    plt.subplot(222)
    plt.title('elastic')
    io.imshow(elt_img)
    plt.subplot(223)
    io.imshow(Y_train[ix])
    plt.subplot(224)
    io.imshow(np.squeeze(elt_label))

plt.show()


