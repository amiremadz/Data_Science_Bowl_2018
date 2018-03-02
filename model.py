from keras.layers import Input
from keras.layers.core import Lambda, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import Model
from keras import backend as KBE

# Metric function
def dice_coef(y_true, y_pred, smooth=1.0):
	y_true_ = KBE.flatten(y_true)
	y_pred_ = KBE.flatten(y_pred)
	intersection = KBE.sum(y_true_ * y_pred_)
	return (2.0 * intersection + smooth) / (KBE.sum(y_true_) + KBE.sum(y_pred_) + smooth)

# Loss funtion
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def build_unet(IMG_WIDTH=256, IMG_HEIGHT=256, IMG_CHANNELS=3):
	inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
	s = Lambda(lambda x: x / 255) (inputs)									# normalize features
																			# relu
	c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
	c1 = Dropout(0.1) (c1)
	c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
	p1 = MaxPooling2D((2, 2)) (c1)

	c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
	c2 = Dropout(0.1) (c2)
	c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
	p2 = MaxPooling2D((2, 2)) (c2)

	c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
	c3 = Dropout(0.2) (c3)
	c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
	p3 = MaxPooling2D((2, 2)) (c3)

	c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
	c4 = Dropout(0.2) (c4)
	c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
	p4 = MaxPooling2D((2, 2)) (c4)

	c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
	c5 = Dropout(0.3) (c5)
	c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

	u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
	u6 = concatenate([u6, c4])												# switch order
	c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
	c6 = Dropout(0.2) (c6)
	c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

	u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
	u7 = concatenate([u7, c3])												# switch order
	c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
	c7 = Dropout(0.2) (c7)
	c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

	u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
	u8 = concatenate([u8, c2])												# switch order
	c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
	c8 = Dropout(0.1) (c8)
	c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

	u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
	u9 = concatenate([u9, c1])												# switch order
	c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
	c9 = Dropout(0.1) (c9)
	c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

	outputs = Conv2D(1, (1, 1), activation='relu') (c9)						# sigmoid

	model = Model(inputs=[inputs], outputs=[outputs])
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])

	return model













