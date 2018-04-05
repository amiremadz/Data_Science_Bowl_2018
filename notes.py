# save model

def scale_img_canals(an_img):
for i in range(IMG_CHANNELS):
    canal = an_img[:,:,i]
    canal = canal - canal.min()
    canalmax = canal.max()
    if canalmax &gt; 0:
        factor = 255/canalmax
        canal = (canal * factor).astype(int)
    an_img[:,:,i] = canal

resize anti-aliasing


def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = label(y_true_in > 0.5)
    y_pred = label(y_pred_in > 0.5)
    
    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.array(np.mean(metric), dtype=np.float32)

def my_iou_metric(label, pred):
    metric_value = tf.py_func(iou_metric_batch, [label, pred], tf.float32)
    return metric_value


Thanks for sharing. You should be able to get a faster version by just using more OpenCV functions instead of SciPy. I managed to get about 4x improvement by using:

# include 4 standard deviations in the kernel (the default for ndimage.gaussian_filter)
# OpenCV also requires an odd size for the kernel hence the "| 1" part
blur_size = int(4*sigma) | 1
cv2.GaussianBlur(image, ksize=(blur_size, blur_size), sigmaX=sigma)
instead of ndimage.gaussian_filter(image, sigma)

and cv2.remap(image, dx, dy, interpolation=cv2.INTER_LINEAR) instead of ndimage.map_coordinates(image, (dx, dy), order=1)


# I've tried flipping and random cropping. I haven't tried random rotation because 
#there would be interpolation on the masks. It improved the performance of my model 
#quite a lot. I'm planning to do elastic transformations in the future as well as finding 
#a good way to do random rotation. I created a short kernel here. I did all the preprocessing 
#done in this kernel as well as flipping and a small amount of random 
#cropping(the output size limited the augmentation).

x = np.random.randn(5,5)
x = (x > 0)
x = x.astype(np.int8)
print(x)
print(rl_encoder(x))


#X, Y = read_images()
X, Y = read_train_data()
X_flp, Y_flp = flip_images(X, Y)
X_els, Y_els = eltransform_images(X_flp, Y_flp)


from skimage import data

X = data.camera()
draw_grid(X, 50)
X = np.expand_dims(X, axis=-1)

X_els = elastic_transform(X, X.shape[1] * 2, X.shape[1] * 0.05, IMG_WIDTH=512, IMG_HEIGHT=512, IMG_CHANNELS=1)
X_els = np.squeeze(X_els)
X = np.squeeze(X)

import skimage.io as io

imshow(X)
io.show()
imshow(X_els)
io.show()

X = X[0]
draw_grid(X, 50)

X_els = elastic_transform(X, X.shape[1] * 2, X.shape[1] * 0.08)
X_els = np.squeeze(X_els)

import skimage.io as io

imshow(X)
io.show()
imshow(X_els)
io.show()