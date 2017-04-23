# Script to make hdf5 files from training and test set from own images
import numpy as np
from skimage import io, color, exposure, transform
import pandas as pd
import os
import glob
import h5py

NUM_CLASSES = 43
IMG_SIZE = 64

def preprocess_img(img):
    # Histogram normalization in y
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)

    # central scrop
    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2,
              :]

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))
    return img


def get_class(img_path):
    return int(img_path.split('/')[-2])


if __name__ == '__main__':
    root_dir = 'Traffic_Pictures/Training/'
    imgs = []
    labels = []

    all_img_paths = glob.glob(os.path.join(root_dir, '*/*.jpg'))
    #np.random.shuffle(all_img_paths)
    for img_path in all_img_paths:
        try:
            img = preprocess_img(io.imread(img_path))
            label = get_class(img_path)
            imgs.append(img)
            labels.append(label)
            if len(imgs) % 100 == 0:
                print("Processed Train {}/{}".format(len(imgs),
                                                     len(all_img_paths)))

        except (IOError, OSError):
            print('missed', img_path)
            pass

    X = np.array(imgs, dtype='float32')
    Y = np.array(labels, dtype='uint8')

    with h5py.File('XS.h5', 'w') as hf1:
        hf1.create_dataset('imgs', data=X)
        hf1.create_dataset('labels', data=Y)

    X_test = []
    y_test = []

    root_dir_test = 'Traffic_Pictures/Test/'
    imgs_test = []
    labels_test = []

    all_img_paths_test = glob.glob(os.path.join(root_dir_test, '*/*.jpg'))
    #np.random.shuffle(all_img_paths)
    for img_path_test in all_img_paths_test:
        try:
            img_test = preprocess_img(io.imread(img_path_test))
            label_test = get_class(img_path_test)
            imgs_test.append(img_test)
            labels_test.append(label_test)
            if len(imgs_test) % 100 == 0:
                print("Processed Test {}/{}".format(len(imgs_test),
                                                    len(all_img_paths_test)))

        except (IOError, OSError):
            print('missed', img_path_test)
            pass

    X_test = np.array(imgs_test, dtype='float32')
    y_test = np.array(labels_test, dtype='uint8')

    with h5py.File('XS_test.h5', 'w') as hf2:
        hf2.create_dataset('imgs', data=X_test)
        hf2.create_dataset('labels', data=y_test)
