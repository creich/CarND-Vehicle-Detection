import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from lesson_functions import *
from sklearn.model_selection import train_test_split

import pickle


## load list of training images

cars = []
notcars = []

images = glob.glob('data/vehicles/**/*.png', recursive=True)
for image in images:
    cars.append(image)
images = glob.glob('data/non-vehicles/**/*.png', recursive=True)
for image in images:
    notcars.append(image)

print("found {} car samples and {} non car sample".format(len(cars), len(notcars)))


## ========

# feature parameters
#color_space = 'YCrCb'         # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    # -> 0.9921
#color_space = 'YUV'         # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    # -> getting an error now... 
    # >> hog.py -> invalid value encountered in sqrt -> image = np.sqrt(image)
    # which later leads to...
    # >> ValueError: Input contains NaN, infinity or a value too large for dtype('float64')
    #
    # -> 0.9918 -- using WRONG image value correction
color_space = 'HLS'         # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    # -> 0.9924
orient = 9                  # HOG orientations
pix_per_cell = 8            # HOG pixels per cell
cell_per_block = 2          # HOG cells per block
hog_channel = "ALL"             # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)     # Spatial binning dimensions
hist_bins = 32              # Number of histogram bins
spatial_feat = True         # Spatial features on or off
hist_feat = True            # Histogram features on or off
hog_feat = True             # HOG features on or off


#n_samples = 500
#random_indices = np.random.randint(0, len(cars), n_samples)
#test_cars = np.array(cars)[random_indices]
#random_indices = np.random.randint(0, len(cars), n_samples)
#test_notcars = np.array(notcars)[random_indices]

def save_classifier(svc, X_scaler,
        color_space,
        orient,
        pix_per_cell,
        cell_per_block,
        hog_channel,
        spatial_size,
        hist_bins,
        spatial_feat,
        hist_feat,
        hog_feat
    ):
    ## save classifier and settings
    PICKLE_FILE_NAME = 'classifier_data.p'
    print("saving classifier and metadata to", PICKLE_FILE_NAME)
    pickle_data = {'svc': svc,
                   'X_scaler': X_scaler,
                   'color_space': color_space,
                   'orient': orient,
                   'pix_per_cell': pix_per_cell,
                   'cell_per_block': cell_per_block,
                   'hog_channel': hog_channel,
                   'spatial_size': spatial_size,
                   'hist_bins': hist_bins,
                   'spatial_feat': spatial_feat,
                   'hist_feat': hist_feat,
                   'hog_feat': hog_feat}

    with open(PICKLE_FILE_NAME, 'wb') as f:
        pickle.dump(pickle_data, f)


#TODO write a function that runs feature extaction, learning and testing in a loop, while automatically changing parameters
def train(svc, X_scaler, color_space, orient, pix_per_cell, cell_per_block, hog_channel,
    spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat):


    t = time.time()

    # use all cars
    car_features = extract_features(cars, color_space=color_space, spatial_size=spatial_size,
    # use only n_samples
    #car_features = extract_features(test_cars, color_space=color_space, spatial_size=spatial_size,
                                       hist_bins=hist_bins, orient=orient,
                                       pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                                       spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

    # use all notcars
    notcar_features = extract_features(notcars, color_space=color_space, spatial_size=spatial_size,
    # use only n_samples
    #notcar_features = extract_features(test_notcars, color_space=color_space, spatial_size=spatial_size,
                                          hist_bins=hist_bins, orient=orient,
                                          pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                                          spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)


    print(round(time.time() - t, 2), "seconds to compute features...")

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)
        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X_train)
    # Apply the scaler to X
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)


    # Use a linear SVC 
    svc = LinearSVC()
 
    print('config data:')
    print('svc =', svc)
    print('X_scaler =', X_scaler)
    print('color_space', color_space)
    print('orient =', orient)
    print('pix_per_cell =', pix_per_cell)
    print('cell_per_block =', cell_per_block)
    print('hog_channel =', hog_channel)
    print('spatial_size =', spatial_size)
    print('hist_bins =', hist_bins)
    print('spatial_feat =', spatial_feat)
    print('hist_feat =', hist_feat)
    print('hog_feat =', hog_feat)
    #print('Feature vector length:', len(X_train[0]))

   # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    print(round(time.time() - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC =', round(svc.score(X_test, y_test), 4))
    print()
    print()

    save_classifier(svc, X_scaler, color_space, orient, pix_per_cell, cell_per_block, hog_channel,
                    spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat)

# comment this out if you want to use deviate_config()
# run on global defaults
train(None, None, color_space, orient, pix_per_cell, cell_per_block, hog_channel,
                        spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat)

def deviate_config():
    svc = None  # done within train() for now
    X_scaler = None # done within train() for now

    spatial_feat = True
    hist_feat = True
    hog_feat = True

    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = "ALL"
    orient = 9

    for color_space in ['RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb']:
        for sz in [8, 12, 16, 24, 32]:
            spatial_size = (sz, sz)
            for hist_bins in [8, 12, 16, 24, 32]:
                #print(svc, X_scaler, color_space, orient, pix_per_cell, cell_per_block, hog_channel,
                train(svc, X_scaler, color_space, orient, pix_per_cell, cell_per_block, hog_channel,
                        spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat)

#deviate_config()


