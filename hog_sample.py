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

# choose random car / not-car indices
car_index = np.random.randint(0, len(cars))
notcar_index = np.random.randint(0, len(notcars))

car_image = mpimg.imread(cars[car_index])
notcar_image = mpimg.imread(notcars[notcar_index])

# feature parameters
color_space = 'RGB'         # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 6                  # HOG orientations
pix_per_cell = 8            # HOG pixels per cell
cell_per_block = 2          # HOG cells per block
hog_channel = 0             # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)     # Spatial binning dimensions
hist_bins = 16              # Number of histogram bins
spatial_feat = True         # Spatial features on or off
hist_feat = True            # Histogram features on or off
hog_feat = True             # HOG features on or off

car_features , car_hog_image = single_img_features(car_image, color_space=color_space, spatial_size=spatial_size,
                                        hist_bins=hist_bins, orient=orient,
                                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat, vis=True)

notcar_features , notcar_hog_image = single_img_features(notcar_image, color_space=color_space, spatial_size=spatial_size,
                                        hist_bins=hist_bins, orient=orient,
                                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat, vis=True)


images = [car_image, car_hog_image, notcar_image, notcar_hog_image]
titles = ['car_image', 'car_hog_image', 'notcar_image', 'notcar_hog_image']
fig = plt.figure(figsize=(12,3))
visualize(fig, 1, 4, images, titles)




