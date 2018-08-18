import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from lesson_functions import *
import time

import pickle


## load classifier from pickle file
PICKLE_FILE_NAME = 'classifier_data.p'
with open(PICKLE_FILE_NAME, 'rb') as f:
    pickle_data = pickle.load(f)

svc = pickle_data['svc']
X_scaler = pickle_data['X_scaler']
color_space = pickle_data['color_space']
orient = pickle_data['orient']
pix_per_cell = pickle_data['pix_per_cell']
cell_per_block = pickle_data['cell_per_block']
hog_channel = pickle_data['hog_channel']
spatial_size = pickle_data['spatial_size']
hist_bins = pickle_data['hist_bins']
spatial_feat = pickle_data['spatial_feat']
hist_feat = pickle_data['hist_feat']
hog_feat = pickle_data['hog_feat']


## load example images

image_pathes = glob.glob('test_images/test4*')
images = []
titles = []
y_start_stop = [380, 600]  #TODO use multiples of windowsize? what's the difference?
overlap = 0.7

for path in image_pathes:
    image = mpimg.imread(path)
    window_image = np.copy(image)
    if (path.endswith(".jpg") or path.endswith(".jpeg")):
        image = image.astype(np.float32)/255
    
    windows_64 = slide_window(image, x_start_stop = [300, 1000], y_start_stop = [400, 480],
                               xy_window=(64, 64), xy_overlap=(overlap, overlap))
    window_image = draw_boxes(window_image, windows_64, color=(255, 0, 0), thick=2)

    windows_96 = slide_window(image, x_start_stop = [None, None], y_start_stop = y_start_stop,
                               xy_window=(96, 96), xy_overlap=(overlap, overlap))
    window_image = draw_boxes(window_image, windows_96, color=(0, 255, 255), thick=2)

    windows_128 = slide_window(image, x_start_stop = [None, None], y_start_stop = y_start_stop,
                               xy_window=(128, 128), xy_overlap=(0.75, 0.75))
    window_image = draw_boxes(window_image, windows_128, color=(0, 255, 0), thick=2)

    windows_192 = slide_window(image, x_start_stop = [None, None], y_start_stop = y_start_stop,
                               xy_window=(192, 192), xy_overlap=(0.75, 0.75))
    window_image = draw_boxes(window_image, windows_192, color=(0, 255, 255), thick=2)

    windows = windows_64 + windows_96 + windows_128 + windows_192
    print(len(windows), "windows found")

    images.append(window_image)
    titles.append('')

fig = plt.figure()
visualize(fig, 1, 1, images, titles)



