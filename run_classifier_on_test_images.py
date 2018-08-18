import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from lesson_functions import *
from sliding_window import my_sliding_window_pattern
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

image_pathes = glob.glob('test_images/*')
images = []
titles = []
#y_start_stop = [380, 764]  # use multiples of windowsize
y_start_stop = [380, 600]  # use multiples of windowsize
overlap = 0.7

for path in image_pathes:
    t = time.time()
    image = mpimg.imread(path)
    draw_image = np.copy(image)
    if (path.endswith(".jpg") or path.endswith(".jpeg")):
        image = image.astype(np.float32)/255
    
    windows = my_sliding_window_pattern(image.shape)

    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)

    window_img = draw_boxes(draw_image, hot_windows, color=(255, 255, 0), thick=3)
    images.append(window_img)
    titles.append('')
    print(time.time() - t, "seconds to search", len(windows), "windows")

fig = plt.figure(figsize=(8, 6))
visualize(fig, 2, 3, images, titles)



