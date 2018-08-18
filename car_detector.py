import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from lesson_functions import *
import time

from scipy.ndimage.measurements import label

import pickle
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip


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


#y_start_stop = [380, 764]  #TODO should use multiples of windowsize
y_start_stop = [380, 600]  #TODO should use multiples of windowsize
overlap = 0.7


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (255, 255, 0), 3)
    # Return the image
    return img

heatmap_history = None

def find_cars(image, heatmap_threshold = 7, return_heatmap=False, use_heatmap_history=True, limit_pix_values=True):
    global heatmap_history

    if limit_pix_values == True:
        #TODO might make sense, to convert png data during training, so we can save thos calulations during video processing
        # value conversion to fit into png-trained kernel
        image = image.astype(np.float32)/255
        #print(np.min(image), np.max(image), np.mean(image))

    windows_96 = slide_window(image, x_start_stop = [None, None], y_start_stop = y_start_stop,
                               xy_window=(96, 96), xy_overlap=(overlap, overlap))
    windows_128 = slide_window(image, x_start_stop = [None, None], y_start_stop = y_start_stop,
                               xy_window=(128, 128), xy_overlap=(overlap, overlap))
    #windows_192 = slide_window(image, x_start_stop = [None, None], y_start_stop = y_start_stop,
    #                           xy_window=(192, 192), xy_overlap=(overlap, overlap))
    windows = windows_96 + windows_128
    #windows = windows_128
    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)

    #window_img = draw_boxes(draw_image, hot_windows, color=(255, 255, 0), thick=3)
    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    hist_length = 15
    if (heatmap_history == None) or (use_heatmap_history == False):
        #TODO zero out the history or start with initial values?
        heatmap_history = np.repeat([heat], hist_length, axis=0)

    # Add heat to each box in box list
    heat = add_heat(heat, hot_windows)
        
    heatmap = heat.reshape((1, heat.shape[0], heat.shape[1]))
    # append last measurment to history data
    heatmap_history = np.vstack([heatmap_history, heatmap])[1:]
    # smooth out the data
    heatmap = np.sum(heatmap_history, axis=0)

    # Apply threshold to help remove false positives
    heatmap = apply_threshold(heatmap, heatmap_threshold)

    # smooth heatmap data
    heatmap = np.clip(heatmap, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)

    #TODO might make sense, to convert png data during training, so we can save thos calulations during video processing
    # inverse value conversion to make picture fit into video format again
    image = (image * 255).astype(np.uint8)

    if return_heatmap == True:
        return draw_labeled_bboxes(np.copy(image), labels), heatmap
    return draw_labeled_bboxes(np.copy(image), labels)



VIDEO_MODE = True
# following settings are only used in image mode atm (but would work during video mode as well)
RETURN_HEATMAP = True
USE_HEATMAP_HISTORY = False
USE_CONSECUTIVE_IMAGES = False

if VIDEO_MODE == True:
    video_out = 'video_out.mp4'
    #video_in = VideoFileClip('test_video.mp4')
    video_in = VideoFileClip('project_video.mp4')
    video_in = video_in.subclip(27, 29)

    print("processing video...")

    video_clip = video_in.fl_image(find_cars)
    video_clip.write_videofile(video_out, audio=False)
else:
    ## load example images
    image_pathes = glob.glob('test_images/*')
    #image_pathes = glob.glob('test_images/test4.jpg')
    images = []
    titles = []

    for path in image_pathes:
        t = time.time()
        image = mpimg.imread(path)
        draw_image = np.copy(image)
        limit_pix_values = False
        if (path.endswith(".jpg") or path.endswith(".jpeg")):
            #image = image.astype(np.float32)/255
            limit_pix_values = True

        if RETURN_HEATMAP == False:
            draw_image = find_cars(image, heatmap_threshold = 1, return_heatmap = False, use_heatmap_history = False, limit_pix_values = limit_pix_values)

            images.append(draw_image)
            titles.append('')

        else:
            draw_image, heatmap = find_cars(image, heatmap_threshold = 1, return_heatmap = True, use_heatmap_history = False, limit_pix_values = limit_pix_values)

            images.append(heatmap)
            titles.append('')
            images.append(draw_image)
            titles.append('')

    if RETURN_HEATMAP == False:
        #fig = plt.figure(figsize=(8, 6))
        fig = plt.figure()
        visualize(fig, 2, 3, images, titles)
    else:
        fig = plt.figure(figsize=(8, 6))
        visualize(fig, 3, 4, images, titles)





