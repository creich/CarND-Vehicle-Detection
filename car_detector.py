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


## load example images

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
        cv2.rectangle(img, bbox[0], bbox[1], (1, 1, 0), 3)
    # Return the image
    return img

def find_cars(image):
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

    # Add heat to each box in box list
    heat = add_heat(heat, hot_windows)
        
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 1)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    return draw_labeled_bboxes(np.copy(image), labels)


video_out = 'video_out.mp4'
#video_in = VideoFileClip('test_video.mp4')
video_in = VideoFileClip('project_video.mp4')

print("processing video...")

video_clip = video_in.fl_image(find_cars)
video_clip.write_videofile(video_out, audio=False)



