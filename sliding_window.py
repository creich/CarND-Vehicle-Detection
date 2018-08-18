import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from lesson_functions import *
import time



y_start_stop = [380, 600]  #TODO use multiples of windowsize? what's the difference?
overlap = 0.7



def my_sliding_window_pattern(image_shape, draw_image = None):
    windows_64 = slide_window(image_shape, x_start_stop = [300, 1000], y_start_stop = [400, 480],
                               xy_window=(64, 64), xy_overlap=(overlap, overlap))
    if draw_image is not None:
        draw_image = draw_boxes(draw_image, windows_64, color=(255, 0, 0), thick=2)

    windows_96 = slide_window(image_shape, x_start_stop = [None, None], y_start_stop = y_start_stop,
                               xy_window=(96, 96), xy_overlap=(overlap, overlap))
    if draw_image is not None:
       draw_image = draw_boxes(draw_image, windows_96, color=(0, 255, 255), thick=2)

    windows_128 = slide_window(image_shape, x_start_stop = [None, None], y_start_stop = y_start_stop,
                               xy_window=(128, 128), xy_overlap=(0.75, 0.75))
    if draw_image is not None:
       draw_image = draw_boxes(draw_image, windows_128, color=(0, 255, 0), thick=2)

    windows_192 = slide_window(image_shape, x_start_stop = [None, None], y_start_stop = y_start_stop,
                               xy_window=(192, 192), xy_overlap=(0.75, 0.75))
    if draw_image is not None:
       draw_image = draw_boxes(draw_image, windows_192, color=(0, 255, 255), thick=2)

    if draw_image is not None:
        return windows_64 + windows_96 + windows_128 + windows_192, draw_image
    return windows_64 + windows_96 + windows_128 + windows_192


if __name__ == "__main__":
    DEBUG = False

    ## load example images
    image_pathes = glob.glob('test_images/test4*')
    images = []
    titles = []

    for path in image_pathes:
        image = mpimg.imread(path)
        draw_image = np.copy(image)
        if (path.endswith(".jpg") or path.endswith(".jpeg")):
            image = image.astype(np.float32)/255
        
        windows, draw_image = my_sliding_window_pattern(image.shape, draw_image)

        print(len(windows), "windows found")

        images.append(draw_image)
        titles.append('')

    if DEBUG == True:
        fig = plt.figure()
        visualize(fig, 1, 1, images, titles)



