## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.png
[image3]: ./examples/sliding_window_pattern.png
[image4]: ./examples/sliding_window_results.png
[image5]: ./examples/bboxes_and_heat.png
[video1]: ./video_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

For training i had a set of `8792 car` samples and `8968 non-car` samples. you can find an example of each class in the following image.

![alt text][image1]

First of all i collected most of the stuff shown during the last lesson within a file called `lesson_functions.py`.
Then, to get a rough feeling about the possible parameters and their influence on the HOG features, i created a small script named `hog_sample.py`.

Here is an example using the `RGB` color space and HOG parameters of `orientations=6`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and got a suitable starting set which can be found at `hog_sample.py` lines 38-48. Later, during my first attemts to train an SVC on those features, i changed those values after some additional trial-and-error hours ;)

| parameter         		| values from the first guess	after using `hog_sample.py`| second parameter set after first training attemts |
|:---------------------:|:---------------:|:---------------:|
| color_space |	RGB | HLS 	|
| orient |	6 | 9	|
| pix_per_cell |  8	| 8	|
| cell_per_block |  2	| 2	|
| hog_channel | 0	| ALL	|
| spatial_size |  (16, 16)	| (32, 32)	|
| hist_bins | 16	| 32	|
| spatial_feat | True	| True	|
| hist_feat |	True | True	|
| hog_feat | True	| True	|


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I used the resulting features to train a linear SVC after normalizing them using a StandardScaler, which lead to an accuracy of 99.49% on the test set. You can find the code in a file called `train_classifier.py`. Most intresing here should be function `train(...)` which can be found at line 77++.

since i was not sure about the parameters for feature extraction i created a function called `deviate_config`, in which i generated several combinations of parameter sets (similar to sklearn.grid_search.GridSearchCV) and applied them all onto the full data set, to tweak my parameters.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I composed a pattern of sliding windows which have different sizes and boundaries. Basically use some `64x64` windows on a small area of the image `((300, 400) (1000, 480))`. I did this, because the small window sizes help finding cars that are more far away (and thus seem to be smaller due to the perspective). also the small size leads to a very fine grid, that helps to increase accuracy. unfortunately on the opposite such a small grid results in a high amount of windows, that have to be searched, which can become really slow. Thats why I especially limited the area of applying those small windows.
Additionally i use seom `96x96, 128x128 and 192x192` windows, which i gave a broader region of the image to search for two reasons. 1) cars seem to be bigger when they are closer. and close cars might appear on the edges of the image in case of passing. 2) there won't fit many of those big windows, which lmits the performance impact, while a bigger search area might increase accuracy (to some extend).
All those 'bigger' windows are limited only on the Y-axis `(380, 600)`, whereas they use the whole width of the image. (due to the former explained reasons).

here you can see the pattern i am using with different colors for different window sizes:

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

As shown in the table of parameters above i combine spatially binned color, histograms of color and the HOG-features. The combination of allof those features gave pretty promising results as you can see here (even thogh there are still some false positives):

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

To reduce false positives i created a heatmap history with the length of 15 frames. I choose 15 frames, since it is not too much to make the 'user experience' become sluggish, while still maintaining a good amount of samples to distinguish false from right positives most of the time.
I used this history to sum up the heat for all pixels over the last 15 frames and used the result as a basis for the `scipy.ndimage.measurements.label()` function, to extract possible car positions. I used the labels to draw bounding boxes around the labeled area.


### Here are some frames including the final bounding boxes and their corresponding heatmaps:

![alt text][image5]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

obviously the current pipeline is not yet bullet proof and could not be used within a real life application.
1) the classifier returns to many false positives. which could probably be reduced by e.g. using multiple classifiers at once and combine their results or train on different features. maybe it's even worth combining the detection of vehicles based on handcrafted features with some DNN techniques from earlier lessons.
definitly there is room for improvement and will use this very amazing project for further investigation on this topic even after the end of this course :D
2) despite the false positives, it might make sence to introduce a more sophisticated technique of object tracking, since the heatmap summation allone is not enough. but we should use the knowldege about the real world. like e.g. we limited the sliding window search area in a way, that we skip the sky, since we don't expect cars there. just like that we should enforce the knowledge, that a car (most likely) won't disappear out of the sudden. so if we've been pretty confident about recognising a car in previous frames, we should expect it to be there in future frame (according to physical laws). maybe we could even estimate/approximate the position of a car, by tracking it's last known speed and dircetion.
3) performance is a big issue here. the feature extraction takes a huge amount of time for eacht frame, which makes it practically impossible to use that algorithm in a real car. the current implementation needs 3s per frame!
