# BlobDetection

This repository contains code for detecting blobs in images using various computer vision techniques. Blob detection is useful in many applications, including object recognition, image segmentation, and feature extraction.


The following image was generated using the blob_tracking.py code.

My key insight was that drones are quite consistently spiny. their arms, blades, shape, often contains components between 5-20 pixels in size. So, a band-pass filter that keeps features in that range is likely to be useful.

The DoG was perfect for this. Then, the intermediate thresholding filter essentially says that anything black/white from the DoG is uninteresting - show me only the features that are in between (i.e. the spiny bits of the drone between 5-20pixels).

Then, fill the holes to make a contiguous region.

![img.png](img.png)