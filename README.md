# BlobDetection

This repository contains code for detecting blobs in images using various computer vision techniques. Blob detection is useful in many applications, including object recognition, image segmentation, and feature extraction.


The following image was generated using the blob_tracking.py code.

It seems to me that some combination of Otsu and filled-holes band-pas filtering would be an interesting initial invesitgation. 
Band-pass filtering was done, essentially, using a  difference of Gaussians (DoG) approach. DoG is actually the difference between
a low pass and a lower pass filter (rather than high and low pass filters), but that works to get rid of super-high frequency noise and super-low frequency background variations.

My key insight was that drones are quite consistently spiny. their arms, blades, shape, often contains components between 5-20 pixels in size. So, a band-pass filter that keeps features in that range is likely to be useful.

The intermediate filter essentially says that anything black/white from the DoG is uninteresting - show me the features that are in between (i.e. the spiny bits of the drone between 5-20pixels).

Fill the holes to make a contiguous region.

![img.png](img.png)