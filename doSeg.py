"""
A script containing a function to run OpenCV's selective search on a single image.
Authors: Feras Alshehri
last modified: 5/15/2021
"""

import os
import cv2
import numpy as np

# jupyter and colab don't support OpenCV's imshow(), use this instead.
# see: https://github.com/jupyter/notebook/issues/3935
from google.colab.patches import cv2_imshow

defaultImgPath = os.path.join("/path","to", "myImage.jpg")

def doSeg(inputImgPath = defaultImgPath,
          sigma=0.1, k=300, min_size=1000):
    """
    Run OpenCV's image segmentation on an image.
    @input: sigma: int: The sigma parameter, used to smooth image.
            k: int: The k parameter of the algorythm.
            min_size: int: The minimum size of segments.
    @output: None.
    """
    

    # check file path validity
    if not os.path.isfile(inputImgPath):
        print(f"ERROR: bad input image path. (got '{inputImgPath}')")
        return

    # enable multi-threading to optimize performance
    cv2.setUseOptimized(True)
    cv2.setNumThreads(8)

    # create segmintor instance 
    mySeg = cv2.ximgproc.segmentation.createGraphSegmentation(sigma=sigma, k=k, min_size=min_size)

    # load input image
    inputImg = cv2.imread(inputImgPath)

    # process input image using the segementor created and return a 
    # numpy array contianing the cvoordinates of each segment region.
    segment = mySeg.processImage(inputImg)

    mask = segment.reshape(list(segment.shape) +[1]).repeat(3, axis=2)
    masked = np.ma.masked_array(inputImg, fill_value=0)

    for i in range(np.max(segment)):
        masked.mask = mask != i
        y, x = np.where(segment == i)
        x_i = min(y)
        y_i = max(y)
        x_f = min(x)
        y_f = max(x)
        imgUT = masked.filled()[x_i: y_i + 1 ,x_f : y_f + 1 ]
        # cv2.imwrite('segment_{num}.jpg'.format(num=i),dst) 
        display(cv2_imshow(imgUT))
