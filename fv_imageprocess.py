# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 15:52:02 2023

@author: ok
"""
import cv2
#from skimage import io
from matplotlib import pyplot as plt
import numpy as np 

img=cv2.imread('E:/Gdnask University of Technology/Finger Vein Presentation Attack Detection System Using Deep Learning Model/IDIAPCROPPED/FAKE/001_L_1.png')
cv2.imshow('Finger Vein Image',img)

plt.imshow(img)
# cv2.waitKey(0)

###############################################################
#Example 1: Top-Hat Transform
  
  
# Getting the kernel to be used in Top-Hat
filterSize =(3, 3)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                   filterSize)

# Applying the Top-Hat operation
tophat_img = cv2.morphologyEx(img, 
                              cv2.MORPH_TOPHAT,
                              kernel)

plt.imshow(tophat_img)

#Example 2: Black Hat transform

  
# Getting the kernel to be used in Top-Hat
filterSize =(3, 3)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                   filterSize)

# Applying the Top-Hat operation
tophat_img = cv2.morphologyEx(img, 
                              cv2.MORPH_TOPHAT,
                              kernel)

plt.imshow(tophat_img)
#############################################################


hist,bins = np.histogram(img.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()
plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()


cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')

img2 = cdf[img]
plt.imshow(img2)
##############################################################

equ = cv2.equalizeHist(img)
res = np.hstack((img,equ)) #stacking images side-by-side
cv2.imwrite('res.png',res)

#################################################################

# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)
#################################################################


import cv2
import numpy as np
import pywt
import matplotlib
matplotlib.use('Qt5Agg')  # Set the backend explicitly

import matplotlib.pyplot as plt

# Load the finger vein image
img = cv2.imread('E:/Gdnask University of Technology/Finger Vein Presentation Attack Detection System Using Deep Learning Model/IDIAPCROPPED/FAKE/001_L_1.png', 0)
plt.imshow(img)
plt.show('img')
# Perform wavelet transform on the image
coeffs = pywt.dwt2(img, 'haar')

# Split the coefficients into approximation and details
LL, (LH, HL, HH) = coeffs

# Apply enhancement to the detail coefficients
alpha = 2.0 # Adjust this value to control the amount of enhancement
LH = LH * alpha
HL = HL * alpha
HH = HH * alpha

# Reconstruct the enhanced image
coeffs = LL, (LH, HL, HH)
img_enhanced = pywt.idwt2(coeffs, 'haar')

# Convert the enhanced image to uint8 format for display
img_enhanced = np.uint8(img_enhanced)

# Display the original and enhanced images side by side
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')
ax[1].imshow(img_enhanced, cmap='gray')
ax[1].set_title('Enhanced Image')
ax[1].axis('off')
plt.show()
#################################################################

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('E:/Gdnask University of Technology/Finger Vein Presentation Attack Detection System Using Deep Learning Model/IDIAPCROPPED/FAKE/001_L_1.png', 0)

# Apply Histogram Equalization
equ = cv2.equalizeHist(img)

# Plot the original and equalized images
fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs[0].imshow(img, cmap='gray')
axs[0].set_title('Original Image')
axs[1].imshow(equ, cmap='gray')
axs[1].set_title('Equalized Image')
plt.show()

