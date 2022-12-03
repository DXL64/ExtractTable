import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
try:
    from PIL import Image
except ImportError:
    import Image
from pdf2image import convert_from_path
from PIL import Image
from PIL import ImageEnhance

number_of_pages = 0
path = '/content/drive/Shareddrives/Lục Đại Anh Hùng/Grade/'

img = convert_from_path('ExtractTable/khmt.pdf', poppler_path='/Dependencies/poppler-0.68.0/bin')
number_of_pages = len(img)

img_bin = [None]*number_of_pages
for i in range(number_of_pages):
  img[i] = cv2.cvtColor(np.array(img[i]), cv2.COLOR_RGB2GRAY)
  
  #thresholding the image to a binary image
  thresh,img_bin[i] = cv2.threshold(img[i],128,255,cv2.THRESH_BINARY |cv2.THRESH_OTSU)
  
  #inverting the image 
  img_bin[i] = 255-img_bin[i]

  #Plotting the image to see the output
  plt.figure(figsize=(50,50))
  plotting = plt.imshow(img[i],cmap='gray')
  plt.show()

# Length(width) of kernel as 100th of total width
kernel_len = np.array(img[i]).shape[1]//100
# Defining a vertical kernel to detect all vertical lines of image 
ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
# Defining a horizontal kernel to detect all horizontal lines of image
hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
# A kernel of 2x2
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

