import cv2
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
try:
    from PIL import Image
except ImportError:
    import Image
from PIL import Image
from PIL import ImageEnhance
import pypdfium2 as pdfium
import json
import argparse


def pipeline(path):

  pdf = pdfium.PdfDocument(path)
  version = pdf.get_version()  # get the PDF standard version
  number_of_pages = len(pdf) # get the number of pages in the document

  img = [None]*number_of_pages
  for i in range(number_of_pages):
    img[i] = pdf[i].render_to(
      pdfium.BitmapConv.pil_image,
      scale = 5,                           # 72dpi resolution
      rotation = 0,                        # no additional rotation
      crop = (0, 0, 0, 0),                 # no crop (form: left, right, bottom, top)
      greyscale = False,                   # coloured output
      fill_colour = (255, 255, 255, 255),  # fill bitmap with white background before rendering (form: RGBA)
      colour_scheme = None,                # no custom colour scheme
        # no optimisations (e. g. subpixel rendering)
      draw_annots = True,                  # show annotations
      draw_forms = True,                   # show forms
      no_smoothtext = False,               # anti-alias text
      no_smoothimage = False,              # anti-alias images
      no_smoothpath = False,               # anti-alias paths
      force_halftone = False,              # don't force halftone for image stretching
      rev_byteorder = False,               # don't reverse byte order
      prefer_bgrx = False,                 # don't prefer four channels for coloured output
      force_bitmap_format = None,          # don't force a specific bitmap format
      extra_flags = 0,                     # no extra flags
      allocator = None,                    # no custom allocator
      memory_limit = 2**30,                # maximum allocation (1 GiB)
    )

  img_bin = [None]*number_of_pages
  for i in range(number_of_pages):
    img[i] = cv2.cvtColor(np.array(img[i]), cv2.COLOR_RGB2GRAY)
    #thresholding the image to a binary image
    thresh,img_bin[i] = cv2.threshold(img[i],128,255,cv2.THRESH_BINARY |cv2.THRESH_OTSU)
    #inverting the image 
    img_bin[i] = 255-img_bin[i]

    # Length(width) of kernel as 100th of total width
  kernel_len = np.array(img[i]).shape[1]//100
  # Defining a vertical kernel to detect all vertical lines of image 
  ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
  # Defining a horizontal kernel to detect all horizontal lines of image
  hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
  # A kernel of 2x2
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

  #Use vertical kernel to detect and save the vertical lines in a jpg
  image_1 = [None]*number_of_pages
  vertical_lines = [None]*number_of_pages
  for i in range(number_of_pages):
    image_1[i] = cv2.erode(img_bin[i], ver_kernel, iterations=3)
    vertical_lines[i] = cv2.dilate(image_1[i], ver_kernel, iterations=4)

  #Use horizontal kernel to detect and save the horizontal lines in a jpg
  image_2 = [None]*number_of_pages
  horizontal_lines = [None]*number_of_pages
  for i in range(number_of_pages):
    image_2[i] = cv2.erode(img_bin[i], hor_kernel, iterations=3)
    horizontal_lines[i] = cv2.dilate(image_2[i], hor_kernel, iterations=4)

  # Combine horizontal and vertical lines in a new third image, with both having same weight.
  img_vh = [None]*number_of_pages
  bitxor = [None]*number_of_pages
  bitnot = [None]*number_of_pages
  for i in range(number_of_pages):
    img_vh[i] = cv2.addWeighted(vertical_lines[i], 0.5, horizontal_lines[i], 0.5, 0.0)
    #Eroding and thesholding the image
    img_vh[i] = cv2.erode(~img_vh[i], kernel, iterations=2)
    thresh, img_vh[i] = cv2.threshold(img_vh[i],128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite("/Users/YOURPATH/img_vh.jpg", img_vh[i])
    bitxor[i] = cv2.bitwise_xor(img[i],img_vh[i])
    bitnot[i] = cv2.bitwise_not(bitxor[i])

  # Detect contours for following box detection
  contours = [None]*number_of_pages
  for i in range(number_of_pages):
    contours[i], hierarchy = cv2.findContours(img_vh[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  def sort_contours(cnts, method='left-to-right'):
      # initialize the reverse flag and sort index
      reverse = False
      i = 0
      # handle if we need to sort in reverse
      if method == 'right-to-left' or method == 'bottom-to-top':
        reverse = True
      # handle if we are sorting against the y-coordinate rather than
      # the x-coordinate of the bounding box
      if method == 'top-to-bottom' or method == 'bottom-to-top':
        i = 1
      # construct the list of bounding boxes and sort them from top to bottom
      boundingBoxes = [cv2.boundingRect(c) for c in cnts]
      (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
      key=lambda b:b[1][i], reverse=reverse))
      # return the list of sorted contours and bounding boxes
      return (cnts, boundingBoxes)

  boundingBoxes = [None]*number_of_pages
  for k in range(number_of_pages):
  # Sort all the contours by top to bottom.
    contours[k], boundingBoxes[k] = sort_contours(contours[k], method = 'top-to-bottom')

  heights = [None]*number_of_pages
  mean = [None]*number_of_pages
  for k in range(number_of_pages):
    #Creating a list of heights for all detected boxes
    heights[k] = [boundingBoxes[k][i][3] for i in range(len(boundingBoxes[k]))]
    #Get mean of heights
    mean[k] = np.mean(heights[k])

  box = [None]*number_of_pages
  for k in range(number_of_pages):
    #Create list box to store all boxes in  
    boxS = []
    # Get position (x,y), width and height for every contour and show the contour on image
    for c in contours[k]:
      x, y, w, h = cv2.boundingRect(c)
      if (w<1000 and h<500 and w>20 and h>20):
          image = cv2.rectangle(img[k],(x,y),(x+w,y+h),(0,0,255),2)
          boxS.append([x,y,w,h])
    box[k] = boxS

  row = [None]*number_of_pages
  column = [None]*number_of_pages
  countcol = [None]*number_of_pages
  center = [None]*number_of_pages
  for k in range(number_of_pages):
    #Creating two lists to define row and column in which cell is located
    rowS=[]
    columnS=[]
    j=0
    #Sorting the boxes to their respective row and column
    for i in range(len(box[k])):
      if(i==0):
          column.append(box[k][i])
          previous=box[k][i]
      else:
          if(box[k][i][1]<=previous[1]+mean[k]/6):
              columnS.append(box[k][i])
              previous=box[k][i]
              if(i==len(box[k])-1):
                  rowS.append(columnS)
          else:
              rowS.append(columnS)
              columnS=[]
              previous = box[k][i]
              columnS.append(box[k][i])
    row[k] = rowS
    column[k] = columnS


    #calculating maximum number of cells
    countcol[k] = 0
    for i in range(len(row[k])):
      countcol[k] = len(row[k][i])
      if countcol[k] > countcol[k]:
          countcol[k] = countcol[k]
    # Retrieving the center of each column
    if len(row[k]) > 0:
      center[k] = [int(row[k][i][j][0]+row[k][i][j][2]/2) for j in range(len(row[k][i]))]
      print(center[k])
      center[k]=np.array(center[k])
      center[k].sort()

  finalboxes = [None]*number_of_pages
  for k in range(number_of_pages):
    #Regarding the distance to the columns center, the boxes are arranged in respective order
    finalboxesS = []
    for i in range(len(row[k])):
      lis=[]
      for t in range(countcol[k]):
          lis.append([])
      for j in range(len(row[k][i])):
          diff = abs(center[k]-(row[k][i][j][0]+row[k][i][j][2]/4))
          minimum = min(diff)
          indexing = list(diff).index(minimum)
          lis[indexing].append(row[k][i][j])
      finalboxesS.append(lis)
    finalboxes[k] = finalboxesS

  config = Cfg.load_config_from_name('vgg_transformer')

  config['weights'] = 'C:/Users/a/Documents/GitHub/ExtractTable/Weight/transformerocr.pth'
  # config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
  config['cnn']['pretrained']=False
  config['device'] = 'cpu'
  config['predictor']['beamsearch']=False

  detector = Predictor(config)

  outer=[]
  for z in range(number_of_pages):
  #from every single image-based cell/box the strings are extracted via pytesseract and stored in a list
    for i in range(len(finalboxes[z])):
      for j in range(len(finalboxes[z][i])):
          inner=''
          if(len(finalboxes[z][i][j])==0):
              outer.append(' ')
          else:
              for k in range(len(finalboxes[z][i][j])):
                  y,x,w,h = finalboxes[z][i][j][k][0],finalboxes[z][i][j][k][1], finalboxes[z][i][j][k][2],finalboxes[z][i][j][k][3]
                  finalimg = bitnot[z][x:x+h, y:y+w]
                  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                  border = cv2.copyMakeBorder(finalimg,2,2,2,2,   cv2.BORDER_CONSTANT,value=[255,255])
                  resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                  dilation = cv2.dilate(resizing, kernel,iterations=1)
                  erosion = cv2.erode(dilation, kernel,iterations=1)

                  cell_img = erosion
                  image = Image.fromarray(cell_img)
                  image = image.convert("RGB")
                  out = detector.predict(image)
                  if(len(out)==0):
                      out = 'NULL'
                  inner = inner +" "+ out
              outer.append(inner)

  #Creating a dataframe of the generated OCR list
  arr = np.array(outer)
  row_len = 0
  for i in range(number_of_pages):
    row_len = row_len + len(row[i])
  # print(arr)
  dataframe = pd.DataFrame(arr.reshape(row_len,countcol[0]))
  # data = dataframe.style.set_properties(align="left")
  print(dataframe)
  #Converting it in a excel-file
  result = dataframe.to_json(orient="table")
  parsed = json.loads(result)
  json.dumps(parsed).encode('utf8')  
  return parsed

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--path', type=str, help='model.pt path(s)')
  # parser.add_argument('--output', type=str, help='model.pt path(s)')
  opt = parser.parse_args()
  pipeline(opt)
