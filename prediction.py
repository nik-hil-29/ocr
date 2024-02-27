from imutils.contours import sort_contours 
 # 'sort_contours' function from the 'imutils.contours' module for sorting contours
import numpy as np 
 #  'numpy' library for numerical operations
import argparse 
 #  'argparse' module for parsing command-line arguments
import imutils  
#  'imutils' library used  for image processing convenience functions
import cv2  
# cv2' module from OpenCV for computer vision operations
from keras.models import load_model 
 # 'load_model' is the  func from the 'keras.models' mod for loadng pre-trained models





def prdct_imge(imge_path):
  trned_mdel = load_model('/Users/nikhilkushwaha/ocr/CombineDatasetModels/final_model_resnet/Combined_Resnet_20_Epochs.h5')

  imge = cv2.imread(imge_path)
  #change the image to grayscale
  gry_imge = cv2.cvtColor(imge, cv2.COLOR_BGR2GRAY)
  # using Gaussian blur to reduce the noise
  blrrd_imge = cv2.GaussianBlur(gry_imge, (5, 5), 0)
  #using canny edge detection
  edgd_imge = cv2.Canny(blrrd_imge, 30, 150)
  #searching contours in edged transformed image
  cnts = cv2.findContours(edgd_imge.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)
  # sorting the contours of the edged transfor image left to right
  cnts = sort_contours(cnts, method="left-to-right")[0]
  chrctrs = []

  # process each contours
  for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
      roi = gry_imge[y:y + h, x:x + w]
      
      # using thresholding  so that to change in binary image 
      thrsh = cv2.threshold(roi, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
      (thrsh_hieght, thrsh_wdth) = thrsh.shape
      # Resizing the image to a fix size
      if thrsh_wdth > thrsh_hieght:
        thrsh = imutils.resize(thrsh, width=32)
      else:
        thrsh = imutils.resize(thrsh, height=32)
      (thrsh_hieght, thrsh_wdth) = thrsh.shape
      
      # doing padding to achive fixed size
      x = max(0, 32 - thrsh_wdth)
      y = max(0, 32 - thrsh_hieght) 
      dlta_x = int( x / 2.0)
      dlta_y = int( y / 2.0)
      pdded = cv2.copyMakeBorder(thrsh, top=dlta_y, bottom=dlta_y,
        left=dlta_x, right=dlta_x, borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0))
      pdded = cv2.resize(pdded, (32, 32))
      pdded = pdded.astype("float32")
      pdded = np.expand_dims(pdded, axis=-1)
      chrctrs.append((pdded, (x, y, w, h)))
      
  # extracting bounding boxes of characters 
  bxs = [bx[1] for bx in chrctrs]

  # converting characters to numpy array
  chrctrs = np.array([c[0] for c in chrctrs], dtype="float32")

  # Make Predictions using the model 
  prds = trned_mdel.predict(chrctrs)
  
  # process the prediction
  lblNms = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
  lblNms = [l for l in lblNms]
  otpt=""
  for (prd, (x, y, w, h)) in zip(prds, bxs):
    i = np.argmax(prd)
    lbl = lblNms[i]
    otpt+=lbl
  
  return otpt
print(prdct_imge("/Users/nikhilkushwaha/ocr/contents/test4.jpeg"))
