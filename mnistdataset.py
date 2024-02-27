from keras.datasets import mnist  
# Built-in dataset of handwritten digits for training and testing machine learning models
from keras.preprocessing.image import ImageDataGenerator  
# Image preprocessing utilities module  for deep learning models
import numpy as np  
#Numpy is for numerical computing and supports for large  multi-dimesional arrays and matrices
import datas as dt  
import datasetpath as dsp
import cv2  
# OpenCV library for computer vision tasks and image processing
from sklearn.preprocessing import LabelBinarizer 
# It is the library  for transforming categorical labels into binary vectors
from sklearn.model_selection import train_test_split  
# Function for splitting datasets into training and testing subsets

((trnDta, trnLbls), (tstDta, tstLbls)) = mnist.load_data()
(azDta, azLbls) = dsp.load_custom_dataset('/Users/nikhilkushwaha/ocr/A_Z Handwritten Data.csv')
# print(azData.shape)

cmbnd_dta = np.vstack([trnDta, tstDta])
cmbnd_lbls = np.hstack([trnLbls, tstLbls])


# print(trnDta.shape)
# print(cmbnd_dta[0].shape)
# print(dt.lbls.shape)


azLbls+=10
#  combining all the dataset
lbls1=np.hstack([cmbnd_lbls,dt.labels,azLbls])
dta1=np.vstack([cmbnd_dta,dt.data,azDta])
# print(lbls1.shape)
# print(dta1.shape)

# print(labels1.shape)

# print(data1.shape)

# saving the combined data
np.save('/Users/nikhilkushwaha/ocr/combined_data.npy', dta1)
np.save('/Users/nikhilkushwaha/ocr/combine_labels.npy', lbls1)

# loading the combined data and labels


loded_dta = np.load('/Users/nikhilkushwaha/ocr/combined_data.npy')
loded_lbls = np.load('/Users/nikhilkushwaha/ocr/combine_labels.npy')

cv2.imshow("image" , loded_dta[-50000])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

tmp_image = cv2.resize(loded_dta[0], (32, 32))

# print(tmp_image.shape)

dta = [cv2.resize(image, (32, 32)) for image in loded_dta]
dta = np.array(dta, dtype="float32")

# print(data.shape)

dta2 = np.expand_dims(dta, axis=-1)

# print(data2.shape)


lbl_bin = LabelBinarizer()
bnrized_lbls = lbl_bin.fit_transform(loded_lbls)
class_totals = bnrized_lbls.sum(axis=0)
# print(binarized_labels[0])   

     
# print(labels[0])          

# classweight
class_totals = bnrized_lbls.sum(axis=0)
class_weight = {}
# loop over all classes and calculate the class weight
for i in range(0, len(class_totals)):
	class_weight[i] = class_totals.max() / class_totals[i]
 
# print(class_weight)

# print("Data shape: -",data2.shape)
# print("labels shape: -",labels.shape)

from sklearn.model_selection import train_test_split  # Function for splitting datasets into training and testing subsets
(trn_dta_X, tst_dta_X, trn_dta_Y, tst_dta_Y) = train_test_split(dta2,
	bnrized_lbls, test_size=0.25, stratify=bnrized_lbls, random_state=42)

# print(trn_dta_X.shape)
# print(trn_dta_X.shape)

aug = ImageDataGenerator(
	rotation_range=10,
	zoom_range=0.05,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.15,
	horizontal_flip=False,
	fill_mode="nearest")

# print(le.classes_)