import os
from collections import OrderedDict
import cv2
import imageconv
import numpy as np
from numpy import save
path = '/Users/nikhilkushwaha/ocr/English/Img/GoodImg/Bmp'

# get a list of all the files in the folder
file_list = os.listdir(path)

# iterate over the files and do something with them
for file_name in file_list:
    # construct the full path to the file
    file_path = os.path.join(path, file_name)

    # do something with the file, for example, print the file name
    # print(file_name)

folders = []

for root, dirnames, filenames in os.walk(path):
    for j in dirnames:
        folders.append(j)
    # print("filenames",filenames)
# print('root   ',root,',  dirnames: - ',dirnames,'   filenames',filenames)

# print(len(folders))

files = {}

for i in folders:
    files[i] = [f for f in os.listdir("/Users/nikhilkushwaha/ocr/English/Img/GoodImg/Bmp/" + i) if not f.endswith('.DS_Store')]
    # files.add(os.listdir("/content/English/Img/BadImag/Bmp/"+i))

# sorting files according to their names in accending order
dict1 = OrderedDict(sorted(files.items()))
# print(dict1.keys())
# print(dict1['Sample067'])

f = open("labels.txt", "a")
for i in dict1.keys():
    for j in dict1[i]:
        f.write("English/Img/GoodImg" + i + "/" + j + "\n")
f.close()

l = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

data = []
labels = []
tmp = 0

# img_path = "/Users/nikhilkushwaha/ocr/English/Img/GoodImg/Bmp/Sample067/img067-00006.png"
# img = cv2.imread(img_path)
# cv2.imshow("image", img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# convolution of image to 28x28
# image = imageconv.conTO28x28(img_path)
# cv2.imshow("Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

for i in dict1.keys():
    for j in dict1[i]:
        # labels list contains labels of images
        labels.append(tmp)
        # image=cv2.imread(/Users/nikhilkushwaha/ocr/English/Img/BadImag/Bmp/" + i + '/' + j)
        imag = imageconv.convert_to_28x28("/Users/nikhilkushwaha/ocr/English/Img/GoodImg/Bmp/" + i + '/' + j)
        # Data list containing images in the form of numpy array
        data.append(imag)
    tmp += 1

# print(len(labels))
labels = np.array(labels, dtype="int")
data = np.array(data, dtype='float32')
# print(labels)
# print(tmp)
#  saving the data and labels
save('data.npy', data)
save('labels.npy', labels)

tmp_data = np.load('data.npy')
# print(tmp_data.shape)

