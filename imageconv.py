import cv2
 # OpenCv(cv2) is a popular computer vision library. It is used in image and video processing. 
import numpy as np 
#Numpy is for handling of   multi-dimesional arrays and matrices


def convert_to_28x28(i_p):  # i_p stands for image_path
    # Read and convert the image to grayscale using OpenCV's built-in function
    imge = cv2.cvtColor(cv2.imread(i_p), cv2.COLOR_BGR2GRAY)

    # Conversion of each image to the shape of (28, 28, 1)
    wdth, hieght = imge.shape

    if hieght > 28 or wdth > 28:
        # Resize the image using OpenCV's built-in functions
        (trgt_hieght, trgt_wdth) = imge.shape
        x = max(0, 28 - trgt_wdth)
        y = max(0, 28 - trgt_hieght)
        dlta_x = int(x / 2.0)
        dlta_y = int(y / 2.0)
        cmb = cv2.copyMakeBorder
        # Add zero-pixel borders to the image to match the target size
        imge = cmb(imge, top=dlta_y, bottom=dlta_y,
                   left=dlta_x, right=dlta_x, borderType=cv2.BORDER_CONSTANT,
                   value=(0, 0, 0))

        # Resize the image to the target size of (28, 28)
        imge = cv2.resize(imge, (28, 28))

    wdth, hieght = imge.shape

    if wdth < 28:
        # Concatenate the image with additional rows of zeros using numpy's built-in functions
        addng_zros = np.ones((28 - wdth, hieght)) * 255
        imge = np.concatenate((imge, addng_zros))

    if hieght < 28:
        # Concatenate the image with additional columns of zeros using numpy's built-in functions
        addng_zros = np.ones((28, 28 - hieght)) * 255
        imge = np.concatenate((imge, addng_zros), axis=1)

    # Normalize each image
    return imge
