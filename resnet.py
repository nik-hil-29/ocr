from keras.layers import BatchNormalization  
# It is a technique used to normalize the activations of a previous layer
from keras.layers.convolutional import Conv2D  
# This module of keras.layers is  performs convolutional operations on 2D inputs
from keras.layers.convolutional import AveragePooling2D  
# This layer  performs downsampling using average values
from keras.layers.convolutional import MaxPooling2D  
# Max pooling layer that performs downsampling using maximum values
from keras.layers.convolutional import ZeroPadding2D 
# Zero padding layer that pads the input with zeros
from keras.layers.core import Activation  
# Activation layer that applies a specified activation function
from keras.layers.core import Dense  
# Fully connected dense layer that applies a specified number of output units
from keras.layers import Flatten  
# It Flattens the input to a 1D array
from keras.layers import Input  
# Input layer for defining the input shape
from keras.models import Model  
# Model class that represents a deep learning model
from keras.layers import add  
# Element-wise addition layer
from keras.regularizers import l2  
# L2 regularization that applies a penalty on the layer's weights
from keras import backend as K  
# Backend module for Keras that provides low-level operations and functions


class RsNt:
    @staticmethod
    def rsdual_mod(dta, nm_fltrs, strde, chnnl_dim, rdce=False,
                   regularization=0.0001, bn_epsln=2e-5, bn_momntm=0.9):

        # The shortcut branch of the RsNt module is initialized as the input (identity) data
        shrtct = dta

        # The 1st block of the RsNt module consisting of 1x1 convolutions
        btchnrm1 = BatchNormalization(axis=chnnl_dim, epsilon=bn_epsln,
                                      momentum=bn_momntm)(dta)
        actvtion1 = Activation("relu")(btchnrm1)
        conv1 = Conv2D(int(nm_fltrs * 0.25), (1, 1), use_bias=False,
                       kernel_regularizer=l2(regularization))(actvtion1)

        # The 2nd block of the ResNet module consisting of 3x3 convolutions
        btchnrm2 = BatchNormalization(axis=chnnl_dim, epsilon=bn_epsln,
                                      momentum=bn_momntm)(conv1)
        actvtion2 = Activation("relu")(btchnrm2)
        conv2 = Conv2D(int(nm_fltrs * 0.25), (3, 3), strides=strde,
                       padding="same", use_bias=False,
                       kernel_regularizer=l2(regularization))(actvtion2)

        # The 3rd block of the RsNt module consisting of another set of 1x1 convolutions
        batchnorm3 = BatchNormalization(axis=chnnl_dim, epsilon=bn_epsln,
                                        momentum=bn_momntm)(conv2)
        actvtion3 = Activation("relu")(batchnorm3)
        conv3 = Conv2D(nm_fltrs, (1, 1), use_bias=False,
                       kernel_regularizer=l2(regularization))(actvtion3)

        # If we need to reduce the spatial size, apply a convolutional layer to the shortcut
        if rdce:
            shrtct = Conv2D(nm_fltrs, (1, 1), strides=strde,
                            use_bias=False, kernel_regularizer=l2(regularization))(actvtion1)

        # Add together the shortcut and the final convolution
        x = add([conv3, shrtct])

        # Return the add as the output of the RsNt module
        return x

    @staticmethod
    def build(wdth, hieght, dpth, nm_clsses, stgs, fltrs,
              regularization=0.0001, bn_epsln=2e-5, bn_momntm=0.9):
        # Initialize the input shape to be "channels last" and the channels dimension itself
        input_shape = (hieght, wdth, dpth)
        chnnl_dim = -1

        # If we are using "channels first", update the input shape and channels dimension
        if K.image_data_format() == "channels_first":
            input_shape = (dpth, hieght, wdth)
            chnnl_dim = 1

        # Set the input and apply batch normalization
        inputs = Input(shape=input_shape)
        x = BatchNormalization(axis=chnnl_dim, epsilon=bn_epsln,
                               momentum=bn_momntm)(inputs)

        # Loop over the number of stages
        for i in range(0, len(stgs)):
            # First, we need to initialize the stride, then apply a rsdual_mod
            # It is used to reduce the spatial size of the input volume
            stride = (1, 1) if i == 0 else (2, 2)
            x = RsNt.rsdual_mod(x, fltrs[i + 1], stride,
                                chnnl_dim, rdce=True, regularization=regularization,
                                bn_epsln=bn_epsln, bn_momntm=bn_momntm)

            # Loop over the number of layers in the stage
            for j in range(0, stgs[i] - 1):
                # Apply an RsNt module
                x = RsNt.rsdual_mod(x, fltrs[i + 1],
                                    (1, 1), chnnl_dim,
                                    regularization=regularization,
                                    bn_epsln=bn_epsln, bn_momntm=bn_momntm)

        # Apply batch normalization => activation => pooling
        x = BatchNormalization(axis=chnnl_dim, epsilon=bn_epsln,
                               momentum=bn_momntm)(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((8, 8))(x)

        # Softmax classifier
        x = Flatten()(x)
        x = Dense(nm_clsses, kernel_regularizer=l2(regularization))(x)
        x = Activation("softmax")(x)

        # Create the model
        model = Model(inputs, x, name="resnet")

        # Return the constructed network architecture
        return model