
# Import the 'resnet' module as 'rsnt' for ResNet architecture
import resnet as rsnt

# Import the 'Adam' optimizer from the 'keras.optimizers' library
from keras.optimizers import Adam

# Import the 'mnistdataset' module as 'mndt' for the MNIST dataset
import mnistdataset as mndt

# Import the 'numpy' library as 'np' for numerical computing
import numpy as np

# Import the 'os' library for operating system related functionalities
import os

# Import the 'tensorflow' library as 'tf' for machine learning and deep learning
import tensorflow as tf

# Import the 'load_model' function from the 'keras.models' library for loading pre-trained models
from keras.models import load_model

# Set the  epochs  counts 
nm_epchs = 1

# Set the init learning rate
lrning_rte = 0.01

# Set the batch size
btch_sze = 256

print("[INFO] Compiling the ocr model...")
optmzr = Adam(learning_rate=lrning_rte)

# Build the ResNet model
model = rsnt.RsNt.build(32, 32, 1, len(mndt.lbl_bin.classes_), (3, 3, 3),
                          (64, 64, 128, 256), regularization=0.0001)

# Compile the model with specified loss, optmizr, and metrics
model.compile(loss="categorical_crossentropy", optimizer=optmzr,
              metrics=["accuracy"])

# Define the checkpoint path and directory
chckpnt_pth = "/Users/nikhilkushwaha/ocr/CombineDatasetModels/training_1/cp.h5"
chckpnt_drctry = os.path.dirname(chckpnt_pth)

# Define the callback to save the model's weights
chckpnt_cllbck = tf.keras.callbacks.ModelCheckpoint(filepath=chckpnt_pth, save_weights_only=False, verbose=1)

# Check if the model wieghts  have already been saved or not

if os.path.exists(chckpnt_pth):
    # Load the saved model weights
    model = load_model(chckpnt_pth)
    print("Loaded model weights")

# # # Train the model and save the weights
# hstry = model.fit(
#     mndt.aug.flow(mndt.trn_dta_X, mndt.trn_dta_Y, batch_size=btch_sze),
#     validation_data=(mndt.tst_dta_X, mndt.tst_dta_Y),
#     steps_per_epoch=len(mndt.trn_dta_X) // btch_sze,
#     epochs=nm_epchs,
#     class_weight=mndt.class_weight,
#     verbose=1,
#     callbacks=[chckpnt_cllbck])

# Save the final trained model
save_path = '/Users/nikhilkushwaha/ocr/CombineDatasetModels/final_model_resnet/Combined_Resnet_20_Epochs.h5'
model.save(save_path)