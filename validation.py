import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import mnistdataset as mndt

# Define a range of learning rates
learning_rates = [0.001, 0.01, 0.1, 1.0]

validation_losses = []
validation_accuracies = []

# Load the pretrained model
pretrained_model = keras.models.load_model('/Users/nikhilkushwaha/ocr/CombineDatasetModels/final_model_resnet/Combined_Resnet_20_Epochs.h5')

# Iterate over the learning rates
for lr in learning_rates:
    # Compile the model with the current learning rate
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    pretrained_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Evaluate the model on the validation set
    evaluation = pretrained_model.evaluate(mndt.tst_dta_X,mndt.tst_dta_Y, verbose=0)
    
    # Record the validation loss and accuracy
    validation_loss = evaluation[0]
    validation_accuracy = evaluation[1]
    
    validation_losses.append(validation_loss)
    validation_accuracies.append(validation_accuracy)


# Plot the validation curve
plt.figure(figsize=(12, 6))

# Plot validation loss
plt.plot(learning_rates, validation_losses, '-o', label='Validation Loss')

# Plot validation accuracy
plt.plot(learning_rates, validation_accuracies, '-*', label='Validation Accuracy')

plt.xlabel('Learning Rate')
plt.ylabel('Value')
plt.legend()
plt.title('Validation Loss and Accuracy')
plt.show()




