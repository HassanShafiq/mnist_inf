from keras.models import model_from_json
import numpy
import os
import tensorflow as tf

# Downloading the MNIST Dataset:
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Pre-Processing the Dataset:
x_train.shape

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

# Load json and compile the model:
print("Loading Pre-Trained model (pima_json) from the Disk !")
json_file = open('mnist_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights("mnist_weights.h5")
print("Loaded model from disk")

# Model Summary:
model.summary()

# Model Compilation:
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

_ = input("Press any key to proceed for model Evaluation  ... !")
modelEval = model.evaluate(x_test, y_test)
print("MNIST Model Evaluation Complete !!!")
print(modelEval)

img_rows = 28
img_cols = 28

_ = input("Press any key to proceed for Inferencing/Predictions ... !")
true_pred = 0
false_pred = 0

for image in range(0, len(x_test)):
    pred_npy = model.predict(x_test[image].reshape(1, img_rows, img_cols, 1))
    pred = pred_npy.argmax()
    
    #print("Orinal Image: ", y_test[image])
    #print("Predicted: ", pred)
    if pred == y_test[image]:
        true_pred += 1
    if pred != y_test[image]:
        false_pred += 1

print("Total True Predictions on Test Dataset: ", true_pred)
print("Total False Predictions on Test Dataset: ", false_pred)

#print("Prediction for Input Image: ", y_test[image_index])
#print("Numpy Prediction Output for the Input Sample: ", pred)
#print(pred.argmax())
