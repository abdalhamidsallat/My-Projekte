# import the necessary library for the code "tensorflo and keras and numpy"
from tensorflow.keras.models import Sequential, model_from_json
#from tensorflow.keras.layers import Dense
from keras.models import Model 
from keras.layers import Input, Dense
import numpy as np
import os
import json
import tensorflow as tf
#from keras.layers import Input, Flatten
from random import *
import matplotlib.pyplot as plt

# load the Test Data 

if os.path.exists("test_data.json"):
   with open('test_data.json', 'r') as json_file:
     probe = json.load(json_file)
     #print(probe)
    
else:
    print("Die Datei ist nicht vorhanden")


# save the Data in array 
original_data=np.array(json.loads(probe))
print("hiiiiiiiiiiiiiiiiiiiiiiiiiiiiii")

#Randome function
x = randint(1, 199)

print('',x)
test_1=original_data[2] 
print("Shape of X_train: ", test_1.shape)

# just to print the data on the screen to see the data "not necessary"
print("original_data", original_data)
print("Shape of Original_data",original_data.shape)
 
# Reshape The x_train
X_train = test_1.reshape(-1, 2600)
Y_train = test_1.reshape(-1, 2600)
print("Shape of X_train: ", X_train.shape)
print("Shape of Y_train: ", Y_train.shape)




# Input layer
input_layer = Input(shape=(2600,))
print("Shape of input_layer", input_layer.shape)




input_size = 2600
hidden_1_size = 1024
hidden_2_size_smolar = 512
code_size = 256
input_layer = Input(shape=(input_size,))


def encoder(input_layer, code_size):
    layer1 = Dense(hidden_1_size, activation='relu')(input_layer)
    layer2 = Dense(hidden_2_size_smolar, activation='relu')(layer1)
    code   = Dense(code_size, activation='relu')(layer2)
 
    return code
 
def decoder(encoded_data):
    layer1 = Dense(code_size, activation='relu')(encoded_data)
    layer2 = Dense(hidden_2_size_smolar, activation='relu')(layer1)
    #output = Dense(input_size, activation='relu')(layer2)
    output = Dense(input_size)(layer2)
 
    return output

# # Encoder layers
# encoder1 = Dense(1024, activation='relu')(input_layer)
# encoder2 = Dense(512, activation='relu')(encoder1)
# encoder3 = Dense(256, activation='relu')(encoder2)


# # Decoder layers
# decoder1 = Dense(256, activation='relu')(encoder3)
# decoder2 = Dense(512, activation='relu')(decoder1)
# decoder3 = Dense(2600)(decoder2)

model = Sequential()

# Create the autoencoder model
model = Model(input_layer,decoder(encoder(input_layer,code_size)))
#model = Model(input_layer,decoder3)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# fit the model
history = model.fit(X_train, Y_train, epochs=150, validation_data=(X_train,Y_train),batch_size=32, verbose=1,)


# Loss Funktionen plotten:
# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()     


# evaluate the model
scores = model.evaluate(X_train, Y_train, verbose=0)
for i in range(len(model.metrics_names)):
    print("%s: %.2f" % (model.metrics_names[0][i], scores[i]))


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
print(model.summary())


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
print("nice :)")


