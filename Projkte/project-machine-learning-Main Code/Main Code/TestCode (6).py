# import the necessary library for the code "tensorflo and keras and numpy"
from keras.models import Sequential, model_from_json
from keras.models import Model 
from keras.layers import Input, Dense
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from pylab import *
import time
from numpy import linalg as LA
#import sys
#from random import *

# load the Training Data 
if os.path.exists("training_data.json"):
   with open('training_data.json', 'r') as json_file:
     probe = json.load(json_file)
     #print(probe)
    
else:
    print("---------- Die Datei ist nicht vorhanden ----------\n")



# load the Test Data 
if os.path.exists("test_data.json"):
   with open('test_data.json', 'r') as json_file:
     test_data = json.load(json_file)
     
    
else:
    print("---------- Die Datei ist nicht vorhanden ----------\n")
    
# save Test Data in array
test_array = np.array(json.loads(test_data))
print("\n\n---------- Shape of Test Array ----------\n", test_array.shape)

# Normilize test data
norm_0 = LA.norm(test_array[0], np.inf)
test_array[0] = test_array[0]/norm_0

norm_1 = LA.norm(test_array[1], np.inf)
test_array[1] = test_array[1]/norm_1

norm_2 = LA.norm(test_array[2], np.inf)
test_array[2] = test_array[2]/norm_2



# save the Training Data in array 
original_data=np.array(json.loads(probe))
"""
#Randome function
x = randint(1, 199)
print('',x)
test_1=original_data[x] 
print("Shape of X_train: ", test_1.shape)
"""
# just to print the data on the screen to see the data "not necessary"
print("---------- Original_data ----------\n", original_data)
print("---------- Shape of Original_data ----------\n",original_data.shape)
 
# Reshape The x_train
X_train = original_data.reshape(-1, 2600)
Y_train = original_data.reshape(-1, 2600)
print("---------- Shape of X_train: ----------\n", X_train.shape)
print("---------- Shape of Y_train: ----------\n ", Y_train.shape)

# Reshape The Test Data
test_input = test_array.reshape(-1, 2600)


# Change Type of Array to float32
X_train = X_train.astype('float32')
Y_train = Y_train.astype('float32')
test_input = test_input.astype('float32')


# Input layer

input_size = 2600
hidden_1_size = 1024
hidden_2_size = 512 
code_size = 256
input_layer = Input(shape=(input_size,))
print(" ---------- Shape of input_layer  ----------\n", input_layer.shape)


def encoder(input_layer, code_size):
    layer1 = Dense(hidden_1_size, activation='linear')(input_layer)
    layer2 = Dense(hidden_2_size, activation='linear')(layer1)
    code = Dense(code_size )(layer2)
    return code
 
def decoder(encoded_data):
    layer4 = Dense(code_size, activation='linear')(encoded_data)
    layer5 = Dense(hidden_2_size, activation='linear')(layer4)
    output = Dense(input_size)(layer5)
    return output

model = Sequential()

# Create the autoencoder model
model = Model(input_layer,decoder(encoder(input_layer,code_size)))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# fit the model
history = model.fit(X_train, Y_train, epochs=150, validation_data=(X_train,Y_train),batch_size=32, verbose=1,)

# evaluate the model
scores = model.evaluate(X_train, Y_train, verbose=0)
for i in range(len(model.metrics_names)):
    print("%s: %.2f" % (model.metrics_names[0][i], scores[i]))
    
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


# Create new model in order to get the encoded data only

encoder_model = Model(inputs=model.input, outputs=encoder(input_layer, code_size))

# Get the encoded data from the last encoder layer
encoded_data = encoder_model.predict(test_input)
print("encoded_data", encoded_data, "\nShape", encoded_data.shape, "\nType", encoded_data.dtype, "\n\n\n") 

# Get the output data from the hole autoencoder
start_time = time.time()
decoder_output = model.predict(test_input)
end_time = time.time()
decoded_train = model.predict(X_train)
print("---------- Original Data ----------\n", test_input, "\nShape", test_input.shape, "\nType", test_input.dtype, "\n\n\n")
print("---------- Decoded Output ----------\n", decoder_output, "\nShape", decoder_output.shape, "\nType", decoder_output.dtype, "\n\n\n")

#Get the output figure before and after compressing data.
gen_data = decoder_output[0].reshape(52,50)
gen_data_1 = decoder_output[1].reshape(52,50)
gen_data_2 = decoder_output[2].reshape(52,50)
gen_data_3 = decoded_train[0].reshape(52,50)
fig, (ax1, ax2) = plt.subplots(1, 2)
fig_1, (ax1_1, ax2_1) = plt.subplots(1, 2)
fig_2, (ax1_2, ax2_2) = plt.subplots(1, 2)
fig_3, (ax1_3, ax2_3) = plt.subplots(1, 2)


ax1.imshow(test_array[0], cmap='inferno')
ax1.set_title("Original")

ax2.imshow(gen_data, cmap='inferno')
ax2.set_title("After Decompression")



ax1_1.imshow(test_array[1], cmap='inferno')
ax1_1.set_title("Original")

ax2_1.imshow(gen_data_1, cmap='inferno')
ax2_1.set_title("After Decompression")



ax1_2.imshow(test_array[2], cmap='inferno')
ax1_2.set_title("Original")

ax2_2.imshow(gen_data_2, cmap='inferno')
ax2_2.set_title("After Decompression")



ax1_3.imshow(original_data[0], cmap='inferno')
ax1_3.set_title("Training Data")

ax2_3.imshow(gen_data_3, cmap='inferno')
ax2_3.set_title("After Decompression")




# Compression rate

list_test_array = test_array.tolist()
json_test_array = json.dumps(list_test_array)
with open('test_array.json', 'w') as json_file:
    json.dump(json_test_array, json_file)

list_encoded_data = encoded_data.tolist()
json_encoded_data = json.dumps(list_encoded_data)
with open('Compressed_data.json', 'w') as json_file:
    json.dump(json_encoded_data, json_file)

original_size = os.path.getsize('test_array.json')
compressed_size = os.path.getsize('Compressed_data.json')
comp_ratio = (compressed_size / original_size) * 100
comp_ratio = round(comp_ratio, 2)
print("\n\n\n\n----------Compression Ratio----------\n\n")
print("The Small Array has", comp_ratio, "% of the original Array's size\n\n\n\n")


# Attempt to compare Original and Output Array
difference = np.subtract(test_input, decoder_output)
difference[difference < 0] = -1*difference[difference < 0]
original_copy = test_input
original_copy[original_copy < 0] = -1*original_copy[original_copy < 0]
sum_original = np.sum(original_copy)
sum_difference = np.sum(difference)
changed_value = (sum_difference/sum_original)*100
print("---------- Comparing Original and Output ----------\nThe Decompressed Data is different from the Original by", changed_value, "%\n\n\n" )

# Run Time
running_time = end_time - start_time
print("\n\n\n---------- Computation Time of encoding and Decoding----------\n")
print(running_time, "seconds\n")

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
print("niceeeeeeeee ")