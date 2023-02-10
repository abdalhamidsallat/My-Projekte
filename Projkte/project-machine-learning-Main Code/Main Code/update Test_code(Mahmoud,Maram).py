
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense
from keras.models import Model 
from keras.layers import Input, Dense
import tensorflow
import numpy as np
import os
import json
# dataset Input
with open('funktion_k_1.json', 'r') as json_file:
  probe = json.load(json_file)
if os.path.exists("funktion_k_1.json"):
     print('vorhanden')

else:
    print("Die Datei ist nicht vorhanden")

#other dataset  
#with open('probe.josn', 'r') as json_file:
 # probe = json.load(json_file)
  
print(type(probe))
arr_1=np.asarray(probe)
print(type(arr_1))
expand_1 = np.expand_dims(arr_1, axis=1)

'''
print("---------------------------------------------------------")
print (arr_2)
print(arr_2.ndim)
print("Shape of X_train: ", arr_2.shape)
'''
X_train = expand_1[:,0:9]
Y = expand_1[:,0:8]
print (X_train)

print (Y)

print(X_train.ndim)
print(Y.ndim)
print("Shape of X_train: ", X_train.shape)
print("Shape of y_train: ", Y.shape)

input_data=expand_1
input_size=len(expand_1)
#input_size = 9
hidden_1_size = 12
hidden_2_size_smolar = 8
code_size = 1
input_layer = Input(shape=(input_size,))

def encoder(input_data, code_size):
    layer1 = Dense(hidden_1_size, activation='relu')(input_data)
    layer2 = Dense(hidden_2_size_smolar, activation='relu')(layer1)
    code   = Dense(code_size, activation='sigmoid')(layer2)
 
    return code
 
def decoder(encoded_data):
    layer1 = Dense(hidden_2_size_smolar, activation='relu')(encoded_data)
    layer2 = Dense(hidden_1_size, activation='relu')(layer1)
    output = Dense(input_size, activation='sigmoid')(layer2)
 
    return output

model = Sequential()

model = model = Model(input_layer, encoder(input_layer, 9))

# we can define encoder like this
# model.add(Dense(12, input_dim=9, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X_train, Y, epochs=150, batch_size=10, verbose=0)
# evaluate the model
scores = model.evaluate(X_train, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")



# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


print("niceeeeee")
