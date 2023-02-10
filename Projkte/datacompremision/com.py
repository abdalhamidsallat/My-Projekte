# MLP for Pima Indians Dataset Serialize to JSON and HDF5
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.models import Model 
from keras.layers import Input, Dense
import tensorflow
import numpy
import os
# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X_train = dataset[:,0:8]
Y = dataset[:,8]
print (X_train)
print("---------------------------------------------------------")
print (Y)
# create model 
# Autoencoder
input_size = len(dataset)
hidden_1_size = 12
hidden_2_size_smolar = 8
code_size = 1

def encoder(input_data_shape, code_size):
    layer1 = Dense(hidden_1_size, activation='relu')(input_data_shape)
    layer2 = Dense(hidden_2_size_smolar, activation='relu')(layer1)
    code   = Dense(code_size, activation='sigmoid')(layer2)
 
    return code
 
def decoder(encoded_data):
    layer1 = Dense(hidden_2_size_smolar, activation='relu')(encoded_data)
    layer2 = Dense(hidden_1_size, activation='relu')(layer1)
    output = Dense(input_size, activation='sigmoid')(layer2)
 
    return output

model = Sequential()
# creat model
input_data_shape = Input(shape=(input_size, ))
model = Model(input_data_shape, encoder(input_data_shape, 8))

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

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X_train, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
print("niceeeeee")