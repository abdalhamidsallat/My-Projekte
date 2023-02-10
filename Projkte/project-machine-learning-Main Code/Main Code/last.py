# import the necessary library for the code "tensorflo and keras and numpy"
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense
from keras.models import Model 
from keras.layers import Input, Dense
import numpy as np
import os
import json
import tensorflow as tf
from keras.layers import Input, Flatten
from random import *


#
import matplotlib
# Specifying the backend to be used before importing pyplot
# to avoid "RuntimeError: Invalid DISPLAY variable"
# matplotlib.use('agg')
import matplotlib.pyplot as plt
from IPython.display import clear_output
import keras


class TrainingPlot(keras.callbacks.Callback):
    
    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []
    
    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        
        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))
        
        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:
            
            # Clear the previous plot
            clear_output(wait=True)
            N = np.arange(0, len(self.losses))
            
            # You can chose the style of your preference
            # print(plt.style.available) to see the available options
            plt.style.use("seaborn")
            
            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure()
            plt.plot(N, self.losses, label = "train_loss")
            #plt.plot(N, self.acc, label = "train_acc")
            plt.plot(N, self.val_losses, label = "val_loss")
            #plt.plot(N, self.val_acc, label = "val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            plt.show()

plot_losses = TrainingPlot()
#


# load the Test Data 
if os.path.exists("training_data.json"):
   with open('training_data.json', 'r') as json_file:
     probe = json.load(json_file)
     #print(probe)
    
else:
    print("Die Datei ist nicht vorhanden")


# save the Data in array 
original_data=np.array(json.loads(probe))

#Randome function
x = randint(1, 199)
print('',x)
test_1=original_data[x] 
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


# Encoder layers
encoder1 = Dense(1024, activation='relu')(input_layer)
encoder2 = Dense(512, activation='relu')(encoder1)
encoder3 = Dense(256, activation='relu')(encoder2)


# Decoder layers
decoder1 = Dense(256, activation='relu')(encoder3)
decoder2 = Dense(512, activation='relu')(decoder1)
decoder3 = Dense(2600)(decoder2)

model = Sequential()

# Create the autoencoder model
model = Model(input_layer,decoder3)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# fit the model
model.fit(X_train, Y_train, epochs=26, batch_size=32, verbose=1,validation_data=(X_train,Y_train),callbacks=[plot_losses])

# evaluate the model
scores = model.evaluate(X_train, Y_train, verbose=0)
for i in range(len(model.metrics_names)):
    print("%s: %.2f" % (model.metrics_names[0][i], scores[i]))

# accuracy
accuracy = model.evaluate(X_train, Y_train)
print('Accuracy: %.2f' % (accuracy*100))


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


