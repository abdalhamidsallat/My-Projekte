#
from keras.layers import Input, Dense
from keras.datasets import mnist
from keras.models import Model 
import numpy as np
import os
#import pandas as pd
#temperature = pd.read_csv(r'D:\datacom\mnist.npz')
#temperature = tf.data.TextLineDataset(r'D:\datacom\mnist.npz')
#print(temperature)
import matplotlib.pyplot as plt
# To load the mnist data
from keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
import matplotlib.image as mpimg
import tensorflow as tf
# tf.keras.utils.text_dataset_from_directory(
#     directory= 'D:\datacom\mnist.npz',
#     labels="inferred",
#     label_mode="int",
#     class_names=None,
#     batch_size=32,
#     max_length=None,
#     shuffle=True,
#     seed=None,
#     validation_split=None,
#     subset=None,
#     follow_links=False,
# )
np.random.seed(7)

IMAGE_SIZE = 784 # 28 * 28 pixels
def encoder(input_image, code_dimention):
    layer1 = Dense(64, activation='relu')(input_image)
    layer2 = Dense(code_dimention, activation='sigmoid')(layer1)
    layer3 = Dense(code_dimention, activation='sigmoid')(layer2)
 
    return layer3
 
def decoder(encoded_image):
    layer1 = Dense(64, activation='relu')(encoded_image)
    layer2 = Dense(IMAGE_SIZE, activation='sigmoid')(layer1)
    layer3 = Dense(IMAGE_SIZE, activation='sigmoid')(layer2)
 
    return layer3

input_image = Input(shape=(IMAGE_SIZE, ))
  
model = Model(input_image, decoder(encoder(input_image, 100)))
model.compile(loss='mean_squared_error', optimizer='nadam')
#data = open("D:\datacom\mnist.npz").read()
# data_url = 'https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz'

# dataset_dir = utils.get_file(
#     origin=data_url,
#     untar=True,
#     cache_dir='stack_overflow',
#     cache_subdir='')

# dataset_dir = pathlib.Path(dataset_dir).parent

# local_dir_path = os.path.dirname('D:\datacom')
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# (x_train, _), (x_test, _) = dataset_dir
x_train = dataset[:,0:9]
x_test = dataset[:,9]

#tf.keras.datasets.mnist.load_data(path = 'D:\datacom\mnist.npz')
#.load_data()
  
# Normalize data
x_train = x_train.astype('float32') 
x_train /= 255.0
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
 
x_test = x_test.astype('float32') 
x_test /= 255.0
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Training
model.fit(x_train, x_train, batch_size=64, epochs=4, validation_data=(x_test, x_test), shuffle=True)

# Print the dimensions of the dataset
print('Train: X = ', x_train.shape)
print('Test: X = ', x_test.shape)

# Encide and  decode some digits
# encoded_imags = encoder.predict(x_test)
# dencoded_imags = decoder.predict(encoded_imags)

# import matplotlib.pyplot as plt

# n = 10 
# plt.figure(figsize=(20,4))
# for i in range(n):
#   # Display Orginal
#     ax = plt.subplot(2, n, i+1)
#     plt.imshow(x_test[i].reshape(28,28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

#   # Display reconstrjction
#     ax = plt.subplot(2, n, i+1+n)
#     plt.imshow(dencoded_imags[i].reshape(28,28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)