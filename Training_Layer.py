import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
from tensorflow.keras.callbacks import TensorBoard
from keras import regularizers
import time
from keras import regularizers
from keras.optimizers import Adam


NAME = "GSF.trash{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
X = pickle.load(open("X2.pickle","rb"))
y = pickle.load(open("y2.pickle","rb"))#load images

X = X/255.0

model = Sequential()
model.add(Conv2D(64,(3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())
model.add(Dense(256,activation='relu',kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.2))
'''
model.add(Activation('relu'))

model.add(kernel_regularizer=regularizers.l1(0.01))'''

model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(
    #Adam(lr=0.0001),
              loss="binary_crossentropy",
              optimizer = "adam",
             metrics=['accuracy'])

model.fit(X,y,batch_size=32,epochs=20,validation_split=0.1,callbacks = [tensorboard])
model.save("GSF2.model")
