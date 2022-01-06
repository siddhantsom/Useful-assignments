from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

import numpy as np
import matplotlib.pyplot as plt


from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten

from sklearn.metrics import classification_report

y_cat_test = to_categorical(y_test, 10)
y_cat_train = to_categorical(y_train, 10)

x_train = x_train/x_train.max()
x_test = x_test/x_test.max()


model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (4,4), input_shape = (28, 28, 1), activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2)))

model.add(Flatten())

model.add(Dense(128, activation = 'relu'))

model.add(Dense(10, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

model.fit(x_train, y_cat_train, epochs = 2)

model.evaluate(x_test, y_cat_test)

predict_x=model.predict(x_test) 
classes_x=np.argmax(predict_x,axis=1)

print(classification_report(y_test, classes_x))