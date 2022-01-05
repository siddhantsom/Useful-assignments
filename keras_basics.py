import numpy as np
from numpy import genfromtxt #generates array from textfile

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report

from keras.models import Sequential, load_model
from keras.layers import Dense


data = genfromtxt(r"C:\Personal\Udemy\Computer-Vision-with-Python\DATA\bank_note_data.txt", delimiter = ',')

X = data[:, 0:4]
y = data[:,4]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

scaler_object = MinMaxScaler()
scaler_object.fit(X_train)

scaled_X_train = scaler_object.transform(X_train)
scaled_X_test = scaler_object.transform(X_test)

model = Sequential()
model.add(Dense(4, input_dim = 4, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(scaled_X_train, y_train, epochs = 50, verbose = 2)

predictions = (model.predict(scaled_X_test) > 0.5).astype("int32")
cf = confusion_matrix(y_test, predictions)

print(classification_report(y_test, predictions))