import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
from keras.optimizers import SGD

#far too much repetition, just cleaning the data for the neural net
df = pd.read_csv('titanic.csv')
train, test = train_test_split(df, test_size = 0.2)
train = train.drop(['Name', 'Cabin', 'Ticket', 'PassengerId'], axis=1)
train = train.dropna()
test = test.drop(['Name', 'Cabin', 'Ticket', 'PassengerId'], axis=1)
test = test.dropna()
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
test['Sex'] = test['Sex'].map({'male':0, 'female':1})
train['Embarked'] = train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
test['Embarked'] = test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

X_train = train.drop(['Survived'], 1)
X_train = X_train.as_matrix()
X_test = test.drop(['Survived'], 1)
X_test = X_test.as_matrix()
y_train = train['Survived']
y_test = test['Survived']

# from class vector to binary matrix
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

nb_classes = 2
nb_epoch = 100
batch_size = 32

model = Sequential()
model.add(Dense(32, input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('softmax'))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size = batch_size, nb_epoch=nb_epoch, validation_data=(X_test, y_test), verbose=2)
