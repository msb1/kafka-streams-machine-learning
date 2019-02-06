import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
import seaborn as sns

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers, optimizers
from sklearn.model_selection import train_test_split, GridSearchCV

"""
(1) Use 2-layer NN as equivalent to Logistic Regression Classifier
(2) This program is being used to save a small NN to file to test with DL4J 
    with Kafka Streams and Spring Cloud
(3) Test data has been generated with sklearn make_classification for testing
    logistic regression and other models

"""

numFeatures = 20

def main():
    # read data from csv file
    file='skTestData.csv'
    df=pd.read_csv(file, header=None)
    df.head()
    # change data to numpy X, y 
    data = df.values
    X = data[:, 0: numFeatures]
    y = data[:, numFeatures]
    # split data into train/test samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    # build model - train 2 layer neural net
    seed = 7
    np.random.seed(seed)

    # create 2 layer model
    model = Sequential()
    # model.add(Dense(21, input_dim=20, activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
    # model.add(Dense(1, activation='tanh'))
    model.add(Dense(21, input_dim=20, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callbacks = [keras.callbacks.EarlyStopping(monitor='loss', patience=2)]

    # Fit the model
    model.fit(X_train, y_train, epochs=100, batch_size=5, verbose=1)

    # evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print("\n loss: {0:.2f}   accuracy: {1:.2f} ".format(loss, accuracy))

    # Save model.
    model.save('test_model_gen1.h5')


if __name__ == '__main__':
    main()
    print("model-gen-1 script is complete...")
