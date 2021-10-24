import numpy as np
import tensorflow as tf

# get test and validation data from keras
(trainingImages, trainingLabels), (validationImages, validationLabels) = tf.keras.datasets.mnist.load_data()

# reshape data and add channel (greyscale)
trainingImages = (trainingImages/255).reshape([60000, 28, 28, 1])
validationImages = (validationImages/255).reshape([10000, 28, 28, 1])

# target vectors
trainingLabels = tf.keras.utils.to_categorical(trainingLabels, 10)
validationLabels = tf.keras.utils.to_categorical(validationLabels, 10)

# model
model = tf.keras.Sequential()
dropOutPercentage = 0.4
kernelSize = (3, 3)
poolSize = (2, 2)

model.add(tf.keras.Input(shape=(28, 28, 1)))
# convolutional layer 1
model.add(tf.keras.layers.Conv2D(32, kernel_size=kernelSize, activation='relu'))
# max pooling layer 1
model.add(tf.keras.layers.MaxPooling2D(pool_size=poolSize))
# drop out 1
model.add(tf.keras.layers.Dropout(dropOutPercentage))
# convolutional layer 2
model.add(tf.keras.layers.Conv2D(64, kernel_size=kernelSize, activation='relu'))
# max pooling layer 2
model.add(tf.keras.layers.MaxPooling2D(pool_size=poolSize))
# drop out 2
model.add(tf.keras.layers.Dropout(dropOutPercentage))
# flatten
model.add(tf.keras.layers.Flatten())
# hidden layer 1
model.add(tf.keras.layers.Dense(128, activation='relu'))
# drop out 3
model.add(tf.keras.layers.Dropout(dropOutPercentage))
# output layer
model.add(tf.keras.layers.Dense(10, activation='softmax'))
# model.summary()

# training
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.fit(trainingImages, trainingLabels, epochs=40, validation_data=(validationImages, validationLabels))

# get test data from canvas and reshape
testImages = np.moveaxis(np.load("xTest2.npy")/255, [0, 1, 2, 3], [-3, -2, -1, -4])

# testing
predictions = model.predict(testImages, verbose=0)
result = predictions.argmax(axis=1)
np.savetxt(fname='classifications.csv', X=result, fmt='%s', delimiter=',')
