import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# get test and validation data from keras
(trainingImages, _), (validationImages, validationLabels) = tf.keras.datasets.mnist.load_data()

# reshape data
trainingImages = (trainingImages/255).reshape(60000, -1)
validationImages = (validationImages/255).reshape(10000, -1)

# model
model = tf.keras.Sequential()
initializeWeights = tf.keras.initializers.GlorotNormal()
encodeNeurons = 4
model.add(tf.keras.Input(shape=(784,)))
# fully connected layer 1
model.add(tf.keras.layers.Dense(50, activation='relu', kernel_initializer=initializeWeights))
# bottleneck layer
model.add(tf.keras.layers.Dense(encodeNeurons, activation='relu', kernel_initializer=initializeWeights))
# fully connected layer 2
model.add(tf.keras.layers.Dense(784, activation='relu', kernel_initializer=initializeWeights))

encodedLayer = model.get_layer(name='dense_1')
decodedLayer = model.get_layer(name='dense_2')


# half-mean-squared loss function
def my_loss_fn(y_true, y_prediction):
    squared_difference = tf.square(y_true - y_prediction)*0.5
    return tf.reduce_mean(squared_difference, axis=-1)


# training
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1/1000), loss=my_loss_fn, metrics=['accuracy'])
model.fit(trainingImages, trainingImages, epochs=1000, batch_size=8192, shuffle=True,
          validation_data=(validationImages, validationImages))

# testing
predictions = model.predict(validationImages, verbose=0)


# plot
nDigits = 10
searchDigit = 0
digitIndexes = np.zeros(nDigits, dtype=int)
fig = plt.figure()

# find one of each digit
while searchDigit < 10:
    randomIndex = np.random.choice(validationLabels.shape[0], size=1, replace=False)
    if validationLabels[randomIndex] == searchDigit:
        digitIndexes[searchDigit] = randomIndex
        searchDigit += 1
for i in range(nDigits):
    # original
    ax = plt.subplot(nDigits, 2, 2*i+1)  # 1, 3, 5..
    plt.imshow(validationImages[digitIndexes[i]].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # encoded
    ax = plt.subplot(nDigits, 2, 2*i+2)  # 2, 4, 6..
    plt.imshow(predictions[digitIndexes[i]].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
