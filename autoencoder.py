import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# get test and validation data from keras
(trainingImages, _), (validationImages, validationLabels) = tf.keras.datasets.mnist.load_data()

# reshape data
trainingImages = (trainingImages/255).reshape(60000, -1)
validationImages = (validationImages/255).reshape(10000, -1)

# parameters
neurons = 2
epochs = 800
batchSize = 8192
learningRate = 1/1000
nDigits = 10
autoEncoder = tf.keras.Sequential()
weights0 = tf.keras.initializers.GlorotUniform()

# model
autoEncoder.add(tf.keras.Input(shape=(784,)))
# fully connected layer 1
autoEncoder.add(tf.keras.layers.Dense(50, activation='relu', kernel_initializer=weights0, name='dense_1'))
# bottleneck layer
autoEncoder.add(tf.keras.layers.Dense(neurons, activation='relu', kernel_initializer=weights0, name='bottleneck'))
# fully connected layer 2
autoEncoder.add(tf.keras.layers.Dense(784, activation='relu', kernel_initializer=weights0, name='dense_2'))


# half-mean-squared loss
def my_loss(y_true, y_prediction):
    squared_difference = tf.square(y_true - y_prediction)
    return tf.reduce_mean(squared_difference, axis=-1)/2


# training
autoEncoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate), loss=my_loss)
autoEncoder.fit(trainingImages, trainingImages, epochs=epochs, batch_size=batchSize, shuffle=True,
                validation_data=(validationImages, validationImages))

# testing
autoEncoderOutputs = autoEncoder.predict(validationImages, verbose=0)

# bottleneck outputs
bottleneckLayer = autoEncoder.get_layer(name='bottleneck').output
bottleneckOutputs = tf.keras.models.Model(inputs=autoEncoder.input, outputs=bottleneckLayer).predict(validationImages)

# plot all digits in a montage
repeat = 100
img = 0
digitIndices = np.zeros(nDigits, dtype=int)
layerOutputs = np.zeros((nDigits*repeat, neurons+1))
for n in range(repeat):
    # find one of each digit randomly
    searchDigit = 0
    while searchDigit < 10:
        randIndices = np.random.choice(validationLabels.shape[0], size=1, replace=False)

        if validationLabels[randIndices] == searchDigit:
            digitIndices[searchDigit] = randIndices
            searchDigit += 1

    # plot all digits in a montage
    for i in range(nDigits):
        if n == repeat-1:
            plt.figure(0)

            # original
            ax = plt.subplot(nDigits, 2, 2 * i + 1)  # 1, 3, 5..
            plt.imshow(validationImages[digitIndices[i]].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # auto encoder output
            ax = plt.subplot(nDigits, 2, 2 * i + 2)  # 2, 4, 6..
            plt.imshow(autoEncoderOutputs[digitIndices[i]].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        # bottleneck layer outputs
        layerOutputs[img, :-1] = bottleneckOutputs[digitIndices[i], :]  # store neurons
        layerOutputs[img, -1:] = i
        img += 1

    # plt.show()


# scatter plot for 2 neuron bottleneck
if neurons == 2:
    plt.figure(1)
    digits = layerOutputs[:, 2]
    colors = ['red', 'green', 'darkred', 'orange', 'wheat', 'purple', 'violet', 'navy', 'cornflowerblue', 'springgreen']

    for idx in range(nDigits):
        neuron1 = layerOutputs[np.where(digits == idx)][:, 0]
        neuron2 = layerOutputs[np.where(digits == idx)][:, 1]

        if idx == 1 or idx == 9:
            plt.scatter(neuron1, neuron2, alpha=0.8, c=colors[idx], s=40, linewidths=1, edgecolors='black')
        else:
            plt.scatter(neuron1, neuron2, alpha=0.3, c=colors[idx], s=40, linewidths=1, edgecolors='black')

    plt.legend([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.title('Scatter plot of the output of 2 bottleneck neurons')
    plt.xlabel("Neuron 1 output")
    plt.ylabel("Neuron 2 output")


# test different bottleneck inputs
if neurons == 4:
    inspectDigit = 9  # 1 and 9 were well produced
    digits = layerOutputs[:, 4]
    neuron1 = layerOutputs[np.where(digits == inspectDigit)][:, 0]
    neuron2 = layerOutputs[np.where(digits == inspectDigit)][:, 1]
    neuron3 = layerOutputs[np.where(digits == inspectDigit)][:, 2]
    neuron4 = layerOutputs[np.where(digits == inspectDigit)][:, 3]

    # calculate averages for each neuron
    neuronAverages = [
        np.sum(neuron1)/repeat,
        np.sum(neuron2)/repeat,
        np.sum(neuron3)/repeat,
        np.sum(neuron4)/repeat
    ]

    # model output
    modelOutput = autoEncoder.layers[-1](np.array(neuronAverages).reshape(1, 4))

    # plot image
    plt.figure(1)
    plt.imshow(np.array(modelOutput).reshape(28, 28))
    plt.gray()

# save
plt.show()
np.savetxt(fname='all-layer-outputs.csv', X=layerOutputs, fmt='%s', delimiter=',')
