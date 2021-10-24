import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

# parameters
N = 500  # reservoir neurons
outputNeurons = 3
WInputVariance = 0.002
WVariance = 2 / N
k = 0.01  # ridge parameter
predictionTimeSteps = 500

# get data
trainingData = np.genfromtxt('training-set.csv', delimiter=',')
testData = np.genfromtxt('test-set-10.csv', delimiter=',')
inputNeurons, trainingTimeSteps = trainingData.shape
_, testTimeSteps = testData.shape

# initialize weights and reservoir
W = np.random.normal(loc=0, scale=WVariance**0.5, size=(N, N))
WInput = np.random.normal(loc=0, scale=WInputVariance**0.5, size=(N, inputNeurons))
R = np.zeros((N, trainingTimeSteps))  # reservoir

# feed training data
for t in range(trainingTimeSteps):
    b = np.zeros((N, 2))  # local field b
    b[:, 0] = np.dot(R[:, t].reshape(1, N), W).T.reshape(N, )
    b[:, 1] = np.dot(WInput, trainingData[:, t]).T.reshape(N, )

    # update
    if t < trainingTimeSteps - 1:
        R[:, t + 1] = np.tanh(np.sum(b, axis=1))

# ridge regression
stopExplode = k * np.identity(N)
WOutput = np.dot(np.dot(trainingData, R.T), inv(np.dot(R, R.T) + stopExplode))

# feed test data
result = np.zeros((outputNeurons, testTimeSteps + predictionTimeSteps))
for t in range(testTimeSteps + predictionTimeSteps):

    if t >= testTimeSteps:
        stepResult = np.dot(WOutput, R[:, t])

        b = np.zeros((N, 2))
        b[:, 0] = np.dot(R[:, t].reshape(1, N), W).T.reshape(N, )
        b[:, 1] = np.dot(WInput, stepResult).T.reshape(N, )

        # update
        if t < (testTimeSteps + predictionTimeSteps - 1):
            R[:, t + 1] = np.tanh(np.sum(b, axis=1))

        result[:, t] = stepResult

    else:
        b = np.zeros((N, 2))
        b[:, 0] = np.dot(R[:, t].reshape(1, N), W).T.reshape(N, )
        b[:, 1] = np.dot(WInput, testData[:, t]).T.reshape(N, )

        # update
        R[:, t + 1] = np.tanh(np.sum(b, axis=1))

        result[:, t] = testData[:, t]

# plot
fig = plt.figure()
ax1 = plt.subplot(1, 2, 1, projection='3d')
ax1.plot3D(result[0, :99], result[1, :99], result[2, :99], 'blue')
ax1.plot3D(result[0, 100:], result[1, 100:], result[2, 100:], 'orange')
ax1.title.set_text('Test data (blue) & prediction (orange)')
ax2 = plt.subplot(1, 2, 2, projection='3d')
ax2.plot3D(trainingData[0, :600], trainingData[1, :600], trainingData[2, :600], 'blue')
ax2.title.set_text('Training data 600 time steps')
plt.show()

# save
np.savetxt(fname='prediction.csv', X=result[1, 100:], fmt='%s', delimiter=',')
