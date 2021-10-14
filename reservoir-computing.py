from easyesn import PredictionESN
from easyesn import backend
import numpy as np
import matplotlib.pyplot as plt

# input data
trainingData = np.genfromtxt('training-set.csv', delimiter=',')
testData = np.genfromtxt('test-set-10.csv', delimiter=',')

trainingOutput = trainingData[:, :-1]
trainingInput = trainingData[:, 1:]


esn = PredictionESN(n_input=3, n_output=3, n_reservoir=500, regressionParameters=0.01, activation=backend.tanh)
esn.fit(outputData=trainingOutput, inputData=trainingInput, transientTime="Auto", verbose=1)
prediction = esn.predict(inputData=testData.T, verbose=1)

comb = np.concatenate((testData.T, prediction), axis=0)

# plot to check results
x = np.linspace(0, 200, num=200)
f, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.plot(x, comb[:, 0])
ax1.set_title('Sharing Y axis')
ax2.scatter(x, comb[:, 1])
ax3.scatter(x, comb[:, 2])

plt.show()
