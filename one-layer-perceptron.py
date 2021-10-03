import csv
import numpy as np

# read both csv files
with open('training_set.csv', newline='') as file:
    reader = csv.reader(file)
    trainingSet = []
    for row in reader:
        trainingSet.append([float(i) for i in row])

with open('validation_set.csv', newline='') as file:
    reader = csv.reader(file)
    validationSet = []
    for row in reader:
        validationSet.append([float(i) for i in row])


training = np.array(trainingSet)
validation = np.array(validationSet)

# normalize data
x1Mean = np.mean(training, axis=0)[0]
x2Mean = np.mean(training, axis=0)[1]
x1Std = np.std(training, axis=0)[0]
x2Std = np.std(training, axis=0)[1]

training[:, 0] = (training[:, 0] - x1Mean) / x1Std
training[:, 1] = (training[:, 1] - x2Mean) / x2Std
t = training[:, 2]

# normalize data
validation[:, 0] = (validation[:, 0] - x1Mean) / x1Std
validation[:, 1] = (validation[:, 1] - x2Mean) / x2Std

# variables
eta = 0.005
hiddenNeurons = 20
pVal = len(validation)
C = 1e6
trial = 0
previousLow = 1
# initiate weights etc
w = np.random.normal(loc=0, size=[hiddenNeurons, 2], scale=1)
W = np.random.normal(loc=0, size=[1, hiddenNeurons], scale=1)
theta = np.zeros([hiddenNeurons, 1])
Theta = 0
x = training[:, [0, 1]]
V = np.zeros([hiddenNeurons, 1])
O = 0
B = np.zeros([hiddenNeurons, 1])
VError = 0
OError = 0
def dg(val):
    return 1/np.power(np.cosh(val), 2)


while C > 0.12:
    # choose random mu
    mu = np.random.randint(low=0, high=len(training))

    # propagate forward
    wij = w
    vj = x[mu]
    b = np.reshape(np.sum(wij*vj, axis=1), (hiddenNeurons, 1))
    V = np.tanh(b - theta)

    # propagate forward
    wjk = W
    vk = V
    B = np.dot(wjk, vk)  # scalar
    O = np.tanh(B - Theta)

    # output error
    OError = (t[mu] - O) * dg(B - Theta)

    # propagate backward
    VError = OError*np.transpose(W)*dg(b-theta)

    # change weights
    w += eta * VError * x[mu]
    W += eta * OError * np.reshape(V, (1, hiddenNeurons))
    theta -= eta * VError
    Theta -= eta * OError

    # reset
    trial += 1
    if trial % 1000 == 0:
        C = 0
        x_ = validation[:, [0, 1]]
        t_ = validation[:, 2]
        for mu_ in range(pVal):

            # propagate forward
            wij = w
            vj = x_[mu_]
            b_ = np.reshape(np.sum(wij * vj, axis=1), (hiddenNeurons, 1))
            V_ = np.tanh(b_ - theta)

            # propagate forward
            wjk = W
            vk = V_
            B_ = np.dot(wjk, vk)  # scalar
            O_ = np.tanh(B_ - Theta)

            if np.sign(O_) == 0:
                C += (1 / (2 * pVal)) * np.abs(1 - t_[mu_])
            else:
                C += (1 / (2 * pVal)) * np.abs(np.sign(O_) - t_[mu_])

        if C < previousLow:
            previousLow = C
            print("New low C:{}, epoch:{}".format(C[0], int(trial / 1000)))

        if trial/1000 % 100 == 0:
            print("Passed {} epochs..".format(int(trial / 1000)))

'''
np.savetxt('w1.csv', w, delimiter=',')
np.savetxt('w2.csv', W, delimiter=',')
np.savetxt('t1.csv', theta, delimiter=',')
np.savetxt('t2.csv', Theta, delimiter=',')
'''
