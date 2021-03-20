"""
Implementation  of Error Space of the Sigmoid Neuron
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x, w, b):
    return 1 / (1 + np.exp(-(w * x + b)))

def calc_loss(X, Y, w_est, b_est):
    loss = 0
    for x, y in zip(X, Y):
        loss += (y - sigmoid(x, w_est, b_est))**2
    return loss

w_unknown = 0.5
b_unknown = 0.25

X = np.random.random(25) * 20 - 10
Y = sigmoid(X, w_unknown, b_unknown)

W = np.linspace(0, 2, 100)
B = np.linspace(-1, 1, 100)

WW, BB = np.meshgrid(W, B)

Loss = np.zeros(WW.shape)

for i in range(WW.shape[0]):
    for j in range(WW.shape[1]):
        Loss[i, j] = calc_loss(X, Y, WW[i, j], BB[i, j])


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(WW, BB, Loss, cmap='viridis')
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('Loss')
ax.view_init(35, 245)
plt.savefig('Dev-Training-DL/Exercises/Loss_Graphs/sigmoid_neuron_loss_graph.png')
plt.show()

print(np.argmin(Loss))

ij = np.argmin(Loss)
i = int(np.floor(ij / Loss.shape[1]))
j = int(ij - i * Loss.shape[1])
print(i, j)
print(WW[i, j], BB[i, j])
