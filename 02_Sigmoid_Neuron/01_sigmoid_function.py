import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors


# Prints the Sigmoid value
def sigmoid(x, w, b):
    return 1/(1 + np.exp(-(w*x + b)))


print("sigmoid {:.3f}".format(sigmoid(1, 0.5, 0)))

# Graphical Representation of Sigmoid Function.
w = -1.8
b = -0.5
X = np.linspace(-10, 10, 100)
Y = sigmoid(X, w, b)

plt.plot(X, Y)
plt.show()


# Plotting Gradient Plot for Sigmoid Function(2D)
def sigmoid_2d(x1, x2, w1, w2, b):
    return 1/(1 + np.exp(-(w1*x1 + w2*x2 + b)))


# Creation of 10x10 Meshgrid Input.
X1 = np.linspace(-10, 10, 100)
X2 = np.linspace(-10, 10, 100)

XX1, XX2 = np.meshgrid(X1, X2)

# Displaying the Dimensions of the Meshgrid
print(X1.shape, X2.shape, XX1.shape, XX2.shape)

# 2-D Sigmoid Neuron Parameters
w1 = 2
w2 = -0.5
b = 0
Y = sigmoid_2d(XX1, XX2, w1, w2, b)
print((Y))

# Plotting a Gradient Distribution for the 2-D Sigmoid Neuron
my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "", ["red", "yellow", "green"])
plt.contourf(XX1, XX2, Y, cmap=my_cmap, alpha=0.6)
plt.show()


# 3-D Projection of (2-D)Sigmoid Function.
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(XX1, XX2, Y, cmap='viridis')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')

ax.view_init(30, 270)
plt.show()

# Compute Loss for a given Dataset
w_unknown = 0.5
b_unknown = 0.25

X = np.random.random(25) * 20 - 10
Y = sigmoid(X, w_unknown, b_unknown)

# Plotting the Sigmoid Values of the above dataset
plt.plot(X, Y, '*')
plt.show()


# Mean Square Error Loss Function
def calc_loss(X, Y, w_est, b_est):
    loss = 0
    for x, y in zip(X, Y):
        loss += (y - sigmoid(x, w_est, b_est))**2
    return loss


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
ax.view_init(30, 270)
plt.show()

print(np.argmin(Loss))

ij = np.argmin(Loss)
i = int(np.floor(ij/Loss.shape[1]))
j = int(ij - i * Loss.shape[1])
print(i, j)
print(WW[i, j], BB[i, j])
