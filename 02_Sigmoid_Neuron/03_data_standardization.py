"""
How to perform Standardization of Data using Standard Scaler and MinMax Scaler
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Creation of set of 100 random numbers and plotting it on graph
R = np.random.random([100, 1])
plt.plot(R)
plt.show()

# Calculation of its mean
print(np.mean(R))

# Calculation of Standard Deviation.
print(np.std(R))

# Using the StandardScaler Model to reduce the Mean to 0 and bring
# the Standard Deviation, close to 1.
scaler = StandardScaler()
scaler.fit(R)

# mean value (after fitting the Random set of numbers)
print(scaler.mean_)

# transforming the R Dataset to get the new mean and standard deviation value.
RT = scaler.transform(R)

# Mean value after tranformation using the Standard Scaler
print(np.mean(RT))

# Standard Deviation value after tranformation using the Standard Scaler
print(np.std(RT))

# plotting of Graph after Standardization.
plt.plot(RT)
plt.show()
