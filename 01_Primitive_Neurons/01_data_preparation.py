"""
McCulloch-Pits Neuron
"""
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split

# loading breast cancer dataset
breast_cancer = sklearn.datasets.load_breast_cancer()

# Capturing the data(real world values) and target values(0,1)
x = breast_cancer.data
y = breast_cancer.target


print(x)
print(y)
# printing the shapes of the arrays
print(x.shape, y.shape)

# Creation of the Dataframe using Pandas
data = pd.DataFrame(x, columns=breast_cancer.feature_names)
data['class'] = breast_cancer.target

# View Sample Data and Basic Description
print(data.head)
print(data.describe())

# Check the Value count for the class column.
print(data['class'].value_counts())

# Check the target names to be assigned for breast cancer.
print(breast_cancer.target_names)

# Grouping of data based on class values and calculating the mean.
print(data.groupby('class').mean())


# Train-Test spltting(Splitting of the code)
x = data.drop('class', axis=1)
y = data['class']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, stratify=y, random_state=1)
# Dimensions of the splitted Dataset.
print(x.shape, x_train.shape, x_test.shape)
print(y.shape, y_train.shape, y_test.shape)

# Calculating the mean of the train, testing and actual target values
print(y.mean(), y_train.mean(), y_test.mean())


# Binarization of Input
plt.plot(x_train.T, '*')
plt.xticks(rotation='vertical')
plt.show()

# Manual Binarization of data
x_binarized_3_train = x_train['mean area'].map(lambda x: 0 if x < 1000 else 1)
plt.plot(x_binarized_3_train, '*')
plt.show()

# Binarization of data using Pandas
x_binarized_train = x_train.apply(pd.cut, bins=2, labels=[1, 0])
plt.plot(x_binarized_train.T, '*')
plt.xticks(rotation='vertical')
plt.show()

x_binarized_test = x_test.apply(pd.cut, bins=2, labels=[1, 0])
plt.plot(x_binarized_test.T, '*')
plt.xticks(rotation='vertical')
plt.show()

x_binarized_train = x_binarized_train.values
x_binarized_test = x_binarized_test.values
