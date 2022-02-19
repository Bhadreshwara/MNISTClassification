# MNISTClassification

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
```
### Loading and checking the data
Fetch dataset from openml by name or dataset id. 
Datasets are uniquely identified by either an integer ID or by a combination of name and version 
```python
from sklearn.datasets import fetch_openml
mnist_jay = fetch_openml('mnist_784', version=1, as_frame=True)
mnist_jay.keys()
```
```python
#Assign the data and target to a ndarray
X_jay, y_jay = mnist_jay['data'], mnist_jay['target']
X_jay.shape, y_jay.shape
```

```python
some_digit1 = X_jay.to_numpy()[7]
some_digit2 = X_jay.to_numpy()[5]
some_digit3 = X_jay.to_numpy()[0]

# Use imshow method to plot the values of the three variables you defined in the above point.   
plt.imshow(some_digit1.reshape(28, 28), cmap=mpl.cm.binary, interpolation='nearest')
display(plt.show())
plt.imshow(some_digit2.reshape(28, 28), cmap=mpl.cm.binary, interpolation='nearest')
display(plt.show())
plt.imshow(some_digit3.reshape(28, 28), cmap=mpl.cm.binary, interpolation='nearest')
display(plt.show())
```

### Pre-processing the data
```python
#	The current target values range from 0 to 9 i.e. 10 classes. 
#Transform the target variable to 3 classes as follows:
# a.	Any digit between 0 and 3 inclusive should be assigned a target value of 0
# b.	Any digit between 4 and 7 inclusive should be assigned a target value of 1
# c.	Any digit between 8 and 11 inclusive should be assigned a target value of 9
# d.	Use the following code to do this:
y_jay_new = np.where(y_jay < 4, 0, y_jay)
y_jay_new = np.where((y_jay_new > 3) & (y_jay_new < 8) , 1, y_jay_new)
y_jay_new = np.where((y_jay_new > 7) & (y_jay_new < 12), 9, y_jay_new)
print(y_jay_new)
```

```python
# Print the frequencies of each of the three target classes in y_jay_new
unique, counts = np.unique(y_jay_new, return_counts=True)
print(np.asarray((unique, counts)).T)
```

```python
# Split your data into train test. Assign the first 60,000 records for training and the last 10,000 records for testing. 
X_train, X_test = X_jay[:60000], X_jay[60000:]
y_train, y_test = y_jay_new[:60000], y_jay_new[60000:]
```
### Build the classification model
```python
# Train a KNN classifier using the training data. Name the classifier KNN_clf_firstname.
from sklearn.neighbors import KNeighborsClassifier
KNN_clf_jay = KNeighborsClassifier()
KNN_clf_jay.fit(X_train, y_train)

# Predict the class labels for the test data using the trained classifier. Assign the result to y_pred_firstname.
y_pred_jay = KNN_clf_jay.predict(X_test)
print(y_pred_jay)

# Predict the class labels for the test data using the trained classifier. Assign the result to y_pred_firstname2.
y_pred_jay2 = KNN_clf_jay.predict(X_test)
# Use the classifier to predict the three variables you defined in point 7 above.
print(KNN_clf_jay.predict([some_digit1]))
print(KNN_clf_jay.predict([some_digit2]))
print(KNN_clf_jay.predict([some_digit3]))

print(KNN_clf_jay.score(X_test, y_test) * 100)

```
