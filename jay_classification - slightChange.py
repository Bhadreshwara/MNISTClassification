import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

np.random.seed(40)

# Loading the MNIST dataset from sklearn
from sklearn.datasets import fetch_openml
mnist_jay = fetch_openml('mnist_784', version=1, as_frame=True)
print(mnist_jay.keys())

#Assign the data and target to a ndarray
X_jay, y_jay = mnist_jay['data'], mnist_jay['target']
print(X_jay.shape)
print(y_jay.shape)

# print the type of X_jay
print(type(X_jay))
print(type(y_jay))

some_digit = X_jay.to_numpy()[7]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=mpl.cm.binary)
plt.axis("off")
plt.show()

print(y_jay[7])
y_jay = y_jay.astype(np.uint8)

# Create three variables named: some_digit1, some_digit2, some_digit3. 
# Store in these variables the values from X_firstname indexed 7,5,0 in order.
some_digit1 = X_jay.to_numpy()[7]
some_digit2 = X_jay.to_numpy()[5]
some_digit3 = X_jay.to_numpy()[0]

# Use imshow method to plot the values of the three variables you defined in the above point. 
plt.imshow(some_digit1.reshape(28, 28), cmap=mpl.cm.binary, interpolation='nearest')
plt.show()
plt.imshow(some_digit2.reshape(28, 28), cmap=mpl.cm.binary, interpolation='nearest')
plt.show()
plt.imshow(some_digit3.reshape(28, 28), cmap=mpl.cm.binary, interpolation='nearest')
plt.show()

#Pre-process the data

#	The current target values range from 0 to 9 i.e. 10 classes. Transform the target variable to 3 classes as follows:
# a.	Any digit between 0 and 3 inclusive should be assigned a target value of 0
# b.	Any digit between 4 and 6 inclusive should be assigned a target value of 1
# c.	Any digit between 7 and 9 inclusive should be assigned a target value of 9
# d.	Use the following code to do this:
y_jay_new = np.where(y_jay < 4, 0, y_jay)
y_jay_new = np.where((y_jay_new > 3) & (y_jay_new < 7) , 1, y_jay_new)
y_jay_new = np.where((y_jay_new > 6) & (y_jay_new < 10), 9, y_jay_new)
print(y_jay_new)

# Print the frequencies of each of the three target classes in y_jay_new
unique, counts = np.unique(y_jay_new, return_counts=True)
print(np.asarray((unique, counts)).T)

# Split your data into train test. Assign the first 60,000 records for training and the last 10,000 records for testing. 
X_train, X_test = X_jay[:60000], X_jay[60000:]
y_train, y_test = y_jay_new[:60000], y_jay_new[60000:]


# Build Classification Models

# Train a Naive Bayes classifier using the training data. Name the classifier NB_clf_firstname.
from sklearn.naive_bayes import GaussianNB
NB_clf_jay = GaussianNB()
NB_clf_jay.fit(X_train, y_train)

# Predict the class labels for the test data using the trained classifier. Assign the result to y_pred_firstname.
y_pred_jay = NB_clf_jay.predict(X_test)
print(y_pred_jay)

# Train a SGD classifier using the training data. Name the classifier SGD_clf_firstname.
from sklearn.linear_model import SGDClassifier
SGD_clf_jay = SGDClassifier()
SGD_clf_jay.fit(X_train, y_train)

# Predict the class labels for the test data using the trained classifier. Assign the result to y_pred_firstname.
y_pred_jay = SGD_clf_jay.predict(X_test)
print(y_pred_jay)
# Use the classifier to predict the three variables you defined in point 7 above.
print(NB_clf_jay.predict([some_digit1]))
print(NB_clf_jay.predict([some_digit2]))
print(NB_clf_jay.predict([some_digit3]))

# Predict the class labels for the test data using the trained classifier. Assign the result to y_pred_firstname1.
y_pred_jay1 = SGD_clf_jay.predict(X_test)
print(y_pred_jay1)
# Use the classifier to predict the three variables you defined in point 7 above.
print(SGD_clf_jay.predict([some_digit1]))
print(SGD_clf_jay.predict([some_digit2]))
print(SGD_clf_jay.predict([some_digit3]))

# Print the accuracy of the classifier on the test data.
print(NB_clf_jay.score(X_test, y_test) * 100)

# Print the accuracy of the classifier on the test data.
print(SGD_clf_jay.score(X_test, y_test) * 100)


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
