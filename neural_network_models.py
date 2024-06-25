import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import fashion_mnist

(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
print('Fashion MNIST Dataset Shape:')
print('X_train: ' + str(X_train.shape))
print('Y_train: ' + str(Y_train.shape))
print('X_test:  '  + str(X_test.shape))
print('Y_test:  '  + str(Y_test.shape))

X_train[1]

Y_train

# specify the number of rows and columns you want to see
num_row = 6
num_col = 6

# get a segment of the dataset
num = num_row*num_col
images = X_train[:num]
labels = Y_train[:num]

# plot images
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(num_row*num_col):
    ax = axes[i//num_col, i%num_col]
    ax.imshow(images[i], cmap='gray')
    ax.set_title('Label: {}'.format(labels[i]))
plt.tight_layout()
plt.show()

X_train = X_train / 255
X_test = X_test / 255

X_train[1]

from sklearn.neural_network import MLPClassifier

X = X_train.reshape(60000,784)
y = Y_train
clf = MLPClassifier(solver='adam', alpha=1e-5,
                    hidden_layer_sizes=(4), random_state=1, max_iter = 100)

clf.fit(X, y)

y_pred = clf.predict(X_test.reshape(10000,784))

def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements

class_names = [ "T-shirt/top" , "Trouser" , "Pullover" , "Dress" , "Coat" , "Sandal" , "Shirt" , "Sneaker" , "Bag" , "Ankle boot" ]

np.unique(Y_test)

#Importing Confusion Matrix
from sklearn.metrics import confusion_matrix,classification_report
#Comparing the predictions against the actual observations in y_val
cm = confusion_matrix(Y_test,y_pred,labels=np.unique(Y_test))
cm

#Printing the accuracy
print("Accuracy of MLPClassifier : ", accuracy(cm))

classification_report(Y_test,y_pred,labels=np.unique(Y_test),target_names=class_names)

"""1. Improving accuracy"""

clf = MLPClassifier(solver='adam', alpha=1e-5,
                    hidden_layer_sizes=(1024, 256, 128, 64, 64), random_state=1, max_iter=350)

clf.fit(X_train.reshape(60000, 784), Y_train)
y_pred = clf.predict(X_test.reshape(10000, 784))

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(Y_test, y_pred, labels=np.unique(Y_test))

def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements

print("Accuracy of MLPClassifier: ", accuracy(cm))

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
print(classification_report(Y_test, y_pred, labels=np.unique(Y_test), target_names=class_names))

"""2. Hyper parameter tuning and k-fold cross validation using GridSearchCV"""

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X_train_flatten = X_train.reshape(X_train.shape[0], -1)
X_test_flatten = X_test.reshape(X_test.shape[0], -1)

parameters = {
    'hidden_layer_sizes': [(128, 64, 64), (256, 128, 64, 64)],
    'max_iter': [100, 200],
    'learning_rate_init': [0.001, 0.01]
}

clf = MLPClassifier(solver='adam', alpha=1e-5, random_state=1)

grid_search = GridSearchCV(clf, parameters, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train_flatten, Y_train)

print("Best parameters found:")
print(grid_search.best_params_)

best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(X_test_flatten)

print("Accuracy of MLPClassifier:", accuracy_score(Y_test, y_pred))
print(classification_report(Y_test, y_pred))

cm = confusion_matrix(Y_test, y_pred)
print("Confusion Matrix:")
print(cm)
