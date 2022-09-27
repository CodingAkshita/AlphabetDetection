# cv2 library gives access to computer's camera
import cv2
# numpy library helps perform complex mathematical and list operations
import numpy as np
# pandas library helps us treat our data as dataframes
import pandas as pd
# To prettify the charts that we draw with matplotlib 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# As we know, we can represent any given image in the form of binary.
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
# We are fetching the dataset of images of handwritten digits from openMl datasets.
#The X here would be the data of images represented in binary 
#The y would be the label of that image. Eg - 0, 1, 2..9


# The print function tells us the count of samples for each of the label
print(pd.Series(y).value_counts())
# We are creating a list that contains all the labels
classes = ['0', '1', '2','3', '4','5', '6', '7', '8', '9']
# Here it tells the size or the length of the classes. Length = 10
nclasses = len(classes)

samples_per_class = 5
figure = plt.figure(figsize=(nclasses*2,(1+samples_per_class*2))) #20, 11

idx_cls = 0
for cls in classes:
  idxs = np.flatnonzero(y == cls)
  idxs = np.random.choice(idxs, samples_per_class, replace=False)
  i = 0
  for idx in idxs:
    # We are iterating over these random indices for the given label
    # Following command is used to define the position of the given label
    # Here, idx_cls = columns
    # Here,    i    = rows  
    # For all the samples of label 0, the column idx_cls would remain to be the same while the row i will change
    # This simply helps us form a grid of samples by plotting  
    plt_idx = i * nclasses + idx_cls + 1
    p = plt.subplot(samples_per_class, nclasses, plt_idx);
    p = sns.heatmap(np.array(X.loc[idx]).reshape(28,28), cmap=plt.cm.gray, 
             xticklabels=False, yticklabels=False, cbar=False);
    p = plt.axis('off');
    i += 1
  idx_cls += 1

idxs = np.flatnonzero(y == '0')
print(np.array(X.loc[idxs[0]]))

print(len(X))
print(len(X.loc[0]))

# We will be using only 1000 samples 
# Splitting our data into 7500 for training and 2500 for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=7500, test_size=2500)

#scaling the features
# We are dividing it by 255 so that all the features of the image can be represented with values between 0 and 1
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

print(X.loc[0])
print(y.loc[0])

clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])

p = plt.figure(figsize=(10,10));
#fmt = format, s -- string, d -- decimal, cbar -- color bar
p = sns.heatmap(cm, annot=True, fmt="d", cbar=False)