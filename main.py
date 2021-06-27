import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import csv

# inputdata
dataset = pd.read_excel("Data.xlsx", index_col=0)  # index_col for removing the data from dataset

dummy = pd.get_dummies(dataset['Gender'])
print(dummy)
dataset1 = pd.concat((dataset, dummy), axis=1)
dataset1 = dataset1.drop(['Gender', 'female'], axis=1)
dataset1 = dataset1.rename(columns={"male": "sex"})

dummy1 = pd.get_dummies(dataset['corona_result'])
print(dummy1)
dataset2 = pd.concat((dataset1, dummy1), axis=1)
dataset2 = dataset2.drop(['positive', 'corona_result'], axis=1)
dataset2 = dataset2.rename(columns={"negative": "Test_result"})

print(dataset1)
print(dataset2)


# caluclating values of means&varience

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(dataset2.drop('Test_result', axis=1))
scaled_features = scaler.transform(dataset2.drop('Test_result', axis=1))
dataset_feat = pd.DataFrame(scaled_features, columns=dataset2.columns[:-1])
print(dataset_feat)

# Train Test Split Data and Use KNN model from sklearn library


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(scaled_features, dataset2['Test_result'], test_size=0.30)

# Remember that we are trying to come up
# with a model to predict whether
# someone will TARGET CLASS or not.
# We'll start with k = 1.

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)
pred = knn.predict(X_test)

# Predictions and Evaluations
# Let's evaluate our KNN model !
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, pred))

print(classification_report(y_test, pred))

# Choosing a K Value:

error_rate = []

# Will take some time
for i in range(1, 6):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 6), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='green', markersize=20)

plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()

# FIRST A QUICK COMPARISON TO OUR ORIGINAL K = 1
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)
pred = knn.predict(X_test)

print('WITH K = 3')
print('\n')
print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))

# NOW WITH K = 15
knn = KNeighborsClassifier(n_neighbors=2)

knn.fit(X_train, y_train)
pred = knn.predict(X_test)

print('WITH K = 2')
print('\n')
print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))
