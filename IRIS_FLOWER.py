from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.metrics import accuracy_score

# get the iris dara from the sklean
iris_data = datasets.load_iris()
N_Classifier = KNeighborsClassifier()

# get the features and labels form the above ref
feature = iris_data.data
lable = iris_data.target

# after getting the features and lable we have to split our data
feature_train, feature_test, lable_train, lable_test = train_test_split(feature, lable, test_size=0.5)

# train our model
N_Classifier.fit(feature_train, lable_train)

# make a prediction
prediction = N_Classifier.predict(feature_test)

# print the accuracy
print("Model Accuracy: " , accuracy_score(lable_test, prediction))

# make our own data to feed the model
My_Flower = [[4.2,2.2,4.6,5.2]]

# feed the new unseen data to the model
Flower = N_Classifier.predict(My_Flower)

# organize the response for us
response = ""
if Flower[0] == 0:
    response = "Flower: setosa"
elif Flower[0] == 1:
    response = "Flower: versicolor"
elif Flower[0] == 2:
    response = "Flower: virginica"

# response get from the model
print(response)

