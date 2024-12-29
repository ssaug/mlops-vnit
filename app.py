from sklearn import datasets
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklear.metrics import accuracy_score
from utils import split_data


df = pd.read_csv("dataset/Iris.csv")
X_train,X_test,y_train,y_test = split_data(df)

clf = DecisionTreeClassifier(criterion="gini")

clf.fit(X_train,y_train)

y_pred= clf.predict(X_test)

print(f"Accuracy of model is {accuracy_score(y_test,y_predect)*100}")