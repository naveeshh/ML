# ML
Website phishing
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
df=pd.read_csv('buddymove_holidayiq.csv')
b=df.isnull().sum()*100/len(df)
m=b.keys()
c=list(filter(lambda x:b[x]>60,m))
df=df.dropna(subset=c)
X=df.drop('User Id',axis=1)
y=df['User Id']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=45)
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(max_depth=12,random_state=2)
model.fit(x_train,y_train)
prediction= model.predict(x_train)
print(prediction)
score = accuracy_score(y_train,prediction)
print(confusion_matrix(y_train,prediction))
print("Accuracy for Descision Tree Classifier = ",score)
from sklearn import svm
model=svm.SVC()
model.fit(x_train,y_train)
model.fit(x_train,y_train)
prediction= model.predict(x_train)
print(prediction)
score = accuracy_score(y_train,prediction)
print(confusion_matrix(y_train,prediction))
print("Accuracy for SVM = ",score) 
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=5)
model.fit(x_train,y_train)
model.fit(x_train,y_train)
prediction= model.predict(x_train)
print(prediction)
score = accuracy_score(y_train,prediction)
print(confusion_matrix(y_train,prediction))
print("Accuracy for Random Forest Classifier = ",score) 
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train,y_train)
prediction= model.predict(x_train)
print(prediction)
score = accuracy_score(y_train,prediction)
print(confusion_matrix(y_train,prediction))
print("Accuracy for Logistic Regression = ",score)
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,y_train)
model.fit(x_train,y_train)
prediction= model.predict(x_train)
print(prediction)
score = accuracy_score(y_train,prediction)
print(confusion_matrix(y_train,prediction))
print("Accuracy for KNN classifier = ",score)
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(x_train,y_train)
model.fit(x_train,y_train)
prediction= model.predict(x_train)
print(prediction)
score = accuracy_score(y_train,prediction)
print(confusion_matrix(y_train,prediction))
print("Accuracy for Naive bayes = ",score)  
