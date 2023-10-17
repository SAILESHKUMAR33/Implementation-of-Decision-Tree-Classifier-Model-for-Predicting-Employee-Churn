# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required Libraries.

2.Upload the dataset in the compiler and read the dataset.

3.Find head,info and null elements in the dataset.

4.Using LabelEncoder and DecisionTreeClassifier , find accuracy and prediction for the dataset.

5.End the program.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Saileshkumar A
RegisterNumber:  212222230126
*/
```
```

import pandas as pd
data=pd.read_csv("/Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```


## Output:

![image](https://github.com/SAILESHKUMAR33/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497410/1a7c423e-859c-457e-af85-6a658ac8de1d)

![image](https://github.com/SAILESHKUMAR33/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497410/d3bc224a-9142-4276-9a0b-5b892080c88c)

![image](https://github.com/SAILESHKUMAR33/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497410/3116417d-5de1-4d77-8367-47d278bd4a84)

![image](https://github.com/SAILESHKUMAR33/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497410/05e92eb0-d1a8-4dde-88e9-b9f1242ffeb5)

![image](https://github.com/SAILESHKUMAR33/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497410/12cf6a38-c3d6-4460-bb8e-3072c0d4b682)

![image](https://github.com/SAILESHKUMAR33/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497410/b2807c0d-5262-42dd-bfd3-05f7eeb85d58)

![image](https://github.com/SAILESHKUMAR33/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497410/0e234c15-5c69-4477-9a14-4fa8356aa189)

![image](https://github.com/SAILESHKUMAR33/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497410/ff29262e-62cf-4697-9c8d-2ccb21f0744e)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
