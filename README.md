# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.

2. Analyse the data.

3. Use modelselection and Countvectorizer to preditct the values.

4. Find the accuracy and display the result.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: EZHILARASI N
RegisterNumber: 212224040088
*/

import chardet
file='spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data=pd.read_csv("spam.csv",encoding="Windows-1252")
data.head()
data.info()
data.isnull().sum()
x=data["v2"].values
y=data["v1"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
x_train
x_test
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
x_train
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print (classification_report1)

```

## Output:

<img width="1052" height="62" alt="image" src="https://github.com/user-attachments/assets/dccac94e-94a4-4022-83db-9b23080d9c2f" />

DATA

<img width="897" height="257" alt="image" src="https://github.com/user-attachments/assets/1b20199f-6da2-424f-8c7e-995aefb8559c" />

<img width="687" height="335" alt="image" src="https://github.com/user-attachments/assets/025cbd52-8ca9-4f8f-884c-7cbd5f1158cc" />

X TRAIN

<img width="302" height="176" alt="image" src="https://github.com/user-attachments/assets/5264cca3-d2a8-46f5-85b0-cb5541b01bdb" />

X TEST

<img width="1552" height="262" alt="image" src="https://github.com/user-attachments/assets/86142af6-772c-47d1-9df6-79c8751ed390" />

<img width="1552" height="307" alt="image" src="https://github.com/user-attachments/assets/80689ac6-0c39-4184-967d-e000e17a2c54" />

<img width="886" height="81" alt="image" src="https://github.com/user-attachments/assets/d5640af4-e0d9-4a69-8d3e-299e49feb95b" />

<img width="857" height="47" alt="image" src="https://github.com/user-attachments/assets/8fc46735-3d02-4f0c-8d89-51c6315672cf" />

<img width="292" height="38" alt="image" src="https://github.com/user-attachments/assets/e2542121-5b9e-472b-82c3-27cbe07debae" />

ACCURACY

<img width="449" height="73" alt="image" src="https://github.com/user-attachments/assets/9d149c26-27f8-48a2-8c0a-a2c9f7e2b0cd" />





## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
