import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#lecture fichier

data=pd.read_csv("new_data.csv")
data.head()
data.info()
data.describe()

#visualisation de données

obj=(data.dtypes==object)
object_cols=list(obj[obj].index)
print("categorical variables",len(object_cols))

int_=(data.dtypes==int)
int_cols=list(int_[int_].index)
print("integer variables",len(int_cols))

float_=(data.dtypes==float)
float_cols=list(float_[float_].index)
print("float variables",len(float_cols))

sns.countplot(x='type', data=data)

sns.barplot(x='type',y='amount',data=data)

plt.figure(figsize=(15,6))
##sns.distplot(data['step'],bins=50). ##deprecated function


plt.figure(figsize=(12,6))
sns.heatmap(data.apply(lambda x:pd.factorize(x)[0]).corr(),cmap='BrBG',fmt='.2f',linewidths=2,annot=True)

#data processing

type_new=pd.get_dummies(data['type'],drop_first=True)
data_new=pd.concat([data,type_new],axis=1)
data_new.head()

X = data_new.drop(['isFraud', 'type', 'nameOrig', 'nameDest'], axis=1)
y = data_new['isFraud']

X.shape,y.shape

#splitting the data

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


#Model training

#from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score as ras
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

models=[LogisticRegression(),RandomForestClassifier(n_estimators=7,criterion='entropy',random_state=7)]
for i in range(len(models)):
    models[i].fit(X_train,y_train)
    print(f'{models[i]}')

    train_preds=models[i].predict_proba(X_train)[:,1]
    print('Training accuracy',ras(y_train,train_preds))

    y_preds=models[i].predict_proba(X_test)[:,1]
    print('Validation accuracy',ras(y_test,y_preds))    