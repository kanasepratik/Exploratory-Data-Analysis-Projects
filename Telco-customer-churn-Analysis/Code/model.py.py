import os
import numpy as np
import pandas as pd


pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)


# In[2]:


missing_values = ["n/a", "na", "--",'null','',' ','NA','?','NaN']
data=pd.read_csv(r'C:\Users\PKK\Desktop\p\telco-customer-churn\WA_Fn-UseC_-Telco-Customer-Churn.csv',na_values=missing_values)



data.drop(columns='customerID',inplace=True)


from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()


# In[7]:


for col in data[['gender',
 'Partner',
 'Dependents',
 'PhoneService',
 'MultipleLines',
 'InternetService',
 'OnlineSecurity',
 'OnlineBackup',
 'DeviceProtection',
 'TechSupport',
 'StreamingTV',
 'StreamingMovies',
 'Contract',
 'PaperlessBilling',
 'PaymentMethod',
 'Churn']]:
    data[col]=le.fit_transform(data[col])


# In[8]:
data.head()

X=data.drop(columns='Churn')
y=data['Churn']

X.columns

# In[9]:


X.fillna(X.mean(), inplace=True)
y.fillna(y.mean(), inplace=True)# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.80,random_state=20)


# In[19]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingClassifier


# In[20]:


def run_randomForest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Accuracy: ', accuracy_score(y_test, y_pred))


# In[ ]:

# In[ ]:





# In[21]:


sel = RFE(RandomForestClassifier(n_estimators=100, random_state=0), n_features_to_select = 10)
sel.fit(X_train, y_train)
X_train_rfe = sel.transform(X_train)
X_test_rfe = sel.transform(X_test)
print('Selected Feature: ', 15)
run_randomForest(X_train_rfe, X_test_rfe, y_train, y_test)
print()


# In[ ]:





# In[22]:


features =X_train.columns[sel.get_support()]
features


# In[24]:


X_train_Sel=X_train[['gender', 'tenure', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
       'TechSupport', 'Contract', 'PaymentMethod', 'MonthlyCharges',
       'TotalCharges']]


# In[25]:


X_test_Sel=X_test[['gender', 'tenure', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
       'TechSupport', 'Contract', 'PaymentMethod', 'MonthlyCharges',
       'TotalCharges']]


# In[26]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix


# In[27]:


model=RandomForestClassifier()


# In[28]:


model.fit(X_train_Sel,y_train)


# In[29]:


model.score(X_test_Sel,y_test)


# In[30]:


model.score(X_train_Sel,y_train)


# In[ ]:


# we are tuning three hyperparameters right now, we are passing the different values for both parameters
grid_param = {
    "n_estimators" : [90,100,115],
    'criterion': ['gini', 'entropy'],
    'min_samples_leaf' : [1,2,3,4,5],
    'min_samples_split': [4,5,6,7,8],
    'max_features' : ['auto','log2']
}


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


grid_search = GridSearchCV(estimator=model,param_grid=grid_param,cv=5,n_jobs =-1,verbose = 3)


# In[ ]:


grid_search.fit(X_train,y_train)


# In[ ]:


grid_search.best_params_


# In[31]:


rf=RandomForestClassifier(criterion= 'entropy',max_features='auto',min_samples_leaf=5,min_samples_split=8
                          ,n_estimators=150)


# In[32]:


rf.fit(X_train_Sel,y_train)


# In[33]:


rf.score(X_train_Sel,y_train)
rf.score(X_test_Sel,y_test)


# In[34]:



rf.score(X_test_Sel,y_test)


# In[36]:


y_pred=rf.predict(X_test_Sel)


#
import pickle


# In[98]:

# open a file, where you ant to store the data
file = open('telco.pkl', 'wb')

# dump information to that file
pickle.dump(rf, file)

import os
os.chdir(r'C:\Users\PKK\Desktop\p\telco-customer-churn')


os.getcwd()