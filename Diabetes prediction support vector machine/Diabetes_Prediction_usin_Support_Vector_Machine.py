#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm


# In[3]:


df=pd.read_csv(r"C:\Users\acer\Desktop\UPGRAD\machine learning\Logistic Regression\Diabetes prediction support vector machine\diabetes.csv")


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[8]:


df['Outcome'].value_counts()


# In[7]:


X=df.drop(columns='Outcome',axis=1)
y=df['Outcome']


# In[9]:


scaler=StandardScaler()


# In[10]:


scaler.fit(X)


# In[11]:


stand_data=scaler.transform(X)


# In[12]:


X=stand_data


# In[13]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=2)


# In[14]:


model=svm.SVC(kernel='linear')


# In[15]:


model.fit(X_train,y_train)


# In[16]:


X_train_pred=model.predict(X_train)
data_accuracy=accuracy_score(X_train_pred,y_train)


# In[17]:


print(data_accuracy)


# In[18]:


X_test_pred=model.predict(X_test)
test_accuracy=accuracy_score(X_test_pred,y_test)


# In[19]:


print(test_accuracy)


# In[22]:


input_data=(10,113,0,0,0,36.3,0.134,27)
id=np.asarray(input_data)
id_reshape=id.reshape(1,-1)
stand_id=scaler.transform(id_reshape)
print(stand_id)

pred=model.predict(stand_id)
print(pred)


# In[ ]:




