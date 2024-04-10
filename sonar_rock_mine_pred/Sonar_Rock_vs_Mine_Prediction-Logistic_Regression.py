#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import statsmodels
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from math import sqrt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv(r"C:\Users\acer\Desktop\UPGRAD\machine learning\Logistic Regression\sonar_rock_mine_pred\Copy of sonar data.csv", header=None)
df.head()


# In[3]:


df.info()


# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


df[60].value_counts()


# In[7]:


df.groupby(60).mean()


# In[8]:


# separating data and labels
X=df.drop(columns=60,axis=1)
Y=df[60]


# In[9]:


print(X)
print(Y)


# In[10]:


# train and test split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1)


# # Model Training-Logistic Regression

# In[14]:


model= LogisticRegression()


# In[15]:


model.fit(X_train, y_train)


# # Model Evaluation

# In[16]:


# accuracy on training set
X_train_pred=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_pred, y_train)


# In[17]:


print(training_data_accuracy)


# In[18]:


X_test_pred=model.predict(X_test)
t_d_a=accuracy_score(X_test_pred, y_test)
print(t_d_a)


# # Making a predictive system

# In[21]:


input_data=(0.0200,0.0371,0.0428,0.0205,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032)
# convert input into numpy array
i_d_np= np.asarray(input_data)

#reshape the numpy array as predicting for one instance
i_d_reshape= i_d_np.reshape(1,-1)

prediction= model.predict(i_d_reshape)


# In[22]:


print(prediction)


# In[23]:


if(prediction[0]=='R'):
    print('Rock')
else:
    print('Mine')


# In[ ]:




