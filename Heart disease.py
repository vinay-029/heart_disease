#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv('heart.csv')


# In[3]:


data.head(15)


# In[4]:


sns.heatmap(data.isnull(),yticklabels = False)


# In[5]:


data.isnull().sum()


# In[6]:


sns.set_style('whitegrid')
sns.countplot(x = "cp",data = data)


# In[7]:


sns.set_style('whitegrid')
sns.countplot(x = "target",data = data,hue = "sex" )


# In[8]:


sns.boxplot(x = 'target',y = 'age',data = data)
plt.figure(figsize = (8,6))


# In[9]:


sns.set_style('whitegrid')
sns.countplot(x = 'target',data = data,hue = 'fbs')


# In[10]:


sns.set_style('whitegrid')
sns.countplot(x = 'target',data = data,hue = 'ca')


# In[11]:


sns.set_style('whitegrid')
sns.countplot(x = 'target',data = data)


# In[12]:


sns.distplot(data['chol'],kde = False,bins = 40, color = 'red' )


# In[13]:


sns.distplot(data['thalach'],kde = False,bins = 40,color = 'r')


# In[14]:


plt.figure(figsize=(10,6))
sns.heatmap(data.corr(),cmap = 'rainbow',annot = True)


# In[49]:


x = data.iloc[:,:-1]
y = data.iloc[:,13]


# In[50]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 1/3,random_state = 0)


# In[51]:


type(x_train)


# In[52]:


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)


# In[53]:


x_train = pd.DataFrame(x_train)


# In[54]:


x_test = pd.DataFrame(x_test)


# In[55]:


shape = x_test.shape
shape


# In[56]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
from sklearn.model_selection import GridSearchCV
params = {'n_neighbors':[i for i in range(1,23)]}
grid_search = GridSearchCV(knn,params,cv = 5)


# In[57]:


grid_search.fit(x_train,y_train)


# In[58]:


grid_search.best_params_


# In[59]:


grid_search.best_estimator_


# In[60]:


y_predict = grid_search.predict(x_test)


# In[61]:


from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_score(y_test,y_predict)


# In[62]:


from sklearn.ensemble import RandomForestClassifier
regressor = RandomForestClassifier()
params_1 = {'n_estimators':[10,30,100,300],'max_features':['auto',None]}


# In[63]:


model = GridSearchCV(regressor,params_1,cv = 10)


# In[64]:


model.fit(x_train,y_train)


# In[65]:


model.best_params_


# In[66]:


y_predict_1 = model.predict(x_test)


# In[67]:


accuracy = accuracy_score(y_test,y_predict_1)


# In[68]:


accuracy


# In[40]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV


# In[41]:


params_lg = {'C': np.logspace(-4,4,20), 'solver':['liblinear']}
log_reg = LogisticRegression()
gridsearch_lg = GridSearchCV(log_reg,params_lg,cv = 10)


# In[42]:


gridsearch_lg.fit(x_train,y_train)


# In[43]:


gridsearch_lg.best_params_


# In[72]:


estimator = []
estimator.append(('lr',LogisticRegression(C =0.03359818286283781 , solver = 'liblinear' )))
estimator.append(('rf',RandomForestClassifier(max_features = 'auto',n_estimators = 100)))
estimator.append(('knn',KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=12, p=2,
                     weights='uniform')))
estimator.append(('svc',SVC()))


# In[73]:


vot_classifier = VotingClassifier(estimators = estimator , voting = 'hard')


# In[74]:


vot_classifier.fit(x_train,y_train)


# In[75]:


y_pred = vot_classifier.predict(x_test)


# In[76]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[ ]:




