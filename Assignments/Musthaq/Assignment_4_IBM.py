#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv(r"C:\Users\Desktop\project\Mall_Customers.csv")


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.shape


# # Visualizations

# 
# # Univariate Analysis

# In[8]:


sns.histplot(data['Gender'])


# In[9]:


sns.distplot(data['Annual Income (k$)'])


# # Bivariate analysis

# In[6]:


sns.lineplot(data['Gender'],data['Annual Income (k$)'])


# # Multivariate analysis

# In[13]:


sns.pairplot(data)


# # Descriptive Statistics

# In[14]:


data.describe()


# # Check for missing values

# In[15]:


data.isnull().sum()


# No missing values were found

# # Finding outliers 

# In[7]:


x = data['Gender']
y = data['Age']

plt.plot(x, y)


# # Check for categorical columns and perfom encoding

# In[17]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data.Gender=le.fit_transform(data.Gender)
data.head()


# # Scaling

# In[18]:


from sklearn.preprocessing import MinMaxScaler
scale=MinMaxScaler()
dfscaled=pd.DataFrame(scale.fit_transform(data),columns=data.columns)
dfscaled.head()


# # K means

# In[19]:


from sklearn import cluster


# In[20]:


error=[]
for i in range(1,15):
    kmeans=cluster.KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(dfscaled)
    error.append(kmeans.inertia_)


# In[21]:


error


# In[22]:


plt.plot(range(1,15),error)


# In[23]:


kmmodel=cluster.KMeans(n_clusters=7,init='k-means++',random_state=0)
kmmodel.fit(dfscaled)


# In[24]:


TargetCustomers=kmmodel.predict(dfscaled)
TargetCustomers


# # Adding the clustered to the primary dataset

# In[30]:


data.insert(loc=4,column='TargetCustomers',value=TargetCustomers)


# In[31]:


data.head()


# # Split the data into dependent and independent variables

# In[103]:


x=dfscaled.iloc[:,:-1]
x.head()


# In[104]:


y=dfscaled.TargetCustomers
y.head()


# # Split the data into training and testing

# In[105]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# # Build the Model

# In[109]:


sns.lineplot(dfscaled.Age,dfscaled.TargetCustomers)


# In[110]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()


# # Training the Model

# In[111]:


model.fit(x_train,y_train)


# In[112]:


train_pred=model.predict(x_train)
train_pred


# # Testing the model

# In[113]:


test_pred=model.predict(x_test)
test_pred


# # Measuring Performance

# In[114]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print('test accuracy score: ',accuracy_score(y_test,test_pred))
print('train accuracy score: ',accuracy_score(y_train,train_pred))


# In[115]:


pd.crosstab(y_test,test_pred)


# In[116]:


pd.crosstab(y_train,train_pred)


# In[ ]:




