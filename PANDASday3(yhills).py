#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[4]:


df= pd.read_csv("titanic-training-data.csv")
                


# In[5]:


df


# In[6]:


df.head()


# In[8]:


df.head(10)


# In[10]:


df.tail()


# In[11]:


df.tail(10)


# In[14]:


df.sample()


# In[15]:


df.sample(10)


# In[17]:


df.dtypes


# In[18]:


df.info()


# In[19]:


df.shape


# In[22]:


df.isnull().sum()


# In[23]:


df.describe()


# In[24]:


df.describe(include="all")


# In[25]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[37]:


plt.hist(x=df["Age"],color="skyblue")
plt.title("Distribute of Age",fontsize=18,color='black')
plt.xlabel("Age")
plt.ylabel("Frequency",fontsize=16,color='black')
plt.show()


# In[48]:


sns.boxplot(df["Age"])
plt.title("Distribute of Age")
plt.xlabel("Age")
#plt.ylabel("Frequency",fontsize=16,color='black')
plt.show()


# In[57]:


df["Sex"].value_counts().plot(kind="barh")
plt.title("Sex Distribution",fontsize=16)
plt.xlabel("Sex")
plt.ylabel("Count")
plt.show()


# In[59]:


df["Pclass"].value_counts().plot(kind="barh")
plt.title("Sex Distribution",fontsize=16)
plt.xlabel("Sex")
plt.ylabel("Count")
plt.show()


# In[62]:


from matplotlib import cm
emb=df["Embarked"].value_counts()
keys=emb.keys().to_list()
counts=emb.to_list()
cs=cm.Set1([2,4,6,8])
plt.pie(x=counts,labels=keys,autopct='%1.1f%%',colors=cs)
plt.show()


# In[63]:


plt.scatter(x="Age", y="SibSp" ,data=df)


# In[ ]:




