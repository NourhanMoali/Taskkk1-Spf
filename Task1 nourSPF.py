#!/usr/bin/env python
# coding: utf-8

# # THE SPARKARS FOUNDATION 
# 
# **TASK 1**:**Prediction Using Supervised ML**
# 
# 

# # Nourhan Mohammed Ali Ebrahim

# # Importing all the required libraries..

# In[2]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split


# # Reading data from link:
# 

# In[3]:


url="http://bit.ly/w-data"
df=pd.read_csv(url)
print("Data imported successfully")


# # Analyzing Data..

# In[4]:


df.head(6)


# In[5]:


df.tail(6)


# In[6]:


df.shape


# In[7]:


df.isnull().sum()


# In[8]:


df.describe()


# In[9]:


df.corr()


# # Visualizing Data.. 

# In[14]:


df.plot(x="Hours",y="Scores",style="o",c="red",figsize=(15,7))
plt.title("Hours vs percentage")
plt.xlabel("HOURS STUDIED")
plt.ylabel("Percentage SCORE")


# # Data Processing..

# In[15]:


x=df.iloc[:,:-1].values
y=df.iloc[:,1].values


# # Model Training..

# In[57]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2, random_state=0)
regressor= LinearRegression()
regressor.fit(x_train,y_train)
print("Training of model is complete")


# # plotting Line Of Regression For Test Data..

# In[21]:


line= regressor.coef_*x+regressor.intercept_
plt.figure(figsize=(15,7))
plt.scatter(x,y)
plt.plot(x,line,color="red")
plt.show()


# In[22]:


print(x_test)
y_pred=regressor.predict(x_test)


# In[28]:


dfAP=pd.DataFrame({"Actual":y_test,"predicted":y_pred})
dfAP


# In[47]:


y_pred


# In[48]:


y_test


# In[62]:


print("Training Score: ",regressor.score(x_train,y_train))
print("Testing Score: ",regressor.score(x_test,y_test))


# # predicting Values..

# In[63]:


s=float(input("Enter the Num of Hours"))
self_prediction=regressor.predict(np.array([s]).reshape(-1,1))
print("Nun of Hours ={}\nPredicted Score = {}".format(s,self_prediction[0]))


# # Final Evaluation..

# In[67]:


from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score,accuracy_score
print("Mean Squared Error: ",mean_squared_error(y_test,y_pred))
print("Mean Absolute Error: ",mean_absolute_error(y_test,y_pred))
print("Root Mean Squared Error:",np.sqrt(mean_absolute_error(y_test,y_pred)))
print("R-2 Score:",r2_score(y_test,y_pred))


# In[ ]:




