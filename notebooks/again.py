#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv("Algerian_forest_fires_dataset_UPDATE.csv", header=1)


# In[4]:


df[df.isnull().any(axis = 1)]


# In[5]:


df = df.dropna().reset_index(drop= True)


# In[6]:


df[df.isnull().any(axis = 1)]


# In[7]:


df = df.drop(122).reset_index(drop = True)


# In[8]:


df.iloc[[122]]


# In[9]:


df.info()


# In[10]:


df.columns


# In[11]:


df.columns = df.columns.str.strip()


# In[12]:


df.columns


# In[13]:


df.head(5)


# In[14]:


df["Classes"] = np.where(df["Classes"].str.contains("not fire"), 'not fire', 'fire')


# In[15]:


df.head()


# In[16]:


df.tail()


# In[17]:


df.info()


# In[18]:


df.columns


# In[19]:


df[['day', 'month', 'year', 'Temperature', 'RH', 'Ws']] = df[['day', 'month', 'year', 'Temperature', 'RH', 'Ws']].astype(int)


# In[20]:


df.info()


# In[21]:


df[['Rain', 'FFMC','DMC', 'DC', 'ISI', 'BUI', 'FWI']] = df[['Rain', 'FFMC','DMC', 'DC', 'ISI', 'BUI', 'FWI']].astype(float)


# In[22]:


df.info()


# In[23]:


df[['Classes']] = df[['Classes']].astype("O")


# In[24]:


df.info()


# In[25]:


df.loc[:122,"Region"] = 0
df.loc[122:,"Region"] = 1


# In[26]:


df[["Region"]] = df[["Region"]].astype(int)


# In[27]:


df.info()


# In[32]:


df["Classes"].value_counts()


# In[29]:


plt.style.use("seaborn-v0_8")
df.hist(bins=50, figsize=(12, 8))
plt.show()


# In[30]:


plt.figure(figsize=(12, 8))
plt.pie(df["Classes"].value_counts(normalize=True)*100, labels=["Fire", "Not Fire"], autopct="%1.1f%%")
plt.title("Pie chart for classes")
plt.show()


# In[33]:


df.info()


# In[34]:


df.head()


# In[35]:


df.columns


# In[37]:


df.drop(['day', 'month', 'year'], axis=1, inplace=True)


# In[38]:


df.head()


# In[41]:


df["Classes"].value_counts()


# In[42]:


Y = df["FWI"]


# In[48]:


X = df[['Temperature', 'RH', 'Ws', 'Rain', 'FFMC','DMC', 'DC', 'ISI', 'BUI', 'Classes', 'Region']]


# In[50]:


X.shape


# In[51]:


Y.shape


# In[52]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


# In[53]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[64]:


X_train["Classes"] = np.where(X_train["Classes"].str.contains("not fire"), 0, 1)
X_train["Classes"] = X_train["Classes"].astype(object)

X_test["Classes"] = np.where(X_test["Classes"].str.contains("not fire"), 0, 1)
X_test["Classes"] = X_test ["Classes"].astype(object)


# In[ ]:


X_test["Clases"] = X_test["Clases"].astype(object)
X_test["Clases"] = X_test["Clases"].astype(object)


# In[57]:


X_train.corr()


# In[58]:


sns.heatmap(X_train.corr(), annot=True)


# In[60]:


def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                col_corr.add(corr_matrix.columns[i])
    return col_corr


# In[65]:


rel = correlation(X_train, 0.85)


# In[66]:


X_train.drop(rel,axis= 1, inplace=True)
X_test.drop(rel,axis=1, inplace = True)


# In[68]:


X_train.info()


# In[69]:


X_test.info()


# In[73]:


X_test


# In[74]:


X_test["Classes"] = np.where(X_test["Classes"].str.contains("not fire"), 0, 1)
X_test["Classes"] = X_test ["Classes"].astype('O')


# In[77]:


X_test = scaler.transform(X_test)


# In[78]:


X_test


# In[79]:


plt.subplots(figsize=(15, 7))
sns.boxplot(data=X_train)
plt.title("Effect of scaling")


# In[80]:


from sklearn.linear_model import LinearRegression
regression = LinearRegression()


# In[81]:


regression.fit(X_train, Y_train)


# In[86]:


Y_train


# In[82]:


y_predict = regression.predict(X_test)


# In[87]:


y_predict


# In[84]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(Y_test, y_predict)
mse = mean_squared_error(Y_test, y_predict)
rmse = np.sqrt(mse)
score = r2_score(Y_test, y_predict)


# In[85]:


print(mae)
print(mse)
print(rmse)
print(score)


# In[90]:


plt.scatter(Y_test, y_predict)


# In[ ]:




