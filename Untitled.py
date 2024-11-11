#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import recall_score, f1_score
import matplotlib.pyplot as plt
from pymongo import MongoClient
from sklearn.preprocessing import LabelEncoder


# In[2]:


df = pd.read_csv('New Institutes.csv')


# In[16]:


df.head()


# In[3]:


client = MongoClient('mongodb://localhost:27017/')
db = client['InstituteApproval']
collection = db['Applications']


# In[22]:


df=df.head()


# In[23]:


data_dict = df.to_dict(orient='records') 
print(data_dict)


# In[25]:


collection.insert_many(data_dict)
missing_values = df.isnull().sum()
print(f"Missing values in each column:\n{missing_values}")


# In[9]:


df.fillna(df.median(), inplace=True)
for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)
state_counts = collection.aggregate([
    {"$group": {"_id": "$state", "count": {"$sum": 1}}}
])
for state in state_counts:
    print(f"State: {state['_id']}, Applications: {state['count']}")    


# In[10]:


label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le
X = df.drop('approval', axis=1)  
y = df['approval']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[12]:


nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)


# In[13]:


svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)


# In[14]:


recall_nb = recall_score(y_test, y_pred_nb)
f1_nb = f1_score(y_test, y_pred_nb)

recall_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)

print(f"Naive Bayes - Recall: {recall_nb:.4f}, F1 Score: {f1_nb:.4f}")
print(f"SVM - Recall: {recall_svm:.4f}, F1 Score: {f1_svm:.4f}")


# In[15]:


region_program_counts = df.groupby(['region', 'program']).size().reset_index(name='count')


# In[17]:


plt.figure(figsize=(10, 6))
for region in region_program_counts['region'].unique():
    region_data = region_program_counts[region_program_counts['region'] == region]
    plt.bar(region_data['program'], region_data['count'], label=region)

plt.title('Relationship between Region and Program')
plt.xlabel('Program')
plt.ylabel('Number of Applications')
plt.xticks(rotation=90)
plt.legend(title='Region')
plt.tight_layout()
plt.show()


# In[ ]:




