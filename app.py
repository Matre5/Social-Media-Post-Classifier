#!/usr/bin/env python
# coding: utf-8

# ## Data Processing

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('Data/social_media_posts.csv')


# In[3]:


df.head(10)


# In[4]:


df.info()


# In[6]:


df.shape


# In[7]:


df.isna().sum()


# In[8]:


df.describe()


# #### Feature distribution

# In[9]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df['likes'], kde=True, color='blue', bins=30)
plt.title('Distribution of Likes')
plt.show()

sns.histplot(df['shares'], kde=True, color='green', bins=30)
plt.title('Distribution of Shares')
plt.show()

sns.histplot(df['comments'], kde=True, color='red', bins=30)
plt.title('Distribution of Comments')
plt.show()


# #### Correlation Matrix

# In[11]:


corr_matrix = df.corr(numeric_only=Tr)

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# In[50]:


sns.countplot(x='popular', data=df, palette='Set2')
plt.title('Distribution of Popular vs. Not Popular Posts')
plt.savefig("POPvsNot.png")
plt.show()


# The data is imbalanced, the gap between the popular post and the not popular posts is alot to be omitted

# #### Balancing and Model creation

# In[17]:


# !pip install imbalanced-learn


# In[19]:


from imblearn.over_sampling import SMOTE


# SMOTE works by creating synthetic examples rather than duplicating the minority class.   
# It works by taking an instance of the minority class, finding its nearest neighbors,  
# and then generating synthetic samples that are combinations of the original and the neighbors.  
# 
# 

# In[23]:


X = df.drop('popular', axis=1)
y = df['popular']


# In[25]:


X_encoded = pd.get_dummies(X, drop_first=True)


# In[26]:


smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_encoded, y)


# In[36]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[31]:


X_train, X_test, Y_train, Y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)


# In[37]:


logreg = LogisticRegression()
logreg.fit(X_train, Y_train)


# In[38]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


# In[39]:


y_pred = logreg.predict(X_test)


# In[40]:


print("Accuracy score: ", accuracy_score(Y_test, y_pred))
print("\n Classification report \n", classification_report(Y_test, y_pred))


# In[42]:


cm = confusion_matrix(Y_test, y_pred)

sns.heatmap(cm, annot=True, cmap="Blues", fmt= "d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("CMatrix.png")
plt.show()


# Our Logistic Regression gives a wonderful accuracy of 96.7%, i am going to compare it with Random Forest to see which is higher

# In[43]:


from sklearn.ensemble import RandomForestClassifier


# In[44]:


rf = RandomForestClassifier()
rf.fit(X_train, Y_train)


# In[45]:


Yf_pred = rf.predict(X_test)


# In[46]:


print("Accuracy: ", accuracy_score(Y_test, Yf_pred))
print("\nClassification report: \n", classification_report(Y_test, Yf_pred))


# In[49]:


cm_rf = confusion_matrix(Y_test, Yf_pred)

sns.heatmap(cm_rf, annot=True, cmap="Reds", fmt = 'd')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix_RandomForest")
plt.savefig("CMatrix2.png")
plt.show()


# Random Forest can capture more complexity than simpler models.  
# Synthetic data doesn't have as much noise, even though the creation of the dataset had some noise added, we can never fully mimic 
# real world noise. Therefore, the structure made the classification easier.  
# Getting a 100% accuracy seemed possible because it is not a real world data

# In[51]:


import joblib


# In[53]:


joblib.dump(logreg, "Model/LogisticRegression.pkl")
joblib.dump(rf, "Model/random_forest_classifier.pkl")


# In[ ]:




