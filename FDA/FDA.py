#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
from matplotlib import pyplot as plt
import datetime
import seaborn as sns


# In[24]:


#Top-10 Reviewer
#此output可能會跟Desired output不一樣,但答案是正確的,因為有些Score count一樣,但是排列順序不同,所以取到的前10名不一樣
reviews=pd.read_csv("Reviews.csv")
reviews=reviews[:10000]
data=reviews['UserId'].value_counts().rename_axis('UserId').reset_index(name='Score count')
data=data[:10]
data['Score mean']=0.0
data['ProfileName']=""
for i in range(data.shape[0]):
    temp=reviews[reviews.UserId == data['UserId'][i]].reset_index()
    data['Score mean'][i]=(temp['Score'].sum())/data['Score count'][i]
    data['ProfileName'][i]=temp['ProfileName'][0]
print(data)


# In[25]:


#Plot score distribution for the user with the most number of reviews
reviews=pd.read_csv("Reviews.csv")
reviews=reviews[:10000]
most_reviews=reviews[reviews.UserId == data['UserId'][0]]['Score'].reset_index()
most_reviews=most_reviews['Score'].value_counts().rename_axis('Score').reset_index(name='Score count')
plt.bar(most_reviews['Score'],most_reviews['Score count'])
plt.show()


# In[26]:


#Plot pandas Series DataFrame (Time->Date)
reviews=pd.read_csv("Reviews.csv")
reviews=reviews[:10000]
score_year=reviews
score_year=score_year['Time']
score_year=pd.to_datetime(score_year, unit='s').dt.year
score_year=score_year.value_counts().rename_axis('Year').reset_index(name='Score count')
plt.bar(score_year['Year'],score_year['Score count'])
plt.show()


# In[27]:


#Plot HeatMap using seaborn
reviews=pd.read_csv("Reviews.csv")
reviews=reviews[:10000]
reviews.drop(["ProductId"], axis=1)
reviews.drop(["UserId"], axis=1)
reviews.drop(["ProfileName"], axis=1)
reviews.drop(["Summary"], axis=1)
reviews.drop(["Text"], axis=1)
sns.heatmap(reviews.corr(),annot=True)
plt.show()


# In[28]:


#Helpful percent
reviews=pd.read_csv("Reviews.csv")
reviews=reviews[:10000]
Numerator=reviews['HelpfulnessNumerator']
Denominator=reviews['HelpfulnessDenominator']
result=Numerator/Denominator
result=result.fillna(-1)
result=result[result<=1]
result=result.replace(0,-1)
plt.hist(result)
plt.show()


# In[ ]:




