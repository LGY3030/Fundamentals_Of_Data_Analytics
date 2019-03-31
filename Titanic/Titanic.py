
# coding: utf-8

# In[1]:


import numpy
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense,Dropout


# In[2]:


def readData():   
    get_data=pd.read_excel("Titanic_Data.xls")

    column=["survived","name","pclass","sex","age","sibsp","parch","fare","embarked"]

    get_data=get_data[column]
    get_data=get_data.drop(['name'],axis=1)

    age_mean=get_data['age'].mean()
    get_data['age']=get_data['age'].fillna(age_mean)

    age_mean=get_data['fare'].mean()
    get_data['fare']=get_data['fare'].fillna(age_mean)

    get_data['sex']=get_data['sex'].map({'female':0,'male':1}).astype(int)

    get_data=pd.get_dummies(data=get_data,columns=["embarked"])
    
    return get_data


# In[3]:


#processed data
data=readData()
print(data)


# In[4]:


#age-->survived
data=readData()
columns=['survived','age']
data=data[columns]
data=data[data['survived']==1]
data=data.drop(["survived"], axis=1)
plt.hist(data['age'])
plt.show()


# In[5]:


#sex-->survived
#0-->woman
#1-->man
data=readData()
columns=['survived','sex']
data=data[columns]
data=data[data['survived']==1]
data=data.drop(["survived"], axis=1)
plt.hist(data['sex'])
plt.show()


# In[13]:


#fare-->survived
data=readData()
columns=['survived','fare']
data=data[columns]
data=data[data['survived']==1]
data=data.drop(["survived"], axis=1)
plt.hist(data['fare'])
plt.show()


# In[6]:


#Plot HeatMap using seaborn
data=readData()
sns.heatmap(data.corr(),annot=True)
plt.show()


# In[9]:


#use MLP to predict survive rate

data=readData()

msk=numpy.random.rand(len(data))<0.8

data=data.values
answer=data[:,0]
features=data[:,1:]

scale=preprocessing.MinMaxScaler(feature_range=(0,1))
features=scale.fit_transform(features)


train_x=features[msk]
train_y=answer[msk]
test_x=features[~msk]
test_y=answer[~msk]


model=Sequential()
model.add(Dense(units=40,input_dim=9,kernel_initializer='uniform',activation='relu'))
model.add(Dense(units=30,kernel_initializer='uniform',activation='relu'))
model.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
train_history=model.fit(x=train_x,y=train_y,validation_split=0.1,epochs=30,batch_size=30,verbose=1)

def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()

accuracy=model.evaluate(x=test_x,y=test_y)
print("")
print("")
print("")
print('accuracy:',accuracy[1])

show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')

