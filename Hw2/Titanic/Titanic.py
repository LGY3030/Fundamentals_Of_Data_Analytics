
# coding: utf-8

# In[15]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from graphviz import Source
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[16]:


def readData():   
    get_data=pd.read_excel("Titanic_Data.xls")

    column=["survived","name","pclass","sex","age","sibsp","parch","fare","embarked"]
    # survived:是否存活
    # name: 姓名
    # pclass: 住的艙等
    # sex: 性別
    # age: 年齡
    # sibsp: 兄弟姊妹＋老婆丈夫數量
    # parch: 父母小孩的數量
    # fare: 票的費用
    # embarked: 哪個港口出發

    get_data=get_data[column]
    
    get_data=get_data.drop(['name'],axis=1)

    #填補nan
    age_mean=get_data['age'].mean()
    get_data['age']=get_data['age'].fillna(age_mean)

    #填補nan
    fare_mean=get_data['fare'].mean()
    get_data['fare']=get_data['fare'].fillna(fare_mean)

    #轉換性別為1和0
    get_data['sex']=get_data['sex'].map({'female':0,'male':1}).astype(int)

    #將embarked轉為One hot形式
    get_data=pd.get_dummies(data=get_data,columns=["embarked"])
    
    return get_data


# In[25]:


#讀出資料
data=readData()

#將資料分為features和target(survived)
column_list=list(data.columns.values)
column_list.remove('survived')
data_x=data[column_list]
data_y=data[['survived']]

#將資料分為訓練資料和測試資料
train_x,test_x,train_y,test_y=train_test_split(data_x,data_y,test_size=0.3,random_state=0)


#建立Decision Tree分類器
tree=DecisionTreeClassifier(criterion='entropy',max_depth=3,random_state=0)
tree.fit(train_x,train_y)

#預測測試資料
tree.predict(test_x)

#此為測試資料的正確解答
test_y.values.reshape(-1)

#看有幾個預測錯
error=0
for i,j in enumerate(tree.predict(test_x)):
    if j != test_y.values.reshape(-1)[i]:
        error=error+1
print('Error: '+str(error)+'/'+str(test_y.shape[0])+' 個')

#算出預測的準確度
accuracy=(tree.score(test_x,test_y.values.reshape(-1)))*100
print('Accuracy: '+str(accuracy)+' %')

#輸出Decision Tree圖
export_graphviz(tree,out_file='tree.dot',feature_names=column_list)

#將.dot檔轉換為png檔,並開啟
os.environ["PATH"] += os.pathsep + 'D://Graphviz2.38/bin/'
s=Source.from_file('tree.dot',format="png")
img = mpimg.imread('tree.dot.png')
plt.imshow(img)
plt.show()


# In[28]:


tree.feature_importances_

