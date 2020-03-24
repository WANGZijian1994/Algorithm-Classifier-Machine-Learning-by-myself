#!/usr/bin/env python
# coding: utf-8

# In[90]:


### Author : Zijian


### To normalize the data: to change the values of numeric columns in the dataset to a common scale, 
### without distorting differences in the ranges of values
def Normalization(l):
    Max = max(l)
    Min = min(l)
    res = [(x-Min)/(Max-Min) for x in l]
    return res

### KNN classifer, you can change the value of parameter K as you wish, and you could also decide whether do the normalization or not;
def KNN_Zijian(test_data,value_data,group_name,k=5,Normalise = True):
    test_data = [list(x) for x in test_data]
    value_data = [list(value_data[i]) for i in range(len(value_data))]
    group_name = list(group_name)
    if Normalise:
        value_data = [Normalization(x) for x in value_data]  
        test_data = [Normalization(x) for x in test_data]
    distance = []
    for i in range(len(value_data)):
        diff = float()
        for j in range(len(value_data[i])):
            diff += abs(test_data[i][j]-value_data[i][j])**2 ### using euclidien distance for every group of test data/
        diff = diff**(0.5)
        feature = group_name[i]
        distance.append([diff,feature])
    ### find the most relevant feature between the k nearnest neighbor
    distance.sort()
    candidat = [x[1] for x in distance[:k]]
    res = {}
    for x in candidat:
        res[x]=res.get(x,0)+1
    Max = 0
    result = ""
    for x,y in res.items():
        if y>Max:
            Max = y
            result = x
    return result

### get the value classified by kNN for every element in the test data and get the result.
def KNN_fit(x_test,x_train,y_train,k=5,Normalization=True):
    result = []
    for x in x_test:
        test = [x for i in range(x_train.shape[0])]
        result.append(KNN_Zijian(test,x_train,y_train,k,Normalization))
    return result

def score(result,y_test):
    correct = 0
    taille = len(result)
    for i in range(taille):
        if result[i]==y_test[i]:
            correct+=1
    return str(correct/taille * 100)+"%"

### ---- Test with Iris data in Sklearn dataset

import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()
x_data = data.data
y_data = data.target

X_train,X_test,Y_train,Y_test = train_test_split(x_data,y_data,test_size = 0.2,random_state=0) ### 80% train 20% test

result = KNN_fit(X_test,X_train,Y_train,k=5,Normalization = False)
score_final = score(result,Y_test)
print(score_final)


# In[ ]:




