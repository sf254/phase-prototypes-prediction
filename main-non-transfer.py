# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 00:00:46 2020

@author: fs
"""

#最终结果画图
#成分向量
#人工特征工程
# Load libraries 

import numpy as np
import aux_fun #convert fraction
import matplotlib.pyplot as plt
import pickle

import pandas as pd
import aux_fun_MF as MF

import model_evaluation_utils_V1 as meu1 
# Load libraries 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score

[Prototype,df_filter] = pickle.load(open('phase-prototypes-dataset.txt','rb'))
df_len = len(df_filter)

dim = []
temp = []

for num in range(df_len):
    nrow,mcol = df_filter[num].shape
    dim.append(nrow)
    #temp = []
    for i in range(nrow):
        i0 = df_filter[num].iloc[i,1]
        i1 = i0.split()[0]
        i2= aux_fun.fra_convert(i1)
        temp.append(i2)  
x0 = [i.split(' ')[1] for i in temp]
dataX = []


for i in x0:
    temp = MF.comp_vector(i)      
    dataX.append(temp)    
dataX = np.array(dataX)


y0 = [[Prototype[i]]*dim[i] for i in range(df_len)]
dataY = sum(y0, [])

score=[]

random_state=4
test_ratio = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]

for test_size in test_ratio:#
    dataXtr, dataXte, dataYtr, dataYte = train_test_split(dataX, dataY, test_size=test_size, \
                                                    random_state=random_state,stratify=dataY)
    randomforest = RandomForestClassifier(bootstrap=False,random_state=0, max_depth=20,n_estimators=100,n_jobs=-1,class_weight='balanced')#max_depth=20,
    randomforest.fit(dataXtr, dataYtr)
    yte_pred=randomforest.predict(dataXte)
    ytr_pred=randomforest.predict(dataXtr)
    
    score.append([recall_score( dataYte, yte_pred ,average='macro'),\
                  precision_score( dataYte, yte_pred,average='macro'),\
                  accuracy_score( dataYte, yte_pred)])
               
    report = meu1.display_classification_report(true_labels=dataYte, 
                                          predicted_labels=yte_pred, 
                                          classes=list(set(dataYte))) 
    df_report = pd.DataFrame(report).transpose()

    name0='res-comp'+str(test_size)#
    df_report.to_excel(name0+'.xlsx')
    

        
srf = []
for test_size in test_ratio:    
    namerf = 'res-comp'+str(test_size)#
    df = pd.read_excel(namerf+'.xlsx', index_col=0)
    acc = df.loc['accuracy'].tolist()[0]
    temp = df.loc['weighted avg'].tolist()[0:3]
    temp.extend([acc])
    srf.append(temp)
    
yrf = np.array(srf)
fig=plt.figure(figsize=(8,6),dpi=75)
ax = fig.add_subplot(111)
#ax.scatter(x,ycnn[:,-1],marker='o') 
ax.scatter(test_ratio,yrf[:,-1],marker='*')
ax.set_xlabel('Ratio of test dataset to entire dataset ')
ax.set_ylabel('Accuracy on test datset')
ax.set_ylim(0.42,0.92)  
plt.show()     
    


