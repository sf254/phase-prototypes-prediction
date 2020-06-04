# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 12:25:37 2020

@author: fs
"""

import csv
import re
import pickle


[property_name_list,property_list,element_name,_]=pickle.load(open('C:\\pythonfs\\element_property.txt', 'rb'))
    
with open('C:\\pythonfs\\row_and_col.csv', 'r') as f:
    reader = csv.reader(f)
    RC = list(reader)
new_index=[int(i[4]) for i in RC]#新顺序 

    
def image(i0):#正常次序1通道C,简化化学式没有$$
    #i0='02 Mo50.00Nb50.00 Mo1Nb1 [12]'
    i=i0.split(' ')[1]
    X= [[[0.0 for ai in range(18)]for aj in range(9)] for ak in range(1) ]  
    tx1_element=re.findall('[A-Z][a-z]?', i)#[B, Fe, P,No]
    tx2_temp=re.findall('[0-9.]+', i)#[$_{[50]}$, ] [50 30 20]
    tx2_value=[float(re.findall('[0-9.]+', i_tx2)[0]) for i_tx2 in tx2_temp]
    for j in range(len(tx2_value)):
        index=int(property_list[element_name.index(tx1_element[j])][1])#atomic number
        xi=int(RC[index-1][1])#row num
        xj=int(RC[index-1][2])#col num
        X[0][xi-1][xj-1]=tx2_value[j]/100.0
     
    return X    

#import pprint
#pprint.pprint(x,width=60,compact=True)

def image_rPTR(i0):#正常次序1通道C，加入厚度在PTR中0911修改为随机顺序
    #i='4 La$_{66}$Al$_{14}$Cu$_{10}$Ni$_{10}$ [c][15]'
    i=i0.split(' ')[1]
    X= [[[0.0 for ai in range(18)]for aj in range(9)] for ak in range(1) ]
    tx1_element=re.findall('[A-Z][a-z]?', i)#[B, Fe, P,No]
    tx2_temp=re.findall('[0-9.]+', i)#[$_{[50]}$, ] [50 30 20]简化
    tx2_value=[float(re.findall('[0-9.]+', i_tx2)[0]) for i_tx2 in tx2_temp]
    for j in range(len(tx2_value)):
        index=new_index[int(property_list[element_name.index(tx1_element[j])][1])-1]#atomic number
        xi=int(RC[index-1][1])#row num
        xj=int(RC[index-1][2])#col num
        X[0][xi-1][xj-1]=tx2_value[j]/100.0
    return X

def image_AT(i0):#正常次序1通道C，加入厚度在PTR中0911修改为随机顺序
    #i='4 La$_{66}$Al$_{14}$Cu$_{10}$Ni$_{10}$ [c][15]'
    i=i0.split(' ')[1]
    X= [[[0.0 for ai in range(11)]for aj in range(11)] for ak in range(1) ]
    tx1_element=re.findall('[A-Z][a-z]?', i)#[B, Fe, P,No]
    tx2_temp=re.findall('[0-9.]+', i)#[$_{[50]}$, ] [50 30 20]简化
    tx2_value=[float(re.findall('[0-9.]+', i_tx2)[0]) for i_tx2 in tx2_temp]
    for j in range(len(tx2_value)):
        index=int(property_list[element_name.index(tx1_element[j])][1])#atomic number
        xi=(index-1)//11#row num
        xj=index%11-1#col num
        X[0][xi][xj]=tx2_value[j]/100.0
    return X

    
def image7by32(i0):#正常次序1通道C，加入厚度在PTR中
    with open('C:\\pythonfs\\row_and_col2.csv', 'r') as f:
        reader = csv.reader(f)
        RC = list(reader)
    
    i=i0.split(' ')[1]
    X= [[[0.0 for ai in range(32)]for aj in range(7)] for ak in range(1) ]  
    tx1_element=re.findall('[A-Z][a-z]?', i)#[B, Fe, P,No]
    tx2_temp=re.findall('[0-9.]+', i)#[$_{[50]}$, ] [50 30 20]
    tx2_value=[float(re.findall('[0-9.]+', i_tx2)[0]) for i_tx2 in tx2_temp]
    for j in range(len(tx2_value)):
        index=int(property_list[element_name.index(tx1_element[j])][1])#atomic number
        xi=int(RC[index-1][1])#row num
        xj=int(RC[index-1][2])#col num
        X[0][xi-1][xj-1]=tx2_value[j]/100
    return X