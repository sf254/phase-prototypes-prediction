# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 21:31:11 2020

@author: fs
"""

import re

def fra_convert(i0):
    tx=re.findall('[A-Z][a-z]?[0-9.]*', i0)
    tx_ele=[]
    tx_val=[]
    for i in tx:
        ele=re.findall('[A-Z][a-z]?', i)[0]
        val=re.findall('[0-9.]+', i)
        if len(val)==0:
            val=1.0
        else:
            val=float(val[0])
        tx_ele.append(ele)
        tx_val.append(val)
    
    #Normalize   
    tx_val_sum=sum(tx_val) 
    tx_ele_val=[]
    for j in range(len(tx_ele)):
        #tx_ele_val.append(tx_ele[j]+str(tx_val_100[j]))
        temp_val=tx_val[j]/tx_val_sum*100
        if temp_val>=0.01:
            tx_ele_val.append(tx_ele[j]+'{:05.2f}'.format(temp_val))
        
    tx_ele_val.sort()
    i0_new = "".join(tx_ele_val)
    length = '{:0>2d}'.format(len(tx_ele_val))
    
    return  length +' '+ i0_new


#------------------------------------------------------------------------------
def comp_split(i0):
    tx=re.findall('[A-Z][a-z]?[0-9.]*', i0)
    tx_ele=[]
    tx_val=[]
    for i in tx:
        ele=re.findall('[A-Z][a-z]?', i)[0]
        val=float(re.findall('[0-9.]+', i)[0])
        tx_ele.append(ele)
        tx_val.append(val)
    return tx_ele,tx_val 