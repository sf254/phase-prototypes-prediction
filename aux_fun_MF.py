# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 11:13:37 2020

@author: fs
"""
import pickle
import aux_fun

[property_name_list,property_list,element_name,_]=pickle.load(open('element_property.txt', 'rb'))

def comp_vector(i0):
    comp = [0]*108
    ele,val = aux_fun.comp_split(i0)
    for i in range(len(ele)):
        e_i=int(property_list[element_name.index(ele[i])][property_name_list.index('Atomic number')])
        comp[e_i-1] = val[i] 
    return comp    