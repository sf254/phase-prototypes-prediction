# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 21:14:58 2020

@author: fs
"""

import pickle
import numpy as np
import csv
import re
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam, SGD,Adadelta
from keras.models import load_model
from  keras.callbacks import ModelCheckpoint 

[comp1_set,energy_pa1_ave,_,_,_]=pickle.load(open('comp_energy_pa_oqmdf2b.txt', 'rb'))

[property_name_list,property_list,element_name,_]=pickle.load(open('element_property.txt', 'rb'))
Z_row_column = pickle.load(open('Z_row_column.txt', 'rb'))

    
#------------------------------------------------------------------------------    
#i0='01 Al100.00 Al'
def imageCb(i0):#map compsition to 2D representation with periodic table
    i=i0.split(' ')[1]
    X= [[[0.0 for ai in range(18)]for aj in range(9)] for ak in range(1) ]  
    tx1_element=re.findall('[A-Z][a-z]?', i)#[B, Fe, P,No]
    tx2_temp=re.findall('[0-9.]+', i)#[$_{[50]}$, ] [50 30 20]
    tx2_value=[float(re.findall('[0-9.]+', i_tx2)[0]) for i_tx2 in tx2_temp]
    for j in range(len(tx2_value)):
        index=int(property_list[element_name.index(tx1_element[j])][1])#atomic number
        xi=int(Z_row_column[index-1][1])#row num
        xj=int(Z_row_column[index-1][2])#col num
        X[0][xi-1][xj-1]=tx2_value[j]/100
    return X
#------------------------------------------------------------------------------
x=[]
y=energy_pa1_ave
for i in comp1_set:
   x.append(imageCb(i))

x_all=np.array(x).reshape(-1, 1,9, 18) #new code
y_all=-1*np.array(y) # Ef*(-1) due to ReLU

def CNNmodel(i=1):
    np.random.seed(i)  # for reproducibility1337
    model = Sequential()
    # Conv layer 1 output shape (32, 28, 28)
    model.add(Convolution2D(batch_input_shape=(None, 1, 9, 18),filters=8,
        kernel_size=3,strides=1, padding='same',
        data_format='channels_first', activation='relu'))
    model.add(Convolution2D(8, 3, strides=1, padding='same', 
                            data_format='channels_first',activation='relu'))
    model.add(MaxPooling2D(pool_size=2,strides=2,padding='same',data_format='channels_first',))
    model.add(Convolution2D(16, 3, strides=1, padding='same', 
                            data_format='channels_first',activation='relu'))#16
    model.add(Convolution2D(16, 3, strides=1, padding='same', 
                            data_format='channels_first',activation='relu'))
    model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))
    model.add(Convolution2D(32, 3, strides=1, padding='same', 
                            data_format='channels_first',activation='relu'))
    model.add(Convolution2D(32, 3, strides=1, padding='same', 
                            data_format='channels_first', activation='relu'))
    model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))#32
    model.add(Flatten())
    model.add(Dense(10,activation='relu'))
    model.add(Dense(1,activation='relu'))
    adadelta=Adadelta()
    # We add metrics to get more results you want to see
    model.compile(optimizer=adadelta,loss='mean_absolute_error',metrics=['mean_absolute_error'])
    return model

best_acc=[]
name0='-Ef-'
epoch=2000

#------------------------------------------------------------------------------
#5-fold cross validation
for k in range(5):
    #--------------------------------------------------------------------------
    i_tr=[i for i in range(len(y_all))]
    i_te=[i for i in range(len(y_all))]
    i_te=i_te[k::5]
    for i in i_te:
        i_tr.remove(i)

    x_train=x_all[i_tr]
    y_train=y_all[i_tr]
    x_test=x_all[i_te]
    y_test=y_all[i_te]
    
    name_best = 'CNN'+name0+'best-'+str(k)+'.h5'
    name_bestWb = 'CNN'+name0+'best-'+str(k)+'Wb.h5'
    name_last = 'CNN'+name0+'last-'+str(k)+'.h5'
    callback_lists = [ModelCheckpoint(filepath=name_best,\
                             monitor='val_loss',verbose=1,\
                             save_best_only='True',mode='auto',period=1)]
    
    model=CNNmodel(1)
        
    history=model.fit(x_train, y_train, epochs=epoch, batch_size=200,\
                  validation_data=(x_test, y_test),callbacks=callback_lists)
    model.save(filepath=name_last)
    
    model = load_model(name_best)
    model.save_weights(name_bestWb)
    #Evaluate the model with the metrics we defined earlier
    loss_te, accuracy_te = model.evaluate(x_test, y_test)
    loss_tr, accuracy_tr = model.evaluate(x_train, y_train)
    pred_te = model.predict(x_test, batch_size=200).reshape(-1)
    pred_tr = model.predict(x_train, batch_size=200).reshape(-1)
    r_tr=np.corrcoef(pred_tr,y_train)[0,1]
    r_te=np.corrcoef(pred_te,y_test)[0,1]
    print('\ntrain r={:.3f}'.format(r_tr))
    print('\ntest r={:.3f}'.format(r_te))
    
    best_acc.append([k,loss_tr, loss_te,r_tr, r_te])

    # save history loss & val_loss
    pickle.dump([history.history['mean_absolute_error'],history.history['val_mean_absolute_error']],\
                open(str(k)+'his2000.txt', 'wb'))
    
    with open('best_acc'+name0+'.csv',"a",newline ='') as csvfile: 
        writer = csv.writer(csvfile) 
        writer.writerow([k,loss_tr, loss_te,r_tr, r_te])
    

    fig, ax = plt.subplots(2,1,figsize=(6, 12))
    ax[0].scatter(pred_tr,y_train,s=2) 
    ax[0].set_title('r_tr={:.3f}'.format(r_tr))
    ax[0].set_xlabel('pred')
    ax[0].set_ylabel('GT')
    ax[1].scatter(pred_te,y_test,s=2)
    ax[1].set_title('r_te={:.3f}'.format(r_te))
    ax[1].set_xlabel('pred')
    ax[1].set_ylabel('GT')
    #plt.show() 
    fig.savefig(str(k)+name0+'acc.png',dpi=300) 
    
    # summarize history for accuracy
    fig, ax = plt.subplots(1,1,figsize=(12, 6))
    ax.plot(history.history['mean_absolute_error'][10:])
    ax.plot(history.history['val_mean_absolute_error'][10:])
    ax.set_title('mean_absolute_error')
    ax.set_ylabel('mean_absolute_error')
    ax.set_xlabel('epoch')
    ax.legend(['train', 'test'], loc='upper left')
    fig.savefig(str(k)+name0+'loss.png',dpi=300)
    
    del model