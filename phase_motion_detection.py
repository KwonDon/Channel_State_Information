#%%

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,BatchNormalization ,GRU, Dropout,Flatten,LSTM,Conv2D,MaxPooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import RMSprop
import sys
path = 'G:/내 드라이브/CSI paper/raw_data/'

np.set_printoptions(threshold=np.inf, linewidth=np.inf)
np.set_printoptions(precision=6, suppress=True)

motion_lt = ['walking', 'standing' , 'jumping' , 'sitting']
#%%

x_data = []
y_data = []

for jj in range(0,len(motion_lt)):

    for kk in range(0,30): # kk is filename index
        #df = pd.read_csv(path+str(kk)+'.csv')
        df = pd.read_csv(path + str(motion_lt[jj]+"phase") + str(kk) + '.csv' , encoding='utf-8')
        
        # slicing padding subcarrier and DC component
        df = df.drop([df.columns[0],df.columns[1],df.columns[2],df.columns[3],df.columns[4],df.columns[5]],axis=1)
        df = df.drop([df.columns[-1],df.columns[-2],df.columns[-3],df.columns[-4],df.columns[-5]],axis=1)
        df = df.drop([df.columns[26]],axis=1)
 
        csi_data = df.values
        
        
        no_list = []
        for kk in range(0,98):
            different_phase = []
            for ii in range(0,52):
                different_phase.append(round(csi_data[kk+1][ii] - csi_data[kk][ii],4))
            no_list.append(different_phase)
        # denoise using butterworth filter        
        # N = 2
        # Wn = 0.3

        # B,A =  signal.butter(N,Wn,output='ba',btype='low')

        # output = (signal.filtfilt(B,A,csi_data))
    
        output = no_list
        
    
        # data labeling
        if jj == 0:
            y_label = [1,0,0,0]
        elif jj == 1:
            y_label = [0,1,0,0]
        elif jj == 2:
            y_label = [0,0,1,0]
        elif jj == 3:
            y_label = [0,0,0,1]

        x_data.append(output) 
        y_data.append(y_label)

        
x_data = np.array(x_data)
y_data = np.array(y_data)


x_train , x_test , y_train , y_test = train_test_split(x_data,y_data ,train_size=0.7 , random_state=42)

print(x_data)
print(x_data.shape)

#%%

for jj in range(0,len(motion_lt)):
    df = pd.read_csv(path + str(motion_lt[jj]+"phase") + '10.csv' , encoding='utf-8')
    lt = []
    #df = pd.read_csv(path+str(kk)+'.csv')


    # slicing padding subcarrier and DC component
    df = df.drop([df.columns[0],df.columns[1],df.columns[2],df.columns[3],df.columns[4],df.columns[5]],axis=1)
    df = df.drop([df.columns[-1],df.columns[-2],df.columns[-3],df.columns[-4],df.columns[-5]],axis=1)
    df = df.drop([df.columns[26]],axis=1)

    csi_data = df.values
    
    
    lias = []
    
    for kk in range(0,98):
        different_phase = []
        for ii in range(0,52):
            different_phase.append(csi_data[kk+1][ii] - csi_data[kk][ii])
        lias.append(different_phase)

    lias = np.array(lias)
    print(lias.shape)



    for kk in range(0,98):
        lt.append(kk)
                
    # plt.title(str(motion_lt[jj]))  
    # plt.plot(lt,org , 'ro--',label = 'orignal signal' ,markersize = 3)
    # #plt.plot(lt,fil,'bo--' ,label = 'filtering signal',markersize = 3)
    # plt.xlabel("subcarrier index")
    # plt.ylabel("magnitude")
    # plt.legend(loc = 'upper right')
    # plt.show()

    plt.title(str(motion_lt[jj]))      
    plt.plot(lt,lias ,markersize = 3)
    # plt.plot(lt,dif_phase ,markersize = 3)
    # plt.plot(lt,dif_phase2 ,markersize = 3)
    # plt.plot(lt,dif_phase3 ,markersize = 3)
    # plt.plot(lt,dif_phase4 ,markersize = 3)
    # plt.plot(lt,dif_phase5 ,markersize = 3)
    #fil_sig = plt.plot(lt,filter,'bo--' ,markersize = 3)
    plt.xlabel("Packet Number")
    plt.ylabel("Magnitude")
    plt.legend(loc='upper right')
    plt.show()

#%%

cnn_motion = Sequential()
cnn_motion.add(Dense(1024,input_shape=(99,52) ,activation='relu' ))
cnn_motion.add(BatchNormalization()) 
cnn_motion.add(Dense(1024,activation='relu'))
cnn_motion.add(Dropout(0.05))
# cnn_motion.add(Dense(1024,activation='relu'))
cnn_motion.add(Dropout(0.05))
cnn_motion.add(Dense(512,activation='relu'))
cnn_motion.add(Dense(521,activation='relu'))
cnn_motion.add(Dropout(0.05))
cnn_motion.add(Dense(256,activation='relu'))
cnn_motion.add(Dense(128,activation='relu'))
cnn_motion.add(Dense(64,activation='relu'))
cnn_motion.add(Dropout(0.05))
cnn_motion.add(Dense(32,activation='relu'))
cnn_motion.add(Flatten())
cnn_motion.add(Dense(4,activation='softmax'))

#%%

LSTM_model = Sequential()
LSTM_model.add(LSTM(units=512, activation="relu", return_sequences=True, input_shape = (99,52)))
LSTM_model.add(BatchNormalization())
LSTM_model.add(LSTM(units=1024, activation="relu", return_sequences=True,dropout = 0.05))
LSTM_model.add(LSTM(units=500, activation="relu", return_sequences=True,dropout = 0.1))
LSTM_model.add(LSTM(units=128, activation="relu", return_sequences=True,dropout = 0.05 ))
LSTM_model.add(LSTM(units=100, activation="relu", return_sequences=True,dropout = 0.1))
LSTM_model.add(LSTM(units=64, activation="relu", return_sequences=True,dropout = 0.05 ))
LSTM_model.add(Dense(units = 300, activation = 'softmax' ))
LSTM_model.add(Flatten())
LSTM_model.add(Dense(units = 4, activation = 'softmax' ))

# softmax를 이용하여 classification


#%%

regression_GRU = Sequential()
regression_GRU.add(GRU(units=512, activation="relu", return_sequences=True, input_shape = (98,52)))
regression_GRU.add(BatchNormalization()) 
regression_GRU.add(GRU(units=1024, activation="relu", return_sequences=True,dropout = 0.05))
regression_GRU.add(GRU(units=500, activation="relu", return_sequences=True,dropout = 0.05))
regression_GRU.add(GRU(units=128, activation="relu", return_sequences=True,dropout = 0.05 ))
regression_GRU.add(GRU(units=100, activation="relu", return_sequences=True,dropout = 0.05))
regression_GRU.add(GRU(units=64, activation="relu", return_sequences=True,dropout = 0.05 ))
regression_GRU.add(Flatten())
regression_GRU.add(Dense(units = 4, activation = 'softmax' ))

#%%
print(len(y_data))
print(len(y_test))
print(len(y_train))

print(y_test)
print()
print(x_train)

#%%

# cnn_motion.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
# CNN_history = cnn_motion.fit(x_train, y_train, epochs=45, batch_size=10, validation_data=(x_test, y_test))

regression_GRU.compile(optimizer=RMSprop(learning_rate = 0.0001) , loss='categorical_crossentropy',metrics=['accuracy'])
GRU_history = regression_GRU.fit(x_train, y_train, epochs=20, batch_size=10, validation_data=(x_test, y_test))

# LSTM_model.compile(optimizer=RMSprop(learning_rate = 0.0001) , loss='categorical_crossentropy',metrics=['accuracy'])
# LSTM_model_history = LSTM_model.fit(x_train, y_train, epochs=15, batch_size=10, validation_data=(x_test, y_test))

#%%

# cnn_motion.save('C:/Users/Don/filter_CNN_motion_model.h5')

# # # 손실 함수 그래프
# plt.plot(CNN_history.history['loss'])
# plt.plot(CNN_history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train','Validation'],loc='best')
# plt.grid()
# plt.show()

regression_GRU.save('C:/Users/Don/different_phase_LSTM_motion_model.h5')

# 손실 함수 그래프
plt.plot(GRU_history.history['loss'])
plt.plot(GRU_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'],loc='best')
plt.grid()
plt.show()

# LSTM_model.save('C:/Users/Don/phase_LSTM_motion_model.h5')

# plt.plot(LSTM_model_history.history['loss'])
# plt.plot(LSTM_model_history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train','Validation'],loc='best')
# plt.grid()
# plt.show()

#%%

model = tf.keras.models.load_model(filepath = 'C:/Users/Don/different_phase_LSTM_motion_model.h5')

np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.8f}".format(x)})
# res=model.evaluate(x_test,y_test,verbose=0)


y_pred = model.predict(x_test, batch_size=100)
y_pred=np.argmax(y_pred, axis=1)
y_test=np.argmax(y_test, axis=1)

#%%

print(y_pred.shape)
print(y_data.shape)
    
#%%

from sklearn.metrics import confusion_matrix

cf_matrix = confusion_matrix(y_test, y_pred)


from sklearn import metrics
print(cf_matrix)

accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred,average='macro')
recall = metrics.recall_score(y_test,y_pred,average='macro')
f1 = metrics.f1_score(y_test,y_pred,average='macro')
print("accuracy:", accuracy)
print("precsion:", precision)
print("reacll:", recall)
print("f1:", f1)

#%%
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
