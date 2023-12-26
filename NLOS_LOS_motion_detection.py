#%%
from scipy.signal import butter, lfilter
from CSIKit.filters.passband import lowpass
from CSIKit.filters.statistical import running_mean
from CSIKit.util.filters import hampel
import math
import cmath
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,BatchNormalization ,GRU, Dropout,Flatten,LSTM,Conv2D,MaxPooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import RMSprop

#%%

# motion_list = ['los_walking','los_sitting','los_standing','los_na','nlos_walking','nlos_sitting','nlos_standing','nlos_na']

# np.set_printoptions(threshold=np.inf, linewidth=np.inf)
# np.set_printoptions(precision=6, suppress=True)

# path = 'G:/내 드라이브/CSI paper/raw_data/'

# x_data = []
# y_data = []
# red = []
# los_predata = []

# x_low_data = []
# x_hampel_data =[]
# x_mean_data = []

# for kk in range(len(motion_list)):
#     print(kk)
#     for jj in range(0,100):
        
#         if jj == 0:
#             print(motion_list[kk])
#         lt = []
        
#         df = pd.read_csv(path + motion_list[kk]  +  str(jj) + '.csv', encoding='utf-8')

#         df = df.drop([df.columns[0],df.columns[1],df.columns[2],df.columns[3],df.columns[4],df.columns[5]],axis=1)
#         df = df.drop([df.columns[-1],df.columns[-2],df.columns[-3],df.columns[-4],df.columns[-5]],axis=1)
#         df = df.drop([df.columns[26]],axis=1)

#         csi_data = df.values

#         csi_value = np.zeros(csi_data.shape,dtype = 'complex128')
#         amp_arr = np.zeros(csi_data.shape, dtype='float16')
#         red_arr = np.zeros(csi_data.shape)        
#         for k in range(0,len(csi_data)):
#             for j in range(0,len(csi_data[0])):
#                 csi_data[k][j] = csi_data[k][j].replace('i','j')
#                 csi_value[k][j] = complex(csi_data[k][j])
#                 amp_arr[k][j] = round(abs(csi_value[k][j]),2)
#                 red_arr[k][j] = math.degrees(csi_value[k][j])
#         x_data.append(amp_arr)        
#         red.append(red_arr)
#         ham_lt = []
#         low_lt = []
#         mean_lt = []
                
#         for x in range(0,19):
#             low_lt.append(lowpass(amp_arr[x], 100, 5000, 5 ))
#             ham_lt.append(hampel(amp_arr[x], 10, 3))
#             mean_lt.append(running_mean(amp_arr[x], 10))
        
#         x_low_data.append(low_lt)
#         x_hampel_data.append(ham_lt)
#         x_mean_data.append(mean_lt)
        
#         los_predata.append(csi_value)
        
#         if kk>=0 and kk<=3:
#             if kk == 0:
#                 y_data.append([[1,0],[1,0,0,0]])
#             elif kk == 1:
#                 y_data.append([[1,0],[0,1,0,0]])
#             elif kk == 2:
#                 y_data.append([[1,0],[0,0,1,0]])
#             elif kk == 3:
#                 y_data.append([[1,0],[0,0,0,1]])    
#         else:
#             if kk == 4:
#                 y_data.append([[0,1],[1,0,0,0]])
#             elif kk == 5:
#                 y_data.append([[0,1],[0,1,0,0]])
#             elif kk == 6:
#                 y_data.append([[0,1],[0,0,1,0]])
#             elif kk == 7:
#                 y_data.append([[0,1],[0,0,0,1]])
#         #y_data.append(y_lt) 

# x_data = np.array(x_data)
# y_data = np.array(y_data)
# x_low_data = np.array(x_low_data)
# x_hampel_data = np.array(x_hampel_data)
# x_mean_data = np.array(x_mean_data)


#%%

#IFFF(CSI) -> CIR  
motion_list = ['los_walking','los_sitting','los_standing','los_na','nlos_walking','nlos_sitting','nlos_standing','nlos_na']

np.set_printoptions(threshold=np.inf, linewidth=np.inf)
np.set_printoptions(precision=6, suppress=True)

path = 'G:/내 드라이브/CSI paper/raw_data/'

x_data = []
y_data = []
cfr_data = []
cir_data = []
original_data =[]
for kk in range(len(motion_list)):
    for jj in range(0,100):
        df = pd.read_csv(path + motion_list[kk]  +  str(jj) + '.csv', encoding='utf-8')
        df = df.drop([df.columns[0],df.columns[1],df.columns[2],df.columns[3],df.columns[4],df.columns[5]],axis=1)
        df = df.drop([df.columns[-1],df.columns[-2],df.columns[-3],df.columns[-4],df.columns[-5]],axis=1)
        df = df.drop([df.columns[26]],axis=1)
        df = df.drop([df.columns[5],df.columns[18],df.columns[-6],df.columns[-19]],axis=1)
        df.columns = [str(kk) for kk in range(0,48)]
        print(df.head)

        
        csi_data = df.values

        csi_value = np.zeros(csi_data.shape,dtype = 'complex128')
      
        for k in range(0,len(csi_data)):
            for j in range(0,len(csi_data[0])):
                try:
                    csi_data[k][j] = csi_data[k][j].replace('i','j')
                except:
                    pass
                csi_value[k][j] = complex(csi_data[k][j])
    
        cfr_data.append(csi_value)
                        
        if kk>=0 and kk<=3:
            if kk == 0:
                y_data.append([[1,0],[1,0,0,0]])
            elif kk == 1:
                y_data.append([[1,0],[0,1,0,0]])
            elif kk == 2:
                y_data.append([[1,0],[0,0,1,0]])
            elif kk == 3:
                y_data.append([[1,0],[0,0,0,1]])    
        else:
            if kk == 4:
                y_data.append([[0,1],[1,0,0,0]])
            elif kk == 5:
                y_data.append([[0,1],[0,1,0,0]])
            elif kk == 6:
                y_data.append([[0,1],[0,0,1,0]])
            elif kk == 7:
                y_data.append([[0,1],[0,0,0,1]])
                
y_data = np.array(y_data)            
print(cfr_data)
cfr_data = np.array(cfr_data)
print(cfr_data.shape)
cir_data = np.zeros(cfr_data.shape,dtype = 'complex128')
h_t = []

amp_lt = []
for col in range(len(cfr_data)): # 800
    lt = []
    amp = []   
    for z in range(len(cfr_data[0])): #19
        x = 0.0 
        y = 0.0
        cir_data[col][z] = np.fft.ifft(cfr_data[col][z])
        zz = []
        for kk in range(len(cfr_data[0][0])):
            x += cir_data[col][z][kk].real
            y += cir_data[col][z][kk].imag
            ss = abs(cir_data[col][z][kk])
            zz.append(ss)
        da = complex(x,y)    
        lt.append( abs(da) )
        amp.append(zz)
    h_t.append(lt)
    amp_lt.append(amp)

h_t = np.array(h_t)        
amp_lt = np.array(amp_lt)


# los_data = amp_lt[0:400]
# nlos_data = amp_lt[400:800]


los_data = h_t[0:400]
nlos_data = h_t[400:800]

print(h_t.shape)

#%%

print(cir_data)

#%%

for jj in range(0,len(motion_list)):
    df = pd.read_csv(path + str(motion_list[jj])   + '19.csv' , encoding='utf-8')
    # slicing padding subcarrier and DC component
    df = df.drop([df.columns[0],df.columns[1],df.columns[2],df.columns[3],df.columns[4],df.columns[5]],axis=1)
    df = df.drop([df.columns[-1],df.columns[-2],df.columns[-3],df.columns[-4],df.columns[-5]],axis=1)
    df = df.drop([df.columns[26]],axis=1)
    df = df.drop([df.columns[5],df.columns[18],df.columns[-6],df.columns[-19]],axis=1)
    not_filter = df.values
    
    csi_value = np.zeros(csi_data.shape,dtype = 'complex128')
    amp_arr = np.zeros(csi_data.shape, dtype='float16')
    low = []
    ham = []
    mean = []    
        
    for k in range(0,len(csi_data)):
        for j in range(0,len(csi_data[0])):
            csi_data[k][j] = not_filter[k][j].replace('i','j')
            csi_value[k][j] = complex(csi_data[k][j])
            #csi_value[k] = np.fft.ifft(csi_value[k])
            amp_arr[k][j] = round(abs(csi_value[k][j]),2)
            #red_arr[k][j] = math.radians(csi_value[k][j])

    
            
    for x in range(0,19):
        # b,a = butter(N = 5, Wn = 0.04, btype = 'low')
        # low_lt = lfilter(b, a, amp_arr[x])
        # low.append(low_lt)
        low.append(lowpass(amp_arr[x], 80000, 4000000, 5 )) # Sampling frequency * 2 becasue nyquest sampling theory 
        ham.append(hampel(amp_arr[x], 10, 3))
        mean.append(running_mean(amp_arr[x], 5))

    lt_plot = []
    lt2 = []
    ll = []
    #print(not_filter.shape)
    
    for kk in range(0,19):
        lt_plot.append(kk)

    for kk in range(0,48):
        lt2.append(kk)

    # for kk in range(len(red_arr)):    
    #     ll.append(red_arr[kk][6]) 
        
    
    print(amp_arr)

    plt.title(str(motion_list[jj]))      
    plt.plot(lt_plot,low,markersize = 1,marker = 'o')
    # plt.plot(lt2,amp_arr[1],markersize = 1,marker = 'o',linestyle="--", linewidth = '0.2')
    # plt.plot(lt2,amp_arr[2],markersize = 1,marker = 'o',linestyle="--", linewidth = '0.2')
    # plt.plot(lt2,amp_arr[3],markersize = 1,marker = 'o',linestyle="--", linewidth = '0.2')
    # plt.plot(lt2,amp_arr[4],markersize = 1,marker = 'o',linestyle="--", linewidth = '0.2')
    # plt.plot(lt2,amp_arr[5],markersize = 1,marker = 'o',linestyle="--", linewidth = '0.2')
    plt.xlabel("Packet Number")
    #plt.xticks(np.arange(0,48,1))
    plt.ylabel("Magnitude")
    plt.show()
    
    # plt.title(str(motion_list[jj]))      
    # plt.plot(lt_plot,red_arr ,markersize = 1,marker = 'o',linestyle="--", linewidth = '0.2')
    # #plt.bar(red_arr)
    # plt.xlabel("Packet Number")
    # #plt.xticks(np.arange(0,19,1))
    # plt.ylabel("Magnitude")
    # plt.show()

    # plt.title(str(motion_list[jj])+'_low pass filter')      
    # org_sig = plt.plot(lt_plot,low ,markersize = 3 )
    # plt.xlabel("Packet Number")
    # plt.xticks(np.arange(0,19,1))
    # plt.ylabel("Magnitude")
    # plt.show()
    
    # plt.title(str(motion_list[jj])+'_hampel_filter')      
    # org_sig = plt.plot(lt_plot,ham ,markersize = 3)
    # plt.xlabel("Packet Number")
    # plt.xticks(np.arange(0,19,1))
    # plt.ylabel("Magnitude")
    # plt.show()

    # plt.title(str(motion_list[jj])+'_mean')      
    # org_sig = plt.plot(lt_plot,mean ,markersize = 3)
    # plt.xlabel("Packet Number")
    # plt.xticks(np.arange(0,19,1))
    # plt.ylabel("Magnitude")
    # plt.show()

#%%

####################
# NLoS / LoS model #
####### >_< ########

LoS_NLoS_model = Sequential()
LoS_NLoS_model.add(GRU(units=512, activation="relu", return_sequences=True, input_shape = (19,52)))
# LoS_NLoS_model.add(BatchNormalization())
LoS_NLoS_model.add(GRU(units=512, activation="relu", return_sequences=True,dropout = 0.1))
LoS_NLoS_model.add(GRU(units=256, activation="relu", return_sequences=True,dropout = 0.1))
LoS_NLoS_model.add(Flatten())
LoS_NLoS_model.add(Dense(units = 2, activation = 'softmax' ))

#%%

###############################
# NLoS / LoS model LSTM 대조군 #
####### >_< ###################

LSTM_LoS_NLoS_model = Sequential()
LSTM_LoS_NLoS_model.add(LSTM(units=512, activation="relu", return_sequences=True, input_shape = (19,52)))
LSTM_LoS_NLoS_model.add(BatchNormalization())
LSTM_LoS_NLoS_model.add(LSTM(units=512, activation="relu", return_sequences=True,dropout = 0.1))
LSTM_LoS_NLoS_model.add(LSTM(units=256, activation="relu", return_sequences=True,dropout = 0.1))
LSTM_LoS_NLoS_model.add(Flatten())
LSTM_LoS_NLoS_model.add(Dense(units = 2, activation = 'softmax' ))

#%%
x_train , x_test , y_train , y_test = train_test_split(h_t, y_data ,train_size=0.8 , random_state=42)

LoS_NLoS_y_train_data = []
LoS_NLoS_y_test_data = []
motion_model_y_train_data = []
motion_model_y_test_data = []
for i in range(0,len(y_train)):
    LoS_NLoS_y_train_data.append(y_train[i][0])
    motion_model_y_train_data.append(y_train[i][1])
    
for i in range(0,len(y_test)):
    LoS_NLoS_y_test_data.append(y_test[i][0])
    motion_model_y_test_data.append(y_test[i][1])
    
LoS_NLoS_y_train_data = np.array(LoS_NLoS_y_train_data)
LoS_NLoS_y_test_data = np.array(LoS_NLoS_y_test_data)
motion_model_y_train_data = np.array(motion_model_y_train_data)
motion_model_y_test_data = np.array(motion_model_y_test_data)


print('------------Data_Scale------------')
print(x_train)
print(x_train.shape)
print(len(x_test))
print(len(LoS_NLoS_y_test_data))

print('done')
#%%
n_los_x_train = []
n_los_y_train = []

n_los_x_test = []
n_los_y_test = []

for ii in range(0,len(x_train)):
    if y_train[ii][0] == [0,1]:
        n_los_x_train.append(x_train[ii])
        n_los_y_train.append(y_train[ii][1])    

        
for ii in range(0,len(x_test)):
    if y_test[ii][0] == [0,1]:
        n_los_x_test.append(x_test[ii])
        n_los_y_test.append(y_test[ii][1])

n_los_x_train = np.array(n_los_x_train)
n_los_y_train = np.array(n_los_y_train)
n_los_x_test = np.array(n_los_x_test)
n_los_y_test = np.array(n_los_y_test)

print(len(n_los_x_train))
print(len(n_los_y_train))
print(len(n_los_x_test))
print(len(n_los_y_test))
                
#%%

###########################
#  LoS motion detection   #
########### >_< ###########

LoS_model = Sequential()
LoS_model.add(GRU(units=1024, activation="relu", return_sequences=True, input_shape = (10,19)))
LoS_model.add(BatchNormalization())
LoS_model.add(GRU(units=1024, activation="relu", return_sequences=True,dropout = 0.1))
LoS_model.add(GRU(units=516, activation="relu", return_sequences=True,dropout = 0.1))
LoS_model.add(GRU(units=516, activation="relu", return_sequences=True,dropout = 0.1))
LoS_model.add(GRU(units=258, activation="relu", return_sequences=True,dropout = 0.1))
LoS_model.add(GRU(units=128, activation="relu", return_sequences=True,dropout = 0.1))
LoS_model.add(Flatten())
LoS_model.add(Dense(units = 128, activation = 'relu' ))
LoS_model.add(Dense(units = 4, activation = 'softmax' ))

#%%

####################################
#  LoS motion detection LSTM 대조군 #
########### >_< ####################

LoS_model = Sequential()
LoS_model.add(GRU(units=1024, activation="relu", return_sequences=True, input_shape = (19,52)))
LoS_model.add(BatchNormalization())
LoS_model.add(GRU(units=1024, activation="relu", return_sequences=True,dropout = 0.1))
LoS_model.add(GRU(units=516, activation="relu", return_sequences=True,dropout = 0.1))
LoS_model.add(GRU(units=516, activation="relu", return_sequences=True,dropout = 0.1))
LoS_model.add(GRU(units=258, activation="relu", return_sequences=True,dropout = 0.1))
LoS_model.add(GRU(units=128, activation="relu", return_sequences=True,dropout = 0.1))
LoS_model.add(Flatten())
LoS_model.add(Dense(units = 128, activation = 'relu' ))
LoS_model.add(Dense(units = 4, activation = 'softmax' ))

#%%
los_x_train = []
los_y_train = []

los_x_test = []
los_y_test = []

for ii in range(0,len(x_train)):
    if y_train[ii][0] == [1,0]:
        los_x_train.append(x_train[ii])
        los_y_train.append(y_train[ii][1])    

        
for ii in range(0,len(x_test)):
    if y_test[ii][0] == [1,0]:
        los_x_test.append(x_test[ii])
        los_y_test.append(y_test[ii][1])

los_x_train = np.array(los_x_train)
los_y_train = np.array(los_y_train)
los_x_test = np.array(los_x_test)
los_y_test = np.array(los_y_test)

print(len(los_x_train))
print(len(los_y_train))
print(len(los_x_test))
print(len(los_y_test))

#%%

#########################
# NLoS motion detection #
########## >_< ##########

NLoS_model = Sequential()
NLoS_model.add(GRU(units=1024, activation="relu", return_sequences=True, input_shape = (19,52)))
NLoS_model.add(BatchNormalization())
NLoS_model.add(GRU(units=1024, activation="relu", return_sequences=True,dropout = 0.1))
NLoS_model.add(GRU(units=516, activation="relu", return_sequences=True,dropout = 0.1))
NLoS_model.add(GRU(units=516, activation="relu", return_sequences=True,dropout = 0.05))
NLoS_model.add(GRU(units=258, activation="relu", return_sequences=True,dropout = 0.1))
NLoS_model.add(GRU(units=128, activation="relu", return_sequences=True,dropout = 0.05))
NLoS_model.add(Flatten())
NLoS_model.add(Dense(units = 4, activation = 'softmax' ))

########################################
# NLoS motion detection LSTM 모델 대조군#
########## >_< #########################

LSTM_NLoS_model = Sequential()
LSTM_NLoS_model.add(LSTM(units=1024, activation="relu", return_sequences=True, input_shape = (19,52)))
LSTM_NLoS_model.add(BatchNormalization())
LSTM_NLoS_model.add(LSTM(units=1024, activation="relu", return_sequences=True,dropout = 0.1))
LSTM_NLoS_model.add(LSTM(units=516, activation="relu", return_sequences=True,dropout = 0.1))
LSTM_NLoS_model.add(LSTM(units=516, activation="relu", return_sequences=True,dropout = 0.05))
LSTM_NLoS_model.add(LSTM(units=258, activation="relu", return_sequences=True,dropout = 0.1))
LSTM_NLoS_model.add(LSTM(units=128, activation="relu", return_sequences=True,dropout = 0.05))
LSTM_NLoS_model.add(Flatten())
LSTM_NLoS_model.add(Dense(units = 4, activation = 'softmax' ))


#%%

#####################################
#LoS and NLoS motion detection model#
########### >_< #####################

motion_model = Sequential()
motion_model.add(GRU(units=1024, activation="relu", return_sequences=True, input_shape = (19,52)))
motion_model.add(BatchNormalization())
motion_model.add(GRU(units=1024, activation="relu", return_sequences=True,dropout = 0.1))
motion_model.add(GRU(units=516, activation="relu", return_sequences=True,dropout = 0.1))
motion_model.add(GRU(units=516, activation="relu", return_sequences=True,dropout = 0.1))
motion_model.add(GRU(units=258, activation="relu", return_sequences=True,dropout = 0.1))
motion_model.add(GRU(units=128, activation="relu", return_sequences=True,dropout = 0.1))
motion_model.add(Flatten())
motion_model.add(Dense(units = 128, activation = 'relu' ))
motion_model.add(Dense(units = 4, activation = 'softmax' ))

#%%
motion_model.compile(optimizer=RMSprop(learning_rate = 0.00001) , loss='categorical_crossentropy',metrics=['accuracy'])
motion_model_history = motion_model.fit(x_train, motion_model_y_train_data, epochs=10, batch_size=10, validation_data=(x_test, motion_model_y_test_data))

#%%
motion_model.save('C:/Users/Don/low_motion_detection.h5')

#%%
LoS_NLoS_model.compile(optimizer=RMSprop(learning_rate = 0.00001) , loss='binary_crossentropy',metrics=['accuracy'])
LoS_NLoS_history = LoS_NLoS_model.fit(x_train, LoS_NLoS_y_train_data, epochs=50, batch_size=10, validation_data=(x_test, LoS_NLoS_y_test_data))
#%%
# 모델 구현 완료 저장 방지 주석처리 확인 하기
LoS_NLoS_model.save('C:/Users/Don/low_pass_Los_NLoS_identifcation.h5')
#%%    
NLoS_model.compile(optimizer=RMSprop(learning_rate = 0.00001) , loss='categorical_crossentropy',metrics=['accuracy'])
NLoS_history = NLoS_model.fit(n_los_x_train, n_los_y_train, epochs=35, batch_size=5, validation_data=(n_los_x_test, n_los_y_test))
#%%
NLoS_model.save('C:/Users/Don/low_pass_NLoS_motion_detection.h5')

#%%
LoS_model.compile(optimizer=RMSprop(learning_rate = 0.00001) , loss='categorical_crossentropy',metrics=['accuracy'])
LoS_history = LoS_model.fit(los_x_train, los_y_train, epochs=35, batch_size=5, validation_data=(los_x_test, los_y_test))
#%%
LoS_model.save('C:/Users/Don/low_pass_LoS_motion_detection.h5')

#%%
# LSTM_NLoS_model.compile(optimizer=RMSprop(learning_rate = 0.00001) , loss='categorical_crossentropy',metrics=['accuracy'])
# LSTM_NLoS_history = LSTM_NLoS_model.fit(n_los_x_train, n_los_y_train, epochs=35, batch_size=5, validation_data=(n_los_x_test, n_los_y_test))
# #%%
# LSTM_NLoS_model.save('C:/Users/qaz10/LSTM_NLoS_motion_detection.h5')

#%%
#LOS/NLOS 성능검증 코드

from sklearn.metrics import confusion_matrix

model = tf.keras.models.load_model(filepath = 'C:/Users/Don/low_pass_Los_NLoS_identifcation.h5')

y_pred = model.predict(x_test, batch_size=100)
y_pred = np.argmax(y_pred, axis=1)
LoS_NLoS_y_test_data=np.argmax(LoS_NLoS_y_test_data, axis=1)

cf_matrix = confusion_matrix(LoS_NLoS_y_test_data, y_pred)

print('NLoS / LoS Identification')

from sklearn import metrics
print(cf_matrix)

accuracy = metrics.accuracy_score(LoS_NLoS_y_test_data, y_pred)
precision = metrics.precision_score(LoS_NLoS_y_test_data, y_pred,average='macro')
recall = metrics.recall_score(LoS_NLoS_y_test_data,y_pred,average='macro')
f1 = metrics.f1_score(LoS_NLoS_y_test_data,y_pred,average='macro')

print("accuracy:", accuracy)
print("precsion:", precision)
print("reacll:", recall)
print("f1:", f1)

######################################
# NLOS motion detection 성능검증 코드 #
######################################
from sklearn.metrics import confusion_matrix

model = tf.keras.models.load_model(filepath = 'C:/Users/Don/low_pass_NLoS_motion_detection.h5')

y_pred = model.predict(n_los_x_test, batch_size=100)
y_pred=np.argmax(y_pred, axis=1)
n_los_y_test=np.argmax(n_los_y_test, axis=1)

cf_matrix = confusion_matrix(n_los_y_test, y_pred)

print('NLoS motion detection')
from sklearn import metrics
print(cf_matrix)

accuracy = metrics.accuracy_score(n_los_y_test, y_pred)
precision = metrics.precision_score(n_los_y_test, y_pred,average='macro')
recall = metrics.recall_score(n_los_y_test,y_pred,average='macro')
f1 = metrics.f1_score(n_los_y_test,y_pred,average='macro')

print("accuracy:", accuracy)
print("precsion:", precision)
print("reacll:", recall)
print("f1:", f1)


model = tf.keras.models.load_model(filepath = 'C:/Users/Don/low_pass_LoS_motion_detection.h5')

y_pred = model.predict(los_x_test, batch_size=100)
y_pred=np.argmax(y_pred, axis=1)
los_y_test=np.argmax(los_y_test, axis=1)


cf_matrix = confusion_matrix(los_y_test, y_pred)

print('LoS motion detection')
from sklearn import metrics
print(cf_matrix)

accuracy = metrics.accuracy_score(los_y_test, y_pred)
precision = metrics.precision_score(los_y_test, y_pred,average='macro')
recall = metrics.recall_score(los_y_test,y_pred,average='macro')
f1 = metrics.f1_score(los_y_test,y_pred,average='macro')

print("accuracy:", accuracy)
print("precsion:", precision)
print("reacll:", recall)
print("f1:", f1)

# %%
from sklearn.metrics import confusion_matrix

model = tf.keras.models.load_model(filepath = 'C:/Users/Don/low_motion_detection.h5')

y_pred = model.predict(x_test, batch_size=100)
y_pred = np.argmax(y_pred, axis=1)
motion_model_y_test_data=np.argmax(motion_model_y_test_data, axis=1)

cf_matrix = confusion_matrix(motion_model_y_test_data, y_pred)

print('NLoS / LoS no lebel identification')

from sklearn import metrics
print(cf_matrix)

accuracy = metrics.accuracy_score(motion_model_y_test_data, y_pred)
precision = metrics.precision_score(motion_model_y_test_data, y_pred,average='macro')
recall = metrics.recall_score(motion_model_y_test_data,y_pred,average='macro')
f1 = metrics.f1_score(motion_model_y_test_data,y_pred,average='macro')


print("accuracy:", accuracy)
print("precsion:", precision)
print("reacll:", recall)
print("f1:", f1)
