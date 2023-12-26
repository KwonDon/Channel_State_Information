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
#from keras import Input , Dense
from tensorflow.keras.layers import Concatenate,BatchNormalization ,GRU, Dropout,Flatten,LSTM,Conv2D,MaxPooling2D #,Dense
from keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import RMSprop

#%%

motion_list = ['los_walking','los_sitting','los_standing','los_na','nlos_walking','nlos_sitting','nlos_standing','nlos_na']

np.set_printoptions(threshold=np.inf, linewidth=np.inf)
np.set_printoptions(precision=6, suppress=True)

path = 'G:/내 드라이브/CSI paper/raw_data/'

x_data = []
y_data = []
rad = []
los_predata = []

x_low_data = []
x_hampel_data =[]
x_mean_data = []

for kk in range(len(motion_list)):
    print(kk)
    for jj in range(0,100):
        
        if jj == 0:
            print(motion_list[kk])
        lt = []
        
        df = pd.read_csv(path + motion_list[kk]  +  str(jj) + '.csv', encoding='utf-8')

        df = df.drop([df.columns[0],df.columns[1],df.columns[2],df.columns[3],df.columns[4],df.columns[5]],axis=1)
        df = df.drop([df.columns[-1],df.columns[-2],df.columns[-3],df.columns[-4],df.columns[-5]],axis=1)
        df = df.drop([df.columns[26]],axis=1)

        csi_data = df.values

        csi_value = np.zeros(csi_data.shape,dtype = 'complex128')
        amp_arr = np.zeros(csi_data.shape, dtype='float16')
        rad_arr = np.zeros(csi_data.shape)        
        for k in range(0,len(csi_data)):
            for j in range(0,len(csi_data[0])):
                csi_data[k][j] = csi_data[k][j].replace('i','j')
                csi_value[k][j] = complex(csi_data[k][j])
                amp_arr[k][j] = round(abs(csi_value[k][j]),2)
                rad_arr[k][j] = math.degrees(csi_value[k][j])
        x_data.append(amp_arr)        
        rad.append(rad_arr)
        ham_lt = []
        low_lt = []
        mean_lt = []
                
        for x in range(0,19):
            low_lt.append(lowpass(amp_arr[x], 100, 5000, 5 ))
            ham_lt.append(hampel(amp_arr[x], 10, 3))
            mean_lt.append(running_mean(amp_arr[x], 10))
        
        x_low_data.append(low_lt)
        x_hampel_data.append(ham_lt)
        x_mean_data.append(mean_lt)
        
        los_predata.append(csi_value)
        
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

x_data = np.array(x_data)
y_data = np.array(y_data)
x_low_data = np.array(x_low_data)
x_hampel_data = np.array(x_hampel_data)
x_mean_data = np.array(x_mean_data)
rad = np.array(rad)

#%%

x_trainR , x_testR , y_train , y_test = train_test_split(rad, y_data ,train_size=0.8 , random_state=42)
x_trainM , x_testM , y_train , y_test = train_test_split(x_data, y_data ,train_size=0.8 , random_state=42)

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

x_trainR = np.array(x_trainR)
x_testR = np.array(x_testR)
x_trainM = np.array(x_trainM)
x_testM = np.array(x_testM)
y_test = np.array(y_test)
y_train = np.array(y_train)

# print('------------Data_Scale------------')
# print(len(x_train[0]))
# print(len(LoS_NLoS_y_train_data))
# print(len(x_test))
# print(len(LoS_NLoS_y_test_data))

# print('done')


#%%

rad_input = Input(shape=(19,52),dtype='float64',name = 'radius')
batch_layrt = BatchNormalization()(rad_input)
rad_layer0 = GRU(units = 512 ,activation="relu", return_sequences=True,dropout = 0.05)(batch_layrt)
rad_layer1 = GRU(units = 512 ,activation="relu", return_sequences=True,dropout = 0.05)(rad_layer0)
rad_layer2 = GRU(units = 256 ,activation="relu", return_sequences=True,dropout = 0.05)(rad_layer1)
rad_layer3 = GRU(units = 128 ,activation="relu", return_sequences=True,dropout = 0.05)(rad_layer2)
rad_flatten_layer = Flatten()(rad_layer3)

#tokenized_inputs = [tokenize(segment) for segment in text_inputs]

mag_input = Input(shape=(19,52),dtype='float16',name = 'maginitude')
batch_layrt = BatchNormalization()(mag_input)
mag_input0 = GRU(units = 512 ,activation="relu", return_sequences=True,dropout = 0.05)(batch_layrt)
mag_input1 = GRU(units = 512 ,activation="relu", return_sequences=True,dropout = 0.05)(mag_input0)
mag_input2 = GRU(units = 256 ,activation="relu", return_sequences=True,dropout = 0.05)(mag_input1)
mag_input3 = GRU(units = 128 ,activation="relu", return_sequences=True,dropout = 0.05)(mag_input2)
mag_flatten_layer = Flatten()(mag_input3)

concatenated = Concatenate()([rad_flatten_layer, mag_flatten_layer])
answer = Dense(units = 2, activation='softmax')(concatenated)

#%%
model = Model([rad_input, mag_input], answer)
model.compile(optimizer=RMSprop(learning_rate = 0.00001 ), loss='binary_crossentropy', metrics=['accuracy'])
model_history = model.fit((x_trainR,x_trainM), LoS_NLoS_y_train_data, epochs=30, batch_size=10, validation_data=((x_testR,x_testM), LoS_NLoS_y_test_data))
#%%
model.save('C:/Users/Don/mix_Identification_model.h5')

#%%
#%%
#########################
# NLoS motion detection #
########## >_< ##########

Identification_model = Sequential()
Identification_model.add(GRU(units=512, activation="relu", return_sequences=True, input_shape = (19,52)))
Identification_model.add(BatchNormalization())
Identification_model.add(GRU(units=512, activation="relu", return_sequences=True, ))
Identification_model.add(GRU(units=256, activation="relu", return_sequences=True, ))
Identification_model.add(GRU(units=128, activation="relu", return_sequences=True, ))
Identification_model.add(Flatten())
Identification_model.add(Dense(units = 2, activation = 'softmax' ))

#%%
Identification_model.compile(optimizer=RMSprop(learning_rate = 0.00001) , loss='binary_crossentropy',metrics=['accuracy'])
Identification_model_history = Identification_model.fit(x_train, LoS_NLoS_y_train_data, epochs=30, batch_size=10, validation_data=(x_test, LoS_NLoS_y_test_data))
#%%
Identification_model.save('C:/Users/Don/rad_Identification_model.h5')
#%%
from sklearn.metrics import confusion_matrix

model = tf.keras.models.load_model(filepath = 'C:/Users/Don/mix_Identification_model.h5')

y_pred = model.predict([x_testR,x_testM], batch_size = 10)
y_pred = np.argmax(y_pred, axis=1)
LoS_NLoS_y_test_data=np.argmax(LoS_NLoS_y_test_data, axis=1)

cf_matrix = confusion_matrix(LoS_NLoS_y_test_data, y_pred)

print('NLoS / LoS no lebel identification')

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

#%%
# 결과 실행시 red_arr 차원 변경으로 인해 훈련 데이터셋 차원 깨짐 주의
for jj in range(0,len(motion_list)):
    df = pd.read_csv(path + str(motion_list[jj])   + '20.csv' , encoding='utf-8')
    # slicing padding subcarrier and DC component
    df = df.drop([df.columns[0],df.columns[1],df.columns[2],df.columns[3],df.columns[4],df.columns[5]],axis=1)
    df = df.drop([df.columns[-1],df.columns[-2],df.columns[-3],df.columns[-4],df.columns[-5]],axis=1)
    df = df.drop([df.columns[26]],axis=1)
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
            amp_arr[k][j] = round(abs(csi_value[k][j]),2)
            #rad_arr[k][j] = math.radians(csi_value[k][j])
                        
    for x in range(0,19):
        low.append(lowpass(amp_arr[x], 80000, 4000000, 5 ))
        ham.append(hampel(amp_arr[x], 10, 3))
        mean.append(running_mean(amp_arr[x], 5))

    lt_plot = []
    lt2 = []
    ll = []

    for kk in range(0,19):
        lt_plot.append(kk)

    for kk in range(0,52):
        lt2.append(kk)

    for kk in range(len(rad_arr)):    
        ll.append(rad_arr[kk][6]) 

    plt.title(str(motion_list[jj]))      
    plt.plot(lt_plot,ll ,markersize = 1,marker = 'o',linestyle="--", linewidth = '0.2')
    plt.xlabel("Packet Number")
    plt.xticks(np.arange(0,19,1))
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