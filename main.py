from keras.layers.rnn import GRU
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import Embedding 
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import LSTM
import tensorflow as tf
from sklearn import preprocessing
from google.colab import files
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import load_model


data = pd.read_csv('/content/CBOT.$DJUSFN_071030_110310.csv')

data["<DATE>"] = data["<DATE>"].astype(str)
for i in range(0, len(data["<DATE>"])):
  data["<DATE>"][i] = datetime(int(data["<DATE>"][i][0:4]), int(data["<DATE>"][i][4:6]), int(data["<DATE>"][i][6:8]))
  data["<DATE>"][i] = datetime.timestamp(data["<DATE>"][i])

data = data[['<DATE>','<TIME>','<OPEN>','<HIGH>','<LOW>','<VOL>','<CLOSE>']]
data = data.apply(pd.to_numeric) # преобразуем в числовой формат
X = data.iloc[:,:data.shape[1]-1]
Y = data.iloc[:, data.shape[1]-1]
n=650
X_train, X_test, y_train, y_test = X.iloc[:n], X.iloc[n:], Y.iloc[:n], Y.iloc[n:]

print('Обучающие данные')
print(pd.concat([X_train,y_train], axis=1))


print('Тестовые данные')  
print(pd.concat([X_test,y_test], axis=1))



def create_plot(title, xlabel, ylabel, dataxg1, datayg1, label1, flag, dataxg2, datayg2, label2):
    plt.figure(figsize=(15, 6))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(dataxg1, datayg1, label=label1)
    if flag==1: 
        plt.plot(dataxg2, datayg2, label=label2)
    plt.legend()
    plt.show()

create_plot('Sectors of the US economy: Financials 30.10.2007-10.03.2011', "Time", "Value", data["<DATE>"], data["<CLOSE>"],'overall statistics', 0, None,None, None)

def build_model():
    model = Sequential()
    # изменяет формат данных (делает 3-х мерный массив, чтобы его принял слой GRU)
    # исп. чтобы убедиться, что размерность входных данных совпадает с тем, которые принимает GRU
    model.add(tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]))
    model.add(GRU(256))
    model.add(Dense(256, activation = "relu"))
    model.add(Dense(256, activation = "relu"))
    #исп. линейную ф-ию активации для выходного слоя
    model.add(Dense(1, activation='linear'))
    #производим компиляцию нашей модели
    model.compile(optimizer="adam", loss='mae',  metrics=['mae']) 

    return model


plt.figure(figsize=(10, 6))
plt.xlabel("epochs")
plt.ylabel("loss")
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# Оценка качества
train_mse = model.evaluate(X_train, y_train, verbose=0)
test_mse = model.evaluate(X_test, y_test, verbose=0)
print('Train: ',  train_mse, 'Test: ',test_mse)

predict = model.predict(X_test)
create_plot('Test data validation', "Time", "Value", X["<DATE>"], Y,'validation set', 1,X_test["<DATE>"],predict, "predicted")

pred = model.predict(X_test).flatten() 
for i in range(y_test.shape[0]):
   print("дата:", datetime.fromtimestamp(X_test.iloc[i,0]), ", вычисленное значение НС: ", round(pred[i],3), ", верный результат: ", 
         round(y_test.iloc[i],3), ", разница: ", round(y_test.iloc[i] - pred[i],3))


#сохраняем НС в файл
model.save("NS.h5")
files.download("NS.h5")

#ипользование уже обученной НС
files.upload()
model = load_model("NS.h5")