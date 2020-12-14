# -*- coding: utf-8 -*-
"""
step01_keras_dnn_binary.py

Tensorflow2.0 keara
-DNN 모델 생성을 위한 고수준 API

iris dataset 이항분류기
- X변수 : scaling
- Y변수  : one hot encoding
- compile : model 학습 환경 
    loss = 'binary_crossentropy'
    metrics = ['accuracy']
    
"""

import tensorflow as tf
from sklearn.datasets import load_iris # dataset 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  minmax_scale # scaling(0~1)
from tensorflow.keras.utils import to_categorical # one hot encoding
from tensorflow.keras import Sequential #keras model 생성
from tensorflow.keras.layers import Dense #DNN layer 추가 

print(tf.keras.__version__) #2.2.4-tf

# 1. dataset load
X,y = load_iris(return_X_y= True)
X = X[:100]
y = y[:100]

print(X.shape) #(100, 4)
print(y.shape) #(100,) 0 or 1


# 전처리                                 
x_data = minmax_scale(X) # x변수 : scaling
y_data = to_categorical(y) # y변수 : one hot encoding

# 2. train / val split
x_train, x_val, y_train, y_val = train_test_split(
    x_data, y_data, test_size = 0.3)

# 3. keras model
model = Sequential()
print(model)

# 4. dnn layers
'''
model.add() : layer 추가
Dense(node 수,input_shape= 입력수, activation = '활성함수')
'''

# hidden layer1 : [4, 12] -> [in, out]
model.add(Dense(12, input_shape=(4,), activation='relu')) #1층

# hidden layer1 : [12, 6] -> [in, node]
model.add(Dense(6, activation='relu')) #2층

# output layer
model.add(Dense(2, activation= 'sigmoid')) # 3층 : 이항분류

# 5. model compile :학습환경
model.compile(optimizer = 'adam', #최적화 알고리즘
              loss = 'binary_crossentropy', # 손실 :one hot encoding
              metrics = ['accuracy']) # 평가 : 분류정확도

# model dnn layer 확인
model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_9 (Dense)              (None, 12)                60(4*12)+12      
_________________________________________________________________
dense_10 (Dense)             (None, 6)                 78(12*6)+6
_________________________________________________________________
dense_11 (Dense)             (None, 2)                 14(6*2)+2        
=================================================================
Total params: 152
Trainable params: 152
Non-trainable params: 0
_________________________________________________________________
'''


# 6. model training : 1epoch -> train(70) -> val(30)
model.fit(x_train, y_train,  #훈련셋
         epochs = 300, #반복학습 수 : 100-> 300
         verbose = 1, #출력
         validation_data = (x_val, y_val)) #검증셋

# 7. model evaluaion : val dataset

model.evaluate(x_val, y_val)
# 100 ->  0s 166us/sample - loss: 0.2426
# 300 -> 
























