#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 20:50:20 2018

@author: liqikai
"""

#此脚本先调用强agent生成一批数据，根据此结果训练出初步的模型，并存起来，后续的训练在此上训练
import os
#os.environ['KERAS_BACKEND']='tensorflow'
import keras

BATCH_SIZE = 128   
NUM_CLASSES = 4     #分类数目
NUM_EPOCHS = 20   #训练的迭代次数

from game2048.game import Game
from game2048.displays import Display
from game2048.agents import ExpectiMaxAgent,MyOwnAgent
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
import numpy as np
from sklearn.model_selection import train_test_split

image=[]
label=[]

display1 = Display()
display2 = Display()

stop_number = 2048
size = int(np.log2(stop_number)) +1    #跑到stop number时所需的one-hot编码位数

for i in range(0,500):   #跑500次棋盘，跑到stop_number停止
    game = Game(4, score_to_win=2048, random=False)
    agent = ExpectiMaxAgent(game, display=display1)  #使用强Agent
    
    while game.end==False:
        a=np.array(game.board)
        
        direction=agent.step()
        image.append(game.board)
        label.append(direction)
        game.move(direction)
        if np.amax(a)==stop_number:
            break
       
    display1.display(game)
    
image=np.array(image)   #将得到的数据和标签转换为numpy数组
label=np.array(label)


#划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(image, label, test_size = 0.1, random_state= 30)

size = int(np.log2(stop_number)) +1    #跑到stop number时所需的one-hot编码位数

input_shape = (4, 4, size)
x_train=np.log2(x_train+1)
x_train=np.trunc(x_train)
x_train = keras.utils.to_categorical(x_train, size) # one-hot编码
x_test=np.log2(x_test+1)
x_test=np.trunc(x_test)   #截断，即取整
x_test = keras.utils.to_categorical(x_test, size)    # one-hot编码


#print(x_train.shape)

y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)       #对标签one-hot编码
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

# define the model
model=Sequential() # 第一个卷积层，卷积核，大小，卷积模式SAME,激活函数relu,输入张量的大小 
model.add(Conv2D(filters= 128, kernel_size=(4,1),kernel_initializer='he_uniform', padding='Same', activation='relu',input_shape=input_shape)) 

model.add(Conv2D(filters= 128, kernel_size=(1,4), kernel_initializer='he_uniform',padding='Same', activation='relu')) 

model.add(Conv2D(filters= 128, kernel_size=(2,2),kernel_initializer='he_uniform', padding='Same', activation='relu')) 

model.add(Conv2D(filters= 128, kernel_size=(3,3),kernel_initializer='he_uniform', padding='Same', activation='relu')) 

model.add(Conv2D(filters= 128, kernel_size=(4,4),kernel_initializer='he_uniform', padding='Same', activation='relu'))  

model.add(Flatten()) 

model.add(Dense(4, activation='softmax')) 

# define the object function, optimizer and metrics
optimizer = RMSprop(lr = 0.001, decay=0.0)


model.compile(optimizer=optimizer, loss = 'categorical_crossentropy',metrics=['accuracy'])
# train
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=2)

# evaluate
score_train = model.evaluate(x_train, y_train, verbose=0)
print('Training loss: %.4f, Training accuracy: %.2f%%' % (score_train[0]*100,score_train[1]*100))
score_test = model.evaluate(x_test, y_test, verbose=0)
print('Testing loss: %.4f, Testing accuracy: %.2f%%' % (score_test[0]*100,score_test[1]*100))

model.save('first_model.h5')  #保存初步训练好的模型


##########################################################################################################
##########################################################################################################
##########################################################################################################
####################以上为初步训练的模型，后续在此基础上改进##############################################

count = 0
image=[]
label=[]
while count<190:   #训练200次有190次到达1024就停止
    
    image=[]
    label=[]   #每次image和label都用来储存400次棋盘和标签的数据
    count = 0
    
    for k in range(0,200):#跑200次棋盘作为train_on_batch的数据
    
        game = Game(4, score_to_win=2048, random=False)
        agent = ExpectiMaxAgent(game, display=display1)  #使用强Agent
        my_agent = MyOwnAgent(game, display=display2)  #使用自己的agent
    
        while game.end==False:
            direction1=agent.step()  #强agent跑棋盘的方向
            
            x=np.array(game.board)
            temp=np.amax(x)
        
            x=np.log2(x+1)
            x=np.trunc(x)
            x = keras.utils.to_categorical(x, size)
            x = x.reshape(1, 4, 4, size)
            pred=model.predict(x,batch_size=128)
            r=pred[0]
            r1=r.tolist()
            direction2=r1.index(max(r1))         #自己训练模型判断的方向，用自己的agent继续跑下去
        
            image.append(game.board)
            label.append(direction1)   #正确的标签是强agent的方向
            game.move(direction2)     #跑下去用的是自己的agent
            
        display1.display(game)    #每次棋盘结束显示一次

        if temp == 1024:          #上一次棋盘的最大值到达1024，count+1
            count +=1
        
    if count >190:           #训练200次有190次到达1024就停止,否则继续训练
        break
    else:
        image=np.array(image)   #将得到的数据和标签转换为numpy数组
        label=np.array(label)
        #划分训练集和测试集
        x_train, x_test, y_train, y_test = train_test_split(image, label, test_size = 0.1, random_state= 30)

        input_shape = (4, 4, size)
        x_train=np.log2(x_train+1)
        x_train=np.trunc(x_train)
        x_train = keras.utils.to_categorical(x_train, size) # one-hot编码
        
        x_test=np.log2(x_test+1)
        x_test=np.trunc(x_test)   #截断，即取整
        x_test = keras.utils.to_categorical(x_test, size)    # one-hot编码

        y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)       #对标签one-hot编码
        y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

        # train
        model.train_on_batch(x_train, y_train)

        model.save('model_2048.h5')  #保存训练好的模型
        
        image=np.reshape(image,(-1,4))
        np.savetxt("image.txt",image)        #注意：从image.txt可以看出当前棋盘能跑到多少
        np.savetxt("label.txt",label)


