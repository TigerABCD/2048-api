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

model = load_model('model_2048.h5')  #导入之前训练好的模型  


##########################################################################################################
##########################################################################################################
##########################################################################################################
####################以上为初步训练的模型，后续在此基础上改进##############################################

count = 0
image=[]
label=[]

board_class = 12
input_shape = (4, 4, board_class)  #棋盘one-hot编码

boards = []
directions = []

while count<45:
    score_train=[]
    score_test=[]
    boards=[]
    directions=[]
    count = 0
    
    print('array deleted')
    
    for i in range(50):
        game = Game(4,score_to_win = 2048,random=False)
        agent1 = ExpectiMaxAgent(game, display=display1)
        #agent2 = MineAgent(game, display=display1)
        
        while game.end==False:
            #if np.sum(game.board) > 2500:
                #break
            a = np.array(game.board)
            a = np.log2(a+1)
            a = np.trunc(a)
            a = keras.utils.to_categorical(a, board_class)
            a = a.reshape(1,4,4,board_class)
            prediction = model.predict(a,batch_size = 128)
            b = prediction[0]
            b = b.tolist()
            direction2=b.index(max(b))
            direction1=agent1.step()
            
            boards.append(game.board)
            directions.append(direction1)
            game.move(direction2)
        display1.display(game)
        if np.amax(game.board) == 1024:
            count +=1
        
    if count>=45:
        break
    else:
        boards = np.array(boards)
        directions = np.array(directions)
        
        x_train, x_test, y_train, y_test = train_test_split(boards, directions, test_size = 0.01, random_state= 30)
        x_train = np.log2(x_train+1)
        x_train = np.trunc(x_train)
        x_train = keras.utils.to_categorical(x_train, board_class)
        y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    
        x_test = np.log2(x_test+1)
        x_test = np.trunc(x_test)
        x_test = keras.utils.to_categorical(x_test, board_class)
        y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
        
        model.train_on_batch(x_train, y_train)
        
        model.save('model_2048.h5')
        print('model saved')
        
        boards=np.reshape(boards,(-1,4))
        np.savetxt("boards.txt",boards)
        np.savetxt("directions.txt",directions)