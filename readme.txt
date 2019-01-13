有关文件说明如下：
first_try.py:最初进行训练调用的脚本，包括模型的建立,可以直接运行；
first_try_256.py:分层训练到256之前的棋盘，需要调用first_try.py生成的model_2048.h5模型文件；
first_try_512.py:分层训练到512之前的棋盘；
first_try_1024.py:分层训练到1024之前的棋盘；
model_2048.h5:在keras框架上训练后保存的模型文件，会在agents.py里面调用.
