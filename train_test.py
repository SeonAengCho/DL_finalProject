import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from LSTMlayers import * 
from optimizer import *
from util import *
from functions import remove_duplicate


lr = 0.001
max_grad = 0.25

print("data load")

X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')


print("parameter")

rn = np.random.randn
lstm_Wx = (rn(100, 4 * 10) / np.sqrt(100)).astype('f')
lstm_Wh = (rn(10, 4 * 10) / np.sqrt(10)).astype('f')
lstm_b = np.zeros(4 * 10).astype('f')
affine_W = (rn(10, 1) / np.sqrt(10)).astype('f')
affine_b = np.zeros(1).astype('f')

print("model select")


layer1 = TimeLSTM(lstm_Wx,lstm_Wh,lstm_b)
layer2 = TimeAffine(affine_W, affine_b)
optimizer = Adam(lr)

# train
for epoch in tqdm(range(1)):
    
    loss_per_epoch = []
    
    for data in range(10):
        
        # forward
        hs = layer1.forward((X_train[data]))
        print(hs)
        score = layer2.forward(hs)
        
        
        # MSEloss 
        loss = 0.5 * ((score-y_train[data])**2)
        loss_per_epoch.append(loss)
        
        # backward
        dout = layer2.backward(loss)
        dout = layer1.backward(dout)
        
        params = []
        grads = []
        
        param1 = layer1.params
        param2 = layer2.params
        
        grad1 = layer1.grads
        grad2 = layer2.grads
        
        params += param1
        params += param2
        
        grads += grad1
        grads += grad2
        
        params, grads = remove_duplicate(params, grads)
        if max_grad is not None:
            clip_grads(grads, max_grad)
            
        optimizer.update(params, grads)
        
        
  
    print("epoch{}".format(epoch))
    print(" -> Loss Value: {}".format(sum(loss_per_epoch)/len(X_train)))

layer1.reset_state()

# test


print("data load")

X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')  

final_loss = []

for data in range(len(X_test)):
    
    hs = layer1.forward((X_test[data]))
    score = layer2.forward(hs)
    
    loss = 0.5 * ((score-y_test[data])**2)
    final_loss.append(loss)
    
print(" -> Loss Value: {}".format(sum(final_loss)/len(X_test)))
