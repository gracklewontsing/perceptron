import random
import numpy as np
import pandas as pd
from pandas.io.parsers import read_csv

#get dataframes with xi and y results
column_names = ["x1","x2","x3","x4","y"]
df = read_csv("phishing2.csv", names=column_names)
x_1 = df.x1.to_list()
x_2 = df.x2.to_list()
x_3 = df.x3.to_list()
x_4 = df.x4.to_list()
y = df.y.to_list()

w_1 = []
w_2 = []
w_3 = []
w_4 = []

#create dataframes with random weights
for row in x_1:
    w_1.append(random.choice([-1,1]))    

for row in x_2:
    w_2.append(random.choice([-1,1]))    

for row in x_3:
    w_3.append(random.choice([-1,1]))    

for row in x_4:
    w_4.append(random.choice([-1,1]))    


#def threshold for adjust and determination
threshold = 0
def step(weighted_sum):
    if weighted_sum >= threshold:
        return 1
    else:
        return -1

#define prediction dataframe and perceptron function
p = []
def perceptron():
    weighted_sum = 0
    for x1,w1,x2,w2,x3,w3,x4,w4 in zip(x_1, w_1,x_2,w_2,x_3,w_3,x_4,w_4):
        weighted_sum = x1*w1 + x2*w2 + x3*w3 + x4*w4 + 0.5        
        p.append(step(weighted_sum))                    
perceptron()

#compare and train
def train():      
    epoch = 0      
    while p != y and epoch != 25:   
        print("Epoch: ", epoch)                     
        for i in range(len(y)):                                                            
            if p[i] != y[i]:         
                #print(i,j,w1,w2,w3,w4)                          
                w_1[i] += (y[i]*x_1[i])
                w_2[i] += (y[i]*x_2[i])
                w_3[i] += (y[i]*x_3[i])
                w_4[i] += (y[i]*x_4[i])
        perceptron()                
        epoch = epoch+1
    res = "\n".join("{} {}".format(x, y) for x,y in zip(p,y))
    print(res)

train()

