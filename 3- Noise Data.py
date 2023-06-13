import numpy as np
import os
import glob
#import random


#train________________________________________________________________________
target = {'A':[1,-1,-1,-1,-1,-1,-1],
          'B':[-1,1,-1,-1,-1,-1,-1],
          'C':[-1,-1,1,-1,-1,-1,-1],
          'D':[-1,-1,-1,1,-1,-1,-1],
          'E':[-1,-1,-1,-1,1,-1,-1],
          'J':[-1,-1,-1,-1,-1,1,-1],
          'K':[-1,-1,-1,-1,-1,-1,1]}

alpha = 0.01
tetha = 0.0
w = np.array(63*[[-0.02]*len(target)]) #weight (size: 63*7)
#w = np.array(63*[[random.uniform(0, 1)]*len(target)]) #weight (size: 63*7)
b = [0.0]*len(target) #bias (size: 1*7)

#reading files
path = 'Characters-TrainSet/'
extension = 'txt'
os.chdir(path)
file_name = glob.glob('*.{}'.format(extension))
stop = False
counter = 0

while not stop and counter<=2000:
    pre_w = np.copy(w)
    
    counter+=1
    
    for file in file_name:
        
        #x: list of all characters in a file, size: 1*63
        x = []
        with open(file,'r') as f:
            for line in f:
                for word in line.split():
                   x+=[*word] 
                   
        
        #replace all characters of a file with 1 and -1
        for i in range(len(x)):
            x[i]=x[i].replace('.','-1')
            x[i]=x[i].replace('#','1')
        x = [int(i) for i in x]

        y_in = b+ np.dot(x,w)

        y = [0]*len(target)

        for i in range(len(y_in)):
            if y_in[i] >tetha :
                y[i] = 1
            elif -tetha <= y_in[i] <= tetha:
                y[i] = 0
            else:
                y[i] = -1
        
        
         
        
        for j in range(7):
            if y[j] != target[file[0]][j]:
                w.T[:][j] += alpha * np.array(x)*target[file[0]][j]
                b[j] += alpha*target[file[0]][j]
        
    if (pre_w == w).all():
        stop = True

#train Error__________________________________________________________________
num_true = 0             #number of correctly predicted output neurons 
num_false = 0            #number of incorrectly predicted output neurons

for file in file_name:
    
    #x: list of all characters in a file, size: 1*63
    x = []
    with open(file,'r') as f:
        for line in f:
            for word in line.split():
               x+=[*word] 

    #replace all characters of a file with 1 and -1
    for i in range(len(x)): 
        x[i]=x[i].replace('.','-1')
        x[i]=x[i].replace('#','1')
    x = [int(i) for i in x]
    
    y_in = b+ np.dot(x,w)
    
    y = [-1]*len(target)
    ind = np.argmax(y_in)
    y[ind] = 1
    
    for i in range(len(y)):
        if y[i] == target[file[0]][i]:
            num_true += 1
            
        else:
            num_false += 1

print('Train Error: ',num_false/(num_false+num_true)*100)
        

#test_________________________________________________________________________        
num_true = 0             #number of correctly predicted output neurons 
num_false = 0            #number of incorrectly predicted output neurons

#reading files
path = '../Characters-TestSet/'
extension = 'txt'
os.chdir(path)
file_name = glob.glob('*.{}'.format(extension))

for file in file_name:
    
    #x: list of all characters in a file, size: 1*63
    x = []
    with open(file,'r') as f:
        for line in f:
            for word in line.split():
               x+=[*word] 

    #replace all characters of a file with 1 and -1
    for i in range(len(x)): 
        x[i]=x[i].replace('.','-1')
        x[i]=x[i].replace('o','-1')
        x[i]=x[i].replace('#','1')
        x[i]=x[i].replace('@','1')
    x = [int(i) for i in x]
    
    y_in = b+ np.dot(x,w)
    
    y = [-1]*len(target)
    ind = np.argmax(y_in)
    y[ind] = 1
        
        
    
    for i in range(len(y)):
        if y[i] == target[file[0]][i]:
            num_true += 1
            
        else:
            num_false += 1

print('Test Error: ',num_false/(num_false+num_true)*100)


