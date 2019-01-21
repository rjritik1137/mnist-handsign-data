import numpy as np
import math
import h5py
import tensorflow as tf
import math
import pandas as pd
def random_mini_batches(X,Y,minibatch_size):
    
    
    m=X.shape[1]
    mini_batches=[]
    
    permutation=list(np.random.permutation(m))
    
    shuffled_X=X[:,permutation]
    shuffled_Y=Y[:,permutation].reshape(Y.shape[0],m)
    
    
    number_of_minibatches=math.floor(m/minibatch_size)
    
    
    
    for k in range(number_of_minibatches):
        mini_batch_X=shuffled_X[:,k*minibatch_size:k*minibatch_size+minibatch_size]
        mini_batch_Y=shuffled_Y[:,k*minibatch_size:k*minibatch_size+minibatch_size]
        minibatch=(mini_batch_X,mini_batch_Y)
        mini_batches.append(minibatch)
        
    
    
    if m%minibatch_size != 0:
        mini_batch_X = shuffled_X[:,number_of_minibatches*minibatch_size:m]
        mini_batch_Y = shuffled_Y[:,number_of_minibatches*minibatch_size:m]
        minibatch=(mini_batch_X,mini_batch_Y)
    
    
    
    mini_batches.append(minibatch)
    return mini_batches


def load_dataset():
    train_dataset=h5py.File('E:\\spider\\Practice\\train_signs.h5',"r")
    trainset_x_orig=np.array(train_dataset["train_set_x"][:])
    trainset_y_orig=np.array(train_dataset["train_set_y"][:])
    test_dataset=h5py.File('E:\\spider\\Practice\\test_signs.h5',"r")
    testset_x_orig=np.array(test_dataset["test_set_x"][:])
    testset_y_orig=np.array(test_dataset["test_set_y"][:])
    
    classes=np.array(test_dataset["list_classes"][:])
    
    trainset_y_orig=trainset_y_orig.reshape(1,trainset_y_orig.shape[0])
    testset_y_orig=testset_y_orig.reshape(1,testset_y_orig.shape[0])
    
    return trainset_x_orig,trainset_y_orig,testset_x_orig,testset_y_orig, classes


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def forward_predict(X,parameter,n_h):
    Z=dict()
    A=dict()
    A[str(0)]=X
    for i in range(1,n_h):
        Z[str(i)]=tf.add(tf.matmul(parameter["W"+str(i)],A[str(i-1)]),parameter["b"+str(i)])
        A[str(i)]=tf.nn.relu(Z[str(i)])
    return Z[str(n_h-1)]
def load_mnist_data():
    train_path="Location"
    dataset1=pd.read_csv(train_path)
    y1=dataset1['label']
    y_train=np.array(y1).reshape(1,-1)
    xt1=[]
    for i in range(1,785):
        x1=dataset1['pixel'+str(i)]
        xt1.append(np.array(x1))
    x_t1=np.array(xt1) 
    test_path="Location"
    dataset2=pd.read_csv(test_path)
    y2=dataset2['label']
    y_test=np.array(y2).reshape(1,-1)
    xt2=[]
    for i in range(1,785):
       x1=dataset2['pixel'+str(i)]
       xt2.append(np.array(x1))
    x_t2=np.array(xt2)
    return x_t1.T.reshape(-1,28,28),y_train,x_t2.T.reshape(-1,28,28),y_test
     