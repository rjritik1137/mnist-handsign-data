import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import create_batch_load_data
import math

# Define training model for the mnist handsign dataset
def model(X_train,Y_train,X_test,Y_test,learning_rate=0.001,nepoch=1000,minibatch_size=1445,print_cost=True):
     ops.reset_default_graph()
     (n_x,m)=X_train.shape
     n_y=Y_train.shape[0]
     costs=[]
     X,Y=create_placeholders(n_x,n_y)
     n_h=np.array([n_x,35,23,43,n_y])
     parameters=initialize_parameters(n_h)
     Z=forward_propagation(X,parameters,len(n_h))
     cost=compute_cost(Z,Y)
     optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
     init=tf.global_variables_initializer()
     
     with tf.Session() as sess:
         sess.run(init)
         
         for epoch in range(nepoch):
             epoch_cost=0.
             minibatch_cost=0.
             num_mini_batch=int(m/minibatch_size)
             
             minibatches=random_mini_batches(X_train,Y_train,minibatch_size)
             for minibatch in minibatches:
                 (minibatch_X,minibatch_Y)=minibatch
                 _,minibatch_cost=sess.run([optimizer,cost],feed_dict={X:minibatch_X,Y:minibatch_Y})
                 epoch_cost = epoch_cost+minibatch_cost/num_mini_batch
                 
             if epoch%100==0:
                 print("Cost after epoch %i: %f" %(epoch,epoch_cost))
             if epoch%5==0:
                 costs.append(epoch_cost)
         plt.plot(np.squeeze(costs))
         plt.ylabel('cost')
         plt.xlabel('iterations (per tens)')
         plt.title("Learning rate =" + str(learning_rate))
         plt.show()

        # lets save the parameters in a variable
         parameters = sess.run(parameters)
         print ("Parameters have been trained!")

        # Calculate the correct predictions
         correct_prediction = tf.equal(tf.argmax(Z), tf.argmax(Y))

        # Calculate accuracy on the test set
         accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
 
         print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
         print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
         return parameters
        
        
        
        
        
        
        
        
        
        
        
# Create placeholders for Input layer and  Y      
def create_placeholders(n_x,n_y):
     X=tf.placeholder("float32",[n_x,None],name='X')
     Y=tf.placeholder("float32",[n_y,None],name='Y')
     return X,Y
 
    #Initialize Parameters for neural network
def initialize_parameters(l):
    L=len(l)

    parameter=dict()
    for i in range(1,L):
        parameter["W"+str(i)]=tf.get_variable(name="W"+str(i),shape=[l[i],l[i-1]],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        parameter["b"+str(i)]=tf.get_variable(name="b"+str(i),shape=[l[i],1],dtype=tf.float32,initializer=tf.zeros_initializer())
        #print(parameter["W"+str(i)].shape,parameter["b"+str(i)].shape)
        
    return parameter

def forward_propagation(X,parameter,n_h):
    Z=dict()
    A=dict()
    A[str(0)]=X
    for i in range(1,n_h):
        Z[str(i)]=tf.add(tf.matmul(parameter["W"+str(i)],A[str(i-1)]),parameter["b"+str(i)])
        A[str(i)]=tf.nn.relu(Z[str(i)])
    return Z[str(n_h-1)]

#Return the cost
def compute_cost(Z,Y):
    logits=tf.transpose(Z)
    labels=tf.transpose(Y)
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=labels))
    return cost



#Load dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig = load_mnist_data()
    
index = 0
plt.imshow(X_test_orig[index])
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))
Y_train=convert_to_one_hot(Y_train_orig,25)
Y_test=convert_to_one_hot(Y_test_orig,25)
X_train_flatten=X_train_orig.reshape(X_train_orig.shape[0],-1).T
X_test_flatten=X_test_orig.reshape(X_test_orig.shape[0],-1).T
X_train=X_train_flatten/255
X_test=X_test_flatten/255
parameters = model(X_train, Y_train, X_test, Y_test)


def predict(X,parameter):
    n_h=5
    param=dict()
    for i in range(1,n_h):
        param['W'+str(i)]=tf.convert_to_tensor(parameter['W'+str(i)])
        param['b'+str(i)]=tf.convert_to_tensor(parameter['b'+str(i)])
    x=tf.placeholder(tf.float32,[X.shape[0],None])
    z=forward_predict(x,param,n_h)
    p=tf.argmax(z)
    sess=tf.Session()
    prediction=sess.run(p,feed_dict={x:X})
    return prediction
    

import scipy
from PIL import Image
from scipy import ndimage

predict_image=predict(X_test[:,0].reshape(-1,1),parameters)
plt.imshow(X_test_orig[0])
print("Your algorithm predicts: y = " + str(np.squeeze(predict_image)))
