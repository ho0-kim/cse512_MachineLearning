
# coding: utf-8

# HW1 Q3. Linear Regression for Bitcoin/Ethereum Price
# =============
# 
# ### (a) Develop a linear regression model to predict Ethereum price. Print the last 10 values of {step, cost, W, b}

# In[1]:

# import packages
import tensorflow as tf
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt


# In[2]:

# load data
file_path = os.path.join('hw1data/bitcoin', 'export-EtherPrice.csv')
load_data = pd.read_csv(file_path, sep=',')
dataset = load_data.values[:, 1:]

plt.scatter(dataset[:,0], dataset[:,1])
plt.title('DATA')
plt.xlabel('Unix Time Stamp')
plt.ylabel('Ether Price')
plt.show()

x_train = dataset[:, 0]
y_train = dataset[:, 1]

# too big numbers to put it in float64
# need to be rescaled
x_train = x_train/100000000.0
y_train = y_train/100.0

plt.scatter(x_train, y_train)
plt.title('RESCALED DATA')
plt.xlabel('Unix Time Stamp/(10^8)')
plt.ylabel('Ether Price/100')
plt.show()


# In[3]:

learning_rate = 15**(-2)
epochs = 8000001
#epochs = 500000
g1 = tf.Graph()
with g1.as_default():
    X = tf.placeholder(tf.float64, shape = [None])
    Y = tf.placeholder(tf.float64, shape = [None])
    
    W = tf.Variable(tf.random_normal([1], dtype=tf.float64), name='weight')
    b = tf.Variable(tf.random_normal([1], dtype=tf.float64), name='bias')

    hypothesis = tf.add(tf.multiply(X, W), b)

    cost = tf.reduce_mean(tf.square(tf.subtract(hypothesis, Y)))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    train = optimizer.minimize(cost)

with tf.Session(graph=g1) as sess:
    print("step | cost | W | b")
    sess.run(tf.global_variables_initializer())

    for step in range(epochs):
        cost_val, W_val, b_val, _ =             sess.run([cost, W, b, train],
                     feed_dict={X: x_train, Y: y_train})
        if step > (epochs-11):
        #if step % 50000 == 0:
            print(step, cost_val, W_val, b_val)
   
    predict_price = sess.run(hypothesis, feed_dict={X: [(1520294400+13.0*86400.0)/(10.0**8.0)], Y: [1]})


# ### (b) Predict the price on March 19th?

# In[4]:

print("The Ether Price on March 19th : ", predict_price*100.0)


# ### Use transaction growth "export-TxGrowth.csv" as a feature to predict the Ethereum price.
# ### (c) Develop a linear regression model and print the last 10 values of {step, cost, W, b}.

# In[5]:

# load data
file_path = os.path.join('hw1data/bitcoin', 'export-TxGrowth.csv')
load_data = pd.read_csv(file_path, sep=',')
tx_growth = load_data.values[:, 2]
#print(tx_growth)

plt.scatter(tx_growth, y_train*100.0)
plt.title('DATA')
plt.xlabel('Transaction Growth')
plt.ylabel('Ether Price')
plt.show()

tx_growth = tx_growth/100000.0

plt.scatter(tx_growth, y_train)
plt.title('RESCALED DATA')
plt.xlabel('Transaction Growth/(10^5)')
plt.ylabel('Ether Price/100')
plt.show()


# In[6]:

learning_rate = 10**(-3)
epochs = 25001

g2 = tf.Graph()
with g2.as_default():
    X = tf.placeholder(tf.float64, shape = [None])
    Y = tf.placeholder(tf.float64, shape = [None])
    
    W = tf.Variable(tf.random_normal([1], dtype=tf.float64), name='weight')
    b = tf.Variable(tf.random_normal([1], dtype=tf.float64), name='bias')

    hypothesis = tf.add(tf.multiply(X, W), b)

    cost = tf.reduce_mean(tf.square(tf.subtract(hypothesis, Y)))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    train = optimizer.minimize(cost)

with tf.Session(graph=g2) as sess:
    print("step | cost | W | b")
    sess.run(tf.global_variables_initializer())

    for step in range(epochs):
        cost_val, W_val, b_val, _ =             sess.run([cost, W, b, train],
                     feed_dict={X: tx_growth, Y: y_train})
        if step > (epochs-11):
        #if step % 5000 == 0:
            print(step, cost_val, W_val, b_val)


# ### (e) Any correlations between transaction growth and price?
# 
# According to the plots, the price proportionally increases as the transaction growth increases even though high values of the transation growth are scattered. 
# ### Now, use price to predict the transaction growth.
# ### (f) Develop a linear regression model and print the last 10 values of {step, cost, W, b}

# In[7]:

learning_rate = 10**(-3)
epochs = 25001

g3 = tf.Graph()
with g3.as_default():
    X = tf.placeholder(tf.float64, shape = [None])
    Y = tf.placeholder(tf.float64, shape = [None])
    
    W = tf.Variable(tf.random_normal([1], dtype=tf.float64), name='weight')
    b = tf.Variable(tf.random_normal([1], dtype=tf.float64), name='bias')

    hypothesis = tf.add(tf.multiply(X, W), b)

    cost = tf.reduce_mean(tf.square(tf.subtract(hypothesis, Y)))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    train = optimizer.minimize(cost)

with tf.Session(graph=g3) as sess:
    print("step | cost | W | b")
    sess.run(tf.global_variables_initializer())

    for step in range(epochs):
        cost_val, W_val, b_val, _ =             sess.run([cost, W, b, train],
                     feed_dict={X: y_train, Y: tx_growth})
        if step > (epochs-11):
        #if step % 5000 == 0:
            print(step, cost_val, W_val, b_val)


# In[ ]:



