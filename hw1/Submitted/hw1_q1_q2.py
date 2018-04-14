
# coding: utf-8

# HW1 Q1. Linear Regression
# =============
# From the given password1.train file (the first column is the password and the second column is the strength), train your best linear regression model to predict the password strength from the password length.

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
file_path = os.path.join('hw1data', 'password1.train')
load_data = pd.read_csv(file_path, sep='\t', header = None)
dataset = np.array(load_data)

new_dataset = []
for i in range(dataset.shape[0]):
    new_dataset.append([len(dataset[i][0]), dataset[i][-1]])
new_dataset = np.array(new_dataset)


# In[3]:

amount_train = int(0.75*dataset.shape[0])

sc = StandardScaler() # for data standardization

x_train = new_dataset[:amount_train,0]
y_train = new_dataset[:amount_train,-1]

x_train = x_train.reshape(x_train.shape[0], 1)
y_train = y_train.reshape(y_train.shape[0], 1)

x_test = new_dataset[amount_train+1:,0]
y_test = new_dataset[amount_train+1:,-1]

x_test = x_test.reshape(x_test.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)

plt.scatter(x_train[:], y_train[:])
plt.title('TRAINING DATASET')
plt.xlabel('PASSWORD LENGTH')
plt.ylabel('STRENGTH')
plt.show()

plt.scatter(x_test[:], y_test[:])
plt.title('TEST DATASET')
plt.xlabel('PASSWORD LENGTH')
plt.ylabel('STRENGTH')
plt.show()

y_train = sc.fit_transform(y_train)
y_test = sc.fit_transform(y_test)

plt.scatter(x_train[:], y_train[:])
plt.title('TRAINING DATASET(STANDARDIZATION)')
plt.xlabel('PASSWORD LENGTH')
plt.ylabel('STANDARDIZED STRENGTH')
plt.show()

plt.scatter(x_test[:], y_test[:])
plt.title('TEST DATASET(STANDARDIZATION)')
plt.xlabel('PASSWORD LENGTH')
plt.ylabel('STANDARDIZED STRENGTH')
plt.show()


# ### (a) Print the last 10 values of {step, cost, W, b}

# In[4]:

learning_rate = 10**(-3)
epochs = 50001

g1 = tf.Graph()
with g1.as_default():
    X = tf.placeholder(tf.float32, shape = [None, 1])
    Y = tf.placeholder(tf.float32, shape = [None, 1])
    
    W = tf.Variable(tf.random_normal([1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

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
        #if step % 10000 == 0:
            print(step, cost_val, W_val, b_val)

    predictions = sess.run(hypothesis, feed_dict={X: x_test, Y: y_test})
    print("MSE: ", metrics.mean_squared_error(predictions, y_test))


# ### (b) Take a log10 for strengh values. And return the regression to predict log10(strength). Print the last 10 values of {step, cost, W, b}

# In[5]:

x_train = new_dataset[:amount_train,0]
y_train = np.log10(new_dataset[:amount_train,-1])
plt.scatter(x_train, y_train)
plt.title('TRAINING DATASET')
plt.xlabel('PASSWORD LENGTH')
plt.ylabel('LOG(STRENGTH)')
plt.show()

x_test = new_dataset[amount_train+1:,0]
y_test = np.log10(new_dataset[amount_train+1:,-1])
plt.scatter(x_test, y_test)
plt.title('TEST DATASET')
plt.xlabel('PASSWORD LENGTH')
plt.ylabel('LOG(STRENGTH)')
plt.show()

x_train = x_train.reshape(x_train.shape[0], 1)
y_train = y_train.reshape(y_train.shape[0], 1)
y_train = sc.fit_transform(y_train)

x_test = x_test.reshape(x_test.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)
y_test = sc.fit_transform(y_test)

plt.scatter(x_train, y_train)
plt.title('TRAINING DATASET(STANDARDIZATION)')
plt.xlabel('PASSWORD LENGTH')
plt.ylabel('LOG(STRENGTH)')
plt.show()

plt.scatter(x_test, y_test)
plt.title('TEST DATASET(STANDARDIZATION)')
plt.xlabel('PASSWORD LENGTH')
plt.ylabel('LOG(STRENGTH)')
plt.show()


# In[6]:

learning_rate = 10**(-4)
epochs = 300001

g2 = tf.Graph()
with g2.as_default():
    X = tf.placeholder(tf.float32, shape = [None, 1])
    Y = tf.placeholder(tf.float32, shape = [None, 1])

    W = tf.Variable(tf.random_normal([1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    hypothesis = tf.add(tf.multiply(X, W), b)

    cost = tf.reduce_mean(tf.square(tf.subtract(hypothesis, Y)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    train = optimizer.minimize(cost)
    
    accuracy = tf.reduce_mean(tf.square(tf.subtract(hypothesis, Y)))

with tf.Session(graph=g2) as sess:
#sess = tf.Session()
    print("step | cost | W | b")
    sess.run(tf.global_variables_initializer())

    for step in range(epochs):
        cost_val, W_val, b_val, _ =             sess.run([cost, W, b, train],
                     feed_dict={X: x_train, Y: y_train})
        if step > (epochs-11):
            print(step, cost_val, W_val, b_val)
    MSE_Len = sess.run(accuracy, feed_dict={X: x_test, Y: y_test})
    print("MSE: ", MSE_Len)


# ### (c) Between (a) and (b), which is a better model or representation? Explain

# As can be seen from the plots, data preprocessing, taking log10, made the gap between each Y values smaller, made the train easier.  
# When log10 is taken, it is possible that linear regression model is able to be applied even without Standardization. 

# Q2. Multivariable Linear Regression
# =====
# 
# From the given password1.train file, train your best linear regression model to predict the password strength from password length, number of digits, number of symbols, and the number of upper_class letter in a password.

# ### (a) Print the last 10 values of {step, cost, W, b}

# In[7]:

def data_preprocess(dataset):
    out = []
    for i in range(dataset.shape[0]):
        s = str(dataset[i][0])
        symbols = set(['`','~','!','@','#','$','%','^','&','*','(',')','_','-','+','=','[',']','{','}','|',':',';','<','>',',','.','/','?'])
        l = len(s)
        dg = 0
        sb = 0
        up = 0
        for c in s:
            if c.isdigit():
                dg = dg+1
            elif c in symbols:
                sb = sb+1
            elif c.isupper():
                up = up+1
        out.append([l, dg, sb, up, dataset[i][1]])
    return out

q2_dataset = np.array(data_preprocess(dataset))

amount_train = int(0.75*dataset.shape[0])

x_train = q2_dataset[:amount_train,:4]
y_train = q2_dataset[:amount_train,-1]

x_test = q2_dataset[amount_train+1:,:4]
y_test = q2_dataset[amount_train+1:,-1]

y_train = y_train.reshape(y_train.shape[0], 1)
y_train_sc = sc.fit_transform(y_train)

y_test = y_test.reshape(y_test.shape[0], 1)
y_test_sc = sc.fit_transform(y_test)


# In[8]:

learning_rate = 10**(-3)
epochs = 20001

g3 = tf.Graph()
with g3.as_default():
    X = tf.placeholder(tf.float32, shape = [None, 4])
    Y = tf.placeholder(tf.float32, shape = [None, 1])
    
    W = tf.Variable(tf.random_normal([4, 1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    hypothesis = tf.add(tf.matmul(X, W), b)

    cost = tf.reduce_mean(tf.square(tf.subtract(hypothesis, Y)))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    train = optimizer.minimize(cost)

with tf.Session(graph=g3) as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(epochs):
        cost_val, W_val, b_val, _ =             sess.run([cost, W, b, train],
                     feed_dict={X: x_train, Y: y_train_sc})
        if step > (epochs-11):
        #if step % 10000 == 0:
            print('step: ', step,' cost : ', cost_val, '\nW: ', np.array2string(W_val).replace('\n', ''), '\nb: ', b_val, '\n-------------')
            
    predictions = sess.run(hypothesis, feed_dict={X: x_test, Y: y_test_sc})
    print("MSE: ", metrics.mean_squared_error(predictions, y_test_sc))


# ### (b) Take a log10 for strength values. And rerun the regression to predict log10(strength). Print the last 10 values of {step, cost, W, b}

# In[9]:

y_train = np.log10(y_train)
y_train_sc = sc.fit_transform(y_train)

y_test = np.log10(y_test)
y_test_sc = sc.fit_transform(y_test)


# In[10]:

learning_rate = 10**(-4)
epochs = 390001

g4 = tf.Graph()
with g4.as_default():
    X = tf.placeholder(tf.float32, shape = [None, 4])
    Y = tf.placeholder(tf.float32, shape = [None, 1])
    
    W = tf.Variable(tf.random_normal([4, 1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    hypothesis = tf.add(tf.matmul(X, W), b)

    cost = tf.reduce_mean(tf.square(tf.subtract(hypothesis, Y)))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    train = optimizer.minimize(cost)

with tf.Session(graph=g4) as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(epochs):
        cost_val, W_val, b_val, _ =             sess.run([cost, W, b, train],
                     feed_dict={X: x_train, Y: y_train_sc})
        if step > (epochs-11):
        #if step % 10000 == 0:
            print('step: ', step,' cost : ', cost_val, '\nW: ', np.array2string(W_val).replace('\n', ''), '\nb: ', b_val, '\n-------------')
            
    predictions = sess.run(hypothesis, feed_dict={X: x_test, Y: y_test_sc})
    MSE_All = metrics.mean_squared_error(predictions, y_test_sc)
    print("MSE: ", MSE_All)


# ### (c) Choose the best features among length, number of digits, number of symbols and number of upper-class letter. And print the last 10 values of {step, cost, W, b}

#  We saw the result of length model at Q1(b). So, I will build new models with the information of digits, symbols and capitals and compare all the results to Q2(b). And then, I am going to figure out which feature is the most related to the password strength.
#  
# #### A model with the number of digits (digit model)

# In[11]:

# with digits.
x_train_dg = x_train[:,1]
x_train_dg = x_train_dg.reshape(x_train_dg.shape[0], 1)

x_test_dg = x_test[:,1]
x_test_dg = x_test_dg.reshape(x_test_dg.shape[0], 1)


# In[12]:

learning_rate = 10**(-4)
epochs = 140000

g5 = tf.Graph()
with g5.as_default():
    X = tf.placeholder(tf.float32, shape = [None, 1])
    Y = tf.placeholder(tf.float32, shape = [None, 1])

    W = tf.Variable(tf.random_normal([1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    hypothesis = tf.add(tf.multiply(X, W), b)

    cost = tf.reduce_mean(tf.square(tf.subtract(hypothesis, Y)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    train = optimizer.minimize(cost)
    
    accuracy = tf.reduce_mean(tf.square(tf.subtract(hypothesis, Y)))

with tf.Session(graph=g5) as sess:
    print("step | cost | W | b")
    sess.run(tf.global_variables_initializer())

    for step in range(epochs):
        cost_val, W_val, b_val, _ =             sess.run([cost, W, b, train],
                     feed_dict={X: x_train_dg, Y: y_train_sc})
        if step > (epochs-11):
        #if step % 10000 == 0:
            print(step, cost_val, W_val, b_val)
        
    MSE_Digit = sess.run(accuracy, feed_dict={X: x_test_dg, Y: y_test_sc})
    print("MSE: ", MSE_Digit)


# #### A model with the number of symbols (symbol model)

# In[13]:

# with symbols
x_train_sb = x_train[:,2]
x_train_sb = x_train_sb.reshape(x_train_sb.shape[0], 1)

x_test_sb = x_test[:,2]
x_test_sb = x_test_sb.reshape(x_test_sb.shape[0], 1)


# In[14]:

learning_rate = 10**(-4)
epochs = 120000

g6 = tf.Graph()
with g6.as_default():
    X = tf.placeholder(tf.float32, shape = [None, 1])
    Y = tf.placeholder(tf.float32, shape = [None, 1])

    W = tf.Variable(tf.random_normal([1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    hypothesis = tf.add(tf.multiply(X, W), b)

    cost = tf.reduce_mean(tf.square(tf.subtract(hypothesis, Y)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    train = optimizer.minimize(cost)
    
    accuracy = tf.reduce_mean(tf.square(tf.subtract(hypothesis, Y)))

with tf.Session(graph=g6) as sess:
    print("step | cost | W | b")
    sess.run(tf.global_variables_initializer())

    for step in range(epochs):
        cost_val, W_val, b_val, _ =             sess.run([cost, W, b, train],
                     feed_dict={X: x_train_sb, Y: y_train_sc})
        if step > (epochs-11):
        #if step % 10000 == 0:
            print(step, cost_val, W_val, b_val)
    MSE_Sym = sess.run(accuracy, feed_dict={X: x_test_sb, Y: y_test_sc})
    print("MSE: ", MSE_Sym)


# #### A model with the number of capitals (capital model)

# In[15]:

# with upper-class letters
x_train_up = x_train[:,3]
x_train_up = x_train_up.reshape(x_train_up.shape[0], 1)

x_test_up = x_test[:,3]
x_test_up = x_test_up.reshape(x_test_up.shape[0], 1)


# In[16]:

learning_rate = 11**(-4)
epochs = 180000

g7 = tf.Graph()
with g7.as_default():
    X = tf.placeholder(tf.float32, shape = [None, 1])
    Y = tf.placeholder(tf.float32, shape = [None, 1])

    W = tf.Variable(tf.random_normal([1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    hypothesis = tf.add(tf.multiply(X, W), b)

    cost = tf.reduce_mean(tf.square(tf.subtract(hypothesis, Y)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    train = optimizer.minimize(cost)
    
    accuracy = tf.reduce_mean(tf.square(tf.subtract(hypothesis, Y)))

with tf.Session(graph=g7) as sess:
    print("step | cost | W | b")
    sess.run(tf.global_variables_initializer())

    for step in range(epochs):
        cost_val, W_val, b_val, _ =             sess.run([cost, W, b, train],
                     feed_dict={X: x_train_up, Y: y_train_sc})
        if step > (epochs-11):
        #if step % 10000 == 0:
            print(step, cost_val, W_val, b_val)
    MSE_Cap = sess.run(accuracy, feed_dict={X: x_test_up, Y: y_test_sc})
    print("MSE: ", MSE_Cap)


# #### A model with password length and the number of symbols

# In[17]:

x_train_LS = np.array([[row[0], row[2]] for row in x_train])
x_test_LS = np.array([[row[0], row[2]] for row in x_test])


# In[18]:

learning_rate = 10**(-4)
epochs = 280000

g8 = tf.Graph()
with g8.as_default():
    X = tf.placeholder(tf.float32, shape = [None, 2])
    Y = tf.placeholder(tf.float32, shape = [None, 1])

    W = tf.Variable(tf.random_normal([2, 1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    hypothesis = tf.add(tf.matmul(X, W), b)

    cost = tf.reduce_mean(tf.square(tf.subtract(hypothesis, Y)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    train = optimizer.minimize(cost)
    
    accuracy = tf.reduce_mean(tf.square(tf.subtract(hypothesis, Y)))

with tf.Session(graph=g8) as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(epochs):
        cost_val, W_val, b_val, _ =             sess.run([cost, W, b, train],
                     feed_dict={X: x_train_LS, Y: y_train_sc})
        if step > (epochs-11):
        #if step % 10000 == 0:
            print('step: ', step,' cost : ', cost_val, ' W: ', np.array2string(W_val).replace('\n', ''), ' b: ', b_val)
    MSE_LenSym = sess.run(accuracy, feed_dict={X: x_test_LS, Y: y_test_sc})
    print("MSE: ", MSE_LenSym)


# #### A model with password length and the number of digits

# In[19]:

x_train_LD = np.array([[row[0], row[1]] for row in x_train])
x_test_LD = np.array([[row[0], row[1]] for row in x_test])


# In[20]:

learning_rate = 10**(-3)
epochs = 40001

g9 = tf.Graph()
with g9.as_default():
    X = tf.placeholder(tf.float32, shape = [None, 2])
    Y = tf.placeholder(tf.float32, shape = [None, 1])

    W = tf.Variable(tf.random_normal([2, 1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    hypothesis = tf.add(tf.matmul(X, W), b)

    cost = tf.reduce_mean(tf.square(tf.subtract(hypothesis, Y)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    train = optimizer.minimize(cost)
    
    accuracy = tf.reduce_mean(tf.square(tf.subtract(hypothesis, Y)))

with tf.Session(graph=g9) as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(epochs):
        cost_val, W_val, b_val, _ =             sess.run([cost, W, b, train],
                     feed_dict={X: x_train_LD, Y: y_train_sc})
        if step > (epochs-11):
        #if step % 10000 == 0:
            print('step: ', step,' cost : ', cost_val, ' W: ', np.array2string(W_val).replace('\n', ''), ' b: ', b_val)
    MSE_LenDig = sess.run(accuracy, feed_dict={X: x_test_LD, Y: y_test_sc})
    print("MSE: ", MSE_LenDig)


# #### A model with password length and the number of caplitals

# In[21]:

x_train_LC = np.array([[row[0], row[3]] for row in x_train])
x_test_LC = np.array([[row[0], row[3]] for row in x_test])


# In[22]:

learning_rate = 10**(-3)
epochs = 50001

g10 = tf.Graph()
with g10.as_default():
    X = tf.placeholder(tf.float32, shape = [None, 2])
    Y = tf.placeholder(tf.float32, shape = [None, 1])

    W = tf.Variable(tf.random_normal([2, 1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    hypothesis = tf.add(tf.matmul(X, W), b)

    cost = tf.reduce_mean(tf.square(tf.subtract(hypothesis, Y)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    train = optimizer.minimize(cost)
    
    accuracy = tf.reduce_mean(tf.square(tf.subtract(hypothesis, Y)))

with tf.Session(graph=g10) as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(epochs):
        cost_val, W_val, b_val, _ =             sess.run([cost, W, b, train],
                     feed_dict={X: x_train_LC, Y: y_train_sc})
        if step > (epochs-11):
        #if step % 10000 == 0:
            print('step: ', step,' cost : ', cost_val, ' W: ', np.array2string(W_val).replace('\n', ''), ' b: ', b_val)
    MSE_LenCap = sess.run(accuracy, feed_dict={X: x_test_LC, Y: y_test_sc})
    print("MSE: ", MSE_LenCap)


# #### A model with digits and symbols

# In[23]:

x_train_DS = np.array([[row[1], row[2]] for row in x_train])
x_test_DS = np.array([[row[1], row[2]] for row in x_test])


# In[24]:

learning_rate = 10**(-3)
epochs = 30001

g11 = tf.Graph()
with g11.as_default():
    X = tf.placeholder(tf.float32, shape = [None, 2])
    Y = tf.placeholder(tf.float32, shape = [None, 1])

    W = tf.Variable(tf.random_normal([2, 1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    hypothesis = tf.add(tf.matmul(X, W), b)

    cost = tf.reduce_mean(tf.square(tf.subtract(hypothesis, Y)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    train = optimizer.minimize(cost)
    
    accuracy = tf.reduce_mean(tf.square(tf.subtract(hypothesis, Y)))

with tf.Session(graph=g11) as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(epochs):
        cost_val, W_val, b_val, _ =             sess.run([cost, W, b, train],
                     feed_dict={X: x_train_DS, Y: y_train_sc})
        if step > (epochs-11):
        #if step % 10000 == 0:
            print('step: ', step,' cost : ', cost_val, ' W: ', np.array2string(W_val).replace('\n', ''), ' b: ', b_val)
    MSE_DigSym = sess.run(accuracy, feed_dict={X: x_test_DS, Y: y_test_sc})
    print("MSE: ", MSE_DigSym)


# #### A model with symbols and capitals

# In[25]:

x_train_SC = np.array([[row[2], row[3]] for row in x_train])
x_test_SC = np.array([[row[2], row[3]] for row in x_test])


# In[26]:

learning_rate = 10**(-3)
epochs = 20001

g12 = tf.Graph()
with g12.as_default():
    X = tf.placeholder(tf.float32, shape = [None, 2])
    Y = tf.placeholder(tf.float32, shape = [None, 1])

    W = tf.Variable(tf.random_normal([2, 1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    hypothesis = tf.add(tf.matmul(X, W), b)

    cost = tf.reduce_mean(tf.square(tf.subtract(hypothesis, Y)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    train = optimizer.minimize(cost)
    
    accuracy = tf.reduce_mean(tf.square(tf.subtract(hypothesis, Y)))

with tf.Session(graph=g12) as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(epochs):
        cost_val, W_val, b_val, _ =             sess.run([cost, W, b, train],
                     feed_dict={X: x_train_SC, Y: y_train_sc})
        if step > (epochs-11):
        #if step % 10000 == 0:
            print('step: ', step,' cost : ', cost_val, ' W: ', np.array2string(W_val).replace('\n', ''), ' b: ', b_val)
    MSE_SymCap = sess.run(accuracy, feed_dict={X: x_test_SC, Y: y_test_sc})
    print("MSE: ", MSE_SymCap)


# #### A model without password length.

# In[27]:

x_train_DSC = np.array([[row[1], row[2], row[3]] for row in x_train])
x_test_DSC = np.array([[row[1], row[2], row[3]] for row in x_test])


# In[28]:

learning_rate = 10**(-3)
epochs = 20001

g13 = tf.Graph()
with g13.as_default():
    X = tf.placeholder(tf.float32, shape = [None, 3])
    Y = tf.placeholder(tf.float32, shape = [None, 1])

    W = tf.Variable(tf.random_normal([3, 1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    hypothesis = tf.add(tf.matmul(X, W), b)

    cost = tf.reduce_mean(tf.square(tf.subtract(hypothesis, Y)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    train = optimizer.minimize(cost)

    accuracy = tf.reduce_mean(tf.square(tf.subtract(hypothesis, Y)))

with tf.Session(graph=g13) as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(epochs):
        cost_val, W_val, b_val, _ =             sess.run([cost, W, b, train],
                     feed_dict={X: x_train_DSC, Y: y_train_sc})
        if step > (epochs-11):
        #if step % 1000 == 0:
            print('step: ', step,' cost : ', cost_val, ' W: ', np.array2string(W_val).replace('\n', ''), ' b: ', b_val)
    MSE_DigSymCap = sess.run(accuracy, feed_dict={X: x_test_DSC, Y: y_test_sc})
    print("MSE: ", MSE_DigSymCap)


# In[29]:

print('        < Mean Square Error >\n')
print('Four features : ', "%22.10f" %(MSE_All))
print('Digit, Symbol & Capital : ', "%0.10f" %(MSE_DigSymCap))
print('Length : ', "%29.10f" %(MSE_Len))
print('Length & Capital : ', "%19.10f" %(MSE_LenCap))
print('Length & Digit : ', "%21.10f" %(MSE_LenDig))
print('Length & Symbol : ', "%20.10f" %(MSE_LenSym))
print('Capital : ', "%28.10f" %(MSE_Cap))
print('Digit : ', "%30.10f" %(MSE_Digit))
print('Digit & Symbol : ', "%21.10f" %(MSE_DigSym))
print('Symbol : ', "%29.10f" %(MSE_Sym))
print('Symbol & Capital : ', "%19.10f" %(MSE_SymCap))


# The model without only password length as a feature has lower MSE than four features model.  
# MSE of the models with password length is close to four features model.  
# Therefore, the information of password length is the stongest feature.

# In[ ]:



