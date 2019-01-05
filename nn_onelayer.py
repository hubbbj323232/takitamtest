import tensorflow as tf
import numpy as np

def nn_onelayer(num_iters, display_step): 

    data = np.load('data1.npz')
    nn_input = data['nn_input']
    nn_target = data['nn_target']
    frame_len = data['frame_len']

    X     = tf.placeholder(tf.float32, [None, frame_len, 1025])
    Y_hat = tf.placeholder(tf.float32, [None, frame_len, 1025])

    W = tf.Variable(np.random.randn(frame_len*1025, frame_len*1025).astype(np.float32))
    b = tf.Variable(np.random.randn(frame_len*1025).astype(np.float32))

    XX     = tf.reshape(X,     [-1, frame_len*1025])
    YY_hat = tf.reshape(Y_hat, [-1, frame_len*1025])

    Y = tf.nn.sigmoid(tf.matmul(XX,W)+b)

    cost = tf.reduce_mean((YY_hat - Y)**2)
    update = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)
    
        for i in range (num_iters+1):

            if i%display_step == 0:
                sess.run(update, {X : nn_input, Y_hat : nn_target})
                cost_val = sess.run(cost, {X : nn_input, Y_hat : nn_target})
                print("#{}, cost={}".format(i,cost_val))
    
            if i%num_iters == 0:
                y_out = sess.run(Y, {X : nn_input, Y_hat : nn_target})

            sess.run(update, {X : nn_input, Y_hat : nn_target})

    np.savez('data2.npz', y_out = y_out)
            
