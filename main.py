import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import matplotlib.pyplot as plt
import numpy as np

n_input=784
batch_size=100
examples_to_show=10

mnist=input_data.read_data_sets("/Users/mac/Documents/dataSet")
x_train,y_train=mnist.train.next_batch(100)

X=tf.placeholder(tf.float32,shape=[None,784])
weight={
    "encoder_h1":tf.Variable(tf.random_normal(shape=[n_input,256])),
    "encoder_h2":tf.Variable(tf.random_normal(shape=[256,128])),
    "decoder_h1":tf.Variable(tf.random_normal(shape=[128,256])),
    "decoder_h2":tf.Variable(tf.random_normal(shape=[256,n_input]))
}
biases={
    "encoder_b1":tf.Variable(tf.random_normal(shape=[256])),
    "encoder_b2":tf.Variable(tf.random_normal(shape=[128])),
    "decoder_b1":tf.Variable(tf.random_normal(shape=[256])),
    "decoder_b2":tf.Variable(tf.random_normal(shape=[n_input]))
}

def encoder(x):
    layer_1=tf.nn.sigmoid(tf.add(tf.matmul(x,weight["encoder_h1"]),biases["encoder_b1"]))
    layer_2=tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weight["encoder_h2"]),biases["encoder_b2"]))
    return layer_2

def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weight["decoder_h1"]) , biases["decoder_b1"]))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weight["decoder_h2"]), biases["decoder_b2"]))
    return layer_2

y_pred_=encoder(X)
y_pred=decoder(y_pred_)
loss=tf.reduce_mean(tf.pow(y_pred-X,2))
optimizer=tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        for j in range(mnist.train.num_examples//batch_size):
            x_train,y_train=mnist.train.next_batch(batch_size)
            _,res=sess.run([optimizer,loss],feed_dict={X:x_train})
            print(res)

    print("Optimization Finished!")

    encode_decode=sess.run(y_pred,feed_dict={X:mnist.test.images[:examples_to_show]})
    f,a=plt.subplots(2,10,figsize=(12,8))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i],[28,28]))
        a[1][i].imshow(np.reshape(encode_decode[i],[28,28]))
    plt.show()

    


