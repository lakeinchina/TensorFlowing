# -*- coding: UTF-8 -*-
"""Trains and Evaluates the MNIST network using a feed dictionary.

TensorFlow install instructions:
https://tensorflow.org/get_started/os_setup.html

MNIST tutorial:
https://tensorflow.org/tutorials/mnist/tf/index.html

"""

import tensorflow as tf
import input_data




def run_training():
    #读数据
    mnist = input_data.read_data_sets('data', one_hot=True)
    #占位符
    x = tf.placeholder("float", shape=[None, 784])
    y_ = tf.placeholder("float", shape=[None, 10])
    #神经网络的W和b
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))

    #模型定义,784->10->10
    y = tf.nn.softmax(tf.matmul(x,W)+b)

    #损失函数,10个真实和预测值乘积的和
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))

    #训练方法
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())

    for i in range(1000):
        #加载50个数据
        batch = mnist.train.next_batch(50)
        #进行一次训练
        train_step.run(feed_dict={x:batch[0],y_:batch[1]})

    #评估方法
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    #求平均数即准确的比例
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
    print accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels})


def main(_):
  run_training()


if __name__ == '__main__':
  tf.app.run()
