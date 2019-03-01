# -*- coding: utf-8 -*-


'''
    训练集：[mnist.train.images，mnist.train.labels]
    测试集：[mnist.test.images，mnist.test.labels]
    验证集：[mnist.validation.images，mnist.validation.labels]
'''

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

print('输入数据:',mnist.train.images.shape)
print("标签:",mnist.train.labels.shape)
print('输入数据:',mnist.train.labels.shape)




