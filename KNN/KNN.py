# -*- coding: utf-8 -*-
import numpy as np 
from sklearn.model_selection import train_test_split
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

class KNN():  
    def fit(self,x_train,y_train):
        '''
        x_train数据格式：每一列表示一个属性，每一行表示一个样本
        y_train数据格式：一维数组，表示标签，与X_train相对应
        '''
        self.x_train = x_train  
        self.y_train = y_train

    def predict(self,x_test,k = 1):
        self.k = k
        #计算欧式距离
        distance = (np.sum((self.x_train - x_test) ** 2,1)) ** 0.5
        sortindex = np.argsort(distance)
        sortindex_k = sortindex[:self.k]
        lable_k = self.y_train[sortindex_k]
        labelCount = {}
        for i in lable_k:
            if i in labelCount:
                labelCount[i] += 1
            else:
                labelCount[i] = 1
        result = sorted(labelCount.items(), key=lambda k:k[1], reverse=True)
        return result[0][0]
    def accuracy(self,x,y,k = 1):
        y_predict = []
        size = len(y)
        i = 0
        while i < size:
            y_predict.append(self.predict(x[i],k))
            i += 1
        j = 0
        count = 0
        while j < size:
            if y_predict[j] == y[j]:
                count += 1
            j += 1
        return count/size


if __name__ == '__main__':

    x_input = mnist.train.images
    y_input = mnist.train.labels
    x_test = mnist.test.images
    y_test = mnist.test.labels

    knn = KNN()
    #由于训练样本较多，可以考虑选择部分样本作为输入
    # 样本划分，选择部分作为参考点
    x_train,x,y_train,y = train_test_split(x_input,y_input,test_size=0.8)
    knn.fit(x_train,y_train)
    print(x_train.shape)
    print(x_test.shape)

    y_predict = []
    #选择了测试集前10个样本做测试
    for i in range(50): 
        y_predict.append(knn.predict(x_test[i],2))

    print('预测值：',y_predict)
    print('实际结果：',y_test[:50])
    # 选取部分数据求准确率，
    print('准确率：',knn.accuracy(x_test[:100],y_test[:100]))



        




        