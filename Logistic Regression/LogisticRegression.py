# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 15:10:27 2019

@author: xinglin
"""

import matplotlib.pyplot as plt
import numpy as np

class MyLogisticRegression:
    def __init__(self):
        pass

    def  sigmoid(self,x):
        return 1.0 / (1.0 + np.exp(-x))
    # 展示sigmoid
    def show_sigmoid(self):
        x = np.linspace(-10, 10)
        y = self.sigmoid(x)
        plt.plot(x,y)
        plt.show()
    # 加载数据
    def loadDataSet(self):
        '''
        return---np array
        trainDataSet:带常数项的数据
        dataSet:原始数据
        dataLabel：标签，类别列表
        
        '''
        data = np.loadtxt('Logistic Regression/data1.txt', delimiter=',')
        dataSet = data[:,0:2]
        #为了便于进行矩阵运算，每个样本增加一列 1 ，表示常数项
        b = np.ones((dataSet.shape[0],1))
        trainDataSet = np.concatenate([dataSet,b],axis = 1)  
        dataLabel = data[:,2]
        return trainDataSet,dataSet,dataLabel
    def showData(self):
        dataMat,data,labelMat = self.loadDataSet()  # 加载数据集
        pos = np.where(labelMat == 1)
        neg = np.where(labelMat == 0)

        plt.scatter(dataMat[pos, 0], dataMat[pos, 1], marker='o', c='b')
        plt.scatter(dataMat[neg, 0], dataMat[neg, 1], marker='x', c='r')
        plt.show()
    # 训练权重weights
    def gradAscent(self):
        dataMatIn,orgData,classLabels = self.loadDataSet()
        dataMatrix = np.mat(dataMatIn)                            #转换成numpy的mat
        labelMat = np.mat(classLabels).transpose()                #转换成numpy的mat,并进行转置
        m, n = np.shape(dataMatrix)                               #返回dataMatrix的大小。m为行数,n为列数。
        alpha = 0.001                                              #学习速率,控制更新的幅度。
        maxCycles = 500000                                       #迭代次数
        weights = np.ones((n,1))
        wList = []
        for k in iter(range(maxCycles)):
            h = self.sigmoid(dataMatrix * weights)                #梯度上升矢量化公式
            error = labelMat - h
            weights = weights + alpha * dataMatrix.transpose() * error
        return weights.getA()                                     #将矩阵转换为数组，返回权重数组
    # 根据权重绘制逻辑回归线
    def plotBestFit(self,weights):
        dataMat,data,labelMat = self.loadDataSet()  # 加载数据集

        pos = np.where(labelMat == 1)
        neg = np.where(labelMat == 0)

        plt.scatter(dataMat[pos, 0], dataMat[pos, 1], marker='o', c='b')
        plt.scatter(dataMat[neg, 0], dataMat[neg, 1], marker='x', c='r')
        x = np.arange(0, 100, 0.1)
        y = (-weights[0] * x - weights[2]) / weights[1]
        plt.plot(x,y)
        plt.show()


if __name__ == '__main__':
    Model = MyLogisticRegression()
    # Model.show_sigmoid()
    weights = Model.gradAscent()
    print(weights)
    Model.showData()
    Model.plotBestFit(weights)