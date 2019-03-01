# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 20:50:30 2019

@author: xinglin
"""
import numpy as np 
import matplotlib.pyplot as plt 

class LinearRegression:
    def __init__(self):
        self.weight = None
    def creatData(self,pointNumber = 100):
        #生成模拟数据
        train_X = np.linspace(-1, 1,pointNumber)
        train_Y = -2 * train_X + 10 + np.random.randn(*train_X.shape) * 0.5 # y=2x，但是加入了噪声
        #显示模拟数据点
        # plt.plot(train_X, train_Y, 'ro', label='Original data')
        # plt.legend()
        # plt.show()
        resX = train_X.reshape(pointNumber,1)
        b = np.ones((pointNumber,1))
        # 在特征前曾加列，值全为1，表示常数项
        trainDataSet = np.concatenate([b,resX],axis = 1)
        return trainDataSet,train_Y,train_X

    # 基本方法，xTx矩阵不可逆时无法求出结果 w = (XTX)^-1XTy
    def standRegres(self,xArr,yArr):
        xMat = np.mat(xArr)
        yMat = np.mat(yArr)
        xTx = xMat.T*xMat
        if np.linalg.det(xTx) == 0:
            print('This matrix is singular, cannot do inverse')
            return
        ws=np.linalg.solve(xTx,xMat.T*yMat.T)
        self.weight = ws
        return ws
        
    #w = (XTX + lamI)^-1XTy 增加参数，使(XTX + lamI)始终可逆，岭回归，当lam为0时和standRegres方法一样
    def linear_regression(self,x_arr, y_arr, lam=0.2):
        x_mat = np.mat(x_arr)
        y_mat = np.mat(y_arr)
    
        x_tx = x_mat.T * x_mat
        denom = x_tx + np.eye(np.shape(x_mat)[1]) * lam
    
        # if lam == 0.0
        if np.linalg.det(denom) == 0.0:
            print('This matrix is singular, cannot do inverse')
            return
        ws = np.linalg.solve(denom,x_mat.T*y_mat.T)
        self.weight = ws
        return ws
    # 接受原始数据点，绘制回归线
    def showResult(self,train_x,train_y):
        plt.scatter(train_x,train_y)
        x = np.arange(-1, 2)
        y =  self.weight[1,0]  * x +  self.weight[0,0]
        plt.plot(x,y,'r')
        plt.show()
    
    # 预测
    def predict(self,test_x):
        y =  self.weight[1,0]  * test_x +  self.weight[0,0]
        return y


if __name__=='__main__':
    Model = LinearRegression()
    train_X,train_Y,point_x = Model.creatData()
    #方法一
    Model.standRegres(train_X,train_Y)
    Model.showResult(point_x,train_Y)
    #方法二
    Model.linear_regression(train_X,train_Y)
    # 预测
    print(Model.predict(0.5))

    Model.showResult(point_x,train_Y)

