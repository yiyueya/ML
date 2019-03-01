# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 17:15:43 2019

@author: xinglin
"""
from math import log

class Trees:
    def __init__(self):
        pass
    # 计算信息熵
    def calcShannonEnt(self,dataSet):
        numEntries = len(dataSet)
        labelCounts = {}
        for featVec in dataSet:
            currentLabel = featVec[-1]
            if currentLabel not in labelCounts.keys():
                labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1
        shannonEnt = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key])/numEntries
            shannonEnt -= prob*log(prob,2)
        return shannonEnt
    # 按照特征划分数据集
    def splitDataSet(self,dataSet,axis,value):
        retDataSet = []
        for i in dataSet:
            if i[axis] == value:
                ret = i[:axis]
                ret += i[axis+1:]
                retDataSet.append(ret)
        return retDataSet
    # 选择最佳划分依据，依据：信息熵减少最多的特征
    def chooseBestFeatureToSplit(self,dataSet):
        numFeatures = len(dataSet[0]) - 1
        baseEntropy = self.calcShannonEnt(dataSet)
        bestInfoGain = 0.0
        bestFeature = -1
        for i in range(numFeatures):
            #列出特征i的所有可能取值 val
            featValList = [example[i] for example in dataSet] 
            featValSet = set(featValList)
            newEntropy = 0.0    
            for value in featValSet:    #求出每种val的信息熵的和
                splitData = self.splitDataSet(dataSet,i,value)
                prob = len(splitData)/float(len(dataSet))
                newEntropy += prob*self.calcShannonEnt(splitData)
            infoGain = baseEntropy - newEntropy
            if infoGain >= bestInfoGain:
                bestInfoGain = infoGain
                bestFeature = i
        return bestFeature
    # 返回出现次数最多的类别，原函数采用了operator模块的一些方法，本人对该模块不怎么了解
    #这里稍作改动，直接遍历列表求出现次数最多的
    def majorityCnt(self,classList):
        temp = 0
        for i in set(classList):
            if classList.count(i) > temp:
                maxClassList = i
                temp = classList.count(i)
        return maxClassList
    # 资料的原函数
    # {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
    def createTree(self,dataSet,labels):
        classList = [i[-1] for i in dataSet]
        if classList.count(classList[0]) == len(classList):
            return classList[0]
        if len(dataSet[0]) == 1:
            return self.majorityCnt(classList)
        bestFeat = self.chooseBestFeatureToSplit(dataSet)
        bestFeatLabel = labels[bestFeat]
        myTree = {bestFeatLabel:{}}
        del(labels[bestFeat])
        featValues = [i[bestFeat] for i in dataSet]
        uniqueVals = set(featValues)
        for value in uniqueVals:
            subLabels = labels[:]
            myTree[bestFeatLabel][value] = \
                self.createTree(self.splitDataSet(dataSet,bestFeat,value),subLabels)
        return myTree
    # 每一分支节点改为键值feat记录当前节点划分依据，featVal记录子节点，以feat为依据遍历featVal可以直接得到结果
    # {'feat': 0, 'featVal': {0: 'no', 1: {'feat': 0, 'featVal': {0: 'no', 1: 'yes'}}}}
    def myCreateTree(self,data):    #输入数据最后一列为标签
        label = [i[-1] for i in data]
        if label.count(label[0]) == len(label):
            return label[0]
        if len(data[0]) == 1:
            return self.majorityCnt(label)
        bestFeat = self.chooseBestFeatureToSplit(data)
        myTree = {'feat':bestFeat,'featVal':{}}
        featValues = [i[bestFeat] for i in data]
        uniqueVals = set(featValues)
        for val in uniqueVals:
            myTree['featVal'][val] = self.myCreateTree(self.splitDataSet(data,bestFeat,val))
        return myTree
    # 训练,返回训练好的模型
    def fit(self,x_train,y_train):
        '''
        x_train数据格式：每一列表示一个属性，每一行表示一个样本
        y_train数据格式：一维数组，表示标签，与X_train相对应
        '''
        lence = len(y_train)
        for i in range(lence):
            x_train[i].append(y_train[i])
        return self.myCreateTree(x_train)

    # 预测predict，
    def predict(self,inputTree,test_data):
        if inputTree is None:
            return 'Decision tree not created !please run fit()'
        index = inputTree['feat']
        valueKey = test_data[index]
        nextTree = inputTree['featVal'][valueKey]
        if type(nextTree) is dict:
            del(test_data[index])
            return self.predict(nextTree,test_data)
        else:
            return nextTree

    # 示例数据
    def creatDataSet(self):
        dataSet = [
            [1,1,'yes'],
            [1,1,'yes'],
            [1,0,'no'],
            [0,1,'no'],
            [0,1,'no'],
        ]
        labels = ['no surfacing','flippers']
        return dataSet,labels

if __name__=='__main__':
    Model = Trees()
    myData,labels = Model.creatDataSet()
    print('myData的信息熵：',Model.calcShannonEnt(myData))
    # print(Model.myCreateTree(myData))
    data_x = [
        [1,1],
        [1,1],
        [1,0],
        [0,1],
        [0,1]
    ]
    data_y = ['yes','yes','no','no','no']
    itree = Model.fit(data_x,data_y)
    print('决策树结构：',itree)
    print('[1,0]预测结果：',Model.predict(itree,[1,0]))

