# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 09:44:55 2019

@author: xinglin
"""
class NaiveBayes:
    def __init__(self):
        pass

    def loadDataSet(self):
        postingList = [['my','dog','has','flea','problems','help','please'],
                    ['maybe','not','take','him','to','dog','park','stupid'],
                    ['my','dalmation','is','so','cute','I','love','him'],
                    ['stop','posting','stupid','worthless','garbage'],
                    ['mr','licks','ate','my','steak','how','to','stop','him'],
                    ['quit','buying','worthless','dog','food','stupid']
        ]
        classVec = [0,1,0,1,0,1]    #1代表侮辱性言语，0代表正常言论
        return postingList,classVec
    # 统计词条,
    def creatVocabList(self,dataSet):
        vocabList = []
        for document in dataSet:
            vocabList += document
        return set(vocabList)
    
    # 