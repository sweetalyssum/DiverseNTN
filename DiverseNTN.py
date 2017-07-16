"""
Created on 2015-7-21

class: DiverseNTN

@author: XiaLong
@contact: xl.1988.life@gmail.com
"""

# !/usr/bin/python
# -*- coding:utf-8 -*-

import yaml
import sys
import string
from numpy import *
import Util


class DiverseNTN(object):
    """docstring for DiverseNTN"""
    def __init__(self, trainFile, testFile, idealFile, confFile):
        super(DiverseNTN, self).__init__()
        self.trainFile = trainFile
        self.testFile = testFile
        self.idealFile = idealFile
        self.confFile = confFile
        self.selectedSet = []
        self.selectedMatrix = array([])

    def __del__(self):
        self.confFile.close()

    def InitConfFile(self):
        self.confFile = open(self.confFile)
        self.dictConf = yaml.load(self.confFile)

    def InitParameter(self):
        self.w_r = random.random((6))
        self.l_d = self.dictConf['tensor']
        self.w_u = random.random((self.l_d))
        self.w_d = random.random((6,6,self.l_d))
        self.learningRate = self.dictConf['learning_rate']
        self.convergence = self.dictConf['convergence']
        self.trainList = [{} for i in xrange(self.dictConf['query'])]
        self.testList = [{} for i in xrange(self.dictConf['query'])]
        self.idealRanking = [[] for i in xrange(self.dictConf['query'])]
        self.sumLoss = 0.0

    def InitSGD(self):
        self.w_r_change = zeros((6))
        self.w_u_change = zeros((self.l_d))
        self.w_d_change = zeros((6,6,self.l_d))

    def InitSelected(self):
        self.selectedSet = []
        self.selectedMatrix = array([])

    def MiniSelect(self, inputArray):
        return inputArray.min()

    def AverSelect(self, inputMatrix):
        return inputMatrix.sum()/inputMatrix.size

    def MaxiSelect(self, inputMatrix):
        return inputMatrix.max()

    def RankingFunction(self, i, doc):
        relevanceScore = dot(self.w_r, self.trainList[i][doc])
        self.selectedSet.append(doc)
        if len(self.selectedSet) == 1:
            self.selectedMatrix = append(self.selectedMatrix, self.trainList[i][doc])
            return relevanceScore
        elif len(self.selectedSet) == 2:
            self.selectedMatrix = array([self.selectedMatrix, self.trainList[i][doc]])
        else:
            self.selectedMatrix = append(self.selectedMatrix, [self.trainList[i][doc]], axis=0)
        tensorScore = dot(self.w_u, self.CalculateTensor(i, self.trainList[i][doc]))
        return relevanceScore+tensorScore

    def RankingScore(self, i, doc):
        relevanceScore = dot(self.w_r, self.testList[i][doc])
        if len(self.selectedSet) == 0:
            return relevanceScore
        tensorScore = dot(self.w_u, self.CalculateTensor(i, self.testList[i][doc]))
        return relevanceScore + tensorScore
        
    def RelevanceFeature(self, inputFile, inputList):
        featureFile = open(inputFile)
        for line in featureFile.readlines():
            listItem = line.split(" ")
            queryID = Util.strLatter(listItem[1], ":")
            queryID = string.atoi(queryID)
            docID = Util.strLatter(listItem[-1], "=")
            docID = docID.strip()
            docID = "clueweb09-en" + docID[0:4] + "-" + docID[4:6] + "-" + docID[6:]
            listFeature = []
            for i in range(2, 8):
                strFeature = Util.strLatter(listItem[i], ":")
                floatFeature = string.atof(strFeature)
                listFeature.append(floatFeature)
            inputList[queryID-1][docID] = listFeature
        for i in xrange(self.dictConf['query']):
            if len(inputList[i]) == 0:
                continue
            sum = 0.0
            for j in xrange(6):
                for k in inputList[i].keys():
                    sum += inputList[i][k][j]
                for k in inputList[i].keys():
                    inputList[i][k][j] = inputList[i][k][j]/sum
        featureFile.close()

    def IdealRanking(self):
        self.idealFile = open(self.idealFile)
        for line in self.idealFile.readlines():
            if line.find('clueweb') != -1:
                listItem = line.split("\t")
                self.idealRanking[int(listItem[0])-1].append(listItem[1])
        self.idealFile.close()

    def InitDoc(self, inputList):
        for i in xrange(self.dictConf['query']):
            if len(inputList[i]) != 0:
                self.idealRanking[i] = [doc for doc in self.idealRanking[i] if doc in inputList[i]]

    def CalculateTensor(self, i, doc):
        tensorArray = array([], dtype=float)
        for j in xrange(self.l_d):
            transposeMatrix = self.selectedMatrix.transpose()
            frontScore = dot(doc, self.w_d[...,j])
            behindScore = dot(frontScore, transposeMatrix)
            tensorScore = self.MiniSelect(behindScore)
            tensorArray = append(tensorArray, ((math.exp(tensorScore)-math.exp(-tensorScore))/(math.exp(tensorScore)+math.exp(-tensorScore))))
        return tensorArray

    def CalculateLoss(self, i):
        queryLoss = 0.0
        sumScore = 0.0
        for doc in self.idealRanking[i]:
            finalScore = math.exp(self.RankingFunction(i, doc))
            sumScore += finalScore
            queryLoss -= math.log(finalScore/sumScore)
        self.InitSelected()
        return queryLoss

    def SGD(self, i):
        sumScore = 0.0
        changeScore = 0.0
        w_r_change = zeros((6))
        w_u_change = zeros((self.l_d))
        w_d_change = zeros((6,6,self.l_d))
        for doc in self.idealRanking[i]:
            docScore = math.exp(self.RankingFunction(i, doc))
            sumScore += docScore
            w_r_change = self.SGD_w_r(i, doc, docScore, sumScore, w_r_change)
            w_u_change = self.SGD_w_u(i, doc, docScore, sumScore, w_u_change)
            w_d_change = self.SGD_w_d(i, doc, docScore, sumScore, w_d_change)
        self.w_r -= self.learningRate * self.w_r_change
        self.w_u -= self.learningRate * self.w_u_change
        self.w_d -= self.learningRate * self.w_d_change
        self.InitSGD()
        self.InitSelected()

    def SGD_w_r(self, i, doc, docScore, sumScore, change):
        deltaScore = array(self.trainList[i][doc])
        change += docScore * deltaScore
        self.w_r_change += change/sumScore - deltaScore
        return change

    def SGD_w_u(self, i, doc, docScore, sumScore, change):
        deltaScore = self.CalculateTensor(i, self.trainList[i][doc])
        change += docScore * deltaScore
        self.w_u_change += change/sumScore - deltaScore
        return change

    def SGD_w_d(self, i, doc, docScore, sumScore, change):
        deltaScore = zeros((6,6,self.l_d))
        for l in xrange(6):
            for m in xrange(6):
                for n in xrange(self.l_d):
                    transposeMatrix = self.selectedMatrix.transpose()
                    frontScore = dot(self.trainList[i][doc], self.w_d[...,n])
                    behindScore = dot(frontScore, transposeMatrix)
                    tensorScore = self.MiniSelect(behindScore)
                    e_S_change = array([], dtype=float)
                    if self.selectedMatrix.size == 0:
                        deltaScore[l][m][n] = 0.0
                        continue
                    if self.selectedMatrix.size == 6:
                        e_S_change = append(e_S_change, self.trainList[i][doc][l] * self.selectedMatrix[m])
                        deltaScore[l][m][n] = self.w_u[n] * 4 / (math.exp(tensorScore) + math.exp(-tensorScore)) *e_S_change[0]
                        continue
                    for j in xrange(self.selectedMatrix[...,0].size):
                        e_S_change = append(e_S_change, self.trainList[i][doc][l] * self.selectedMatrix[j][m])
                    deltaScore[l][m][n] = self.w_u[n] * 4 / (math.exp(tensorScore) + math.exp(-tensorScore)) *self.MiniSelect(e_S_change)
        change += docScore * deltaScore
        self.w_d_change += change/sumScore - deltaScore
        return change

    def TrainNTN(self):
        self.RelevanceFeature(self.trainFile, self.trainList)
        self.IdealRanking()
        self.InitDoc(self.trainList)
        lossFile = open("loss.txt", "w")
        n=1
        while True:
            for i in xrange(self.dictConf['query']):
                if len(self.trainList[i]) != 0:
                    self.SGD(i)
            self.sumLoss = 0.0
            for i in xrange(self.dictConf['query']):
                if len(self.trainList[i]) != 0:
                    self.sumLoss += self.CalculateLoss(i)
            print 'Sum Loss:' + str(self.sumLoss)
            lossFile.write(str(self.sumLoss))
            lossFile.write("\n")
            lossFile.flush()
            self.TestNTN(n)
            n+=1

    def TestNTN(self, n):
        self.RelevanceFeature(self.testFile, self.testList)
        resultFile = open("result\\result"+str(n)+".txt", "w")
        for i in xrange(self.dictConf['query']):
            if len(self.testList[i]) != 0:
                bestResult = 101
                rank = 1
                while len(self.testList[i].keys()) != 0:
                    bestScore = -10000.0 
                    bestDoc = ""
                    for key in self.testList[i].keys():
                        rankingScore = self.RankingScore(i, key)
                        if rankingScore > bestScore:
                            bestScore = rankingScore
                            bestDoc = key
                    self.selectedSet.append(bestDoc)
                    if len(self.selectedSet) == 1:
                        self.selectedMatrix = append(self.selectedMatrix, self.testList[i][bestDoc])
                    elif len(self.selectedSet) == 2:
                        self.selectedMatrix = array([self.selectedMatrix, self.testList[i][bestDoc]])
                    else:
                        self.selectedMatrix = append(self.selectedMatrix, [self.testList[i][bestDoc]], axis=0)
                    resultFile.write(str(i+1) + " Q0 " + str(bestDoc) + " " + str(rank) + " " + str(bestResult) + " xialong" + "\n")
                    self.testList[i].pop(bestDoc)
                    bestResult -= 1
                    rank += 1
                self.InitSelected()
        pass

    def Main(self):
        self.InitConfFile()
        self.InitParameter()
        self.InitSGD()
        self.InitSelected()
        self.TrainNTN()
        #self.TestNTN()


def main():
    pass


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print "Error: params number is 4!"
        print "Need: train relevance feature file, test relevance feature file, ideal ranking file, and configure file!"
        sys.exit(-1)

    carpe_diem = DiverseNTN(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    carpe_diem.Main()
    del carpe_diem
    print "Game over!"
