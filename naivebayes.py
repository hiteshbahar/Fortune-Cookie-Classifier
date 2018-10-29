import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

def yLabelProcessin(label):
    with open(label,'r') as fp:
        label = fp.read().split()
        label = [int(i) for i in label]
    return label

def preprocessing(data,label,stopList):
    with open(data,'r') as fp:
        data = fp.readlines()
    with open(stopList,'r') as fp:
        stopData = fp.read().split()
    vocabulary =set()
    for item in data:
        word = item.split()
        for i in list(word):
            if i not in stopData:
                vocabulary.add(i)
    vocabulary = sorted(vocabulary)
    featureSet = []
    for item in data:
        individualWord = item.split()
        featureVector = [0]*len(vocabulary)
        for i in individualWord:
            if i in vocabulary:
                position = vocabulary.index(i)
                featureVector[position] = 1
        featureSet.append(featureVector)
    featureDataframe = pd.DataFrame(featureSet,columns = vocabulary)
    #print(featureDataframe)
    return featureDataframe,label

def NaiveB(featureDataframe,ylabel):
    indexes_0 = [index for index in range(len(ylabel)) if ylabel[index] == 0]
    indexes_1 = [index for index in range(len(ylabel)) if ylabel[index] == 1]
    subset0 = pd.DataFrame()
    subset1 = pd.DataFrame()

    subset0 = (featureDataframe.iloc[152:,])
    subset1 = (featureDataframe.iloc[0:152,])
    
    columns = subset0.columns
    prob_present0 = {}
    prob_absent0 = {}
    for i in range(subset0.shape[1]):
        counter1 = 0
        counter0 =0
        for j in range(subset0.shape[0]):
            if subset0.iloc[j,i] == 1:
                counter1 += 1
            elif subset0.iloc[j,i] == 0:
                counter0 += 1
        prob_present0[columns[i]] = (counter1 + 1)/(len(subset0) + 2)
        prob_absent0[columns[i]] = (counter0 +1 )/(len(subset0) + 2)
    prob_present0
    prob_present1 = {}
    prob_absent1 = {}
    for i in range(subset1.shape[1]):
        counter1 = 0
        counter0 =0
        for j in range(subset1.shape[0]):
            if subset1.iloc[j,i] == 1:
                counter1 += 1
            elif subset1.iloc[j,i] == 0:
                counter0 += 1
        prob_present1[columns[i]] = (counter1 +1) / (len(subset1) + 2)
        prob_absent1[columns[i]] = (counter0 +1) / (len(subset1) + 2)
    
    return prob_present0,prob_present1

def probCal(prob_present0,prob_present1,data,ylabel,file1) :
    predictions ={}
    with open(data,'r') as fp:
        line = fp.readlines()
        
    problist0 =[]
    problist1 =[]
    error = []
    for i in line:
        predict_prob0 = 1
        x = i.split()
        for word in x:
            if word in prob_present0.keys():
                tmp = prob_present0.get(word)
                predict_prob0 = tmp * predict_prob0
        problist0.append(predict_prob0)
    for j in line:
        predict_prob1 = 1
        y = j.split()
        for word1 in y:
            if word1 in prob_present1.keys():
                tmp = prob_present1.get(word1)
                predict_prob1 = tmp * predict_prob1
        problist1.append(predict_prob1)
    loopiter = 0
    for i in line:
        if problist0[loopiter] > problist1[loopiter]:
            predictions[i] = 'wise'
            error.append(0)
        else:
            predictions[i] = 'future'
            error.append(1)
        loopiter += 1
    
    print("**************        Future/Wise        ***************",file = file1)
    print(predictions,file = file1)
    #Predict The Accuracy of the model
    count = 0
    for i in range(len(ylabel)):
        if error[i] == ylabel[i]:
            count +=1
    accuracy = (count/len(ylabel)) * 100
    return accuracy
    
    

def main():
    file1 = open('output.txt','wt')
    print("**************************************        Results of Naive Bayes Algorithm Implementation      ***************************",file =file1)

    filepath = "fortunecookiedata/traindata.txt"
    stopList = "fortunecookiedata/stoplist.txt"
    labelList = "fortunecookiedata/trainlabels.txt"
    probDataTrain = "fortunecookiedata/traindata.txt"
    ylabel = yLabelProcessin(labelList)
    featureDataframe,label = preprocessing(filepath,ylabel,stopList)
    prob_present0,prob_present1 = NaiveB(featureDataframe,label)
    accuracy = probCal(prob_present0,prob_present1,probDataTrain,ylabel,file1)
    print("Train Accuracy ",accuracy,file = file1)
    ylabelTest = "fortunecookiedata/testlabels.txt"
    
    ylabelT = yLabelProcessin(ylabelTest)
    probDataTest = "fortunecookiedata/testdata.txt"
    accuracyT = probCal(prob_present0,prob_present1,probDataTrain,ylabelT,file1)
    print("*****************        Train Accuracy   ***************",file = file1)
    
    print("*****************        Test Accuracy   ***************",file = file1)
    print("Test Accuracy",accuracyT,file = file1)
    
    # Logistic Regression Part b of the exercise
    logreg = LogisticRegression()
    logreg.fit(featureDataframe,label)
    
    sklearnAccuracy = logreg.score(featureDataframe,label)
    print("*****************       SK Learn Train Accuracy   ***************",file = file1)
    
    print('Accuracy Using Logistic Regression SKlearn - Trainng  {:.3f}'.format(sklearnAccuracy*100),file = file1)
    
    featureDataframeT,labelT = preprocessing(probDataTest,ylabelT,stopList)
    
    logreg = LogisticRegression()
    logreg.fit(featureDataframeT,labelT)
    
    sklearnAccuracy = logreg.score(featureDataframeT,labelT)
    
    print("*****************     SK Learn   Test Accuracy   ***************",file = file1)
    print('Accuracy Using Logistic Regression SKlearn - Testing  {:.3f}'.format(sklearnAccuracy*100),file = file1)
    file1.close()

main()