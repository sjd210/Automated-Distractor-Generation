import math
import numpy as np

def recallAt(dataframe, rank, questionNo, allDists): 
    total = 0
    for i in range(1, rank+1):
        total += dataframe.iloc[len(dataframe)-i]['distractor'] in [str(allDists[questionNo][0]), str(allDists[questionNo][1]), str(allDists[questionNo][2])]
    return total/3

def meanRecallAt(dataframeList, rank, allDists):
    
    size = len(dataframeList)
    total = [0] * size
    for q in range(size):
        total[q] = recallAt(dataframeList[q], rank, q, allDists)
    return sum(total)/size


def precisionAt(dataframe, rank, questionNo, allDists):
    total = 0
    for i in range(1, rank+1):
        total += dataframe.iloc[len(dataframe)-i]['distractor'] in [str(allDists[questionNo][0]), str(allDists[questionNo][1]), str(allDists[questionNo][2])]
    return total/rank

def meanPrecisionAt(dataframeList, rank, allDists):
    size = len(dataframeList)
    total = [0] * size
    for q in range(size):
        total[q] = precisionAt(dataframeList[q], rank, q, allDists)
    return sum(total)/size

def averagePrecisionAt(dataframe, rank, questionNo, allDists):
    total = 0
    for i in range(1, rank+1):
        total += precisionAt(dataframe, i, questionNo, allDists)
    return total/rank

def meanAveragePrecisionAt(dataframeList, rank, allDists):
    size = len(dataframeList)
    total = [0] * size
    for q in range(size):
        total[q] = averagePrecisionAt(dataframeList[q], rank, q, allDists)
    return sum(total)/size

def tTestMAP(dataframeList1, dataframeList2, rank, allDists, tRank):
    size = len(dataframeList1)
    if (size != len(dataframeList2)):
        print("Invalid dataframe sizes")
    diff = [0] * tRank
    diffSquared = [0] * tRank
    randomChoices = np.random.choice(size, tRank, replace=False)
    i = 0
    for q in randomChoices:
        total1 = averagePrecisionAt(dataframeList1[q], rank, q, allDists)
        total2 = averagePrecisionAt(dataframeList2[q], rank, q, allDists)
        diff[i] = abs(total1 - total2)
        diffSquared[i] = diff[i] * diff[i]
        i += 1

    t = sum(diff)/(math.sqrt(((tRank)*sum(diffSquared) - sum(diff) ** 2)/(tRank-1)))
    return t

def unpairedTTestMAP(dataframeList1, dataframeList2, rank, allDists1, allDists2, tRank):
    size = len(dataframeList1)
    if (size != len(dataframeList2)):
        print("Invalid dataframe sizes")

    MAP1 = meanAveragePrecisionAt(dataframeList1, rank, allDists1)
    MAP2 = meanAveragePrecisionAt(dataframeList2, rank, allDists2)

    randomChoices = np.random.choice(size, tRank, replace=False)
    i = 0
    sum1 = 0
    sum2 = 0
    for q in randomChoices:
        sum1 += (averagePrecisionAt(dataframeList1[q], rank, q, allDists1) - MAP1) ** 2
        sum2 += (averagePrecisionAt(dataframeList2[q], rank, q, allDists2) - MAP2) ** 2
        i += 1

    s1 = sum1/(i**2)
    s2 = sum2/(i**2)
    t = (MAP1 - MAP2)/math.sqrt(s1 + s2)
    return t


def dcgAt(dataframe, rank, questionNo, allDists):
    total = 0
    for i in range(1, rank+1):
        total += (dataframe.iloc[len(dataframe)-i]['distractor'] in [str(allDists[questionNo][0]), str(allDists[questionNo][1]), str(allDists[questionNo][2])])/math.log(i+1, 2)
    return total

def idcgAt(rank):
    total = 0
    for i in range(1, rank+1):
        if (i < 4):
            total += 1/math.log(i+1, 2)
    return total

def ndcgAt(dataframe, rank, questionNo, allDists):
    return dcgAt(dataframe, rank, questionNo, allDists)/idcgAt(rank)

def meanNdcgAt(dataframeList, rank, allDists):
    size = len(dataframeList)
    total = [0] * size
    for q in range(size):
        total[q] = ndcgAt(dataframeList[q], rank, q, allDists)
    return sum(total)/size


def reciprocalRank(dataframe, questionNo, allDists):
    for i in range(1, min(len(dataframe), 1000)):
        if (dataframe.iloc[len(dataframe)-i]['distractor'] in [str(allDists[questionNo][0]), str(allDists[questionNo][1]), str(allDists[questionNo][2])]):
            return 1/i
    return 0

def meanReciprocalRank(dataframeList, allDists): 
    size = len(dataframeList)
    total = [0] * size
    for q in range(size):
        total[q] = reciprocalRank(dataframeList[q], q, allDists)
    return sum(total)/size

def calculate_metrics(dfDict, allDists, setName):
  print("====== ", setName, " =======")

  for name in dfDict:
    print("======")
    print(name)
    print("R@10: ", meanRecallAt(dfDict[name], 10, allDists))
    print("P@1: ", meanPrecisionAt(dfDict[name], 1, allDists))
    print("P@3: ", meanPrecisionAt(dfDict[name], 3, allDists))
    print("MAP@10: ", meanAveragePrecisionAt(dfDict[name], 10, allDists))
    print("NDCG@10: ", meanNdcgAt(dfDict[name], 10, allDists))
    print("MRR: ", meanReciprocalRank(dfDict[name], allDists))