import math

def recallAt(dataframe, rank, questionNo, allDists): # "we report top recall (R@10)" - so find the best one??
    total = 0
    for i in range(1, rank+1):
        total += dataframe.iloc[len(dataframe)-i]['distractor'] in [str(allDists[questionNo][0]), str(allDists[questionNo][1]), str(allDists[questionNo][2])]
    return total/3

def precisionAt(dataframe, rank, questionNo, allDists):
    total = 0
    for i in range(1, rank+1):
        total += dataframe.iloc[len(dataframe)-i]['distractor'] in [str(allDists[questionNo][0]), str(allDists[questionNo][1]), str(allDists[questionNo][2])]
    return total/rank

def averagePrecisionAt(dataframe, rank, questionNo, allDists):
    total = 0
    for i in range(1, rank+1):
        total += precisionAt(dataframe, i, questionNo, allDists)
    return total/rank

def meanAveragePrecisionAt(dataframeList, rank, allDists):
    size = len(dataframeList[0])//3
    total = [0] * size
    for q in range(size):
        total[q] = averagePrecisionAt(dataframeList[q], rank, q, allDists)
    return sum(total)/size



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

def ndcgAt(dataframe, rank, questionNo, allDists): #https://www.evidentlyai.com/ranking-metrics/ndcg-metric
    return dcgAt(dataframe, rank, questionNo, allDists)/idcgAt(rank)


def reciprocalRank(dataframe, questionNo, allDists):
    for i in range(1, len(dataframe)):
        if (dataframe.iloc[len(dataframe)-i]['distractor'] in [str(allDists[questionNo][0]), str(allDists[questionNo][1]), str(allDists[questionNo][2])]):
            return 1/i
    return 0

def meanReciprocalRank(dataframeList, allDists): # only care about first relevent item
    size = len(dataframeList[0])//3
    total = [0] * size
    for q in range(size):
        total[q] = reciprocalRank(dataframeList[q], q, allDists)
    return sum(total)/size
