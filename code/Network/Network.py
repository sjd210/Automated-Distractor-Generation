import numpy as np
import pandas as pd
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import math
import Model

import Read
import pickle
import datetime

import sys
sys.path.append('../Feature_Based')
import Evaluators

# env: 3.11.8 'cuda'

featureSize = 32
batchSize = 32 
poolsSize = 64
weightDecay = 0.01
d_learnRate = 0.01
g_learnRate = 0.01
temperature = 0.2
lambdaVal = 0.5
negativeSampleSize = 16
maxSeqLength = 34
embeddingDim = 300

prefix = "sciq"
dataset = 'sciq_processed_2'
embeddingsPath = "data_embeddings/glove.840B.300d.txt"
fill = '<->'


print("1. Loading embeddings and vocab")
vocab, embd = Read.build_vocab(dataset)
if len(embeddingsPath) > 0:
    assert(embd.shape[1] == embeddingDim)
    with open('{}/embd.pkl'.format(dataset), 'wb') as fout:
        pickle.dump(embd, fout)
with open('{}/vocab.pkl'.format(dataset), 'wb') as fout:
    pickle.dump(vocab, fout)


print("2. Converting data to local format")
answerList = Read.read_answer_list(dataset, "vocab.txt", maxSeqLength, fill)
data, dataDict = Read.read_data('{}/train.data'.format(dataset))
testData, testDataDict = Read.read_data('{}/newtest.data'.format(dataset), allDists=True) # smaller test-set
# validData, validDataDict = data_helpers.Read_data_with_all_dists('{}/valid.data'.format(dataset))


print("3. Creating Models")
cuda = torch.device('cuda:0')
# Gradient Learners
with torch.cuda.device('cuda:0'):
    G_v1 = torch.autograd.Variable((torch.ones(featureSize, featureSize)).cuda(), requires_grad=True)
    G_v2 = torch.autograd.Variable((torch.ones(featureSize, 1)).cuda(), requires_grad=True)
    
    D_v1 = torch.autograd.Variable((torch.ones(featureSize, featureSize)).cuda(), requires_grad=True)
    D_v2 = torch.autograd.Variable((torch.ones(featureSize, 1)).cuda(), requires_grad=True)

    evaluateLoss = torch.nn.DataParallel(Model.Loss(), [0])
    evaluateLoss.cuda()

for autovar in [G_v1, G_v2, D_v1, D_v2]:
    torch.nn.init.normal_(autovar, mean=0, std=0.1)

# Models
generator = torch.nn.DataParallel(Model.Generator(G_v1, G_v2, temperature), device_ids=[0])
generator.cuda()

discriminator = torch.nn.DataParallel(Model.Discriminator(D_v1, D_v2), device_ids=[0])
discriminator.cuda()

# Optimisers
gp = list(generator.parameters()) + [G_v1, G_v2]
g_opt = torch.optim.SGD(gp, lr = g_learnRate, momentum=0.9)

dp = list(generator.parameters()) + [D_v1, D_v2]
d_opt = torch.optim.SGD(dp, lr = d_learnRate, momentum=0.9)

dEpochCount = 1
gEpochCount = 1
bigEpochCount = 30
def main():
    for epoch in range(bigEpochCount):
        print("BIG EPOCH", epoch)
        if epoch >= 0:
            print("Distractor Training")
            for d_epoch in range(dEpochCount):
                lenData = sum([1 for _ in enumerate(data)])
                for index, values in enumerate(data):
                    q,a,d = values[1], values[2], values[3]

                    positiveSampleSize = min(4, len(dataDict[q]))
                    negativeIndexes = np.random.choice(np.arange(len(answerList)), (32-positiveSampleSize), replace=False)
                    negativePool = np.array(answerList)[negativeIndexes]
                    positiveIndexes = np.random.choice(np.arange(len(dataDict[q])), positiveSampleSize, replace=False)
                    positivePool = np.array(dataDict[q])[positiveIndexes]
                    answerPool = np.concatenate((negativePool, positivePool))

                    samples = Read.load_samples(q, a, d, answerPool, vocab, maxSeqLength)
                    predicteds = []
                    for batch in Read.batch(samples, batchSize):
                        predicted = generator.module.get_score(torch.tensor(batch).cuda())

                        predicteds += [predicted.reshape(-1).data.cpu().numpy()]
                    predicteds = np.array(predicteds)
                    predicteds *= temperature
                    normPredicteds = predicteds - np.max(predicteds)
                    expPredicteds = np.exp(normPredicteds)
                    prob = expPredicteds / np.sum(expPredicteds)
                    prob = np.nan_to_num(prob) + 1e-7
                    prob = (prob / np.sum(prob)).flatten()

                    genWeightedIndexes = np.random.choice(np.arange(len(answerPool)), size=negativeSampleSize, p=prob)
                    subsamples = np.array(Read.load_samples(q,a,d, answerPool[genWeightedIndexes], vocab, maxSeqLength))

                    predData = []
                    for [qss,ass,dss,nss] in subsamples:
                        predData += [[qss,ass,dss]]
                    for [qss,ass,dss,nss] in subsamples:
                        predData += [[qss,ass,nss]]

                    dss = subsamples[:, 2].tolist()
                    nss = subsamples[:, 3].tolist()

                    predDataLabel = [1.0] * len(dss)
                    predDataLabel.extend([0.0] * len(nss))
                    predDataLabel = np.asarray(predDataLabel)

                    loss = discriminator(torch.tensor(predData), torch.tensor(predDataLabel)) + weightDecay * (evaluateLoss(D_v1) + evaluateLoss(D_v2))
                    
                    d_opt.zero_grad()
                    loss.backward()
                    d_opt.step()

                    if index % 10 == 0:
                        print(datetime.datetime.now().strftime("%X"), "Discriminator Epoch: %d/%d - %d/%d Loss:" %(d_epoch+1, dEpochCount, index, lenData), loss.item())


        print("Generator Training")
        baselineList = []
        baselineVal = 0.05
        for g_epoch in range(gEpochCount):
            lenData = sum([1 for _ in enumerate(data)])
            for index, values in enumerate(data):
                q,a,d = values[1], values[2], values[3]

                positiveSampleSize = min(4, len(dataDict[q]))
                negativeIndexes = np.random.choice(np.arange(len(answerList)), (32-positiveSampleSize), replace=False)
                negativePool = np.array(answerList)[negativeIndexes]
                positiveIndexes = np.random.choice(np.arange(len(dataDict[q])), positiveSampleSize, replace=False)
                positivePool = np.array(dataDict[q])[positiveIndexes]
                answerPool = np.concatenate((negativePool, positivePool))

                samples = Read.load_samples(q, a, d, answerPool, vocab, maxSeqLength)
                for batch in Read.batch(samples, batchSize):
                    predicted = generator.module.get_score(torch.tensor(batch).cuda())
                    predicteds += [predicted.reshape(-1).data.cpu().numpy()]
                predicteds = np.array(predicteds) * temperature
                normPredicteds = predicteds - np.max(predicteds)
                expPredicteds = np.exp(normPredicteds)
                prob = expPredicteds / np.sum(expPredicteds)
                prob = np.nan_to_num(prob) + 1e-7
                prob = (prob / np.sum(prob)).flatten()

                probS = prob * (1 - lambdaVal)
                for i in range(len(answerPool)):
                    if answerPool[i] in positivePool:
                        probS[i] += lambdaVal / len(positivePool)
                
                choiceIndexes = np.random.choice(np.arange(len(answerPool)), [32], p=probS.reshape(-1))
                choiceSamples = Read.load_samples(q,a,d, answerPool[choiceIndexes], vocab, maxSeqLength)
                choiceS = np.array(prob)[choiceIndexes] / np.array(probS)[choiceIndexes]
                choiceReward = discriminator.module.get_reward(np.asarray(choiceSamples))
                choiceReward.detach_()

                baselineList.append(np.mean(choiceReward.data.cpu().numpy()))
                choiceReward = choiceReward - baselineVal

                loss = generator(torch.tensor(choiceSamples).cuda(), torch.tensor(choiceIndexes).cuda(), choiceReward, torch.tensor(probS).cuda().long() + weightDecay * (evaluateLoss(G_v1) + evaluateLoss(G_v2)))
                g_opt.zero_grad()
                loss.backward()
                g_opt.step()

                if index % 10 == 0:
                    print(datetime.datetime.now().strftime("%X"), "Generator Epoch: %d/%d - %d/%d Loss:" %(g_epoch+1, gEpochCount, index, lenData), loss.item())

                    dfList = []
                    for i in range(1, len(testDataDict.keys())):
                        alltest_labels = []
                        b = list(testDataDict.keys())[i]
                        
                        candidates = Read.load_samples(b, testDataDict[b][0], testDataDict[b][1], testDataDict["all_dists"], vocab, maxSeqLength)
                        
                        alltest_labels = [0] * (32 * math.ceil(len(testDataDict["all_dists"])/32))
                        for j in range(1, len(testDataDict[b])):
                            alltest_labels[testDataDict["all_dists"].index(testDataDict[b][j])] = 1

                        preds = []
                        for batch in Read.batch(candidates, batchSize):
                            preds += generator.module.get_score(torch.tensor(batch).cuda())
                        preds_tensor = torch.tensor(preds, device = 'cpu')

                        dfList += pd.DataFrame({"value": preds_tensor, "label": alltest_labels}).sort_values(by="value")

                    dfDict = {"Neural Network": dfList}
                    allDists = [[1] * 3] * (len(testDataDict.keys())-1)
                    Evaluators.calculate_metrics(dfDict, allDists, "Test Set")


if __name__ == '__main__':
    main()