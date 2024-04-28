import Read
import Extractors as Extractors
import Evaluators as Evaluators
import numpy as np
import pandas as pd
import xgboost
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from gensim.models import KeyedVectors
#from wikipedia2vec import Wikipedia2Vec
import spacy

import zlib
import pickle

# env 3.6.3 nevEnv

dataset = "sciq"

print("Loading Wikipedia Embeddings")

wiki2vec2 = KeyedVectors.load_word2vec_format("WikipediaEmbeddings/small_test.txt", binary = False)
#wiki2vec2 = KeyedVectors.load_word2vec_format("WikipediaEmbeddings/enwiki_20180420_100d.txt", binary = False)

print("Loading SpaCy Tokenisations")
tokenise = spacy.load("en_core_web_lg")

def logit2prob(logr, X):
  log_odds = logr.intercept_[0]
  for i in range(len(col_names)):
    log_odds += logr.coef_[0][i] * X[col_names[i]]
  odds = np.exp(log_odds)
  probability = odds / (1 + odds)
  return(probability)

print("Loading Arc Dictionary")

with open(os.path.dirname(__file__)+ "/arcDict", 'rb') as file:
  a = file.read()
  b = zlib.decompress(a)
  arcDict = pickle.loads(b)
  arcDictSum = sum(arcDict.values())

print("Loading SciQ Dictionary")

with open(os.path.dirname(__file__)+ "/sciqSupportDict", 'rb') as file:
  a = file.read()
  b = zlib.decompress(a)
  sciqSupportDict = pickle.loads(b)
  sciqSupportSum = sum(sciqSupportDict.values())

def dictToTokenLists(dict, allDistsListALL, rank):
  tokenQuestion = []
  tokenAnswer = []
  tokenDist = []
  tokenDistList = []
  
  question, answer, allDist, allDistsList = Read.extract_all_fields(dict, rank)

  for i in range(rank):
    print("= ", i)
    tokenAnswer += [tokenise(answer[i])]
    tokenQuestion += [tokenise(question[i])]
    tokenDist += [[[], [], []]]
    #for j in range(3):
    # tokenDist[i][j] = tokenise(str(allDists[i][j]))
  
  for i in range(len(allDistsListALL)):
    print("==", i)
    tokenDistList += [tokenise(allDistsListALL[i])]

  return ((tokenQuestion, tokenAnswer, tokenDist, tokenDistList), (question, answer, allDists, allDistsListALL))

def writeToPickleFile(name, data):
  with open(name, 'wb') as file:
    pickledData = pickle.dumps(data)
    compressedPickle = zlib.compress(pickledData)
    file.write(compressedPickle)

def dictToX(fileNames, rank, col_names, tokens, strings):
  X2 = []
  savedSize = 0
  print(fileNames, len(fileNames))
  if len(fileNames) != 0:
    for fileName in fileNames:
      if os.path.isfile(fileName):
        with open(fileName, 'rb') as file:
          a = file.read()
          b = zlib.decompress(a)
          c = pickle.loads(b)
          savedSize += len(c)
          print(fileName, " read")
          embDict = [[]] * rank
          i = 0
          s = (savedSize % 500)
          if s == 0:
            s = 500
          for q in range((savedSize - s), savedSize):
            print("emb ", q, i)
            X = {}
            X["emb_sim_ad"] = c[i]["emb_sim_ad"]
            X["emb_sim_qd"] = c[i]["emb_sim_qd"]
            embDict[q] = pd.DataFrame.from_dict(X)
            if ((q+1) % 500 == 0 and q != 0):
              writeToPickleFile(generate_file_name("progressive"), embDict[(q-499):(q+1)])
              print(generate_file_name("progressive"), " written")
              embDict = [[]] * rank
            elif ((q+1) == rank):
              writeToPickleFile(generate_file_name("progressive"), embDict[(rank - (rank % 500)):rank])
              print(generate_file_name("progressive"), " written")
              embDict = [[]] * rank
            i += 1
      else:
        print("Failed to read", fileName)
  else:
    X2 = [[]] * (rank)

  X2 = [[]] * rank

  (tokenQuestion, tokenAnswer, tokenDist, tokenDistList), (question, answer, allDists, allDistsList) = tokens, strings

  preX = {"character_length_d": [], "token_length_d": [], "word_freq_support_d": [], "word_freq_arc_d": [], "number_d": []}
  for i in range(len(allDistsList)):
    preX["character_length_d"] += [Extractors.character_length(allDistsList[i])] # notSpecific
    preX["token_length_d"] += [Extractors.token_length(tokenDistList[i])] # notSpecific
    preX["word_freq_support_d"] += [Extractors.word_freq(tokenDistList[i], sciqSupportDict, sciqSupportSum)] # notSpecific
    preX["word_freq_arc_d"] += [Extractors.word_freq(tokenDistList[i], arcDict, arcDictSum)] # notSpecific
    preX["number_d"] += [Extractors.does_string_contain_number(tokenDistList[i])] # notSpecific

  for q in range(savedSize, rank):
    X = {}
    for col in col_names:
      X[col] = []
      X["character_length_d"] = preX["character_length_d"]
      X["token_length_d"] = preX["token_length_d"]
      X["word_freq_support_d"] = preX["word_freq_support_d"]
      X["word_freq_arc_d"] = preX["word_freq_arc_d"]
      X["number_d"] = preX["number_d"]

    for i in range(len(allDistsList)):
      X["emb_sim_qd"] += [Extractors.embedding_sim(tokenQuestion[q], tokenDistList[i])]
      X["emb_sim_ad"] += [Extractors.embedding_sim(tokenAnswer[q], tokenDistList[i])] # warning + 0 if not word in dictionary
      X["pos_sim_ad"] += [Extractors.pos_similarity(tokenAnswer[q], tokenDistList[i])]
      X["edit_dist"] += [Extractors.edit_distance(answer[q], allDistsList[i])]
      X["token_sim_qd"] += [Extractors.token_similarity(tokenQuestion[q], tokenDistList[i])]
      X["token_sim_ad"] += [Extractors.token_similarity(tokenAnswer[q], tokenDistList[i])]
      #X["character_length_d"] += [Extractors.character_length(allDistsList[i])] # notSpecific
      X["character_length_diff"] += [Extractors.character_length_diff(answer[q], allDistsList[i])] 
      #X["token_length_d"] += [Extractors.token_length(tokenDistList[i])] # notSpecific
      X["token_length_diff"] += [Extractors.token_length_diff(tokenAnswer[q], tokenDistList[i])]
      X["abs_comm_suffix"] += [Extractors.longest_suffix_length(answer[q], allDistsList[i])]
      X["rel_comm_suffix"] += [Extractors.longest_suffix_length(answer[q], allDistsList[i])/len(allDistsList[i])] 
      #X["word_freq_support_d"] += [Extractors.word_freq(tokenDistList[i], sciqSupportDict, sciqSupportSum)] # notSpecific
      X["word_freq_support_comp_ad"] += [Extractors.word_freq_comp(tokenAnswer[q], tokenDistList[i], sciqSupportDict, sciqSupportSum)]
      #X["word_freq_arc_d"] += [Extractors.word_freq(tokenDistList[i], arcDict, arcDictSum)] # notSpecific
      X["word_freq_comp_arc_ad"] += [Extractors.word_freq_comp(tokenAnswer[q], tokenDistList[i], arcDict, arcDictSum)] 
      X["sing_plur"] += [Extractors.singular_plural_consistency(tokenAnswer[q], tokenDistList[i])]
      #X["number_d"] += [Extractors.does_string_contain_number(tokenDistList[i])] # notSpecific
      X["number_diff"] += [Extractors.number_consistency(tokenAnswer[q], tokenDistList[i])]
      X["wikisim_entity"] += [Extractors.wiki_sim_entity(answer[q], allDistsList[i], wiki2vec2)]
      X["wikisim_ad"] += [Extractors.wiki_sim_base_similarity(tokenAnswer[q], tokenDistList[i], wiki2vec2)]
      X["wikisim_qd"] += [Extractors.wiki_sim_base_similarity(tokenQuestion[q], tokenDistList[i], wiki2vec2)] 
    X2[q] = pd.DataFrame.from_dict(X)
    print(q)
    if ((q+1) % 500 == 0 and q != 0):
      writeToPickleFile(generate_file_name("progressive"), X2[(q-499):(q+1)])
      X2 = [[]] * rank
    elif ((q+1) == rank):
      writeToPickleFile(generate_file_name("progressive"), X2[(rank - (rank % 500)):rank])
      X2 = [[]] * rank
  return X2

def X_from_files(fileNames):
  X = []
  for fileName in fileNames:
    if os.path.isfile(fileName):
      with open(fileName, 'rb') as file:
        a = file.read()
        b = zlib.decompress(a)
        X += pickle.loads(b)
      print(fileName, "READ")
  return X

if dataset == "sciq":
  sciqTrainDict = Read.open_json_as_dict("../../data/SciQ dataset-2 3/train.json")
  sciqValidateDict = Read.open_json_as_dict("../../data/SciQ dataset-2 3/valid.json")
  sciqTestDict = Read.open_json_as_dict("../../data/SciQ dataset-2 3/test.json")

  rankTrain = len(sciqTrainDict)
  rankValidate = len(sciqValidateDict)
  rankTest = len(sciqTestDict)

  question, answer, allDists, allDistsList = Read.extract_all_fields(sciqTrainDict, rankTrain)
  questionB, answerB, allDistsB, allDistsListB = Read.extract_all_fields(sciqValidateDict, rankValidate)
  questionC, answerC, allDistsC, allDistsListC = Read.extract_all_fields(sciqTestDict, rankTest)

  trainFileNames = []
  for i in range(1, 25):
    trainFileNames += ["PrecomputedFeatures/sciqTrain{0}".format(i)]
  validFileNames = ["PrecomputedFeatures/sciqValid1", "PrecomputedFeatures/sciqValid2"]
  #validFileNames = ["PrecomputedFeatures/allSciqValid1", "PrecomputedFeatures/allSciqValid2"]
  testFileNames = ["PrecomputedFeatures/sciqTest1", "PrecomputedFeatures/sciqTest2"]
  #testFileNames = ["PrecomputedFeatures/allSciqTest1", "PrecomputedFeatures/allSciqTest2"]

if dataset == "arcCombined":
  arcCombinedTrainDict = Read.open_json_as_dict("../../data/ARC-Combined/train.json")
  arcCombinedValidateDict = Read.open_json_as_dict("../../data/ARC-Combined/valid.json")
  arcCombinedTestDict = Read.open_json_as_dict("../../data/ARC-Combined/test.json")

  rankTrain = len(arcCombinedTrainDict)
  rankValidate = len(arcCombinedValidateDict)
  rankTest = len(arcCombinedTestDict)

  question, answer,  allDists, allDistsList = Read.extract_all_fields(arcCombinedTrainDict, rankTrain)
  questionB, answerB, allDistsB, allDistsListB = Read.extract_all_fields(arcCombinedValidateDict, rankValidate)
  questionC, answerC, allDistsC, allDistsListC = Read.extract_all_fields(arcCombinedTestDict, rankTest)

  trainFileNames = []
  for i in range(1,13):
    trainFileNames += ["PrecomputedFeatures/arcCombinedTrain{0}".format(i)]
  validFileNames = ["PrecomputedFeatures/arcCombinedValid1", "PrecomputedFeatures/arcCombinedValid2"]
  testFileNames = ["PrecomputedFeatures/arcCombinedTest1", "PrecomputedFeatures/arcCombinedTest2"]

if dataset == "arcEasy":
  arcEasyTrainDict = Read.open_json_as_dict("../../data/ARC-Easy-2/train.json")
  arcEasyValidateDict = Read.open_json_as_dict("../../data/ARC-Easy-2/valid.json")
  arcEasyTestDict = Read.open_json_as_dict("../../data/ARC-Easy-2/test.json")

  rankTrain = len(arcEasyTrainDict)
  rankValidate = len(arcEasyValidateDict)
  rankTest = len(arcEasyTestDict)

  question, answer,  allDists, allDistsList = Read.extract_all_fields(arcEasyTrainDict, rankTrain)
  questionB, answerB, allDistsB, allDistsListB = Read.extract_all_fields(arcEasyValidateDict, rankValidate)
  questionC, answerC, allDistsC, allDistsListC = Read.extract_all_fields(arcEasyTestDict, rankTest)

  trainFileNames = ["PrecomputedFeatures/arcEasyTrain1", "PrecomputedFeatures/arcEasyTrain2", "PrecomputedFeatures/arcEasyTrain3", "PrecomputedFeatures/arcEasyTrain4", "PrecomputedFeatures/arcEasyTrain5"]
  validFileNames = ["PrecomputedFeatures/arcEasyValid1", "PrecomputedFeatures/arcEasyValid2"]
  testFileNames = ["PrecomputedFeatures/arcEasyTest1", "PrecomputedFeatures/arcEasyTest2", "PrecomputedFeatures/arcEasyTest3", "PrecomputedFeatures/arcEasyTest4", "PrecomputedFeatures/arcEasyTest5"]

if dataset == "arcEasy3":
  arcEasyTrainDict = Read.open_json_as_dict("../../data/ARC-Easy-3/train.json")
  arcEasyValidateDict = Read.open_json_as_dict("../../data/ARC-Easy-3/valid.json")
  arcEasyTestDict = Read.open_json_as_dict("../../data/ARC-Easy-3/test.json")

  rankTrain = len(arcEasyTrainDict)
  rankValidate = len(arcEasyValidateDict)
  rankTest = len(arcEasyTestDict)

  question, answer,  allDists, allDistsList = Read.extract_all_fields(arcEasyTrainDict, rankTrain)
  questionB, answerB, allDistsB, allDistsListB = Read.extract_all_fields(arcEasyValidateDict, rankValidate)
  questionC, answerC, allDistsC, allDistsListC = Read.extract_all_fields(arcEasyTestDict, rankTest)

  trainFileNames = []
  validFileNames = []
  testFileNames = []

if dataset == "arcChallenge":
  arcChallengeTrainDict = Read.open_json_as_dict("../../data/ARC-Challenge-2/train.json")
  arcChallengeValidateDict = Read.open_json_as_dict("../../data/ARC-Challenge-2/valid.json")
  arcChallengeTestDict = Read.open_json_as_dict("../../data/ARC-Challenge-2/test.json")

  rankTrain = len(arcChallengeTrainDict)
  rankValidate = len(arcChallengeValidateDict)
  rankTest = len(arcChallengeTestDict)

  question, answer, allDists, allDistsList = Read.extract_all_fields(arcChallengeTrainDict, rankTrain)
  questionB, answerB, allDistsB, allDistsListB = Read.extract_all_fields(arcChallengeValidateDict, rankValidate)
  questionC, answerC, allDistsC, allDistsListC  = Read.extract_all_fields(arcChallengeTestDict, rankTest)

  trainFileNames = ["PrecomputedFeatures/arcChallengeTrain1", "PrecomputedFeatures/arcChallengeTrain2", "PrecomputedFeatures/arcChallengeTrain3"]
  validFileNames = ["PrecomputedFeatures/arcChallengeValid1"]
  testFileNames = ["PrecomputedFeatures/arcChallengeTest1", "PrecomputedFeatures/arcChallengeTest2", "PrecomputedFeatures/arcChallengeTest3"]

allDistsListALL = sorted(list(set(allDistsList + allDistsListB + allDistsListC)))

col_names = ["emb_sim_qd", "emb_sim_ad", "pos_sim_ad", "edit_dist", "token_sim_qd", "token_sim_ad", "character_length_d",
             "character_length_diff", "token_length_d", "token_length_diff", "abs_comm_suffix", "rel_comm_suffix", "word_freq_support_d",
             "word_freq_support_comp_ad", "word_freq_arc_d", "word_freq_comp_arc_ad",
             "sing_plur", "number_d", "number_diff", "wikisim_entity", "wikisim_ad", "wikisim_qd"]

#col_names = ["emb_sim_qd", "emb_sim_ad"]

             # wikisim_ad, token_sim_ad, edit_dist, "pos_sim_ad", "emb_sim_ad", "sing_plur", "rel_comm_suffix", "wikisim_entity", "token_sim_qd", "token_length_diff"

logreg = LogisticRegression(C=1, solver = 'newton-cg', max_iter=10000) #rank)
logregScale = LogisticRegression(C=1, solver = 'newton-cg', max_iter=10000) #rank)
randomForest = RandomForestClassifier(n_estimators=750, criterion='gini') 
randomForestScale = RandomForestClassifier(n_estimators=750, criterion='gini') 
lambdaMart4 = xgboost.XGBRanker(tree_method="hist",  objective="rank:map", lambdarank_num_pair_per_sample=10, lambdarank_pair_method="topk")
lambdaMartScale = xgboost.XGBRanker(tree_method="hist",  objective="rank:map", lambdarank_num_pair_per_sample=10, lambdarank_pair_method="topk")

def model_from_file(fileName):
  if os.path.isfile(fileName):
    with open(fileName, 'rb') as file:
      a = file.read()
      b = zlib.decompress(a)
      model = pickle.loads(b)
    print(fileName, "READ")

  return model

logreg = model_from_file("PrecomputedModels/logReg/sciq-2000")
logregScale = model_from_file("PrecomputedModels/logReg/sciqScaled-2000")
randomForest = model_from_file("PrecomputedModels/randomForest/sciq-2000")
randomForestScale = model_from_file("PrecomputedModels/randomForest/sciqScaled-2000")
lambdaMart4 = model_from_file("PrecomputedModels/lambdaMart/sciq-2000")
lambdaMartScale = model_from_file("PrecomputedModels/lambdaMart/sciqScaled-2000")

from datetime import datetime

def generate_file_name(name):
  dt_string = datetime.now().strftime("%d.%m.%Y.%H.%M.%S")
  return "PrecomputedFeatures/" + name + dt_string

writing = False
if writing:
  tokens, strings = dictToTokenLists(sciqTrainDict, allDistsListALL, rankTrain)
  XTrain = dictToX([], rankTrain, col_names, tokens, strings) 

  #tokens, strings = dictToTokenLists(arcEasyValidateDict, allDistsListB, rankValidate)
  #XValid = dictToX(validFileNames, rankValidate, col_names, tokens, strings)

  #tokens, strings = dictToTokenLists(arcEasyTestDict, allDistsListC, rankTest)
  #XTest = dictToX(testFileNames, rankTest, col_names, tokens, strings)

  print(1/0)

totLen = 0 
prevQ = 0
loopStopper = 0

#pretrainedModels = ["PrecomputedModels/logReg/sciqScaled-2000", "PrecomputedModels/randomForest/sciq-500", "PrecomputedModels/lambdaMart/sciqScaled-11679"]
pretrainedModels = []
for i in range(500, 12000, 500):
  pretrainedModels += ["PrecomputedModels/randomForest/sciqScaled-%d" %(i)]
pretrainedModels += ["PrecomputedModels/randomForest/sciqScaled-11679"]
models = []

with open("PrecomputedModels/scaler/sciq-11679", 'rb') as file:
  a = file.read()
  b = zlib.decompress(a)
  scaler  = pickle.loads(b)
#scaler = StandardScaler()

y = [[]] * rankTrain
for q in range(prevQ, rankTrain):
  y[q] = [0] * (rankTrain*3)
  y[q][q*3 + 0] = 1
  y[q][q*3 + 1] = 1
  y[q][q*3 + 2] = 1

if len(pretrainedModels) == 0:
  for fileName in trainFileNames:
    if os.path.isfile(fileName) and (loopStopper < rankTrain):
      with open(fileName, 'rb') as file:
        a = file.read()
        b = zlib.decompress(a)
        c = pickle.loads(b)
      print(fileName, "READ")

    loopStopper = min((prevQ+len(c)), rankTrain)
    used = False
    loopSize = loopStopper-prevQ
    
    X = [[]] * loopSize
    for q in range(prevQ, loopStopper):
      X[q-prevQ] = c[q-prevQ][:(rankTrain*3)] #OHHHHHH ERROR HERE WOAHHH
      #scaler.fit(X[q-prevQ])

    scaledX = [[]] * loopSize
    for q in range(prevQ, loopStopper):
      used = True
      print("train", q)

      logreg.fit(X[q-prevQ], y[q])
      randomForest.fit(X[q-prevQ], y[q])

      scaledX[q-prevQ] = scaler.transform(X[q-prevQ])
      logregScale.fit(scaledX[q-prevQ], y[q])
      randomForestScale.fit(scaledX[q-prevQ], y[q])
      
    
    if used:
      qida = [0] * (rankTrain*3) * (loopSize)
      val = prevQ
      for i in range(loopSize):
        for j in range(rankTrain*3):
          qida[i*rankTrain*3 + j] = val
        val += 1

      ya = np.array(y[prevQ:loopStopper]).flatten()
      xa = np.array(X).reshape((-1,len(col_names)))
      scaledXa = np.array(scaledX).reshape((-1,len(col_names)))
      lambdaMart4.fit(xa, ya, qid=qida)
      lambdaMartScale.fit(scaledXa, ya, qid=qida)
      print("Lambda Done")
    
    prevQ += loopSize

    writeToPickleFile('./PrecomputedModels/logReg/%s-%d' %(dataset, prevQ), logreg)
    writeToPickleFile('./PrecomputedModels/randomForest/%s-%d' %(dataset, prevQ), randomForest)
    writeToPickleFile('./PrecomputedModels/lambdaMart/%s-%d' %(dataset, prevQ), lambdaMart4)
    writeToPickleFile('./PrecomputedModels/logReg/%sScaled-%d' %(dataset, prevQ), logregScale)
    writeToPickleFile('./PrecomputedModels/randomForest/%sScaled-%d' %(dataset, prevQ), randomForestScale)
    writeToPickleFile('./PrecomputedModels/lambdaMart/%sScaled-%d' %(dataset, prevQ), lambdaMartScale)
    #writeToPickleFile('./PrecomputedModels/scaler/%s-%d' %(dataset, prevQ), scaler)
else:
  for pretrainedModel in pretrainedModels:
    if os.path.isfile(pretrainedModel):
      with open(pretrainedModel, 'rb') as file:
        a = file.read()
        b = zlib.decompress(a)
        models += [pickle.loads(b)]
      print(pretrainedModel, "READ")

feature_importance = False
if feature_importance:
  importance = models[len(models)].feature_importances_

  for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))

def predict_values(X, classifier, model, allDistsList, scalerBool=False): 
  df = [[]] * len(X)
  if scalerBool:
    for q in range(len(X)):
      df[q] = pd.DataFrame({"value": classifier(model, scaler.transform(X[q])), "distractor": allDistsList}).sort_values(by="value")
  else:
    for q in range(len(X)):
      df[q] = pd.DataFrame({"value": classifier(model, X[q]), "distractor": allDistsList}).sort_values(by="value")
  return df

def sklearnPredict(model, val): 
  return np.array(model.predict_proba(val)).T[1]

def xgboostPredict(model, val):
  return model.predict(val)

XValid = X_from_files(validFileNames)
XTest = X_from_files(testFileNames)

dfDict = {}
"""dfDict["Logistic Regression"] = predict_values(XValid, sklearnPredict, models[0], allDistsListB, True) 
fDict["Random Forest"] = predict_values(XValid, sklearnPredict, models[1],allDistsListB, False)
dfDict["LambdaMART"] = predict_values(XValid, xgboostPredict, models[2], allDistsListB, True)

print("t =", Evaluators.tTestMAP(dfDict["Logistic Regression"], dfDict["Random Forest"], 10, allDistsB, 1000))
print("t =", Evaluators.tTestMAP(dfDict["Logistic Regression"], dfDict["LambdaMART"], 10, allDistsB, 1000))
print("t =", Evaluators.tTestMAP(dfDict["Random Forest"], dfDict["LambdaMART"], 10, allDistsB, 1000))

#print(dfDict["Logistic Regression"][0:10])
#print(dfDict["Random Forest"][0:10])
#print(dfDict["LambdaMART"][0:10])

Evaluators.calculate_metrics(dfDict, allDistsB, "Validate Set")"""

"""dfDictB = {}
dfDictB["Logistic Regression"] = predict_values(XTest, sklearnPredict, models[0], allDistsListC, True)
dfDictB["Random Forest"] = predict_values(XTest, sklearnPredict, models[1], allDistsListC, False)
dfDictB["LambdaMART"] = predict_values(XTest, xgboostPredict, models[2], allDistsListC, True)"""

dfDictB = {}
for i in range(0, len(models)):
  dfDictB["Logistic Regression %d" %i] = predict_values(XTest, sklearnPredict, models[i], allDistsListC, True)

#print("t =", Evaluators.tTestMAP(dfDictB["Logistic Regression"], dfDictB["Random Forest"], 10, allDistsC, 1000))
#print("t =", Evaluators.tTestMAP(dfDictB["Logistic Regression"], dfDictB["LambdaMART"], 10, allDistsC, 1000))
#print("t =", Evaluators.tTestMAP(dfDictB["Random Forest"], dfDictB["LambdaMART"], 10, allDistsC, 1000))

#print(dfDictB["Logistic Regression"][0:10])
#print(dfDictB["Random Forest"][0:10])
#print(dfDictB["LambdaMART"][0:10])

Evaluators.calculate_metrics(dfDictB, allDistsC, "Test Set")