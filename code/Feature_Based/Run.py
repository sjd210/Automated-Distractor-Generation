import Read
import Extractors as Extractors
import Evaluators as Evaluators
import numpy as np
import pandas as pd
import xgboost
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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

def dictToTokenLists(dict, rank):
  tokenQuestion = []
  tokenAnswer = []
  tokenDist = []
  
  question, answer, allDistsList, allDists = Read.extract_all_fields(dict, rank)

  for i in range(rank):
    print("= ", i)
    tokenAnswer += [tokenise(answer[i])]
    tokenQuestion += [tokenise(question[i])]
    tokenDist += [[[], [], []]]
    for j in range(3):
      tokenDist[i][j] = tokenise(str(allDists[i][j]))
  
  return ((tokenQuestion, tokenAnswer, tokenDist), (question, answer, allDistsList, allDists))

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
      else:
        print("Failed to read", fileName)
  else:
    X2 = [[]] * (rank)

  X2 = [[]] * rank

  (tokenQuestion, tokenAnswer, tokenDist), (question, answer, allDistsList, allDists) = tokens, strings

  for q in range(savedSize, rank):
    X = {}
    for col in col_names:
      X[col] = []
    for i in range(len(question)):
      for j in range(3):
        X["emb_sim_qd"] += [Extractors.embedding_sim(tokenQuestion[q], tokenDist[i][j])]
        X["emb_sim_ad"] += [Extractors.embedding_sim(tokenAnswer[q], tokenDist[i][j])] # warning + 0 if not word in dictionary
        X["pos_sim_ad"] += [Extractors.pos_similarity(tokenAnswer[q], tokenDist[i][j])]
        X["edit_dist"] += [Extractors.edit_distance(answer[q], allDists[i][j])]
        X["token_sim_qd"] += [Extractors.token_similarity(tokenQuestion[q], tokenDist[i][j])] # very lower coeff (?)
        X["token_sim_ad"] += [Extractors.token_similarity(tokenAnswer[q], tokenDist[i][j])]
        X["character_length_d"] += [Extractors.character_length(allDists[i][j])] # notSpecific
        X["character_length_diff"] += [Extractors.character_length_diff(answer[q], allDists[i][j])] 
        X["token_length_d"] += [Extractors.token_length(tokenDist[i][j])] # notSpecific
        X["token_length_diff"] += [Extractors.token_length_diff(tokenAnswer[q], tokenDist[i][j])]
        X["abs_comm_suffix"] += [Extractors.longest_suffix_length(answer[q], allDists[i][j])]
        X["rel_comm_suffix"] += [Extractors.longest_suffix_length(answer[q], allDists[i][j])/len(allDists[i][j])] 
        X["word_freq_d"] += [Extractors.word_freq(tokenDist[i][j], sciqSupportDict, sciqSupportSum)] # notSpecific
        X["word_freq_comp_ad"] += [Extractors.word_freq_comp(tokenAnswer[q], tokenDist[i][j], sciqSupportDict, sciqSupportSum)] # NEWWWW
        X["word_freq_arc_d"] += [Extractors.word_freq(tokenDist[i][j], arcDict, arcDictSum)] # NEWWWW
        X["word_freq_comp_arc_ad"] += [Extractors.word_freq_comp(tokenAnswer[q], tokenDist[i][j], arcDict, arcDictSum)] # NEWWWW
        X["sing_plur"] += [Extractors.singular_plural_consistency(tokenAnswer[q], tokenDist[i][j])]
        X["number_d"] += [Extractors.does_string_contain_number(tokenDist[i][j])] # notSpecific
        X["number_diff"] += [Extractors.number_consistency(tokenAnswer[q], tokenDist[i][j])]
        X["wikisim_entity"] += [Extractors.wiki_sim_entity(answer[q], allDists[i][j], wiki2vec2)]
        X["wikisim_ad"] += [Extractors.wiki_sim_base_similarity(tokenAnswer[q], tokenDist[i][j], wiki2vec2)]
        X["wikisim_qd"] += [Extractors.wiki_sim_base_similarity(tokenQuestion[q], tokenDist[i][j], wiki2vec2)] # NEWWWW
    X2[q] = pd.DataFrame.from_dict(X)
    print(q)
    if ((q+1) % 500 == 0 and q != 0):
      with open(generate_file_name("progressive"), 'wb') as file:
        pickledData = pickle.dumps(X2[(q-499):(q+1)])
        compressedPickle = zlib.compress(pickledData)
        file.write(compressedPickle)
      X2 = [[]] * rank
    elif ((q+1) == rank):
      with open(generate_file_name("progressive"), 'wb') as file:
        pickledData = pickle.dumps(X2[(rank - (rank % 500)):rank])
        compressedPickle = zlib.compress(pickledData)
        file.write(compressedPickle)
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
  sciqTrainDict = Read.open_json_as_dict("/SciQ dataset-2 3/train.json")
  sciqValidateDict = Read.open_json_as_dict("/SciQ dataset-2 3/valid.json")
  sciqTestDict = Read.open_json_as_dict("/SciQ dataset-2 3/test.json")

  rankTrain = len(sciqTrainDict)
  rankValidate = len(sciqValidateDict)
  rankTest = len(sciqTestDict)

  question, answer, allDistsList, allDists = Read.extract_all_fields(sciqTrainDict, rankTrain)
  questionB, answerB, allDistsListB, allDistsB = Read.extract_all_fields(sciqValidateDict, rankValidate)
  questionC, answerC, allDistsListC, allDistsC = Read.extract_all_fields(sciqTestDict, rankTest)

  sciqTrainFileNames = []
  for i in range(1,6): #25):
    sciqTrainFileNames += ["PrecomputedFeatures/sciqTrain{0}".format(i)]
  sciqValidFileNames = ["PrecomputedFeatures/sciqValid1", "PrecomputedFeatures/sciqValid2"]
  sciqTestFileNames = ["PrecomputedFeatures/sciqTest1", "PrecomputedFeatures/sciqTest2"]

  XValid = X_from_files(sciqValidFileNames)
  XTest = X_from_files(sciqTestFileNames)

if dataset == "arcCombined":
  arcCombinedTrainDict = Read.open_json_as_dict("/ARC-Combined/train.json")
  arcCombinedValidateDict = Read.open_json_as_dict("/ARC-Combined/valid.json")
  arcCombinedTestDict = Read.open_json_as_dict("/ARC-Combined/test.json")

  rankTrain = len(arcCombinedTrainDict)
  rankValidate = len(arcCombinedValidateDict)
  rankTest = len(arcCombinedTestDict)

  question, answer, allDistsList, allDists = Read.extract_all_fields(arcCombinedTrainDict, rankTrain)
  questionB, answerB, allDistsListB, allDistsB = Read.extract_all_fields(arcCombinedValidateDict, rankValidate)
  questionC, answerC, allDistsListC, allDistsC = Read.extract_all_fields(arcCombinedTestDict, rankTest)

  arcCombinedTrainFileNames = []
  arcCombinedValidFileNames = ["PrecomputedFeatures/arcCombinedValid1", "PrecomputedFeatures/arcCombinedValid2"]
  arcCombinedTestFileNames = ["PrecomputedFeatures/arcCombinedTest1", "PrecomputedFeatures/arcCombinedTest2"]

  #XValid = X_from_files(arcCombinedValidFileNames)
  #XTest = X_from_files(arcCombinedTestFileNames)

if dataset == "arcEasy":
  arcEasyTrainDict = Read.open_json_as_dict("/ARC-Easy-2/train.json")
  arcEasyValidateDict = Read.open_json_as_dict("/ARC-Easy-2/valid.json")
  arcEasyTestDict = Read.open_json_as_dict("/ARC-Easy-2/test.json")

  rankTrain = len(arcEasyTrainDict)
  rankValidate = len(arcEasyValidateDict)
  rankTest = len(arcEasyTestDict)

  question, answer, allDistsList, allDists = Read.extract_all_fields(arcEasyTrainDict, rankTrain)
  questionB, answerB, allDistsListB, allDistsB = Read.extract_all_fields(arcEasyValidateDict, rankValidate)
  questionC, answerC, allDistsListC, allDistsC = Read.extract_all_fields(arcEasyTestDict, rankTest)

  arcEasyTrainFileNames = ["PrecomputedFeatures/arcEasyTrain1", "PrecomputedFeatures/arcEasyTrain2", "PrecomputedFeatures/arcEasyTrain3", "PrecomputedFeatures/arcEasyTrain4", "PrecomputedFeatures/arcEasyTrain5"]
  arcEasyValidFileNames = ["PrecomputedFeatures/arcEasyValid1", "PrecomputedFeatures/arcEasyValid2"]
  arcEasyTestFileNames = ["PrecomputedFeatures/arcEasyTest1", "PrecomputedFeatures/arcEasyTest2", "PrecomputedFeatures/arcEasyTest3", "PrecomputedFeatures/arcEasyTest4", "PrecomputedFeatures/arcEasyTest5"]

  #XValid = X_from_files(arcEasyValidFileNames)
  #XTest = X_from_files(arcEasyTestFileNames)

if dataset == "arcChallenge":
  arcChallengeTrainDict = Read.open_json_as_dict("/ARC-Challenge-2/train.json")
  arcChallengeValidateDict = Read.open_json_as_dict("/ARC-Challenge-2/valid.json")
  arcChallengeTestDict = Read.open_json_as_dict("/ARC-Challenge-2/test.json")

  rankTrain = len(arcChallengeTrainDict)
  rankValidate = len(arcChallengeValidateDict)
  rankTest = len(arcChallengeTestDict)

  question, answer, allDistsList, allDists = Read.extract_all_fields(arcChallengeTrainDict, rankTrain)
  questionB, answerB, allDistsListB, allDistsB = Read.extract_all_fields(arcChallengeValidateDict, rankValidate)
  questionC, answerC, allDistsListC, allDistsC = Read.extract_all_fields(arcChallengeTestDict, rankTest)

  arcChallengeTrainFileNames = ["PrecomputedFeatures/arcChallengeTrain1", "PrecomputedFeatures/arcChallengeTrain2", "PrecomputedFeatures/arcChallengeTrain3"]
  arcChallengeValidFileNames = ["PrecomputedFeatures/arcChallengeValid1"]
  arcChallengeTestFileNames = ["PrecomputedFeatures/arcChallengeTest1", "PrecomputedFeatures/arcChallengeTest2", "PrecomputedFeatures/arcChallengeTest3"]

  XValid = X_from_files(arcChallengeValidFileNames)
  XTest = X_from_files(arcChallengeTestFileNames)

col_names = ["emb_sim_qd", "emb_sim_ad", "pos_sim_ad", "edit_dist", "token_sim_qd", "token_sim_ad", "character_length_d",
             "character_length_diff", "token_length_d", "token_length_diff", "abs_comm_suffix", "rel_comm_suffix", "word_freq_d",
             "word_freq_comp_ad", "word_freq_arc_d", "word_freq_comp_arc_ad",
             "sing_plur", "number_d", "number_diff", "wikisim_entity", "wikisim_ad", "wikisim_qd"]

logreg = LogisticRegression(solver = 'newton-cg', max_iter=10000) #rank)
randomForest = RandomForestClassifier(n_estimators=1000) # set number of trees to 500
lambdaMart4 = xgboost.XGBRanker(tree_method="hist",  objective="rank:ndcg")#, lambdarank_num_pair_per_sample=8, lambdarank_pair_method="topk")

from datetime import datetime

def generate_file_name(name):
  dt_string = datetime.now().strftime("%d.%m.%Y.%H.%M.%S")
  return "PrecomputedFeatures/" + name + dt_string

writing = False
if writing:
  #tokens, strings = dictToTokenLists(sciqTrainDict, rankTrain)
  #XTrain = dictToX(sciqTrainFileNames, rankTrain, col_names, tokens, strings) 

  #tokens, strings = dictToTokenLists(sciqValidateDict, rankValidate)
  #XValid = dictToX([], rankValidate, col_names, tokens, strings)

  #tokens, strings = dictToTokenLists(sciqTestDict, rankTest)
  #XTest = dictToX([], rankTest, col_names, tokens, strings)

  tokens, strings = dictToTokenLists(arcCombinedTrainDict, rankTrain)
  XTrain = dictToX([], rankTrain, col_names, tokens, strings) 

  print(1/0)

  #tokens, strings = dictToTokenLists(arcCombinedValidateDict, rankValidate2)
  #XValid = dictToX([], rankValidate2, col_names, tokens, strings)

  #tokens, strings = dictToTokenLists(arcCombinedTestDict, rankTest2)
  #XTest = dictToX([], rankTest2, col_names, tokens, strings)

            # 0                                                    1                                                      2                                                    3                                                     4                                                     5                                                      6                                                    7                                                     8                                                     9         # no 22?                                    10                                                    11                                                    12                                                    13                                                    14                                                    15                                                    16                                                    17                                                    18                                                    19                                                    20                                                    21                                                    22                                                   # no end 25? (wait - where is that even from?)                       
#fileNames = ["PrecomputedFeatures/progressive31.01.2024.23.02.50", "PrecomputedFeatures/progressive31.01.2024.23.37.31", "PrecomputedFeatures/progressive01.02.2024.00.07.38", "PrecomputedFeatures/progressive01.02.2024.00.36.52", "PrecomputedFeatures/progressive01.02.2024.01.06.40", "PrecomputedFeatures/progressive01.02.2024.01.37.05", "PrecomputedFeatures/progressive01.02.2024.02.07.11", "PrecomputedFeatures/progressive01.02.2024.02.37.03", "PrecomputedFeatures/progressive01.02.2024.09.42.30", "PrecomputedFeatures/progressive01.02.2024.10.11.56", "PrecomputedFeatures/progressive01.02.2024.11.41.26", "PrecomputedFeatures/progressive01.02.2024.12.19.21", "PrecomputedFeatures/progressive01.02.2024.12.49.22", "PrecomputedFeatures/progressive01.02.2024.13.19.21", "PrecomputedFeatures/progressive01.02.2024.13.53.32", "PrecomputedFeatures/progressive01.02.2024.14.43.14", "PrecomputedFeatures/progressive01.02.2024.15.16.55", "PrecomputedFeatures/progressive01.02.2024.15.50.32", "PrecomputedFeatures/progressive01.02.2024.16.22.05", "PrecomputedFeatures/progressive01.02.2024.16.51.51", "PrecomputedFeatures/progressive01.02.2024.17.21.25", "PrecomputedFeatures/progressive01.02.2024.18.05.10", "PrecomputedFeatures/progressive01.02.2024.18.16.49"]
#fileNames = ["PrecomputedFeatures/progressive31.01.2024.23.02.50", "PrecomputedFeatures/progressive31.01.2024.23.37.31", "PrecomputedFeatures/progressive01.02.2024.00.07.38", "PrecomputedFeatures/progressive01.02.2024.00.36.52", "PrecomputedFeatures/progressive01.02.2024.01.06.40", "PrecomputedFeatures/progressive01.02.2024.01.37.05", "PrecomputedFeatures/progressive01.02.2024.02.07.11", "PrecomputedFeatures/progressive01.02.2024.02.37.03", "PrecomputedFeatures/progressive01.02.2024.11.41.26", "PrecomputedFeatures/progressive01.02.2024.12.19.21", "PrecomputedFeatures/progressive01.02.2024.12.49.22", "PrecomputedFeatures/progressive01.02.2024.13.19.21", "PrecomputedFeatures/progressive01.02.2024.13.53.32", "PrecomputedFeatures/progressive01.02.2024.14.43.14", "PrecomputedFeatures/progressive01.02.2024.15.16.55", "PrecomputedFeatures/progressive01.02.2024.15.50.32", "PrecomputedFeatures/progressive01.02.2024.16.22.05", "PrecomputedFeatures/progressive01.02.2024.16.51.51", "PrecomputedFeatures/progressive01.02.2024.17.21.25", "PrecomputedFeatures/progressive01.02.2024.18.05.10", "PrecomputedFeatures/progressive07.02.2024.22.05.11"]

totLen = 0 # [-1][0] == [-2][-1]
#fileNames = ["PrecomputedFeatures/progressive31.01.2024.23.02.50"]
prevQ = 0
loopStopper = 0

pretrainedModels = ["PrecomputedModels/logReg", "PrecomputedModels/randomForest", "PrecomputedModels/lambdaMart"]
pretrainedModels = []

y = [[]] * rankTrain
for q in range(prevQ, rankTrain):
  y[q] = [0] * (rankTrain*3)
  y[q][q*3 + 0] = 1
  y[q][q*3 + 1] = 1
  y[q][q*3 + 2] = 1

if len(pretrainedModels) == 0:
  for fileName in sciqTrainFileNames:
    if os.path.isfile(fileName) and (loopStopper < rankTrain):
      with open(fileName, 'rb') as file:
        a = file.read()
        b = zlib.decompress(a)
        c = pickle.loads(b)
      print(fileName, "READ")

    """logreg.fit(X, y)
  randomForest.fit(X, y)
  qida = [0] * (rank*3) * (rank)
  for i in range(rank):
    for j in range(rank*3):
      qida[i*rank*3 + j] = i
  #ya = np.array(y[prevQ:loopStopper]).flatten()
  ya = np.array(y).flatten()
  xa = np.array(X).reshape((-1,len(col_names)))
  lambdaMart4.fit(xa, ya, qid=qida)"""
    loopStopper = min((prevQ+len(c)), rankTrain)
    used = False
    loopSize = loopStopper-prevQ
    
    X = [[]] * loopSize
    for q in range(prevQ, loopStopper):
      used = True
      print("train", q)

      X[q-prevQ] = c[q-prevQ][:(rankTrain*3)]
      logreg.fit(X[q-prevQ], y[q])
      randomForest.fit(X[q-prevQ], y[q])
    
    if used:
      qida = [0] * (rankTrain*3) * (loopSize)
      val = prevQ
      for i in range(loopSize):
        for j in range(rankTrain*3):
          qida[i*rankTrain*3 + j] = val
        val += 1

      ya = np.array(y[prevQ:loopStopper]).flatten()
      xa = np.array(X).reshape((-1,len(col_names)))
      lambdaMart4.fit(xa, ya, qid=qida)
      print("Lambda Done")
      #logreg.fit(xa, ya)
      #print("Logit Done")
      #randomForest.fit(xa, ya)
      #print("Random Forest Done")
    
    prevQ += loopSize

  with open("./PrecomputedModels/logReg2", 'wb') as file:
    pickledData = pickle.dumps(logreg)
    compressedPickle = zlib.compress(pickledData)
    file.write(compressedPickle)

  with open("./PrecomputedModels/randomForest2", 'wb') as file:
    pickledData = pickle.dumps(randomForest)
    compressedPickle = zlib.compress(pickledData)
    file.write(compressedPickle)

  with open("./PrecomputedModels/lambdaMart2", 'wb') as file:
    pickledData = pickle.dumps(lambdaMart4)
    compressedPickle = zlib.compress(pickledData)
    file.write(compressedPickle)
else:
  if os.path.isfile(pretrainedModels[0]):
    with open(pretrainedModels[0], 'rb') as file:
      a = file.read()
      b = zlib.decompress(a)
      logreg = pickle.loads(b)
    print(pretrainedModels[0], "READ")

  if os.path.isfile(pretrainedModels[1]):
    with open(pretrainedModels[1], 'rb') as file:
      a = file.read()
      b = zlib.decompress(a)
      randomForest = pickle.loads(b)
    print(pretrainedModels[1], "READ")

  if os.path.isfile(pretrainedModels[2]):
    with open(pretrainedModels[2], 'rb') as file:
      a = file.read()
      b = zlib.decompress(a)
      lambdaMart4 = pickle.loads(b)
    print(pretrainedModels[2], "READ")

def predict_values(X, rank, classifier, allDistsList): # remove X2 and allDistsList as parameters
  df = [[]] * rank

  for q in range(rank):
    a = classifier(X[q]) # SELECT 100 negative samples and the positive one instead
    df[q] = pd.DataFrame({"value": classifier(X[q]), "distractor": allDistsList}).sort_values(by="value")
  return df

def calculate_metrics(dfDict, allDists, setName):
  print("====== ", setName, " =======")

  for name in dfDict:
    print("======")
    print(name)
    print("R@10: ", Evaluators.meanRecallAt(dfDict[name], 10, allDists))
    print("P@1: ", Evaluators.meanPrecisionAt(dfDict[name], 1, allDists))
    print("P@3: ", Evaluators.meanPrecisionAt(dfDict[name], 3, allDists))
    print("MAP@10: ", Evaluators.meanAveragePrecisionAt(dfDict[name], 10, allDists))
    print("NDCG@10: ", Evaluators.meanNdcgAt(dfDict[name], 10, allDists))
    print("MRR: ", Evaluators.meanReciprocalRank(dfDict[name], allDists))

def logRegPredict(val): 
  #return np.array(logit2prob(logreg, val))
  return np.array(logreg.predict_proba(val)).T[1]

def ranForPredict(val): 
  return np.array(randomForest.predict_proba(val)).T[1]

dfDict = {}
dfDict["Logistic Regression"] = predict_values(XValid, rankValidate, logRegPredict, allDistsListB) #
dfDict["Random Forest"] = predict_values(XValid, rankValidate, ranForPredict, allDistsListB)
dfDict["LambdaMART"] = predict_values(XValid, rankValidate, lambdaMart4.predict, allDistsListB)

print(dfDict["Logistic Regression"][0:10])
print(dfDict["Random Forest"][0:10])
print(dfDict["LambdaMART"][0:10])

calculate_metrics(dfDict, allDistsB, "Validate Set")

dfDictB = {}
dfDictB["Logistic Regression"] = predict_values(XTest, rankTest, logRegPredict, allDistsListC)
dfDictB["Random Forest"] = predict_values(XTest, rankTest, ranForPredict, allDistsListC)
dfDictB["LambdaMART"] = predict_values(XTest, rankTest, lambdaMart4.predict, allDistsListC)

print(dfDictB["Logistic Regression"][0:10])
print(dfDictB["Random Forest"][0:10])
print(dfDictB["LambdaMART"][0:10])

calculate_metrics(dfDictB, allDistsC, "Test Set")

"""dfB = [[]] * rank2 # FROM HERE, REDEFINE
df2B = [[]] * rank2
df3B = [[]] * rank2

for q in range(rank2):
  print("eval2", q)
  a = np.array(logit2prob(logreg, X3[q]))
  b = np.array(randomForest.predict_proba(X3[q])).T[1]
  c = lambdaMart.predict(X3[q])
  dfB[q] = pd.DataFrame({"value": a, "distractor": allDistsList}).sort_values(by="value")
  df2B[q] = pd.DataFrame({"value": b, "distractor": allDistsList}).sort_values(by="value")
  df3B[q] = pd.DataFrame({"value": c, "distractor": allDistsList}).sort_values(by="value")

calculate_metrics(dfB, df2B, df3B, allDists, "Test Set")"""