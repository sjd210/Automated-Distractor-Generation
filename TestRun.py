import TestRead
import TestCompareDistractors
import TestEvaluate
import numpy as np
import pandas as pd
import spacy
import xgboost
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from gensim.models import KeyedVectors
from wikipedia2vec import Wikipedia2Vec

import zlib
import ast
import pickle

print("Loading Wikipedia Embeddings")
wiki2vec2 = KeyedVectors.load_word2vec_format("WikipediaEmbeddings/small_test.txt", binary = False)
#wiki2vec2 = KeyedVectors.load_word2vec_format("WikipediaEmbeddings/enwiki_20180420_100d.txt", binary = False)
#wiki2vec2 = KeyedVectors.load_word2vec_format(pickle.loads("WikipediaEmbeddings/enwiki_20180420_100d.pkl"), binary = True) # TRY FIX THIS
#wiki2vec2 = Wikipedia2Vec.load("WikipediaEmbeddings/enwiki_20180420_100d.pkl")
print("Loading SpaCy Tokenisations")
tokenise = spacy.load("en_core_web_lg")

def logit2prob(logr, X):
  log_odds = logr.intercept_[0]
  for i in range(len(col_names)):
    log_odds += logr.coef_[0][i] * X[col_names[i]]
  odds = np.exp(log_odds)
  probability = odds / (1 + odds)
  return(probability)

def dictToTokenLists(dict, rank):
  tokenQuestion = []
  tokenAnswer = []
  tokenDist = []
  
  question, answer, allDistsList, allDists, allSupportDict = TestRead.extract_all_fields(dict, rank)

  for i in range(rank):
    print("= ", i)
    tokenAnswer += [tokenise(answer[i])]
    tokenQuestion += [tokenise(question[i])]
    tokenDist += [[[], [], []]]
    for j in range(3):
      tokenDist[i][j] = tokenise(str(allDists[i][j])) # pretokenising - maybe I should store this in a file and just read that?
  
  return ((tokenQuestion, tokenAnswer, tokenDist), (question, answer, allDistsList, allDists, allSupportDict))

def dictToX(fileNames, rank, col_names, tokens, strings):
  X2 = []
  """for fileName in fileNames:
    if os.path.isfile(fileName):
      with open(fileName, 'rb') as file:
        a = file.read()
        b = zlib.decompress(a)
        c = pickle.loads(b)
        X2 += c
        print("file READ")"""

  print("wo", len(X2), rank)
  savedSize = rank#len(X2)
  X2 = [[]] * (rank)

  (tokenQuestion, tokenAnswer, tokenDist), (question, answer, allDistsList, allDists, allSupportDict) = tokens, strings

  for q in range(savedSize, rank):
    X = {}
    for col in col_names:
      X[col] = []
    for i in range(len(question)): # TO COMPLETE FULL LIST
      for j in range(3):
        X["emb_sim_qd"] += [TestCompareDistractors.embedding_sim(tokenQuestion[q], tokenDist[i][j])]
        X["emb_sim_ad"] += [TestCompareDistractors.embedding_sim(tokenAnswer[q], tokenDist[i][j])] # warning + 0 if not word in dictionary
        X["pos_sim_ad"] += [TestCompareDistractors.pos_similarity(tokenAnswer[q], tokenDist[i][j])]
        X["edit_dist"] += [TestCompareDistractors.edit_distance(answer[q], allDists[i][j])]
        X["token_sim_qd"] += [TestCompareDistractors.token_similarity(tokenQuestion[q], tokenDist[i][j])] # very lower coeff (?)
        X["token_sim_ad"] += [TestCompareDistractors.token_similarity(tokenAnswer[q], tokenDist[i][j])]
        X["character_length_d"] += [TestCompareDistractors.character_length(allDists[i][j])] # notSpecific
        X["character_length_diff"] += [TestCompareDistractors.character_length_diff(answer[q], allDists[i][j])] 
        X["token_length_d"] += [TestCompareDistractors.token_length(tokenDist[i][j])] # notSpecific
        X["token_length_diff"] += [TestCompareDistractors.token_length_diff(tokenAnswer[q], tokenDist[i][j])]
        X["abs_comm_suffix"] += [TestCompareDistractors.longest_suffix_length(answer[q], allDists[i][j])]
        X["rel_comm_suffix"] += [TestCompareDistractors.longest_suffix_length(answer[q], allDists[i][j])/len(allDists[i][j])] 
        X["word_freq_d"] += [TestCompareDistractors.word_freq(tokenDist[i][j], allSupportDict)] # notSpecific
        X["sing_plur"] += [TestCompareDistractors.singular_plural_consistency(tokenAnswer[q], tokenDist[i][j])]
        X["number_d"] += [TestCompareDistractors.does_string_contain_number(tokenDist[i][j])] # notSpecific
        X["number_diff"] += [TestCompareDistractors.number_consistency(tokenAnswer[q], tokenDist[i][j])]
        X["wikisim_entity"] += [TestCompareDistractors.wiki_sim_entity(answer[q], allDists[i][j], wiki2vec2)]
        X["wikisim"] += [TestCompareDistractors.wiki_sim_base_similarity(tokenAnswer[q], tokenDist[i][j], wiki2vec2)]
    X2[q] = pd.DataFrame.from_dict(X)
    print(q)
    if (q % 500 == 1 and q != 1):
      with open(generate_file_name("progressive"), 'wb') as file:
        pickledData = pickle.dumps(X2[(q-500):q])
        compressedPickle = zlib.compress(pickledData)
        file.write(compressedPickle)
    if (q == rank-1):
      with open(generate_file_name("progressive"), 'wb') as file:
        pickledData = pickle.dumps(X2[(rank - (rank % 500)):rank])
        compressedPickle = zlib.compress(pickledData)
        file.write(compressedPickle)
  return X2

trainDict = TestRead.open_json_as_dict("train.json")
validateDict = TestRead.open_json_as_dict("valid.json")
testDict = TestRead.open_json_as_dict("test.json")

rank = 1000#len(trainDict)
rank2 = 1000
rank3 = 1000

col_names = ["emb_sim_qd", "emb_sim_ad", "pos_sim_ad", "edit_dist", "token_sim_qd", "token_sim_ad", "character_length_d",
             "character_length_diff", "token_length_d", "token_length_diff", "abs_comm_suffix", "rel_comm_suffix", "word_freq_d",
             "sing_plur", "number_d", "number_diff", "wikisim_entity", "wikisim"]

logreg = LogisticRegression(random_state=1, max_iter=rank)
randomForest = RandomForestClassifier(random_state=1, n_estimators=500) # set number of trees to 500
# (we want 500 rounds of boosting in lambdaMart)
lambdaMart = xgboost.XGBRanker(tree_method="hist",  objective="rank:ndcg")#, lambdarank_num_pair_per_sample=8, lambdarank_pair_method="topk")

from datetime import datetime

def generate_file_name(name):
  dt_string = datetime.now().strftime("%d.%m.%Y.%H.%M.%S")
  return "PrecomputedFeatures/" + name + dt_string

question, answer, allDistsList, allDists, allSupportDict = TestRead.extract_all_fields(trainDict, rank)
questionB, answerB, allDistsListB, allDistsB, allSupportDictB = TestRead.extract_all_fields(validateDict, rank2)
questionC, answerC, allDistsListC, allDistsC, allSupportDictC = TestRead.extract_all_fields(testDict, rank3)
#tokens, strings = dictToTokenLists(trainDict, rank)
tokens, strings = (0,0,0), (0,0,0,0,0)
X = dictToX(["PrecomputedFeatures/progressive31.01.2024.23.02.50", "PrecomputedFeatures/progressive31.01.2024.23.37.31", "PrecomputedFeatures/progressive01.02.2024.00.07.38", "PrecomputedFeatures/progressive01.02.2024.00.36.52", "PrecomputedFeatures/progressive01.02.2024.01.06.40", "PrecomputedFeatures/progressive01.02.2024.01.37.05", "PrecomputedFeatures/progressive01.02.2024.02.07.11", "PrecomputedFeatures/progressive01.02.2024.02.37.03", "PrecomputedFeatures/progressive01.02.2024.09.42.30", "PrecomputedFeatures/progressive01.02.2024.10.11.56", "PrecomputedFeatures/progressive01.02.2024.11.41.26", "PrecomputedFeatures/progressive01.02.2024.12.19.21", "PrecomputedFeatures/progressive01.02.2024.12.49.22", "PrecomputedFeatures/progressive01.02.2024.13.19.21", "PrecomputedFeatures/progressive01.02.2024.13.53.32", "PrecomputedFeatures/progressive01.02.2024.14.43.14", "PrecomputedFeatures/progressive01.02.2024.15.16.55", "PrecomputedFeatures/progressive01.02.2024.15.50.32", "PrecomputedFeatures/progressive01.02.2024.16.22.05", "PrecomputedFeatures/progressive01.02.2024.16.51.51", "PrecomputedFeatures/progressive01.02.2024.17.21.25", "PrecomputedFeatures/progressive01.02.2024.18.05.10", "PrecomputedFeatures/progressive01.02.2024.18.16.49"], rank, col_names, tokens, strings) 
#X = dictToX([], rank, col_names, tokens, strings) 

# NEED TO SPLIT THIS INTO PROGRESSIVE LOADING FOR ML STUFF

fileName = "PrecomputedFeatures/valid31.01.2024.17.03.51"
if os.path.isfile(fileName):
    with open(fileName, 'rb') as file:
      a = file.read()
      b = zlib.decompress(a)
      X2 = pickle.loads(b)
      print("Validate File READ")
else:
  tokens, strings = dictToTokenLists(validateDict, rank2)
  X2 = dictToX([], rank2, col_names, tokens, strings)
  #X2 = dictToX("PrecomputedFeatures/valid31.01.2024.17.03.51", rank2, col_names, tokens, strings)

fileName = "PrecomputedFeatures/test31.01.2024.17.09.52"
if os.path.isfile(fileName):
    with open(fileName, 'rb') as file:
      a = file.read()
      b = zlib.decompress(a)
      X3 = pickle.loads(b)
      print("Test File READ")
else:
  tokens, strings = dictToTokenLists(testDict, rank3)
  #X3 = dictToX(generate_file_name("test"), rank3, col_names, tokens, strings)
  X3 = dictToX([], rank3, col_names, tokens, strings)

y = [[]] * rank
qid = [0] * (rank*3) * rank

for i in range(rank):
  for j in range(rank*3):
    qid[i*rank*3 + j] = i

fileNames = ["PrecomputedFeatures/progressive31.01.2024.23.02.50", "PrecomputedFeatures/progressive31.01.2024.23.37.31", "PrecomputedFeatures/progressive01.02.2024.00.07.38", "PrecomputedFeatures/progressive01.02.2024.00.36.52", "PrecomputedFeatures/progressive01.02.2024.01.06.40", "PrecomputedFeatures/progressive01.02.2024.01.37.05", "PrecomputedFeatures/progressive01.02.2024.02.07.11", "PrecomputedFeatures/progressive01.02.2024.02.37.03", "PrecomputedFeatures/progressive01.02.2024.09.42.30", "PrecomputedFeatures/progressive01.02.2024.10.11.56", "PrecomputedFeatures/progressive01.02.2024.11.41.26", "PrecomputedFeatures/progressive01.02.2024.12.19.21", "PrecomputedFeatures/progressive01.02.2024.12.49.22", "PrecomputedFeatures/progressive01.02.2024.13.19.21", "PrecomputedFeatures/progressive01.02.2024.13.53.32", "PrecomputedFeatures/progressive01.02.2024.14.43.14", "PrecomputedFeatures/progressive01.02.2024.15.16.55", "PrecomputedFeatures/progressive01.02.2024.15.50.32", "PrecomputedFeatures/progressive01.02.2024.16.22.05", "PrecomputedFeatures/progressive01.02.2024.16.51.51", "PrecomputedFeatures/progressive01.02.2024.17.21.25", "PrecomputedFeatures/progressive01.02.2024.18.05.10", "PrecomputedFeatures/progressive01.02.2024.18.16.49"]
#fileNames = ["PrecomputedFeatures/progressive31.01.2024.23.02.50"]
prevQ = 0

for q in range(rank):
  y[q] = [0] * (rank*3)
  y[q][q*3 + 0] = 1
  y[q][q*3 + 1] = 1
  y[q][q*3 + 2] = 1

X = [[]] * rank

for fileName in fileNames:
  if os.path.isfile(fileName):
    with open(fileName, 'rb') as file:
      a = file.read()
      b = zlib.decompress(a)
      c = pickle.loads(b)
      print("file READ")

      print(len(y[0]), "I AM HERE")
      loopStopper = min((prevQ+len(c)), rank)

      for q in range(prevQ, loopStopper):
        print("train", q)

        """y[q] = [0] * (rank*3)
        y[q][q*3 + 0] = 1
        y[q][q*3 + 1] = 1
        y[q][q*3 + 2] = 1"""

        X[q] = c[q-prevQ][:(rank*3)]
        # print("WOAH", X[q])
        logreg.fit(X[q], y[q])
        randomForest.fit(X[q], y[q])
        #qid = [q] * (rank*3)
        #lambdaMart.fit(X[q], y[q], qid=qid) # EITHER we want qid for each question, or all same qid?
      prevQ = prevQ + len(c)
  
print("COEF", logreg.coef_)

ya = np.array(y).flatten()
xa = np.array(X).reshape((-1,len(col_names)))
lambdaMart.fit(xa, ya, qid=qid) # EITHER we want qid for each question, or all same qid?
# we will have to figure out what to do here :((
# perhaps by implementing the multi-stage framework thing?

def predict_values(X, rank, classifier, allDistsList): # remove X2 and allDistsList as parameters
  df = [[]] * rank2

  for q in range(rank):
    #print(":(", q)
    a = classifier(X[q])
   # print(len(a), len(allDistsListB))
    df[q] = pd.DataFrame({"value": a, "distractor": allDistsList}).sort_values(by="value")

  return df

"""b = np.array(randomForest.predict_proba(X2[q])).T[1]
c = np.array(lambdaMart.predict(X2[q]))
df[q] = pd.DataFrame({"value": a, "distractor": allDistsList}).sort_values(by="value")
df2[q] = pd.DataFrame({"value": b, "distractor": allDistsList}).sort_values(by="value")
df3[q] = pd.DataFrame({"value": c, "distractor": allDistsList}).sort_values(by="value")"""
# y_pred = logreg.predict(X2)
# print(y_pred)

def calculate_metrics(df, df2, df3, allDists, setName):
  print("====== ", setName, " =======")

  print("Logistic Regression")
  print("R@10: ", TestEvaluate.meanRecallAt(df, 10, allDists))
  print("P@1: ", TestEvaluate.meanPrecisionAt(df, 1, allDists))
  print("P@3: ", TestEvaluate.meanPrecisionAt(df, 3, allDists))
  print("MAP@10: ", TestEvaluate.meanAveragePrecisionAt(df, 10, allDists))
  print("NDCG@10: ", TestEvaluate.meanNdcgAt(df, 10, allDists))
  print("MRR: ", TestEvaluate.meanReciprocalRank(df, allDists))

  print("======")

  print("Random Forest")
  print("R@10: ", TestEvaluate.meanRecallAt(df2, 10, allDists))
  print("P@1: ", TestEvaluate.meanPrecisionAt(df2, 1, allDists))
  print("P@3: ", TestEvaluate.meanPrecisionAt(df2, 3, allDists))
  print("MAP@10: ", TestEvaluate.meanAveragePrecisionAt(df2, 10, allDists))
  print("NDCG@10: ", TestEvaluate.meanNdcgAt(df2, 10, allDists))
  print("MRR: ", TestEvaluate.meanReciprocalRank(df2, allDists))

  print("======")

  print("LambdaMART")
  print("R@10: ", TestEvaluate.meanRecallAt(df3, 10, allDists))
  print("P@1: ", TestEvaluate.meanPrecisionAt(df3, 1, allDists))
  print("P@3: ", TestEvaluate.meanPrecisionAt(df3, 3, allDists))
  print("MAP@10: ", TestEvaluate.meanAveragePrecisionAt(df3, 10, allDists))
  print("NDCG@10: ", TestEvaluate.meanNdcgAt(df3, 10, allDists))
  print("MRR: ", TestEvaluate.meanReciprocalRank(df3, allDists))

def logRegPredict(val): 
  #return np.array(logit2prob(logreg, val))
  return np.array(logreg.predict_proba(val)).T[1]

def ranForPredict(val): 
  return np.array(randomForest.predict_proba(val)).T[1]

df = predict_values(X2, rank2, logRegPredict, allDistsListB)
df2 = predict_values(X2, rank2, ranForPredict, allDistsListB)
df3 = predict_values(X2, rank2, lambdaMart.predict, allDistsListB)

calculate_metrics(df, df2, df3, allDistsB, "Validate Set")

dfB = predict_values(X3, rank3, logRegPredict, allDistsListC)
df2B = predict_values(X3, rank3, ranForPredict, allDistsListC) #MAYYYYBE not using logit2prob was the reason this was underperforming?
df3B = predict_values(X3, rank3, lambdaMart.predict, allDistsListC)

calculate_metrics(dfB, df2B, df3B, allDistsC, "Test Set")

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