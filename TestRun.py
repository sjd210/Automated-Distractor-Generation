import TestRead
import TestCompareDistractors
import TestEvaluate
import numpy as np
import pandas as pd
import spacy
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from gensim.models import KeyedVectors

#wiki2vec2 = KeyedVectors.load_word2vec_format("WikipediaEmbeddings/small_test.txt", binary = False)
wiki2vec2 = KeyedVectors.load_word2vec_format("WikipediaEmbeddings/enwiki_20180420_100d.txt", binary = False)
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

def dictToX(dict, rank, col_names, tokens, strings):
  X2 = [[]] * rank

  (tokenQuestion, tokenAnswer, tokenDist), (question, answer, allDistsList, allDists, allSupportDict) = tokens, strings

  for q in range(rank):
    X = {}
    for col in col_names:
      X[col] = []
    for i in range(len(question)):
      for j in range(3):
        X["emb_sim_qd"] += [TestCompareDistractors.embedding_sim(tokenQuestion[q], tokenDist[i][j])]
        X["emb_sim_ad"] += [TestCompareDistractors.embedding_sim(tokenAnswer[q], tokenDist[i][j])] # warning + 0 if not word in dictionary
        X["pos_sim_ad"] += [TestCompareDistractors.pos_similarity(tokenAnswer[q], tokenDist[i][j])]
        X["edit_dist"] += [TestCompareDistractors.edit_distance(answer[q], allDists[i][j])]
        X["token_sim_qd"] += [TestCompareDistractors.token_similarity(tokenQuestion[q], tokenDist[i][j])]
        X["token_sim_ad"] += [TestCompareDistractors.token_similarity(tokenAnswer[q], tokenDist[i][j])]
      #  X["token_sim_qa"] += [TestCompareDistractors.token_similarity(tokenQuestion[q], tokenAnswer[q])] Maybe make a relative thing?
        X["character_length_a"] += [TestCompareDistractors.character_length(answer[q])] 
        X["character_length_d"] += [TestCompareDistractors.character_length(allDists[i][j])]
        X["character_length_diff"] += [TestCompareDistractors.character_length_diff(answer[q], allDists[i][j])] 
        X["token_length_a"] += [TestCompareDistractors.token_length(tokenAnswer[q])]
        X["token_length_d"] += [TestCompareDistractors.token_length(tokenDist[i][j])]
        X["token_length_diff"] += [TestCompareDistractors.token_length_diff(tokenAnswer[q], tokenDist[i][j])]
        X["abs_comm_suffix"] += [TestCompareDistractors.longest_suffix_length(answer[q], allDists[i][j])]
        X["rel_comm_suffix"] += [TestCompareDistractors.longest_suffix_length(answer[q], allDists[i][j])/len(allDists[i][j])] 
        X["word_freq_a"] += [TestCompareDistractors.word_freq(tokenAnswer[q], allSupportDict)]
        X["word_freq_d"] += [TestCompareDistractors.word_freq(tokenDist[i][j], allSupportDict)]
        X["sing_plur"] += [TestCompareDistractors.singular_plural_consistency(tokenAnswer[q], tokenDist[i][j])]
        X["number_d"] += [TestCompareDistractors.does_string_contain_number(tokenDist[i][j])]
        X["number_diff"] += [TestCompareDistractors.number_consistency(tokenAnswer[q], tokenDist[i][j])]
        X["wikisim"] += [TestCompareDistractors.wiki_sim_entity(answer[q], allDists[i][j], wiki2vec2)] # HERE?
    X2[q] = pd.DataFrame.from_dict(X)
    print(q)
  return X2

trainDict = TestRead.open_json_as_dict("train.json")
validateDict = TestRead.open_json_as_dict("valid.json")

rank = 1000#len(trainDict)
rank2 = 1000#len(validateDict)

col_names = ["emb_sim_qd", "emb_sim_ad", "pos_sim_ad", "edit_dist", "token_sim_qd", "token_sim_ad", "character_length_a", "character_length_d",
             "character_length_diff", "token_length_a", "token_length_d", "token_length_diff", "abs_comm_suffix", "rel_comm_suffix", "word_freq_a", "word_freq_d",
             "sing_plur", "number_d", "number_diff", "wikisim"]

logreg = LogisticRegression(random_state=1, max_iter=10000)
randomForest = RandomForestClassifier(random_state=1, n_estimators=500) # set number of trees to 500
# (we want 500 rounds of boosting in lambdaMart)
lambdaMart = xgboost.XGBRanker(tree_method="hist",  objective="rank:ndcg")#, lambdarank_num_pair_per_sample=8, lambdarank_pair_method="topk")

(tokenQuestion, tokenAnswer, tokenDist), (question, answer, allDistsList, allDists, allSupportDict) = dictToTokenLists(trainDict, rank)
tokens, strings = (tokenQuestion, tokenAnswer, tokenDist), (question, answer, allDistsList, allDists, allSupportDict)
X = dictToX(trainDict, rank, col_names, tokens, strings)

(tokenQuestion, tokenAnswer, tokenDist), (question, answer, allDistsList, allDists, allSupportDict) = dictToTokenLists(validateDict, rank2)
tokens, strings = (tokenQuestion, tokenAnswer, tokenDist), (question, answer, allDistsList, allDists, allSupportDict)
X2 = dictToX(validateDict, rank2, col_names, tokens, strings)

qid = [0] * (rank*3) * rank

for i in range(rank):
  for j in range(rank*3):
    qid[i*rank*3 + j] = i

y = [[]] * rank

for q in range(rank):
  print("train", q)

  y[q] = [0] * (rank*3)
  y[q][q*3 + 0] = 1
  y[q][q*3 + 1] = 1
  y[q][q*3+ 2] = 1

  logreg.fit(X[q], y[q])
  randomForest.fit(X[q], y[q])
  
ya = np.array(y).flatten()
xa = np.array(X).reshape((-1,20))

#print(xa[150], ya[150], qid[150])

#print("lens", len(xa), len(ya), len(qid))
lambdaMart.fit(xa, ya, qid=qid) # EITHER we want qid for each question, or all same qid?
# qid must be sorted in non-decreasing order
# can mostly use GradientBoostingRegressor for LambdaMART?
df = [[]] * rank2
df2 = [[]] * rank2
df3 = [[]] * rank2

for q in range(rank2):
  print("eval", q)
  a = np.array(logit2prob(logreg, X2[q]))
  b = np.array(randomForest.predict_proba(X2[q])).T[1]
  c = lambdaMart.predict(X2[q])
  df[q] = pd.DataFrame({"value": a, "distractor": allDistsList}).sort_values(by="value")
  df2[q] = pd.DataFrame({"value": b, "distractor": allDistsList}).sort_values(by="value")
  df3[q] = pd.DataFrame({"value": c, "distractor": allDistsList}).sort_values(by="value")
  #print(df[q].to_string())
  #print(df2[q])
  #print(allDists[q])

# y_pred = logreg.predict(X2)
# print(y_pred)
  
print("======")

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

#In addition, we also study the following unsupervised baselines that measure similarities between the key and distractors: (i) pointwise mutual
#information (PMI) based on co-occurrences; (ii) edit distance (ED), which measures the spelling similarity; and (iii) GloVe embedding similarity (Emb Sim)

# https://medium.com/the-rise-of-unbelievable/what-is-evaluation-metrics-and-when-to-use-which-metrics-23d16446d690 
# PRECISION@k 
# For evaluation, we report top recall (R@10), precision (P@1, P@3), mean average precision (MAP@10), 
# normalized discounted cumulative gain (NDCG@10), and mean reciprocal rank (MRR).
