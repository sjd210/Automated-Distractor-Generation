import TestRead
import TestCompareDistractors
import TestEvaluate
import numpy as np
import sklearn
import pandas as pd
import spacy
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from gensim.models import KeyedVectors

def logit2prob(logr, X):
  log_odds = logr.intercept_[0]
  for i in range(len(col_names)):
    log_odds += logr.coef_[0][i] * X[col_names[i]]
  odds = np.exp(log_odds)
  probability = odds / (1 + odds)
  return(probability)

dict = TestRead.open_json_as_dict("train.json")
question = TestRead.extract_field("question", dict)[0:100]
dist1 = TestRead.extract_field("distractor1", dict)
dist2 = TestRead.extract_field("distractor2", dict)
dist3 = TestRead.extract_field("distractor3", dict)
answer = TestRead.extract_field("correct_answer", dict)
support = TestRead.extract_field("support", dict)

allSupport = TestRead.combine_evidence(dict)
allSupportDict = TestRead.count_text(allSupport)

allDistsList = [val for a in zip(dist1, dist2, dist3) for val in a][0:len(question)*3]
allDists = np.transpose([dist1,dist2,dist3])

col_names = ["emb_sim_qd", "emb_sim_ad", "pos_sim_ad", "edit_dist", "token_sim_qd", "token_sim_ad", "character_length_a", "character_length_d",
             "character_length_diff", "token_length_a", "token_length_d", "token_length_diff", "abs_comm_suffix", "rel_comm_suffix", "word_freq_a", "word_freq_d",
             "sing_plur", "number_d", "number_diff", "wikisim"]

tokenAnswer = []
tokenQuestion = []
tokenDist = []
tokenise = spacy.load("en_core_web_lg")
#wiki2vec2 = KeyedVectors.load_word2vec_format("WikipediaEmbeddings/enwiki_20180420_100d.txt", binary = False)
wiki2vec2 = KeyedVectors.load_word2vec_format("WikipediaEmbeddings/small_test.txt", binary = False)

for i in range(len(question)):
  print("= ", i)
  tokenAnswer += [tokenise(answer[i])]
  tokenQuestion += [tokenise(question[i])]
  tokenDist += [[[], [], []]]
  for j in range(3):
    tokenDist[i][j] = tokenise(str(allDists[i][j])) # pretokenising - maybe I should store this in a file and just read that?

logreg = LogisticRegression(random_state=1, max_iter=10000)
randomForest = RandomForestClassifier(max_depth=2, random_state=1)
lambdaMart = xgboost.XGBRanker(tree_method="hist", lambdarank_num_pair_per_sample=8, objective="rank:ndcg", lambdarank_pair_method="topk")

X2 = [[]] * len(question)
y = [[]] * len(question)
for q in range(len(question)):
  X = {}
  for col in col_names:
    X[col] = []
  for i in range(len(question)):
    for j in range(3):
      X["emb_sim_qd"] += [TestCompareDistractors.embedding_sim(tokenQuestion[q], tokenDist[i][j])]
      X["emb_sim_ad"] += [TestCompareDistractors.embedding_sim(tokenAnswer[q], tokenDist[i][j])]
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
  #print(X2)
  print(q)

  y[q] = [0] * (len(question)*3)
  y[q][q*3 + 0] = 1
  y[q][q*3 + 1] = 1
  y[q][q*3+ 2] = 1
 #  y2 = pd.DataFrame.from_dict(y)

  logreg.fit(X2[q], y[q])
  randomForest.fit(X2[q], y[q])
  # lambdaMart.fit(X2[q],y)
  # can mostly use GradientBoostingRegressor for LambdaMART?

"""
  y = {"Outcome": [0] * (len(question)*3)}
  y["Outcome"][q*3 + 0] = 1
  y["Outcome"][q*3 + 1] = 1
  y["Outcome"][q*3+ 2] = 1 """

# print(logreg.coef_)

# print(X2[2])

a = [[]] * len(question)
b = [[]] * len(question)
df = [[]] * len(question)
df2 = [[]] * len(question)
top1 = [[]] * len(question)
top3 = [[]] * len(question)

for q in range(len(question)):
  a[q] = np.array(logit2prob(logreg, X2[q]))
  b[q] = np.array(randomForest.predict_proba(X2[q])).T[1]
  # c = np.array(lambdaMart.predict_proba(X2[q])).T[1]
  df[q] = pd.DataFrame({"value": a[q], "distractor": allDistsList}).sort_values(by="value")
  df2[q] = pd.DataFrame({"value": b[q], "distractor": allDistsList}).sort_values(by="value")
  # df3 = pd.DataFrame({"value": c, "distractor": allDistsList}).sort_values(by="value")
  print(df[q].to_string())
  # print(df2[q])
  #  df[q].iloc[len(question)*3-3]['distractor'] print(df3)
  print(question[q])
  print(allDists[q])
  print(answer[q]) # we should bar the correct answer from showing up here

# y_pred = logreg.predict(X2)
# print(y_pred)

print(TestEvaluate.meanAveragePrecisionAt(df, 1, allDists))
print(TestEvaluate.meanAveragePrecisionAt(df, 3, allDists))
print(TestEvaluate.meanAveragePrecisionAt(df, 10, allDists))
print(TestEvaluate.recallAt(df[10], 10, 10, allDists))
print(TestEvaluate.meanReciprocalRank(df, allDists))
print(TestEvaluate.ndcgAt(df[10], 10, 10, allDists))

# https://medium.com/the-rise-of-unbelievable/what-is-evaluation-metrics-and-when-to-use-which-metrics-23d16446d690 
# PRECISION@k 
# For evaluation, we report top recall (R@10), precision (P@1, P@3), mean average precision (MAP@10), 
# normalized discounted cumulative gain (NDCG@10), and mean reciprocal rank (MRR).
