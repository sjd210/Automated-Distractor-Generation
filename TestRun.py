import TestRead
import TestCompareDistractors
import numpy as np
from sklearn import linear_model
import pandas as pd
import spacy
from sklearn.linear_model import LogisticRegression

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
# support = TestRead.extract_field("correct_answer", dict)

all_dists_list = [val for a in zip(dist1, dist2, dist3) for val in a][0:len(question)*3]
all_dists = np.transpose([dist1,dist2,dist3])
# print(all_dists)

col_names = ["emb_sim_qd", "emb_sim_ad", "pos_sim_ad", "edit_dist", "token_sim_qd", "token_sim_ad", "token_sim_qa", "character_length_a", "character_length_d",
             "character_length_diff", "token_length_a", "token_length_d", "token_length_diff", "abs_comm_suffix", "rel_comm_suffix", "word_freq_a", "word_freq_d",
             "sing_plur", "number_a", "number_d", "number_diff", "wikisim"]

tokenAnswer = []
tokenQuestion = []
tokenDist = []
tokenise = spacy.load("en_core_web_sm")

for i in range(len(question)):
  print("= ", i)
  tokenAnswer += [tokenise(answer[i])]
  tokenQuestion += [tokenise(question[i])]
  tokenDist += [[[], [], []]]
  for j in range(3):
    tokenDist[i][j] = tokenise(str(all_dists[i][j])) # pretokenising - maybe I should store this in a file and just read that?

logreg = LogisticRegression(random_state=16, max_iter=10000)
X2 = [[]] * len(question)
for q in range(len(question)):
  X = {"emb_sim_qd": [], "emb_sim_ad": [], "pos_sim_ad": [], "edit_dist": [], "token_sim_qd": [], "token_sim_ad": [], "token_sim_qa": [], "character_length_a": [], "character_length_d": [],
              "character_length_diff": [], "token_length_a": [], "token_length_d": [], "token_length_diff": [], "abs_comm_suffix": [], "rel_comm_suffix": [], "word_freq_a": [], "word_freq_d": [],
              "sing_plur": [], "number_a": [], "number_d": [], "number_diff": [], "wikisim": []}
  for i in range(len(question)):
    for j in range(3):
      X["emb_sim_qd"] += [0]
      X["emb_sim_ad"] += [0]
      X["pos_sim_ad"] += [0]
      X["edit_dist"] += [TestCompareDistractors.edit_distance(answer[q], all_dists[i][j])]
      X["token_sim_qd"] += [TestCompareDistractors.pos_similarity(tokenQuestion[q], tokenDist[i][j])]
      X["token_sim_ad"] += [TestCompareDistractors.pos_similarity(tokenAnswer[q], tokenDist[i][j])]
      X["token_sim_qa"] += [TestCompareDistractors.pos_similarity(tokenQuestion[q], tokenAnswer[q])]
      X["character_length_a"] += [TestCompareDistractors.character_length(answer[q])] 
      X["character_length_d"] += [TestCompareDistractors.character_length(all_dists[i][j])]
      X["character_length_diff"] += [TestCompareDistractors.character_length_diff(answer[q], all_dists[i][j])] 
      X["token_length_a"] += [TestCompareDistractors.token_length(tokenAnswer[q])]
      X["token_length_d"] += [TestCompareDistractors.token_length(tokenDist[i][j])]
      X["token_length_diff"] += [TestCompareDistractors.token_length_diff(tokenAnswer[q], tokenDist[i][j])]
      X["abs_comm_suffix"] += [TestCompareDistractors.longest_suffix_length(answer[q], all_dists[i][j])]
      X["rel_comm_suffix"] += [TestCompareDistractors.longest_suffix_length(answer[q], all_dists[i][j])/len(all_dists[i][j])] # relative to the distractor?
      X["word_freq_a"] += [0]
      X["word_freq_d"] += [0]
      X["sing_plur"] += [TestCompareDistractors.singular_plural_consistency(tokenAnswer[q], tokenDist[i][j])]
      X["number_a"] += [TestCompareDistractors.does_string_contain_number(tokenAnswer[q])]
      X["number_d"] += [TestCompareDistractors.does_string_contain_number(tokenDist[i][j])]
      X["number_diff"] += [TestCompareDistractors.number_consistency(tokenAnswer[q], tokenDist[i][j])]
      X["wikisim"] += [0]

  X2[q] = pd.DataFrame.from_dict(X)
  #print(X2)
  print(q)

  y = [0] * (len(question)*3)
  y[q*3 + 0] = 1
  y[q*3 + 1] = 1
  y[q*3+ 2] = 1
 #  y2 = pd.DataFrame.from_dict(y)

  logreg.fit(X2[q], y)

"""
  y = {"Outcome": [0] * (len(question)*3)}
  y["Outcome"][q*3 + 0] = 1
  y["Outcome"][q*3 + 1] = 1
  y["Outcome"][q*3+ 2] = 1 """

print(logreg.coef_)

# print(X2)

a = np.array(logit2prob(logreg, X2[2]))
df = pd.DataFrame({"value": a, "distractor": all_dists_list}).sort_values(by="value")
print(df)

# y_pred = logreg.predict(X2)
# print(y_pred)
