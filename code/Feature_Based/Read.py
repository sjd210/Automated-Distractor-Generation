import os
import json 
import numpy as np
import zlib
import pickle
from sklearn.model_selection import train_test_split

def open_json_as_dict(filename):
    f = open(filename)

    return json.loads(f.read())

def extract_field(field, dict):
    list = []
    for d in dict:
        if field in d:
            list += [d[field].lower()]
    return list

def extract_all_distractors(dictIn):
    testDist1 = extract_field("distractor1", dictIn)
    testDist2 = extract_field("distractor2", dictIn)
    testDist3 = extract_field("distractor3", dictIn)

    testDistAll = testDist1 + testDist2 + testDist3
    return list(dict.fromkeys(testDistAll))

def combine_evidence(dictIn):
    support = extract_field("support", dictIn)
    out = ""
    for i in range(len(support)):
        out += " " + support[i].lower()
    return out

def count_text(text):
    totals = {}
    for word in text.split():
        if word in totals:
            totals[word] += 1
        else:
            totals[word] = 1
    return totals

def extract_all_fields(dict, rank, includeAnswers=False):
  question = extract_field("question", dict)[0:rank]
  answer = extract_field("correct_answer", dict)[0:rank]
  dist1 = extract_field("distractor1", dict)[0:rank]
  dist2 = extract_field("distractor2", dict)[0:rank]
  dist3 = extract_field("distractor3", dict)[0:rank]

  if includeAnswers:
    allDistsList = list(set([val for a in zip(dist1, dist2, dist3, answer) for val in a][0:rank*4]))
  else:
    allDistsList = [val for a in zip(dist1, dist2, dist3) for val in a][0:rank*3]
  allDists = np.transpose([dist1,dist2,dist3])
  return (question, answer, allDists, allDistsList)

def arcEntry_to_sciqEntry(arc):
    sciq = {}
    sciq["question"] = arc["question"]["stem"]
    distractorNum = 1
    for i in range(len(arc["question"]["choices"])):
        if arc["question"]["choices"][i]["label"] == arc["answerKey"]:
            sciq["correct_answer"] = arc["question"]["choices"][i]["text"]
        elif distractorNum < 4:
            sciq["distractor%d" %(distractorNum)] = arc["question"]["choices"][i]["text"]
            distractorNum += 1
    if (len(arc["question"]["choices"]) == 3):
        sciq["distractor3"] = sciq["distractor2"]
    sciq["support"] = ""
    return sciq

def arc_to_sciq(arcFile, arcDest):
    arcDict = []
    with open(arcFile, 'r', encoding='utf-8') as f:
        for line in f:
            arcDict.append(json.loads(line))
    sciqDict = [arcEntry_to_sciqEntry(d) for d in arcDict]
    with open(arcDest, 'w') as fp:
        json.dump(sciqDict, fp, indent=2)

def run_arc_conversion():
    arc_to_sciq("ARC-Easy/ARC-Easy-Dev.jsonl", "ARC-Easy-2/valid.json")
    arc_to_sciq("ARC-Easy/ARC-Easy-Test.jsonl", "ARC-Easy-2/test.json")
    arc_to_sciq("ARC-Easy/ARC-Easy-Train.jsonl", "ARC-Easy-2/train.json")

    arc_to_sciq("ARC-Challenge/ARC-Challenge-Dev.jsonl", "ARC-Challenge-2/valid.json")
    arc_to_sciq("ARC-Challenge/ARC-Challenge-Test.jsonl", "ARC-Challenge-2/test.json")
    arc_to_sciq("ARC-Challenge/ARC-Challenge-Train.jsonl", "ARC-Challenge-2/train.json")

def arc_combined_split(X):
    XTrain, XTestValid = train_test_split(X, test_size=1200)
    XTest, XValid = train_test_split(XTestValid, test_size=200)
    with open("ARC-Easy-3/train.json", 'w') as fp:
        json.dump(XTrain, fp, indent=2)
    with open("ARC-Easy-3/test.json", 'w') as fp:
        json.dump(XTest, fp, indent=2)
    with open("ARC-Easy-3/valid.json", 'w') as fp:
        json.dump(XValid, fp, indent=2)

#arc_combined_split(open_json_as_dict("ARC-Easy-2/all.json"))

if False:
    a = open_json_as_dict("../../data/ARC-Combined/train.json")
    b = open_json_as_dict("../../data/ARC-Combined/valid.json")
    c = open_json_as_dict("../../data/ARC-Combined/test.json")
    (question, answer, allDists, allDistsList) = extract_all_fields(a, len(a))
    (question, answer, allDists, allDistsListb) = extract_all_fields(b, len(b))
    (question, answer, allDists, allDistsListc) = extract_all_fields(c, len(c))
    allDistsListALL = sorted(list(set(allDistsList + allDistsListb + allDistsListc)))

if False:
    a = combine_evidence(open_json_as_dict("/SciQ dataset-2 3/train.json"))
    b = combine_evidence(open_json_as_dict("/SciQ dataset-2 3/valid.json"))
    c = combine_evidence(open_json_as_dict("/SciQ dataset-2 3/test.json"))
    d = a + " " + b + " " + c
    with open("sciqSupportDict", 'wb') as file:
        pickledData = pickle.dumps(count_text(d))
        compressedPickle = zlib.compress(pickledData)
        file.write(compressedPickle)

if False:
    with open("sciqSupportDict", 'rb') as file:
        a = file.read()
        b = zlib.decompress(a)
        c = pickle.loads(b)
        print(c)

if False:
    f = open(os.path.dirname(__file__) + "/ARC_Corpus.txt", "r", encoding='utf-8')
    a = f.read()
    with open("arcDict", 'wb') as file:
        pickledData = pickle.dumps(count_text(a))
        compressedPickle = zlib.compress(pickledData)
        file.write(compressedPickle)