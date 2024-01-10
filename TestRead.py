import os
import json 
import numpy as np

def open_json_as_dict(filename):
    cur_path = os.path.dirname(__file__)
    f = open(cur_path + "/SciQ/SciQ dataset-2 3/" + filename)
    # f = open(cur_path + "\\SciQ\\SciQ dataset-2 3\\" + filename)

    return json.loads(f.read())

def extract_field(field, dict):
    return [d[field] for d in dict]

def extract_all_distractors(dictIn): # extracts the set of all unique distractors from dictIn
    testDist1 = extract_field("distractor1", dictIn)
    testDist2 = extract_field("distractor2", dictIn)
    testDist3 = extract_field("distractor3", dictIn)

    testDistAll = testDist1 + testDist2 + testDist3
    return list(dict.fromkeys(testDistAll))

def combine_evidence(dictIn):
    support = extract_field("support", dictIn)
    out = ""
    for i in range(len(support)):
        out += " " + support[i]
    return out

def count_text(text):
    totals = {}
    for word in text.split():
        if word in totals:
            totals[word] += 1
        else:
            totals[word] = 1
    return totals

def extract_all_fields(dict, rank):
  question = extract_field("question", dict)[0:rank]
  answer = extract_field("correct_answer", dict)[0:rank]
  dist1 = extract_field("distractor1", dict)[0:rank]
  dist2 = extract_field("distractor2", dict)[0:rank]
  dist3 = extract_field("distractor3", dict)[0:rank]
  allSupport = combine_evidence(dict)[0:rank]
  allSupportDict = count_text(allSupport)

  allDistsList = [val for a in zip(dist1, dist2, dist3) for val in a][0:rank*3]
  allDists = np.transpose([dist1,dist2,dist3])
  return (question, answer, allDistsList, allDists, allSupportDict)

print()

testDict = open_json_as_dict("test.json")
validateDict = open_json_as_dict("valid.json")
trainDict = open_json_as_dict("train.json")

print(trainDict[0])
print("\n-=+=-\n") 

for i in range(10):
    print(f"{i}: {validateDict[i]['question']}")

trainQuestions = extract_field("question", trainDict)
print(trainQuestions[0:9], len(trainQuestions))

testDists = extract_all_distractors(trainDict)

print(len(testDists))