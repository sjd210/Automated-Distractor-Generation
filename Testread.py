import os
import json 

def open_json_as_dict(filename):
    cur_path = os.path.dirname(__file__)
    f = open(cur_path + "\\SciQ\\SciQ dataset-2 3\\" + filename)
    return json.loads(f.read())

def extract_field(field, dict):
    return [d[field] for d in dict]

def extract_all_distractors(dictIn): # extracts the set of all unique distractors from dictIn
    testDist1 = extract_field("distractor1", dictIn)
    testDist2 = extract_field("distractor2", dictIn)
    testDist3 = extract_field("distractor3", dictIn)

    testDistAll = testDist1 + testDist2 + testDist3
    return list(dict.fromkeys(testDistAll))

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