import pickle
import math

def build_vocab(dataset, embeddings_path):
    vocab = None
    print(" 1.1. Loading saved vocab")
    with open('{}/vocab.pkl'.format(dataset), 'rb') as vocabFile:
        vocab = pickle.load(vocabFile)

    embd = None
    print(" 1.2. Loading saved embeddings") # MUST EDIT THIS
    with open('{}/embd.pkl'.format(dataset), 'rb') as embedFile:
        embd = pickle.load(embedFile)

    return vocab, embd

def read_answer_list(dataset, answerFile, maxSeqLength, fill='<->'):
    answerList = []
    print(" 2.1 Reading answer file")
    for line in open('{}/{}'.format(dataset, answerFile), encoding="utf8"):
        items = line.strip().lower().split(' ')
        if len(items) > maxSeqLength:
            items = items[:maxSeqLength]
        else:
            items = items + [fill] * (maxSeqLength - len(items))
        answerList += ['_'.join(items)]
    
    return answerList

def read_data(filepath, allDists=False):
    data = []
    dataDict = {} 
    if (allDists):
        dataDict["all_dists"] = []
    print(" 2.2 Reading training data")
    with open(filepath, 'r', encoding="utf8") as dataFile:
        for _, line in enumerate(dataFile):
            items = line.strip().lower().split(' ')
            if items[0] == '1':  
                data.append(items)
                if (allDists):
                    dataDict["all_dists"] += [items[3]]

                if items[1] in dataDict:
                    dataDict[items[1]] += [items[3]] 
                else:
                    dataDict[items[1]] = [items[2], items[3]] # correct answer in position 0
    return data, dataDict

def get_encoding(vocab, target, maxSeqLength):
    x = []
    words = target.strip().split('_')
    for word in words:
        if word in vocab:
            x.append(vocab[word])
        else:
            x.append(vocab['UNKNOWN'])
    for i in range(maxSeqLength - len(words)):
        x.append(vocab['UNKNOWN'])
    return x

def load_samples(q, a, d, ns, vocab, maxSeqLength=100):
    samples = []
    for n in ns:
        samples.append((get_encoding(vocab, q, maxSeqLength),
                        get_encoding(vocab, a, maxSeqLength),
                        get_encoding(vocab, d, maxSeqLength),
                        get_encoding(vocab, n, maxSeqLength)))
    return samples

def batch(data, batchSize):
    for i in range(int(math.ceil(len(data) / batchSize))):
        endIndex = min((i + 1) * batchSize, len(data))
        yield data[endIndex-batchSize:endIndex]
