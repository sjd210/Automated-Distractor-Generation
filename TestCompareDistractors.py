from wikipedia2vec import Wikipedia2Vec
import os
import spacy
import TestRead

cur_path = os.path.dirname(__file__)
f = open(cur_path + "/WikipediaEmbeddings/enwiki_20180420_100d.pkl", 'r', errors = "ignore")
wiki2vec = Wikipedia2Vec.load(cur_path + "/WikipediaEmbeddings/enwiki_20180420_100d.pkl")
wiki2vec.get_word_vector('the')

# print(wiki2vec.most_similar(wiki2vec.get_word('lint'), 5))

# Emb Sim !!! 
# POS (part of speech) tagging
# plurality
# frequency
nlp = spacy.load("en_core_web_sm")

word1 = 'hello1'
word2 = 'world1'

def is_string_plural(tok):
    for token in tok:
        if (token.tag_ == "NNS"):
            return (1)
    return 0

def singular_plural_consistency(tok1, tok2):
    if (is_string_plural(tok1) == is_string_plural(tok2)):
        return 1
    return 0

def does_string_contain_number(tok):
    for token in tok:
        if (token.pos_ == "NUM"):
            return (1)
    return 0

def number_consistency(tok1, tok2):
    if (does_string_contain_number(tok1) == does_string_contain_number(tok2)):
        return 1
    return 0

def longest_suffix_length(str1, str2):
    if ((str1 != "") & (str2 != "")):
        if (str1[-1] == str2[-1]):
            return longest_suffix_length(str1[:-1], str2[:-1]) + 1
    return 0

def character_length(str):
    return len(str)

def character_length_diff(str1, str2): 
    return abs(character_length(str1)-character_length(str2))

def token_length(tok):
    return len(tok)

def token_length_diff(tok1, tok2): 
    return abs(token_length(tok1)-token_length(tok2))

def pos_similarity(tok1, tok2):
    tok1Totals = {}
    tok2Totals = {}
    andTotals = {}
    for token in tok1:
        if token in tok1Totals:
            tok1Totals[token] += 1
        else:
            tok1Totals[token] = 1
    for token in tok2:
        if token in tok2Totals:
            tok2Totals[token] += 1
        else:
            tok2Totals[token] = 1
        if token in tok1Totals:
            andTotals[token] = min(tok1Totals[token], tok2Totals[token])

    andSum = sum(andTotals.values())
    orSum = sum(tok1Totals.values()) + sum(tok2Totals.values()) - andSum
    return andSum/orSum

def word_freq(tok, dict):
    total = 0
    for token in tok:
        if token in dict:
            total += dict[token]
    return total/sum(dict.values())

def edit_distance(str1, str2):  
    m, n = len(str1), len(str2)  
    matrix = [[0] * (n + 1) for i in range(m + 1)]  
    for i in range(m + 1):  
        for j in range(n + 1):  
            if (i == 0):  
                matrix[i][j] = j  
            elif (j == 0):  
                matrix[i][j] = i  
            elif (str1[i - 1] == str2[j - 1]):  
                matrix[i][j] = matrix[i - 1][j - 1]  
            else:  
                matrix[i][j] = 1 + min(matrix[i - 1][j], matrix[i][j - 1], matrix[i - 1][j - 1])  
  
    return matrix[m][n]  