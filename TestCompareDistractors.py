from wikipedia2vec import Wikipedia2Vec
import os
from gensim.models import KeyedVectors

'''cur_path = os.path.dirname(__file__)
#wiki2vec = Wikipedia2Vec.load(cur_path + "/WikipediaEmbeddings/enwiki_20180420_100d.pkl")
wiki2vec2 = KeyedVectors.load_word2vec_format("WikipediaEmbeddings/enwiki_20180420_100d.txt", binary = False)
#wiki2vec2 = KeyedVectors.load_word2vec_format("WikipediaEmbeddings/small_test.txt", binary = False)
#wiki2vec3 = KeyedVectors.load_word2vec_format("WikipediaEmbeddings/enwiki_20180420_100d.pkl", binary = True)
print(wiki2vec2)
print(wiki2vec2.similarity('the', 'of'))
print(wiki2vec2.most_similar('ENTITY/Gravity'))

#print(wiki2vec.most_similar(wiki2vec.get_entity('Scarlett Johansson'), 5)) '''

# Emb Sim !!! 
# POS (part of speech) tagging
# plurality
# frequency
# nlp = spacy.load("en_core_web_sm")

def embedding_sim(tok1, tok2):
    return tok1.similarity(tok2)

def name_to_entity_name(name):
    capitalName = name.capitalize()
    underscoredName = capitalName.replace(" ", "_")
    return 'ENTITY/' + underscoredName

def wiki_sim_entity(str1, str2, wv):
    try:
      #  print(name_to_entity_name(str1), name_to_entity_name(str2))
        return wv.similarity(name_to_entity_name(str1), name_to_entity_name(str2))
    except:
       return 0
    
def wiki_sim_word(str1, str2, wv):
    try:
        return wv.similarity(str1, str2) # TODO
    except:
        return 0

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
            return 1
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
        if token.pos_ in tok1Totals:
            tok1Totals[token.pos_] += 1
        else:
            tok1Totals[token.pos_] = 1
    for token in tok2:
        if token.pos_  in tok2Totals:
            tok2Totals[token.pos_] += 1
        else:
            tok2Totals[token.pos_] = 1
        if token.pos_  in tok1Totals:
            andTotals[token.pos_] = min(tok1Totals[token.pos_], tok2Totals[token.pos_])

    andSum = sum(andTotals.values())
    orSum = sum(tok1Totals.values()) + sum(tok2Totals.values()) - andSum
    return andSum/orSum

def token_similarity(tok1, tok2):
    tok1Totals = {}
    tok2Totals = {}
    andTotals = {}
    for token in tok1:
        if token.text in tok1Totals:
            tok1Totals[token.text] += 1
        else:
            tok1Totals[token.text] = 1
    for token in tok2:
        if token.text in tok2Totals:
            tok2Totals[token.text] += 1
        else:
            tok2Totals[token.text] = 1
        if token.text in tok1Totals:
            andTotals[token.text] = min(tok1Totals[token.text], tok2Totals[token.text])

    andSum = sum(andTotals.values())
    orSum = sum(tok1Totals.values()) + sum(tok2Totals.values()) - andSum
    return andSum/orSum

def word_freq(tok, dict):
    total = 0
    for token in tok:
        if token.text in dict:
            total += dict[token.text]
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