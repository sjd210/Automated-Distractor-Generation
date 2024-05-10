import numpy as np
from numpy import dot
from numpy.linalg import norm

def embedding_sim(tok1, tok2):
    return tok1.similarity(tok2)

def name_to_entity_name(name):
    capitalName = name.capitalize()
    underscoredName = capitalName.replace(" ", "_")
    return 'ENTITY/' + underscoredName

def wiki_sim_entity(str1, str2, wv):
    try:
        return wv.similarity(name_to_entity_name(str1), name_to_entity_name(str2))
    except:
       return 0
    
def wiki_sim_base_similarity(tok1, tok2, wv):
    sum = np.zeros(100)
    validTotal = 0
    for token1 in tok1:
        try:
            sum += np.array(wv[token1.text.lower()])
            validTotal += 1
        except:
            sum = sum
    a = sum/validTotal

    sum2 = np.zeros(100)
    validTotal2 = 0
    for token2 in tok2:
        try:
            sum2 += np.array(wv[token2.text.lower()])
            validTotal2 += 1
        except:
            sum2 = sum2
    b = sum2/validTotal2

    return(min(1,dot(a, b)/(norm(a)*norm(b)))) 

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

def word_freq(tok, dict, sum):
    total = 0
    for token in tok:
        if token.text in dict:
            total += dict[token.text]
    return (total/len(tok))/sum

def word_freq_comp(tok1, tok2, dict, sum):
    freq1 = word_freq(tok1, dict, sum)
    freq2 = word_freq(tok2, dict, sum)
    if freq1 == 0 and freq2 == 0:
        return 0
    elif freq1 == 0 or freq2 ==0:
        return 1
    else:
        return abs(freq1/freq2)

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