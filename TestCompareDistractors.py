word1 = 'hello1'
word2 = 'world1'

def is_string_plural(str):
    return (str[-1] == "s") #https://pypi.org/project/inflect/0.2.5/

def does_string_contain_number(str):
    number_list = ["1","2","3","4","5","6","7","8","9","0"] # add word versions
    return any(number in str for number in number_list)

def longest_suffix_length(str1, str2):
    if ((str1 != "") & (str2 != "")):
        if (str1[-1] == str2[-1]):
            return longest_suffix_length(str1[:-1], str2[:-1]) + 1
    return 0

def length_diff(str1, str2): 
    return abs(len(str1)-len(str2))

#def edit_distance(str1, str2):
#    return 0 # https://www.javatpoint.com/edit-distance-in-python#:~:text=The%20edit%20distance%20between%20two,to%20transform%20str1%20into%20str2.

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

str1 = "kitten"  
str2 = "sitting"  
print(edit_distance(word1, word2))  

print(longest_suffix_length(word1, word1))