"""
Author: Alfredo Bernal Luna
Date: April 03rd 2023
Project: Bag-of-words model implementation - NLP class

Relevant functions:

1.- clean_data -> "clean" the input text data 
2.- vectorization_frequencies -> vectorization of input texts, as vectors of frequencies 
3.- cos_similarities -> Computation of the cosine similarity between vectors (input texts represented by vectors)

References: https://en.wikipedia.org/wiki/Vector_space_model
            https://en.wikipedia.org/wiki/Tf%E2%80%93idf
            https://en.wikipedia.org/wiki/Bag-of-words_model
            https://en.wikipedia.org/wiki/Cosine_similarity
            https://en.wikipedia.org/wiki/Information_retrieval
            
Songs: 1. "La Llorona"              -> https://www.youtube.com/watch?v=5pqPFMVAIeM
       2. "La Ixhuateca"            -> https://www.youtube.com/watch?v=VHRDLv5Y9Lg
       3. "La niña de Guatemala"    -> https://www.youtube.com/watch?v=XAAP6bfNGK4
"""

import os, sys
import itertools
import math

def clean_data(input_text):
    """
    This function do the cleaning of data, by removing punctuation marks, along with uninteresting
    words that are irrelevant to the analysis of similarities in the input texts. Similarly, this 
    function also pass each word of the text to lower case (as the case is also irrelevant to the 
    performed analysis).
    Notice that we list an adjustable set of punctuations (and uninteresting words); i.e., this 
    values can be modified by the user of the program.
    
    Finally, this function returns a bag of the words passed in the input text. Notice this function
    helped us to obtain the vocabulary we'll be considering. 
    
    """    
    # Here is a list of punctuations and uninteresting words we consider, to process our text
    punctuations = '''¡!()-[]{};:'"\,<>./¿?@#$%^&*_~'''
    uninteresting_words = ["la", "a", "de", "se", "mi", "es", "con", "le", "si", "y", "que",
                           "las", "el", "en", "un", "este", "los", "una", "al", "por", "su",
                           "te", "ya", "ha", "ahí", "era", "quedó", "han", "aquel", "me"]
    input_text_lower_case = input_text.lower()   
    """
    The next for, replace (if it founds a punctuation in the text) that symbol for a white space "";
    this is done, for each punctuation character; i.e., for each punctuation char, if that character
    is in the text, it will replace it with an empty string.
    """
    for char in punctuations:
        input_text_lower_case = input_text_lower_case.replace(char, '')
    #print(text_lower_case)
    words_in_text = input_text_lower_case.split()
    words_in_text = [word for word in words_in_text if word not in uninteresting_words] 
    # print()
    # print(words_in_text)
    return words_in_text

def clean_all_texts(unclean_list_of_texts):
    """
    This function cleans each one of the input texts, by calling the function clean_data above.
    This functions creates smaller (in constrast with the vocabulary) "bags" of words for each input text.     
    """
    clean_texts_list = []
    for i in range(len(unclean_list_of_texts)):
        clean_text = clean_data(''.join(unclean_list_of_texts[i]))
        clean_texts_list.append(clean_text)    
    return clean_texts_list

def vocabulary_creation(list_of_all_clean_texts):
    """
    This function computes the "bag of words" (vocabulary) obtained from all of the input
    clean texts.
    """
    vocabulary = []    
    for i in range(len(list_of_all_clean_texts)):
        vocabulary += list_of_all_clean_texts[i]
    vocabulary = list(dict.fromkeys(vocabulary)) # remove duplicated words from the vocabulary list
    return vocabulary

def frequencies_dict(clean_text_list, vocabulary):
    """
    This function computes the frequencies of the words appearing in the vocabulary, for the input 
    clean text
    """
    frequencies = {}
    for word in vocabulary:
        # print(word)
        if word in clean_text_list:
            if word not in frequencies:
                frequencies[word] = clean_text_list.count(word)
        else:
            frequencies[word] = 0
    return frequencies

def frequencies(clean_text_list):
    """
    This function computes the frequencies of the words appearing in the input 
    clean text. Notice how this function helps us to compute the tf (term frequencies) 
    in a input text.
    """
    frequencies = {}
    for word in clean_text_list:
        if word not in frequencies:
            frequencies[word] = clean_text_list.count(word)
    return frequencies
    

def vectorization_frequencies(list_of_all_clean_texts):
    """
    This function computes the frequencies of the words appearing in the vocabulary, for each
    one of the input texts. Further, this function also vectorize our text, via the computed 
    frequencies.    
    """
    vocabulary = vocabulary_creation(list_of_all_clean_texts)  
    set_of_vectors = []
    for clean_text_list in list_of_all_clean_texts:
        freq_text_dict = frequencies_dict(clean_text_list, vocabulary)       
        set_of_vectors.append(list(freq_text_dict.values()))
    # print(set_of_vectors)
    # print()
    """
    Quick validation for same size of vectors:
    for vector in set_of_vectors:
        print(len(vector))
    """
    return set_of_vectors

def vect_norm(vector):
    norm = 0
    for i in range(1, len(vector)):
        norm += vector[i]**2
    norm = math.sqrt(norm)
    return norm
    
def dot_prod(vector1, vector2):
    dot = 0
    for i in range(1, len(vector1)):
        dot += vector1[i]*vector2[i]
    return dot 

def angle_opening(x, y):
    cos_theta = dot_prod(x, y) / (vect_norm(x) * vect_norm(y))
    theta = math.acos(cos_theta)
    return theta

def cos_similarities(frequencies_vectors):
    print("\n============================================================================")
    print(f"                    Start of Cosine similarities computation:                ")
    print("============================================================================\n")
    print()
    for vector1, vector2 in itertools.combinations(frequencies_vectors, 2):
        theta = angle_opening(vector1, vector2)
        print(f"Angle similarity between:\n\n{vector1} \n\nand \n\n{vector2} \n\nis of: \n\n{theta}\n")

def tf(list_of_all_clean_texts):
    print("\n============================================================================")
    print(f"                    Start of tf's (term frequencies) computation:            ")
    print("============================================================================\n")
    for clean_text_list in list_of_all_clean_texts:
        freq_text_dict = frequencies(clean_text_list)
        print(f"==Term frequencies for text: {freq_text_dict}==\n")
        for word in freq_text_dict:
            print(f"The tf for the word '{word}' is of: {freq_text_dict[word]/len(freq_text_dict)}\n")
                        
def idf(list_of_all_clean_texts):
    print("\n============================================================================")
    print(f"             Start of idf's (inverse document frequencies) computation:      ")
    print("============================================================================\n")
    vocabulary = vocabulary_creation(list_of_all_clean_texts)
    N = len(list_of_all_clean_texts)     # total number of documents in the corpus
    for word in vocabulary:
        idf_word = 0
        for clean_text_list in list_of_all_clean_texts:
            if word in clean_text_list:
                idf_word += 1
        idf_word = math.log(N/(1+idf_word)) # plus one in the denominator, in case the term is not in the corpus (this not happen, as the corpus is the vocabulary)
        print(f"The idf for the word '{word}' is of: {idf_word}\n")            
    
def main():
    """
    Main function: This function creates a list of lists, that have the different input texts that will
    be compared, via the Bag-of-words model. In our particular case, we are passing three
    (3) different files:
    
    1. LaIxhuateca.txt
    2. LaLlorona.txt
    3. LaNiñaDeGuatemala.txt
    
    That correspond to different nice Mexican and Cuban (La niña de Guatemala) songs, although you can passs any file you 
    wish to analyze.
    
    """
    path = os.path.join(os.getcwd(), "InputTexts") # All files are stored in the dir "InputTexts"
    print("\n===========================================================================================")
    print("The directory where your input texts are located is: " + path  )
    print("===========================================================================================\n")
    dirs = os.listdir(path)
    list_of_texts = [] # This list will contain a list of lists having each one of the input texts
    print("============================================================================")
    print("                             Your input files are:                           ") 
    print("============================================================================\n")
    for file in dirs:
        print(file)
        print()
        with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
            text_in_file = []                
            text_in_file.append(f.read()) 
        # print()
        # print(text_in_file)
        list_of_texts.append(text_in_file)
    # print(list_of_texts)
    # song1 = ''.join(list_of_texts[1])
    # print(song1) 
    # print()    
    # print(clean_data(song1))
    clean_texts = clean_all_texts(list_of_texts) # clean all input texts, and return a "bag" of the words of each text
    #print()
    #print(clean_texts)
    #print()
    #print(vocabulary_creation(clean_texts))
    #print(len(vocabulary_creation(clean_texts)))
    frequencies_vectors = vectorization_frequencies(clean_texts)   # vectorization of texts via its frequency     
    """
    for vector in frequencies_vectors:
        print(len(vector))
        print()
        print(vector)        
        print()
    """
    cos_similarities(frequencies_vectors)
    tf(clean_texts)
    idf(clean_texts)
    
    
if __name__ == '__main__':
    main()
    
      
