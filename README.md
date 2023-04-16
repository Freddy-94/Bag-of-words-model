# Bag-of-words-model.  

## Description

This program implements the Bag-of-words-model (https://en.wikipedia.org/wiki/Bag-of-words_model), to study similarities between texts. In this particular example, we selected three different texts containing the lyrics of three nice Mexican and Cuban songs, although, it can be modified to pass any documents we might want to analyze.

## Execution of the program

In your terminal run the command: python BagOfWordsM.py

Notice that the source code must be "outside" the folder "InputTexts" that has the texts we want to analyze.

## Relevant functions:

1. clean_data -> "clean" the input text data 
2. vectorization_frequencies -> vectorization of input texts, as vectors of frequencies 
3. cos_similarities -> Computation of the cosine similarity between vectors (input texts represented by vectors)

