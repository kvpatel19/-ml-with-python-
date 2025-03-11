import nltk 
from nltk.tokenize import word_tokenize 
from nltk import pos_tag 
from nltk.chunk import RegexpParser 
# Download necessary NLTK data 
nltk.download('punkt') 
nltk.download('averaged_perceptron_tagger') 
# Sample sentence 
sentence = "The quick brown fox jumps over the lazy dog" 
# Tokenize the sentence 
tokens = word_tokenize(sentence) 
# Get the part-of-speech tags for each token 
pos_tags = pos_tag(tokens) 
# Define the chunking pattern (rules for noun and verb phrases) 
chunk_grammar = """ 
NP: {<DT>?<JJ>*<NN>}   # Noun Phrase 
VP: {<VB.*>}            
# Verb Phrase 
""" 
# Create the chunk parser with the defined grammar 
chunk_parser = RegexpParser(chunk_grammar) 
# Parse the POS tagged tokens to identify chunks 
chunked_tree = chunk_parser.parse(pos_tags) 
# Display the chunked tree 
chunked_tree.pretty_print() 
