import gensim 
from gensim import corpora 
from nltk.tokenize import word_tokenize 
import string 
# Sample documents 
documents = [ 
"I love programming in Python. Python is great for data analysis.", 
"I enjoy learning machine learning techniques.", 
"Data science is a mix of statistics, programming, and machine learning.", 
"Machine learning is part of the broader field of artificial intelligence.", 
"Statistics and data science are closely related fields.", 
] 
# Preprocessing: Tokenization and removing stopwords/punctuation 
stopwords = set(['is', 'a', 'the', 'for', 'and', 'of', 'in', 'to']) 
def preprocess(text): 
# Tokenize and remove punctuation and stopwords
    tokens = word_tokenize(text.lower())  # lowercase and tokenize 
    tokens = [t for t in tokens if t not in stopwords and t not in string.punctuation]
    return tokens 
# Preprocess all documents 
processed_docs = [preprocess(doc) for doc in documents]
# Create a dictionary from the processed documents 
dictionary = corpora.Dictionary(processed_docs) 
# Create a document-term matrix (DTM) 
corpus = [dictionary.doc2bow(doc) for doc in processed_docs] 
# Build the LDA model 
lda_model = gensim.models.LdaMulticore(corpus, id2word=dictionary, passes=10) 
# Print the topics discovered by the LDA model 
topics = lda_model.print_topics(num_words=4) 
for topic in topics: 
    print(topic) 
