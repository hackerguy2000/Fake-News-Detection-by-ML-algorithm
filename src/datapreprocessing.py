import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_data(data):
    # Replace missing values with empty strings
    data = data.fillna('')

    # Merge "author" and "title" into "content"
    data['content'] = data['author'] + ' ' + data['title']

    # Perform stemming and vectorization
    X = data['content'].apply(stemming)
    Y = data['label'].values

    vectorizer = TfidfVectorizer()
    vectorizer.fit(X)
    X = vectorizer.transform(X)

    return X, Y

def stemming(content):
    # Implement stemming logic
    # ...

    return stemmed_content
