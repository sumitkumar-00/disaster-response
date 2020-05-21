# import libraries
import pandas as pd
import nltk

nltk.download(['punkt', 'wordnet', 'stopwords'])
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# load data from database
engine = create_engine('sqlite:///InsertDatabaseName.db')
# engine.table_names()
df = pd.read_sql_table('InsertTableName', engine)

# Use messages column as predictor and the rest as response
X = df['message']
y = df[df.columns[4:40]]


# tokenization function
def tokenize(text):
    """
    This function tokenizes the text passed as input parameter. Words in
    the text are first normalized by converting to lowercase letters. Then
    it is tokenized and lemmatized. Stop words are also removed before
    returning the tokenized text.
    :param text: text that need to be tokenized (message in this case)
    :return: tokenized list of words
    """
    # Normalize text
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())

    # Tokenize text
    words = word_tokenize(text)

    # Get Rid of Stop Words
    words = [w for w in words if w not in stopwords.words('english')]

    # lemmatize words
    wnl = WordNetLemmatizer()
    words = [wnl.lemmatize(w) for w in words]

    # lemmatize words (verbs)
    words = [wnl.lemmatize(w, pos='v') for w in words]

    return words


# Build a machine learning pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultiOutputClassifier(RandomForestClassifier()))
])


# Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

# fit training data
pipeline.fit(X_train, y_train)

# prediction on test data
