# import sys libraries
import sys
import pandas as pd
import nltk
import pickle

nltk.download(['punkt', 'wordnet', 'stopwords'])
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def load_data(database_filepath):
    """
    Load data from database into dataframe for modeling
    :param database_file: path to database file
    :return: predictor and response data as dataframes
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_response', engine)

    # Use messages column as predictor and the rest as response
    X = df['message']
    y = df[df.columns[4:]]

    return X, y


def build_model():
    """
    build a machine learning pipeline
    :return: pipeline
    """
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('classifier', OneVsRestClassifier(AdaBoostClassifier(random_state=7, n_estimators=50)))
    ])

    return pipeline


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
    # remove urls
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_regex, text)

    for url in urls:
        text = text.replace(url, 'urlplaceholder')

    # Normalize text
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())

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


def evaluate_model(model, X_test, y_test, target_names):
    """
    Evaluate model performance and print it to console
    :param model: Machine learning model used for fit
    :param X_test: test data set
    :param y_test: true response values for test data set
    :param target_names: response class names
    :return: None
    """
    y_test_pred = model.predict(X_test)
    print(classification_report(y_test, y_test_pred, target_names=target_names))

    return None


def export_model(model, model_file):
    """
    Save model in a pickle file that can be imported by web app
    :param model: Machine learning model
    :param model_file: destination for model on the disk
    :return: None
    """
    pickle_file = open(model_file, 'wb')
    pickle.dump(model, pickle_file)
    pickle_file.close()

    return None

def main():
    """
    This function calls all other functions to do the specific tasks
    such as loading data, tokenizing data, building and fitting model
    and finally making prediction
    :return:
    """
    if len(sys.argv) == 3:
        # load data
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data ...\n   DATABASE:{}'.format(database_filepath))
        X, y = load_data(database_filepath)

        # Split data into train and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=43, test_size=0.20)

        # Build a machine learning pipeline
        print('Building model...')
        model = build_model()

        # fit training data
        print('Fitting machine learning model. It may take a while!')
        model.fit(X_train, y_train)

        # evaluate model
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, y.columns.values)

        # export model/pipeline as a pickle file
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        export_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
