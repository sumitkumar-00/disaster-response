import json
import plotly
import pickle
import pandas as pd
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine

app = Flask(__name__)


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


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_response', engine)

# load model
model = pickle.load(open("../models/classifier.pkl", 'rb'))


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals

    # data for plotting Message Genres
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    ## data for plotting Message Class
    class_df = df[df.columns[4:]].sum(axis=0).sort_values(ascending=False)[0:10]
    class_counts = class_df.values
    class_names = class_df.index.values

    # create visuals
    # TODO: Below is an example - modify to create your own visuals

    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=class_names,
                    y=class_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Top 10 Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]


    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(port=3001, debug=True)


if __name__ == '__main__':
    main()
