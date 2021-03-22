import sys 
import requests
import json
import plotly
import pandas as pd
import numpy as np
import nltk
import re
import sklearn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])


app = Flask(__name__)

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def tokenize(text):
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Disasters_massg', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message'].sort_values(ascending=False)
    genre_names = list(genre_counts.index)

    # Show distribution of different category
    categories = df.iloc[:,4:]
    categories_count = categories.sum().sort_values(ascending=False)[1:]
    categories_name = list(categories_count.index)
    # extract data exclude related
    categories = df.iloc[:,4:]
    categories_mean = categories.mean().sort_values(ascending=False)[1:11]
    categories_names = list(categories_mean.index)
    
                        
    # create visuals
      
    graphs = [
        {
           
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    text=genre_counts,
                    textposition='auto',
                    marker=dict(color='rgb(31, 119, 180)')
                   
                )
            ],

            'layout': {
                'title': '<b>Distribution of Message Genres</b>',
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
                    x=categories_name,
                    y=categories_count,
                    text=categories_count,
                    textposition='outside',
                    marker=dict(color='rgb(227, 119, 194)')
                  
                )
            ],

            'layout': {
                'title': '<b>Distribution of Message Categories</b>',
                'yaxis': {
                    'title': "<b>Count</b>"
                },
                'xaxis': {
                    'title': "<b>Category</b>",
                    
                   
                }
            }
        },
        {
            'data': [
                Bar(
                    x=categories_names,
                    y=categories_mean,
                    text=categories_mean,
                    textposition='outside',
                    marker=dict(color='rgb(23, 190, 207)')
                )
            ],

            'layout': {
                'title': '<b>Top 10 Message Categories</b>',
                'yaxis': {
                    'title': "<b>Percentage</b>"
                },
                'xaxis': {
                    'title': "<b>Categories</b>"
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
    #app.run()
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()