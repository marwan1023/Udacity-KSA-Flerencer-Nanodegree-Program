# import librarie
import sys 
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import re , pickle
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sqlalchemy import create_engine
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
import pickle
import sqlite3


def load_data(database_filepath):
    """
    input:
    Database file path: string, data/DisasterResponse.db
    Returns:
    x: containing message data
    Y: other columns
    """
    data_con = 'sqlite:///' + database_filepath
    engine = create_engine(data_con)
    df = pd.read_sql_table('Disasters_massg', con=engine) # is table always called this? 
    print(df.head())
    X = df['message']
    Y = df[df.columns[4:]]
    category_names = Y.columns
    return X, Y, category_names
    
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def tokenize(text):
    """
    inputs:
    messages
       
    Returns:
    list of words into numbers of same meaning
    """

    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    # take out all punctuation while tokenizing
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
     # lemmatize as shown in the nmessgee
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
    
def build_model():
    '''
    Function to build a model, create pipeline, hypertuning as well as gridsearchcv
    A pipeline performs a list of steps in a linear sequence, while a feature union performs a list of steps in parallel and then combines their results.
    Input: N/A
    Output: Returns the model
    '''
    # Create pipeline with Classifier

    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters ={'clf__estimator__max_depth': [10, 50, 100],
                 'clf__estimator__min_samples_leaf':[2, 5, 10],
                 'clf__estimator__min_samples_split': [2, 3, 5],
                 'clf__estimator__n_estimators':[50, 100, 200],
                 'features__text_pipeline__vect__max_features': (None, 5000)
               
                }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=3)    
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function to evaluate a model and return the classificatio and accurancy score.
    Inputs: Model, X_test, y_test, Catgegory_names
    Outputs: Prints the Classification report & Accuracy Score
    '''
    y_pred = model.predict(X_test)
    print(classification_report(y_pred, Y_test.values, target_names=category_names))
    # print raw accuracy score 
    print('Accuracy Score: {}'.format(np.mean(Y_test.values == y_pred)))


def save_model(model, model_filepath):
    '''
    Function to save the model
    Input: model and the file path to save the model
    Output: save the model as pickle file in the give filepath 
    '''
    pkl_filename = '{}'.format(model_filepath)
    with open(pkl_filename ,'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
