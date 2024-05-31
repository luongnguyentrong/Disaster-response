import sys

# import libraries
import pandas as pd

import nltk
nltk.download('punkt')

import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def load_data(database_filepath):
    """
    Load data from an SQLite database and split it into features (X) and targets (y).

    Parameters:
    database_filepath (str): Filepath of the SQLite database.

    Returns:
    X (Series): Features containing messages.
    y (array): Target values containing categories.
    category_names (list): List of category names.
    """

    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("clean_disaster_data", engine)

    X = df["message"]
    y = df.iloc[:, 4:]

    column_names = y.columns.values

    return X, y.values, column_names

def tokenize(text):
    """
    Tokenize and preprocess the text data.

    Parameters:
    text (str): Input text data.

    Returns:
    clean_tokens (list): List of preprocessed tokens.
    """

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
    

def build_model():
    """
    Build and instantiate the machine learning model pipeline.

    Returns:
    cv (GridSearchCV): GridSearchCV object with optimized model pipeline.
    """

    # create pipeline
    pipeline = Pipeline([
        ('feature', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ])),
            ('starting_verb', StartingVerbExtractor()) 
        ])) ,
        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    # optimize hyperparameter with GridSearchCV
    parameters =  {
        'classifier__estimator__learning_rate': [0.5, 1.0],
        'classifier__estimator__n_estimators': [10, 20]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=5, n_jobs=-1, verbose=3) 

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the machine learning model and print classification report for each category.

    Parameters:
    model: Trained machine learning model object.
    X_test (Series): Test features containing messages.
    Y_test (array): Test targets containing categories.
    category_names (list): List of category names.
    """

    y_prediction = model.predict(X_test)

    for idx, column in enumerate(category_names):
        print("Target: ", column)
        print(classification_report(Y_test[:, idx], y_prediction[:, idx]))
        print("---" * 10)


def save_model(model, model_filepath):
    """
    Save the trained model to a file.

    Parameters:
    model: Trained machine learning model object.
    model_filepath (str): Filepath to save the model.
    """

    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

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