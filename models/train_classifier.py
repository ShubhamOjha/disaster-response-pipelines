# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin

print(nltk.download(['punkt', 'stopwords', 'wordnet']))
print(nltk.download('averaged_perceptron_tagger'))

def load_data(database_filepath):
    '''
    Function to load the data from database

    Input : 
     - database_filepath : relative location of database

    Returns :
     - X : training data containg messages
     - Y : categorical information of messages
     - category_names : column names ofo categories
    '''

    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(table_name='DisasterResponse', con=engine)
    X = df['message']
    Y = df.iloc[:,4:]
    Y = Y.apply(pd.to_numeric)
    category_names = list(df.columns[4:])
    
    return X, Y, category_names


def tokenize(text):
    '''
    Function to process and tokenize passed sentence

    Input :
     - text : sentence which need to be processed
    
    Returns :
     - clean_tokens : processed tokenized information
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    tokens = word_tokenize(re.sub(r"[^a-zA-Z0-9]",' ', text).lower())
    tokens = [token for token in tokens if tokens not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token)
        clean_tokens.append(clean_token)
    return clean_tokens


def build_model():
    '''
    Function to build model pipeline

    Input :
     - None

    Returns : 
     - cv : tuned GridSearchCV model
    '''
    pipeline = Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(RandomForestClassifier()))
                        ])

    parameters = {'clf__estimator__n_estimators': [50, 100],
                  'clf__estimator__min_samples_split': [2, 3, 4],
                  'clf__estimator__criterion': ['entropy', 'gini']
                 }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Fucntion to evaluate the performance of model

    Input :
     - model - trained model object
     - X_test - Test disaster messages
     - Y_test - Test cetgorical data of messages
     - category_names : column names ofo categories

     Return :
      - None
    
    It will print the result on terminal

    '''
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns = category_names)
    for column in category_names:
        print('------------------------------------------------------\n')
        print('FEATURE : {}\n'.format(column))
        print(classification_report(Y_test[column],y_pred_df[column]))


def save_model(model, model_filepath):
    '''
    Functiion to save the trained model

    Input :
     - model : trained model object
     - model_filepath : filepath where model pickel file will be saved
    
    Returns :
     - None
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    '''
    Main method for the execution of all functions

    '''

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