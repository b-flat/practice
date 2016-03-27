from pymongo import MongoClient
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

client = MongoClient()
db = client.twitter
coll = db.tweets

def preprocessing(coll):
    
    tweets_raw = [''.join(tweet['text']).lower() for tweet in coll.find()]
    tweets_tokenized = [word_tokenize(tweet) for tweet in tweets_raw]
    wordnet = WordNetLemmatizer()
    tweets = [[wordnet.lemmatize(word) for word in words] for words in tweets_tokenized]
    
    return tweets

def get_y():

    y = np.random.choice([0, 1], size=(len(tweets),))
    
    return y

def model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    vect = TfidfVectorizer(stop_words='english')
    X_train = vect.fit_transform(X_train)
    X_test = vect.transform(X_test)
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)

    print "Logistic Regression Accuracy on test set:", log_reg.score(X_test, y_test)

# def save_model(X, y):

#     with open('model.pkl', 'w') as f:
#         pickle.dump(model, f)

def run():

    X = preprocession(coll)
    y = get_y
    model(X, y)

if __name__ == '__main__':
    run()


