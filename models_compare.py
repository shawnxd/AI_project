import numpy as np
import pandas as pd
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import Constant
from keras.models import Model
from keras.layers import *
from keras.utils.np_utils import to_categorical
import re

df = pd.read_csv('commands.csv', delimiter=',',encoding='gb18030')
df = df[['Phrase', 'Sentiment']]

pd.set_option('display.max_colwidth', -1)
df.head(3)

def clean_str(in_str):
    in_str = str(in_str)
    # replace urls with 'url'
    in_str = re.sub(r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})", "url", in_str)
    in_str = re.sub(r'([^\s\w]|_)+', '', in_str)
    return in_str.strip().lower()


df['text'] = df['Phrase'].apply(clean_str)

df.Sentiment.value_counts()

df_0 = df[df['Sentiment'] == 0].sample(frac=1)
df_1 = df[df['Sentiment'] == 1].sample(frac=1)
df_2 = df[df['Sentiment'] == 2].sample(frac=1)
df_3 = df[df['Sentiment'] == 3].sample(frac=1)
df_4 = df[df['Sentiment'] == 4].sample(frac=1)
df_5 = df[df['Sentiment'] == 5].sample(frac=1)
df_6 = df[df['Sentiment'] == 6].sample(frac=1)
df_7 = df[df['Sentiment'] == 7].sample(frac=1)
df_8 = df[df['Sentiment'] == 8].sample(frac=1)

sample_size = 92

data = pd.concat([df_0.head(sample_size), df_1.head(sample_size), df_2.head(sample_size), df_3.head(sample_size),df_4.head(sample_size),df_5.head(sample_size),df_6.head(sample_size),df_7.head(sample_size),df_8.head(sample_size)]).sample(frac=1)

data['l'] = data['Phrase'].apply(lambda x: len(str(x).split(' ')))
print("mean length of sentence: " + str(data.l.mean()))
print("max length of sentence: " + str(data.l.max()))
print("std dev length of sentence: " + str(data.l.std()))

sequence_length = 20
embedding_dim = 200

max_features = 20000

tokenizer = Tokenizer(num_words=max_features, split=' ', oov_token='<unw>')
tokenizer.fit_on_texts(data['Phrase'].values)

# this takes our sentences and replaces each word with an integer
X = tokenizer.texts_to_sequences(data['Phrase'].values)

# we then pad the sequences so they're all the same length (sequence_length)
X = pad_sequences(X, sequence_length)

y = pd.get_dummies(data['Sentiment']).values

# where there isn't a test set, Kim keeps back 10% of the data for testing, I'm going to do the same since we have an ok amount to play with
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0001)

print("test set size " + str(len(X_test)))

embeddings_index = {}
with open('glove.twitter.27B.200d.txt','r+') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

print('Found %s word vectors.' % len(embeddings_index))

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

num_words = min(max_features, len(word_index)) + 1
print(num_words)

# first create a matrix of zeros, this is our embedding matrix
embedding_matrix = np.zeros((num_words, embedding_dim))

# for each word in out tokenizer lets try to find that work in our w2v model
for word, i in word_index.items():
    if i > max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # we found the word - add that words vector to the matrix
        embedding_matrix[i] = embedding_vector
    else:
        # doesn't exist, assign a random vector
        embedding_matrix[i] = np.random.randn(embedding_dim)

inputs_2 = Input(shape=(sequence_length,), dtype='int32')

# note the `trainable=False`, later we will make this layer trainable
embedding_layer_2 = Embedding(num_words,
                              embedding_dim,
                              embeddings_initializer=Constant(embedding_matrix),
                              input_length=sequence_length,
                              trainable=False)(inputs_2)

reshape_2 = Reshape((sequence_length, embedding_dim, 1))(embedding_layer_2)

row = len(y_train)
col = len(y_train[0])
res = []
for i in range(row):
    for j in range(col):
        if y_train[i][j] == 1:
            res.append(int(j))
y_train = np.array(res)
X_train = np.array(X_train)

print(type(X_train))
print(type(y_train))


# Applying machine learning models, building cosine kernel
from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
def my_kernel(X, Y):
    return cosine_similarity(X,Y)

# Evaluations for model, output is list of accuracy, precision, and recall
def cross_validation(k_fold,clf,X,y,max_feature,ngram_range):
    accuracy_list = []
    precision_list = []
    recall_list = []
    for train, test in k_fold.split(X):
        print(train)
        clf.fit(X[train], y[train])
        y_pred = clf.predict(X[test])
        y_true = y[test]
        accuracy_list.append(accuracy_score(y_true, y_pred))
        precision_list.append(precision_score(y_true, y_pred,y_pred,average = 'macro'))
        recall_list.append(recall_score(y_true, y_pred,average = 'macro'))
    result = []
    for score in [accuracy_list,precision_list,recall_list]:
        result.append(np.mean(score))
    return result

def train_models_svm(train_X, train_y,max_feature,ngram_range):
    k_fold = KFold(n_splits=5,shuffle = False)
    results = []
    models = []
    models.append(svm.SVC(C=1.0, kernel=my_kernel))
    models.append(svm.SVC(C=1.0, kernel='linear'))
    models.append(svm.SVC(C=1.0, kernel='sigmoid'))
    models.append(svm.SVC(C=2.0, kernel=my_kernel))
    models.append(svm.SVC(C=2.0, kernel='linear'))
    models.append(svm.SVC(C=2.0, kernel='sigmoid'))
    models_name = ['SVM(C=1.0, kernel=cosine_kernel)',
                   'SVM(C=1.0, kernel=linear)',
                   'SVM(C=1.0, kernel=sigmoid)',
                   'SVM(C=2.0, kernel=cosine_kernel)',
                   'SVM(C=2.0, kernel=linear)',
                   'SVM(C=2.0, kernel=sigmoid)']
    for i in range(len(models)):
        i = int(i)
        print(models_name[i])
        lis = cross_validation(k_fold,models[i],train_X, train_y,max_feature,ngram_range)
        results.append([models_name[i],lis[0],lis[1],lis[2]])
    results_df = pd.DataFrame(results)
    results_df.columns = ['Model Name & Key Parameters', 'Accuracy','Precision','Recall']
    return results_df

print(train_models_svm(X_train, y_train,1000,(1,1)))

def train_models_NB(train_X, train_y,max_feature,ngram_range):
    k_fold = KFold(n_splits=5,shuffle = False)
    results = []
    models = []
    models.append(BernoulliNB(alpha=0.1))
    models.append(BernoulliNB(alpha=0.01))
    models.append(GaussianNB())
    models.append(GaussianNB())
    models.append(MultinomialNB(alpha=0.1))
    models.append(MultinomialNB(alpha=0.01))
    models_name = ['BernoulliNB(alpha=0.1)',
                   'BernoulliNB(alpha=0.01)',
                   'GaussianNB()',
                   'GaussianNB()',
                   'MultinomialNB(alpha=0.1)',
                   'MultinomialNB(alpha=0.01)']
    for i in range(len(models)):
        i = int(i)
        print(models_name[i])
        lis = cross_validation(k_fold,models[i],train_X, train_y,max_feature,ngram_range)
        results.append([models_name[i],lis[0],lis[1],lis[2]])
    results_df = pd.DataFrame(results)
    results_df.columns = ['Model Name & Key Parameters', 'Accuracy','Precision','Recall']
    return results_df

print(train_models_NB(X_train, y_train,1000,(1,1)))


def train_models_perceptron(train_X, train_y,max_feature,ngram_range):
    k_fold = KFold(n_splits=5,shuffle = False)
    results = []
    models = []
    models.append(Perceptron(penalty='l1', alpha=0.0001, max_iter=1000))
    models.append(Perceptron(penalty='l2', alpha=0.0001, max_iter=1000))
    models.append(Perceptron(penalty='l1', alpha=0.0001, max_iter=500))
    models.append(Perceptron(penalty='l2', alpha=0.0001, max_iter=500))
    models.append(Perceptron(penalty='l1', alpha=0.001, max_iter=1000))
    models.append(Perceptron(penalty='l2', alpha=0.001, max_iter=1000))
    models_name = ['Perceptron(penalty=l1, alpha=0.0001, max_iter=1000)',
                   'Perceptron(penalty=l2, alpha=0.0001, max_iter=1000)',
                   'Perceptron(penalty=l1, alpha=0.0001, max_iter=500)',
                   'Perceptron(penalty=l2, alpha=0.0001, max_iter=500)',
                   'Perceptron(penalty=l1, alpha=0.001, max_iter=1000)',
                   'Perceptron(penalty=l2, alpha=0.001, max_iter=1000)']
    for i in range(len(models)):
        i = int(i)
        print(models_name[i])
        lis = cross_validation(k_fold,models[i],train_X, train_y,max_feature,ngram_range)
        results.append([models_name[i],lis[0],lis[1],lis[2]])
    results_df = pd.DataFrame(results)
    results_df.columns = ['Model Name & Key Parameters', 'Accuracy','Precision','Recall']
    return results_df

print(train_models_perceptron(X_train, y_train,1000,(1,1)))
