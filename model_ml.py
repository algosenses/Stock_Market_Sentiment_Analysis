import os
from time import time
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.utils.extmath import density
from sklearn import svm
from sklearn import naive_bayes
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.utils import shuffle


np.random.seed(42)

comment_file = './data/stock_comments_seg.csv'
data_path = './data'
pos_corpus = 'positive.txt'
neg_corpus = 'negative.txt'
K_Best_Features = 3000

def load_dataset():
    pos_file = os.path.join(data_path, pos_corpus)
    neg_file = os.path.join(data_path, neg_corpus)

    pos_sents = []
    with open(pos_file, 'r', encoding='utf-8') as f:
        for sent in f:
            pos_sents.append(sent.replace('\n', ''))

    neg_sents = []
    with open(neg_file, 'r', encoding='utf-8') as f:
        for sent in f:
            neg_sents.append(sent.replace('\n', ''))

    balance_len = min(len(pos_sents), len(neg_sents))

    pos_df = pd.DataFrame(pos_sents, columns=['text'])
    pos_df['polarity'] = 1
    pos_df = pos_df[:balance_len]

    neg_df = pd.DataFrame(neg_sents, columns=['text'])
    neg_df['polarity'] = 0
    neg_df = neg_df[:balance_len]

    return pd.concat([pos_df, neg_df]).reset_index(drop=True)
#    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)

def load_dataset_tokenized():
    pos_file = os.path.join(data_path, pos_corpus)
    neg_file = os.path.join(data_path, neg_corpus)

    pos_sents = []
    with open(pos_file, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.split(' ')
            sent = []
            for t in tokens:
                if t.strip():
                    sent.append(t.strip())
            pos_sents.append(sent)

    neg_sents = []
    with open(neg_file, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.split(' ')
            sent = []
            for t in tokens:
                if t.strip():
                    sent.append(t.strip())
            neg_sents.append(sent)

    balance_len = min(len(pos_sents), len(neg_sents))

    texts = pos_sents + neg_sents
    labels = [1] * balance_len + [0] * balance_len

    return texts, labels


def KFold_validation(clf, X, y):
    acc = []
    pos_precision, pos_recall, pos_f1_score = [], [], []
    neg_precision, neg_recall, neg_f1_score = [], [], []

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train, test in kf.split(X):
        X_train = [X[i] for i in train]
        X_test = [X[i] for i in test]
        y_train = [y[i] for i in train]
        y_test = [y[i] for i in test]

        # vectorizer = TfidfVectorizer(analyzer='word', tokenizer=lambda x : (w for w in x.split(' ') if w.strip()))
        def dummy_fun(doc):
            return doc

        vectorizer = TfidfVectorizer(analyzer='word',
                                     tokenizer=dummy_fun,
                                     preprocessor=dummy_fun,
                                     token_pattern=None)

        vectorizer.fit(X_train)
        X_train = vectorizer.transform(X_train)
        X_test = vectorizer.transform(X_test)

        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        acc.append(metrics.accuracy_score(y_test, preds))
        pos_precision.append(metrics.precision_score(y_test, preds, pos_label=1))
        pos_recall.append(metrics.recall_score(y_test, preds, pos_label=1))
        pos_f1_score.append(metrics.f1_score(y_test, preds, pos_label=1))
        neg_precision.append(metrics.precision_score(y_test, preds, pos_label=0))
        neg_recall.append(metrics.recall_score(y_test, preds, pos_label=0))
        neg_f1_score.append(metrics.f1_score(y_test, preds, pos_label=0))


    return (np.mean(acc), np.mean(pos_precision), np.mean(pos_recall), np.mean(pos_f1_score),
            np.mean(neg_precision), np.mean(neg_recall), np.mean(neg_f1_score))


def benchmark_clfs():
    print('Loading dataset...')

    X, y = load_dataset_tokenized()

    classifiers = [
        ('LinearSVC', svm.LinearSVC()),
        ('LogisticReg', LogisticRegression()),
        ('SGD', SGDClassifier()),
        ('MultinomialNB', naive_bayes.MultinomialNB()),
        ('KNN', KNeighborsClassifier()),
        ('DecisionTree', DecisionTreeClassifier()),
        ('RandomForest', RandomForestClassifier()),
        ('AdaBoost', AdaBoostClassifier(base_estimator=LogisticRegression()))
    ]

    cols = ['metrics', 'accuracy',  'pos_precision', 'pos_recall', 'pos_f1_score', 'neg_precision', 'neg_recall', 'neg_f1_score']
    scores = []
    for name, clf in classifiers:
        score = KFold_validation(clf, X, y)
        row = [name]
        row.extend(score)
        scores.append(row)

    df = pd.DataFrame(scores, columns=cols).T
    df.columns = df.iloc[0]
    df.drop(df.index[[0]], inplace=True)
    df = df.apply(pd.to_numeric, errors='ignore')

    return df

def dummy_fun(doc):
        return doc

def eval_model():
    print('Loading dataset...')

    X, y = load_dataset_tokenized()

    clf = svm.LinearSVC()

    vectorizer = TfidfVectorizer(analyzer='word',
                                 tokenizer=dummy_fun,
                                 preprocessor=dummy_fun,
                                 token_pattern=None)

    X = vectorizer.fit_transform(X)

    print('Train model...')
    clf.fit(X, y)

    print('Loading comments...')
    df = pd.read_csv(comment_file)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['created_time'] = pd.to_datetime(df['created_time'], format='%Y-%m-%d %H:%M:%S')
    df['polarity'] = 0
    df['title'].apply(lambda x: [w.strip() for w in x.split()])

    texts = df['title']
    texts = vectorizer.transform(texts)

    preds = clf.predict(texts)
    df['polarity'] = preds

    df.to_csv('stock_comments_analyzed.csv', index=False)


if __name__ == '__main__':
    scores = benchmark_clfs()
    print(scores)
    scores.to_csv('model_ml_scores.csv', float_format='%.4f')


    eval_model()
