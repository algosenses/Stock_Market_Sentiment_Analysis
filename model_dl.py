import os
import gc
import pickle
import numpy as np
import pandas as pd
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Dense, Activation, Input
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional
from keras.layers import BatchNormalization
from keras.layers.merge import concatenate
from keras.models import Model, Sequential
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

np.random.seed(42)

data_path = './data'
pos_corpus = 'positive.txt'
neg_corpus = 'negative.txt'

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


print('Loading dataset...')
dataset = load_dataset()

print('Dataset size ', len(dataset))

X = dataset['text']
y = dataset['polarity'].astype(int)

tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")
tokenizer.fit_on_texts(X)
vocab = tokenizer.word_index
print('Vocab size', len(vocab))

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

max_len = 64
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train_word_ids = tokenizer.texts_to_sequences(X_train)
X_test_word_ids = tokenizer.texts_to_sequences(X_test)
X_train_padded_seqs = pad_sequences(X_train_word_ids, maxlen=max_len)
X_test_padded_seqs = pad_sequences(X_test_word_ids, maxlen=max_len)

word_embedding = True

if word_embedding:
    print('Embedding...')
    EMBEDDING_FILE = 'x:/E/CodeBases/AI/NLP/Corpus/word2vec/sgns.baidubaike.bigram-char'
    embed_size = 300

    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE, encoding='utf-8'))

    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(vocab) + 1, embed_size))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
#        print(" — val_f1: % f — val_precision: % f — val_recall % f" % (_val_f1, _val_precision, _val_recall))
        return


def train_model_MLP():
    x_train = tokenizer.sequences_to_matrix(X_train_word_ids, mode='binary')
    x_test = tokenizer.sequences_to_matrix(X_test_word_ids, mode='binary')

    model = Sequential()
    model.add(Dense(128, input_shape=(len(vocab) + 1,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model_checkpoint = ModelCheckpoint('./model-MLP.h5', save_best_only=True)
    metrics = Metrics()
    hist = model.fit(x_train, y_train,
              batch_size=128,
              epochs=40,
              validation_data=(x_test, y_test),
              callbacks=[metrics, early_stopping, model_checkpoint])

    best_acc = max(hist.history['val_acc'])
    idx = np.argmax(hist.history['val_acc'])
    precision = metrics.val_precisions[idx]
    recall = metrics.val_recalls[idx]
    f1score = metrics.val_f1s[idx]

    del model, early_stopping, model_checkpoint, metrics
    gc.collect()

    return (best_acc, precision, recall, f1score)



def train_model_LSTM():
    model = Sequential()
    model.add(Embedding(len(vocab)+1, embed_size, weights=[embedding_matrix], input_length=max_len, trainable=True))
#    model.add(Embedding(len(vocab)+1, embed_size, input_length=max_len))
    model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model_checkpoint = ModelCheckpoint('model-LSTM.h5', save_best_only=True)
    metrics = Metrics()
    hist = model.fit(X_train_padded_seqs, y_train,
              batch_size=128,
              epochs=100,
              validation_data=(X_test_padded_seqs, y_test),
              callbacks=[metrics, early_stopping, model_checkpoint])

    best_acc = max(hist.history['val_acc'])
    idx = np.argmax(hist.history['val_acc'])
    precision = metrics.val_precisions[idx]
    recall = metrics.val_recalls[idx]
    f1score = metrics.val_f1s[idx]

    del model, early_stopping, model_checkpoint, metrics
    gc.collect()

    return (best_acc, precision, recall, f1score)



def train_model_TextCNN():
    main_input = Input(shape=(max_len,), dtype='float64')
    embedder = Embedding(len(vocab) + 1, embed_size, input_length=max_len)
    embed = embedder(main_input)
    cnn1 = Convolution1D(256, 3, padding='same', strides=1, activation='relu')(embed)
    cnn1 = MaxPool1D(pool_size=4)(cnn1)
    cnn2 = Convolution1D(256, 4, padding='same', strides=1, activation='relu')(embed)
    cnn2 = MaxPool1D(pool_size=4)(cnn2)
    cnn3 = Convolution1D(256, 5, padding='same', strides=1, activation='relu')(embed)
    cnn3 = MaxPool1D(pool_size=4)(cnn3)
    cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
    flat = Flatten()(cnn)
    drop = Dropout(0.2)(flat)
    main_output = Dense(1, activation='sigmoid')(drop)
    model = Model(inputs=main_input, outputs=main_output)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model_checkpoint = ModelCheckpoint('model-TextCNN.h5', save_best_only=True)
    metrics = Metrics()

    hist = model.fit(X_train_padded_seqs, y_train,
              batch_size=128,
              epochs=20,
              validation_data=(X_test_padded_seqs, y_test),
              callbacks=[early_stopping, metrics, model_checkpoint])

    best_acc = max(hist.history['val_acc'])
    idx = np.argmax(hist.history['val_acc'])
    precision = metrics.val_precisions[idx]
    recall = metrics.val_recalls[idx]
    f1score = metrics.val_f1s[idx]

    del model, early_stopping, model_checkpoint, metrics
    gc.collect()

    return (best_acc, precision, recall, f1score)


def eval_models():
    scores = []

    score = ['NN(MLP)']
    score.extend(train_model_MLP())
    scores.append(score)

    score = ['CNN(TextCNN)']
    score.extend(train_model_TextCNN())
    scores.append(score)

    score = ['RNN(LSTM)']
    score.extend(train_model_LSTM())
    scores.append(score)

    df = pd.DataFrame(scores).T
    df.index = ['model', 'accuracy', 'precision', 'recall', 'f1score']
    df.columns = df.iloc[0]
    df.drop(df.index[[0]], inplace=True)
    df = df.apply(pd.to_numeric, errors='ignore')

    return df

if __name__ == '__main__':
    df = eval_models()
    df.to_csv('model_dl_scores.csv', float_format='%.4f')
