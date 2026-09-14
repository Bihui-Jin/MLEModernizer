# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Predict the sentiment of phrases.

## Metric
Classification accuracy.

## Submission Format
For each phrase in the test set, predict a label for the sentiment. Your submission should have a header and look like the following:

```
PhraseId,Sentiment
156061,2
156062,2
156063,2
...
```

## Dataset
The dataset is comprised of tab-separated files with phrases. Each phrase has a PhraseId. Each sentence has a SentenceId.

The sentiment labels are:

0 - negative

1 - somewhat negative

2 - neutral

3 - somewhat positive

4 - positive

# 2. Python version

3.7

# 3. Installed packages

gensim==4.4.0
geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
nltk==3.9.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (72 lines)
            sampleSubmission.csv (46819 lines)
            sampleSubmission.csv.zip (146.0 kB)
            test.tsv (46819 lines)
            test.tsv.zip (1.1 MB)
            train.tsv (109243 lines)
            train.tsv.zip (2.7 MB)
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
        input/
            description.md (72 lines)
            sampleSubmission.csv (46819 lines)
            sampleSubmission.csv.zip (146.0 kB)
            test.tsv (46819 lines)
            test.tsv.zip (1.1 MB)
            train.tsv (109243 lines)
            train.tsv.zip (2.7 MB)
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
        working/
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
```

-> data/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv has 46818 rows and 2 columns.
Here is some information about the columns:
PhraseId (int64) has range: 29.00 - 156030.00, 0 nan values
Sentiment (int64) has 1 unique values: [2]

-> data/sampleSubmission.csv has 46818 rows and 2 columns.
Here is some information about the columns:
PhraseId (int64) has range: 29.00 - 156030.00, 0 nan values
Sentiment (int64) has 1 unique values: [2]

-> (stopped after 10 files for performance)

# 5. Target score

0.63392

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os, sys

print(os.listdir("../input"))

path = "../input/train.tsv"
features = ["pid", "sid", "p", "s"]
sms = pd.read_csv(path, names=features, sep="\t", header=0)
print(sms.shape)
print(sms.head(10))



## === cell 1
import matplotlib.pyplot as plt


def plot_bars(df, cols):
    for col in cols:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.gca()
        counts = df[col].value_counts()
        counts.plot.bar(ax=ax, color="blue")
        ax.set_title("Number sentiments " + col)
        ax.set_xlabel(col)
        ax.set_ylabel("freq")
        plt.show()


plot_cols = ["s"]
plot_bars(sms, plot_cols)
print(sms.s.value_counts())



## === cell 2
X = sms.p
Y = sms.s



## === cell 3
import nltk

nltk.download("punkt")
from nltk.stem import PorterStemmer

ps = PorterStemmer()
review = []
for row in X:
    tokens = [ps.stem(w.lower()) for w in nltk.word_tokenize(row)]
    review.append(" ".join(tokens))
X = review
print(X[:1])



## === cell 4
from sklearn.model_selection import train_test_split

X_train, X_inter, Y_train, Y_inter = train_test_split(
    X, Y, test_size=0.3, random_state=1, stratify=Y
)
X_val, X_test, Y_val, Y_test = train_test_split(
    X_inter, Y_inter, test_size=0.5, random_state=1, stratify=Y_inter
)
print(len(X_train), len(X_val), len(X_test))



## === cell 5
import gensim
import nltk

data_words = [nltk.word_tokenize(x) for x in X_train]
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=10)
trigram = gensim.models.Phrases(bigram[data_words], threshold=10)

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


data_words_bigrams = make_trigrams(data_words)
X_train = [" ".join(doc) for doc in data_words_bigrams]

data_words = [nltk.word_tokenize(x) for x in X_val]
data_words_bigrams = make_trigrams(data_words)
X_val = [" ".join(doc) for doc in data_words_bigrams]

data_words = [nltk.word_tokenize(x) for x in X_test]
data_words_bigrams = make_trigrams(data_words)
X_test = [" ".join(doc) for doc in data_words_bigrams]



## === cell 6
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical  # replace deprecated np_utils

max_sentence = len(max(X, key=len))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

encoded_docs = tokenizer.texts_to_sequences(X_train)
train_x = pad_sequences(encoded_docs, maxlen=max_sentence, padding="post")
print(train_x[0])

encoded_docs = tokenizer.texts_to_sequences(X_val)
val_x = pad_sequences(encoded_docs, maxlen=max_sentence, padding="post")
print(val_x[1])

encoded_docs = tokenizer.texts_to_sequences(X_test)
test_x = pad_sequences(encoded_docs, maxlen=max_sentence, padding="post")
print(test_x[1])

encoder = LabelEncoder()
encoder.fit(Y_train)
encoded_Y_train = encoder.transform(Y_train)
dummy_y_train = to_categorical(encoded_Y_train)
print(dummy_y_train[:3])

encoded_Y_val = encoder.transform(Y_val)
dummy_y_val = to_categorical(encoded_Y_val)

vocab_size = len(tokenizer.word_index) + 1



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 7
from keras.models import Sequential
from keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D

model = Sequential()
model.add(
    Embedding(
        input_dim=vocab_size, output_dim=200, input_length=max_sentence, trainable=True
    )
)
model.add(Conv1D(128, 2, strides=1, padding="valid", activation="relu"))
model.add(GlobalMaxPooling1D())
model.add(Dense(5, activation="softmax"))
print(model.summary())
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(
    train_x,
    dummy_y_train,
    validation_data=(val_x, dummy_y_val),
    epochs=2,
    batch_size=128,
    verbose=1,
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2057371669.py in <cell line: 0>()
      5 model.add(
      6     Embedding(
----> 7         input_dim=vocab_size, output_dim=200, input_length=max_sentence, trainable=True
      8     )
      9 )

NameError: name 'vocab_size' is not defined

## === cell 8
import sklearn.metrics as sklm

predictions = model.predict(test_x)
pred = [int(np.argmax(p)) for p in predictions]

print(len(Y_test), len(pred))
print("Accuracy  %0.4f" % sklm.accuracy_score(Y_test, pred))

test_path = "../input/test.tsv"
test_features = ["PhraseId", "sid", "p"]
test_frame = pd.read_csv(test_path, names=test_features, sep="\t", header=0)

review = []
for row in test_frame["p"]:
    tokens = [ps.stem(w.lower()) for w in nltk.word_tokenize(row)]
    review.append(" ".join(tokens))
test_frame["p_stemmed"] = review

data_words = [nltk.word_tokenize(x) for x in test_frame["p_stemmed"]]
data_words_bigrams = make_trigrams(data_words)
test_frame["p_ngrams"] = [" ".join(doc) for doc in data_words_bigrams]

encoded_docs = tokenizer.texts_to_sequences(test_frame["p_ngrams"])
temp_test = pad_sequences(encoded_docs, maxlen=max_sentence, padding="post")
test_preds = model.predict(temp_test)
test_frame["Sentiment"] = [int(np.argmax(p)) for p in test_preds]

submission = test_frame[["PhraseId", "Sentiment"]]
submission.to_csv("submission.csv", index=False)
print("Submission written to submission.csv")

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1691218293.py in <cell line: 0>()
      1 import sklearn.metrics as sklm
      2 
----> 3 predictions = model.predict(test_x)
      4 pred = [int(np.argmax(p)) for p in predictions]
      5 

NameError: name 'test_x' is not defined
