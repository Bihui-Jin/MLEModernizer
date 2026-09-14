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

0.32619

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.54737) has done: 'I fix the runtime errors caused by deprecated scikit-learn imports and the Keras 3 API changes by switching to `sklearn.model_selection.train_test_split` and `tf_keras` equivalents for `Tokenizer`, `pad_sequences`, `to_categorical`, and layers. I also make sure NLTK tokenization works in the Kaggle environment by downloading the required tokenizer resource, and ensure the n-gram helper functions are defined before they’re used for test preprocessing. Finally, I produce a valid submission CSV with exactly the required columns `PhraseId` and `Sentiment`, written with a `.csv` suffix, and keep the model/training core logic (Embedding + Conv1D + GlobalMaxPooling + Dense softmax, same losses/epochs/batch size) unchanged.'
- What this solution (achieved 0.53969) has done: 'I fix the runtime failure in the Keras preprocessing imports by switching from `keras` to `tf_keras`, which is the compatible backend in this Kaggle environment and prevents the protobuf `MessageFactory.GetPrototype` crash. I also keep your existing modeling/training logic intact, but ensure variables like `tokenizer`, `vocab_size`, and `test_x` are defined by making cell 7 complete successfully. Finally, I make the pipeline run end-to-end and always write a valid `submission.csv` with exactly `PhraseId,Sentiment` using the trained model and the same preprocessing steps applied to test.'
- What this solution (achieved 0.54831) has done: 'I fix the crash in cell 7 caused by an incompatibility between `tf_keras` and the environment’s protobuf runtime by switching preprocessing/model imports to `tensorflow.keras`, which is the stable, supported API here. This change is API-level only and preserves your exact core logic: same stemming, same gensim n-grams, same Tokenizer/padding, same CNN architecture, and the same training loop/epochs/batch size. I also add deterministic seeds to keep results stable and make sure the submission rows align exactly with `PhraseId` from `test.tsv`. Since your current score is already well above the target, I’m not making any score-tuning changes—only runtime/stability fixes to ensure a valid `submission.csv` is always produced.'
- What this solution (achieved 0.5418) has done: 'I fix the protobuf-related crash coming from importing/using `tensorflow.keras` in this Kaggle environment by switching Keras usage to `tf_keras`, which is installed and avoids the `MessageFactory.GetPrototype` error. I keep your preprocessing (stemming + gensim n-grams), CNN architecture, and training loop unchanged, only adjusting the Keras import paths so the same logic runs end-to-end. I also make sure the label encoding stays aligned by fitting the encoder on all labels (0–4) so inference always maps correctly, without changing the training targets. Finally, I ensure a valid `submission.csv` is always written with exactly `PhraseId,Sentiment`.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

os.environ["PYTHONHASHSEED"] = "0"
random.seed(0)
np.random.seed(0)

INPUT_DIR = "../input"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "/kaggle/input"

print("INPUT_DIR =", INPUT_DIR)
print("Top-level entries:", os.listdir(INPUT_DIR)[:20])

train_path = os.path.join(INPUT_DIR, "train.tsv")
test_path = os.path.join(INPUT_DIR, "test.tsv")

features = ["pid", "sid", "p", "s"]
sms = pd.read_csv(train_path, names=features, sep="\t", header=0)
print(sms.shape)
print(sms.head(10))



## === cell 1
import matplotlib.pyplot as plt


def plot_bars(auto_prices, cols):
    for col in cols:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.gca()
        counts = auto_prices[col].value_counts()
        counts.plot.bar(ax=ax, color="blue")
        ax.set_title("Number sentiments " + col)
        ax.set_xlabel(col)
        ax.set_ylabel("freq")
        plt.show()


plot_cols = ["s"]
plot_bars(sms, plot_cols)
print(sms.s.value_counts())



## === cell 2
len_checker = []
for row in sms["p"]:
    if isinstance(row, str) and len(row) > 3:
        len_checker.append(1)
    else:
        len_checker.append(0)
sms["dummy"] = len_checker
sms = sms[sms["dummy"] == 1].copy()
sms.drop(columns=["dummy"], inplace=True)
sms.head(10)



## === cell 3
import numpy.random as nr

Labels = np.array(sms.s)
Features = np.array(sms.p)

temp_Labels_4 = Labels[Labels == 4]
temp_Features_4 = Features[Labels == 4]

temp_Labels_2 = Labels[Labels == 2]
temp_Features_2 = Features[Labels == 2]
indx2 = nr.choice(temp_Features_2.shape[0], temp_Features_4.shape[0], replace=True)

temp_Labels_3 = Labels[Labels == 3]
temp_Features_3 = Features[Labels == 3]
indx3 = nr.choice(temp_Features_3.shape[0], temp_Features_4.shape[0], replace=True)

temp_Labels_1 = Labels[Labels == 1]
temp_Features_1 = Features[Labels == 1]
indx1 = nr.choice(temp_Features_1.shape[0], temp_Features_4.shape[0], replace=True)

l0 = Labels[Labels == 0]
f0 = Features[Labels == 0]

X = np.concatenate((f0, temp_Features_2[indx2,]), axis=0)
Y = np.concatenate((l0, temp_Labels_2[indx2,]), axis=0)

X = np.concatenate((X, temp_Features_3[indx3,]), axis=0)
Y = np.concatenate((Y, temp_Labels_3[indx3,]), axis=0)

X = np.concatenate((X, temp_Features_1[indx1,]), axis=0)
Y = np.concatenate((Y, temp_Labels_1[indx1,]), axis=0)

X = np.concatenate((temp_Features_4, X), axis=0)
Y = np.concatenate((temp_Labels_4, Y), axis=0)

sms = pd.DataFrame({"p": X, "s": Y})
print("done, balanced shape:", sms.shape)
print(sms["s"].value_counts())



## === cell 4
import nltk
from nltk.stem import PorterStemmer

nltk.download("punkt", quiet=True)
try:
    nltk.download("punkt_tab", quiet=True)
except Exception:
    pass

ps = PorterStemmer()

X = sms.p
Y = sms.s

l2 = []
review = []
s2 = ""
for row in X:
    for words in nltk.word_tokenize(row):
        l2.append(ps.stem(words.lower()))
        l2.append(" ")
    s2 = "".join(l2)
    review.append(s2)
    s2 = ""
    l2 = []
X = review
print(X[:1])



## === cell 5
from sklearn.model_selection import train_test_split

X_train, X_inter, Y_train, Y_inter = train_test_split(
    X, Y, test_size=0.3, random_state=123, stratify=Y
)
X_val, X_test, Y_val, Y_test = train_test_split(
    X_inter, Y_inter, test_size=0.5, random_state=234, stratify=Y_inter
)
print(len(X_train))
print(len(X_val))
print(len(X_test))



## === cell 6
import gensim

data_words = [nltk.word_tokenize(x) for x in X_train]
print(data_words[:3])

bigram = gensim.models.Phrases(data_words, min_count=5, threshold=10)
trigram = gensim.models.Phrases(bigram[data_words], threshold=10)

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def docs_to_space_joined(docs):
    token2 = []
    token = ""
    for ritem in docs:
        for eachword in ritem:
            token = token + " " + eachword
        token2.append(token)
        token = ""
    return token2


data_words_trigrams = make_trigrams(data_words)
X_train = docs_to_space_joined(data_words_trigrams)
print("X after trigrams:")
print(X_train[:3])

data_words = [nltk.word_tokenize(x) for x in X_val]
data_words_trigrams = make_trigrams(data_words)
X_val = docs_to_space_joined(data_words_trigrams)

data_words = [nltk.word_tokenize(x) for x in X_test]
data_words_trigrams = make_trigrams(data_words)
X_test = docs_to_space_joined(data_words_trigrams)



## === cell 7
import tensorflow as tf
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

tf.random.set_seed(0)

max_sentence = len(max(X_train, key=len))

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
encoder.fit([0, 1, 2, 3, 4])

encoded_Y_train = encoder.transform(Y_train)
dummy_y_train = to_categorical(encoded_Y_train, num_classes=5)
print(dummy_y_train[:3])

encoded_Y_val = encoder.transform(Y_val)
dummy_y_val = to_categorical(encoded_Y_val, num_classes=5)

vocab_size = len(tokenizer.word_index) + 1
print("vocab_size:", vocab_size, "max_sentence:", max_sentence)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 8
from keras.models import Sequential
from keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D

model = Sequential()
model.add(
    Embedding(
        input_dim=vocab_size, output_dim=100, input_length=max_sentence, trainable=True
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



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3119170462.py in <cell line: 0>()
      5 model.add(
      6     Embedding(
----> 7         input_dim=vocab_size, output_dim=100, input_length=max_sentence, trainable=True
      8     )
      9 )

NameError: name 'vocab_size' is not defined

## === cell 9
import sklearn.metrics as sklm

predictions = model.predict(test_x, verbose=0)
pred = [int(np.argmax(v)) for v in predictions]

print(len(Y_test))
print(len(pred))
print(set(Y_test))
print(set(pred))

metrics = sklm.precision_recall_fscore_support(
    Y_test, pred, labels=[0, 1, 2, 3, 4], zero_division=0
)
print("Accuracy  %0.4f" % sklm.accuracy_score(Y_test, pred))



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1637516528.py in <cell line: 0>()
      1 import sklearn.metrics as sklm
      2 
----> 3 predictions = model.predict(test_x, verbose=0)
      4 pred = [int(np.argmax(v)) for v in predictions]
      5 

NameError: name 'test_x' is not defined

## === cell 10
test_frame = pd.read_csv(test_path, names=["PhraseId", "sid", "p"], sep="\t", header=0)

l2 = []
review = []
s2 = ""
for row in test_frame["p"]:
    if not isinstance(row, str):
        row = ""
    for words in nltk.word_tokenize(row):
        l2.append(ps.stem(words.lower()))
        l2.append(" ")
    s2 = "".join(l2)
    review.append(s2)
    s2 = ""
    l2 = []
test_frame["p_stemmed"] = review

data_words = [nltk.word_tokenize(x) for x in test_frame["p_stemmed"]]
data_words_trigrams = make_trigrams(data_words)
test_frame["p_ngrams"] = docs_to_space_joined(data_words_trigrams)

encoded_docs = tokenizer.texts_to_sequences(test_frame["p_ngrams"])
temp_test = pad_sequences(encoded_docs, maxlen=max_sentence, padding="post")

predictions = model.predict(temp_test, verbose=0)
pred = [int(np.argmax(v)) for v in predictions]

submission = pd.DataFrame(
    {
        "PhraseId": test_frame["PhraseId"].astype(int).values,
        "Sentiment": np.asarray(pred, dtype=int),
    }
)
print(submission.head())
print(submission.shape)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3371258755.py in <cell line: 0>()
     20 test_frame["p_ngrams"] = docs_to_space_joined(data_words_trigrams)
     21 
---> 22 encoded_docs = tokenizer.texts_to_sequences(test_frame["p_ngrams"])
     23 temp_test = pad_sequences(encoded_docs, maxlen=max_sentence, padding="post")
     24 

NameError: name 'tokenizer' is not defined

## === cell 11
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with columns:", submission.columns.tolist())
print("Unique predicted sentiments:", sorted(submission["Sentiment"].unique().tolist()))

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/628495362.py in <cell line: 0>()
----> 1 submission.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with columns:", submission.columns.tolist())
      3 print("Unique predicted sentiments:", sorted(submission["Sentiment"].unique().tolist()))

NameError: name 'submission' is not defined
