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

0.54686

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.55239) has done: 'I make minimal compatibility fixes so the notebook runs in this Kaggle environment: replace deprecated `sklearn.cross_validation` with `sklearn.model_selection`, and switch all Keras imports/usages to `tf_keras` (since `keras==3.x` removed `keras.preprocessing` and some legacy module paths). I also ensure NLTK tokenizers are available by downloading `punkt`/`punkt_tab` at runtime. Finally, I keep your existing preprocessing (stemming + n-grams) and CNN model intact, but make sure the pipeline trains once and writes a valid `submission.csv` with the required `PhraseId,Sentiment` columns.'
- What this solution (achieved 0.54686) has done: 'I fix the runtime crash in the tokenization/sequence step by removing the fragile `tf_keras.preprocessing`/`to_categorical` dependency that is triggering the protobuf `MessageFactory.GetPrototype` error in this environment, while keeping the exact same tokenization + padding + CNN architecture and training loop. I replace it with a minimal, equivalent pipeline using `sklearn` for integer-encoding labels and `tf_keras.utils.set_random_seed` for determinism. I also correct a subtle but important bug where `max_sentence` was computed from the full (pre-trigram) `X` instead of the actual trigram-processed training texts, which can unnecessarily inflate sequence length and hurt accuracy; this should nudge score upward toward your target without changing the model. The script still train once and always write a valid `submission.csv` with `PhraseId,Sentiment`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

INPUT_DIR = "/kaggle/input/movie-review-sentiment-analysis-kernels-only"
train_path = os.path.join(INPUT_DIR, "train.tsv")
test_path = os.path.join(INPUT_DIR, "test.tsv")

print("Input dir exists:", os.path.exists(INPUT_DIR))
print("Files:", sorted(os.listdir(INPUT_DIR))[:10])

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
    if isinstance(row, str) and (len(row) > 3):
        len_checker.append(1)
    else:
        len_checker.append(0)
sms["dummy"] = len_checker
sms = sms[sms["dummy"] == 1].copy()
sms.drop(columns=["dummy"], inplace=True)
print("After length filter:", sms.shape)



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
print("Resampled class counts:\n", sms["s"].value_counts().sort_index())



## === cell 4
X = sms.p.astype(str).tolist()
Y = sms.s.astype(int).tolist()

print("Example phrase:", X[0])
print("Example label:", Y[0])



## === cell 5
import nltk
from nltk.stem import PorterStemmer

nltk.download("punkt", quiet=True)
try:
    nltk.download("punkt_tab", quiet=True)
except Exception:
    pass

ps = PorterStemmer()
l2 = []
review = []
s2 = ""
for row in X:
    row = "" if row is None else str(row)
    for words in nltk.word_tokenize(row):
        l2.append(ps.stem(words.lower()))
        l2.append(" ")
    s2 = "".join(l2)
    review.append(s2)
    s2 = ""
    l2 = []
X = review
print(X[:1])



## === cell 6
from sklearn.model_selection import train_test_split

X_train, X_inter, Y_train, Y_inter = train_test_split(
    X, Y, test_size=0.3, random_state=123, stratify=Y
)
X_val, X_test, Y_val, Y_test = train_test_split(
    X_inter, Y_inter, test_size=0.5, random_state=234, stratify=Y_inter
)
print(len(X_train), len(X_val), len(X_test))



## === cell 7
import gensim

data_words = [nltk.word_tokenize(x) for x in X_train]
print(data_words[:2])

bigram = gensim.models.Phrases(data_words, min_count=5, threshold=10)
trigram = gensim.models.Phrases(bigram[data_words], threshold=10)

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


data_words_trigrams = make_trigrams(data_words)

token2 = []
for ritem in data_words_trigrams:
    token = ""
    for eachword in ritem:
        token = token + " " + eachword
    token2.append(token)
X_train = token2

data_words = [nltk.word_tokenize(x) for x in X_val]
data_words_trigrams = make_trigrams(data_words)
token2 = []
for ritem in data_words_trigrams:
    token = ""
    for eachword in ritem:
        token = token + " " + eachword
    token2.append(token)
X_val = token2

data_words = [nltk.word_tokenize(x) for x in X_test]
data_words_trigrams = make_trigrams(data_words)
token2 = []
for ritem in data_words_trigrams:
    token = ""
    for eachword in ritem:
        token = token + " " + eachword
    token2.append(token)
X_test = token2

print("X after trigrams:", X_train[:2])



## === cell 8
import tensorflow as tf
import tf_keras
from tf_keras.layers import TextVectorization
from sklearn.preprocessing import LabelEncoder
from tf_keras.utils import set_random_seed

set_random_seed(123)

max_sentence = int(max(len(s.split()) for s in X_train))
max_sentence = min(max_sentence, 80)

vectorizer = TextVectorization(
    standardize=None,
    split="whitespace",
    output_mode="int",
    output_sequence_length=max_sentence,
)

vectorizer.adapt(tf.data.Dataset.from_tensor_slices(X_train).batch(1024))

train_x = vectorizer(np.array(X_train)).numpy()
val_x = vectorizer(np.array(X_val)).numpy()
test_x = vectorizer(np.array(X_test)).numpy()

encoder = LabelEncoder()
encoder.fit(Y_train)
encoded_Y_train = encoder.transform(Y_train)
encoded_Y_val = encoder.transform(Y_val)

dummy_y_train = encoded_Y_train.astype("int32")
dummy_y_val = encoded_Y_val.astype("int32")

vocab_size = len(vectorizer.get_vocabulary())
print(
    "vocab_size:",
    vocab_size,
    "max_sentence:",
    max_sentence,
    "train_x shape:",
    train_x.shape,
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 9
from tf_keras.models import Sequential
from tf_keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D

model = Sequential()
model.add(
    Embedding(
        input_dim=vocab_size, output_dim=100, input_length=max_sentence, trainable=True
    )
)
model.add(Conv1D(128, 2, strides=1, padding="valid", activation="relu"))
model.add(GlobalMaxPooling1D())
model.add(Dense(5, activation="softmax"))

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
print(model.summary())

model.fit(
    train_x,
    dummy_y_train,
    validation_data=(val_x, dummy_y_val),
    epochs=2,
    batch_size=128,
    verbose=1,
)



## === cell 10
import sklearn.metrics as sklm

predictions = model.predict(test_x, verbose=0)
pred = [int(np.argmax(v)) for v in predictions]

print("Holdout accuracy: %0.4f" % sklm.accuracy_score(Y_test, pred))



## === cell 11
test_features = ["PhraseId", "sid", "p"]
test_frame = pd.read_csv(test_path, names=test_features, sep="\t", header=0)

l2 = []
review = []
s2 = ""
for row in test_frame["p"].astype(str).tolist():
    for words in nltk.word_tokenize(row):
        l2.append(ps.stem(words.lower()))
        l2.append(" ")
    s2 = "".join(l2)
    review.append(s2)
    s2 = ""
    l2 = []
test_frame["p_stemmed"] = review

data_words = [nltk.word_tokenize(x) for x in test_frame["p_stemmed"].tolist()]
data_words_trigrams = make_trigrams(data_words)

token2 = []
for ritem in data_words_trigrams:
    token = ""
    for eachword in ritem:
        token = token + " " + eachword
    token2.append(token)
test_frame["p_ngrams"] = token2

temp_test = vectorizer(np.array(test_frame["p_ngrams"].tolist())).numpy()

predictions = model.predict(temp_test, verbose=0)
pred = [int(np.argmax(v)) for v in predictions]

submission = pd.DataFrame(
    {"PhraseId": test_frame["PhraseId"].astype(int), "Sentiment": pred}
)
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
