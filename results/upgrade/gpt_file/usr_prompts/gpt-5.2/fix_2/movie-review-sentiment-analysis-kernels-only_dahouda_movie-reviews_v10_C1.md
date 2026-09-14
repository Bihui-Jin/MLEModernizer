# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.9

# 3. Installed packages

No external packages required in the script and installed.

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

# 5. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")

pd.set_option("display.max_colwidth", None)

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

seed = 101
np.random.seed(seed)
tf.random.set_seed(seed)

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))



## === cell 1
BASE_PATH = "/kaggle/input/movie-review-sentiment-analysis-kernels-only"

train_path = os.path.join(BASE_PATH, "train.tsv.zip")
test_path = os.path.join(BASE_PATH, "test.tsv.zip")
sub_path = os.path.join(BASE_PATH, "sampleSubmission.csv")

train = pd.read_csv(train_path, sep="\t", encoding="utf-8")
test = pd.read_csv(test_path, sep="\t", encoding="utf-8")
sub = pd.read_csv(sub_path)

print(train.shape)
print(test.shape)
train.head()



## === cell 2
test.head()



## === cell 3
X = train["Phrase"].astype(str)
temp = test["Phrase"].astype(str)

num_classes = train["Sentiment"].nunique()
y = to_categorical(train["Sentiment"].values, num_classes=num_classes)

print("Number of classes:", num_classes)
print("X shape:", X.shape, "y shape:", y.shape)



## === cell 4
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=train["Sentiment"].values, random_state=seed
)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)



## === cell 5
X_train.head()



## === cell 6
X_test.head()



## === cell 7
max_features = 15000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train))

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
temp_seq = tokenizer.texts_to_sequences(temp)

max_words = 50
X_train_pad = sequence.pad_sequences(X_train_seq, maxlen=max_words)
X_test_pad = sequence.pad_sequences(X_test_seq, maxlen=max_words)
temp_pad = sequence.pad_sequences(temp_seq, maxlen=max_words)

print(X_train_pad.shape, X_test_pad.shape, temp_pad.shape)



## === cell 8
X_train_pad



## === cell 9
X_test_pad



## === cell 10
batch_size = 64
epochs = 25


def get_model(max_features_local, embed_dim, embedding_matrix):
    K.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)

    model = Sequential()
    model.add(
        Embedding(
            max_features_local,
            embed_dim,
            input_length=X_train_pad.shape[1],
            weights=[embedding_matrix],
        )
    )
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    print(model.summary())
    return model




## === cell 11
embed_dim = 100

word_index = tokenizer.word_index
num_words = min(max_features, len(word_index) + 1)

rng = np.random.default_rng(seed)
embedding_matrix = rng.normal(loc=0.0, scale=0.05, size=(num_words, embed_dim)).astype(
    "float32"
)

max_features_effective = embedding_matrix.shape[0]
print("Using random embedding matrix with shape:", embedding_matrix.shape)



## === cell 12
model = get_model(max_features_effective, embed_dim, embedding_matrix)
model.fit(
    X_train_pad,
    y_train,
    validation_data=(X_test_pad, y_test),
    epochs=epochs,
    batch_size=batch_size,
    verbose=2,
)



## === cell 13
probs = model.predict(temp_pad, batch_size=batch_size, verbose=0)
preds = np.argmax(probs, axis=1).astype(int)

submission = pd.DataFrame({"PhraseId": test["PhraseId"].values, "Sentiment": preds})
submission.to_csv("submission.csv", index=False)
submission.head()



## === cell 14
print("submission.csv rows:", submission.shape[0])
print("Unique sentiments predicted:", sorted(submission["Sentiment"].unique().tolist()))
submission.head()
