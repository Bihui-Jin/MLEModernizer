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

3.7

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
import numpy as np  # linear algebra
import pandas as pd  # data processing

import os

print(os.listdir("../input"))



## === cell 1
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from tqdm import tqdm

np.random.seed(42)
tf.random.set_seed(42)



## === cell 2
train = pd.read_csv(
    "../input/movie-review-sentiment-analysis-kernels-only/train.tsv", sep="\t"
)
test = pd.read_csv(
    "../input/movie-review-sentiment-analysis-kernels-only/test.tsv", sep="\t"
)
sub = pd.read_csv(
    "../input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv",
    sep=",",
)



## === cell 3
df = train[["Phrase", "Sentiment"]]



## === cell 4
df.head()



## === cell 5
embedding_dim = 300
glove_path = "../input/gloveembeddings/glove.6B.300d.txt"

embedding_values = {}
if os.path.exists(glove_path):
    with open(glove_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f, desc="Loading GloVe"):
            value = line.rstrip().split(" ")
            word = value[0]
            coef = np.asarray(value[1:], dtype="float32")
            if coef.shape[0] == embedding_dim:
                embedding_values[word] = coef
    print(f"Loaded {len(embedding_values)} GloVe vectors.")
else:
    print(f"GloVe file not found at: {glove_path}")
    print(
        "Proceeding with randomly initialized embedding matrix (trainable=False as in original)."
    )



## === cell 6
token = Tokenizer()
x = df["Phrase"].astype(str)
y = df["Sentiment"].astype(int)
y = to_categorical(y, num_classes=5)



## === cell 7
token.fit_on_texts(x)



## === cell 8
seq = token.texts_to_sequences(x)



## === cell 9
pad_seq = pad_sequences(seq, maxlen=300)



## === cell 10
vocab_size = len(token.word_index) + 1
print(vocab_size)



## === cell 11
embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)

if len(embedding_values) > 0:
    for word, i in tqdm(
        token.word_index.items(),
        total=len(token.word_index),
        desc="Building embedding matrix",
    ):
        values = embedding_values.get(word)
        if values is not None:
            embedding_matrix[i] = values
else:
    rng = np.random.RandomState(42)
    embedding_matrix[1:] = rng.normal(
        loc=0.0, scale=0.05, size=(vocab_size - 1, embedding_dim)
    ).astype(np.float32)



## === cell 12
x_train, x_test, y_train, y_test = train_test_split(
    pad_seq, y, test_size=0.3, random_state=42, stratify=np.argmax(y, axis=1)
)



## === cell 13
model = Sequential()



## === cell 14
model.add(
    Embedding(
        vocab_size,
        embedding_dim,
        input_length=300,
        weights=[embedding_matrix],
        trainable=False,
    )
)



## === cell 15
model.add(LSTM(75))



## === cell 16
model.add(Dense(128, activation="relu"))



## === cell 17
model.add(Dense(5, activation="softmax"))



## === cell 18
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])



## === cell 19
history = model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=4,
    validation_data=(x_test, y_test),
    verbose=2,
)



## === cell 20
test.head()



## === cell 21
test["Sentiment"] = ""



## === cell 22
test.head()



## === cell 23
testing_phrase = test["Phrase"].astype(str)



## === cell 24
test_seq = token.texts_to_sequences(testing_phrase)



## === cell 25
pad_test_seq = pad_sequences(test_seq, maxlen=300)



## === cell 26
proba = model.predict(pad_test_seq, batch_size=256, verbose=0)
predict = np.argmax(proba, axis=1).astype(int)



## === cell 27
predict[0]



## === cell 28
test["Sentiment"] = predict



## === cell 29
test.head()



## === cell 30
submission = test[["PhraseId", "Sentiment"]]



## === cell 31
submission.head()



## === cell 32
submission.to_csv("Submission.csv", index=False)
print("Wrote submission to Submission.csv with shape:", submission.shape)
