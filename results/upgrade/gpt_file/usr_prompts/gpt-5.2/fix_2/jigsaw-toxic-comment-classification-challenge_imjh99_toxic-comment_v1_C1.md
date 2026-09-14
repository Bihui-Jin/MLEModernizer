# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of comments from Wikipedia's talk page edits, predict the probability of each comment being toxic.

## Metric
Mean column-wise ROC AUC; the average of the individual AUCs of each predicted column.

## Submission Format
For each `id` in the test set, you must predict a probability for each of the six possible types of comment toxicity (toxic, severe_toxic, obscene, threat, insult, identity_hate). The columns must be in the same order as shown below. The file should contain a header and have the following format:

```
id,toxic,severe_toxic,obscene,threat,insult,identity_hate
00001cee341fdb12,0.5,0.5,0.5,0.5,0.5,0.5
0000247867823ef7,0.5,0.5,0.5,0.5,0.5,0.5
etc.
```

## Dataset 
- **train.csv** - the training set, contains comments with their binary labels
- **test.csv** - the test set, you must predict the toxicity probabilities for these comments.
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        input/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        working/
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
```

-> data/jigsaw-toxic-comment-classification-challenge/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/jigsaw-toxic-comment-classification-challenge/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/jigsaw-toxic-comment-classification-challenge/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np



## === cell 1
BASE_CANDIDATES = [
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge",
    "/kaggle/input/dataset",
    "/kaggle/input",
]


def _find_file(filename: str) -> str:
    for base in BASE_CANDIDATES:
        p = os.path.join(base, filename)
        if os.path.exists(p):
            return p
    for base in BASE_CANDIDATES:
        for root, _, files in os.walk(base):
            if filename in files:
                return os.path.join(root, filename)
    raise FileNotFoundError(
        f"Could not find {filename} under any of: {BASE_CANDIDATES}"
    )


train_path = _find_file("train.csv")
test_path = _find_file("test.csv")
sample_path = _find_file("sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample = pd.read_csv(sample_path)

print("Loaded:", train.shape, test.shape, sample.shape)
print("Paths:", train_path, test_path, sample_path)



## === cell 2
train.head()



## === cell 3
train[train["obscene"] > 0].head()



## === cell 4
train.isnull().any(), test.isnull().any()



## === cell 5
import matplotlib.pyplot as plt
import seaborn as sns

x = train.iloc[:, 2:].sum()
plt.figure(figsize=(8, 4))
ax = sns.barplot(x=x.index, y=x.values, alpha=0.8)
plt.title("# per class")
plt.ylabel("# of Occurrences", fontsize=12)
plt.xlabel("Type", fontsize=12)
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(
        rect.get_x() + rect.get_width() / 2,
        height + 5,
        int(label),
        ha="center",
        va="bottom",
    )
plt.show()



## === cell 6
temp_df = train.iloc[:, 2:]
corr = temp_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr,
    xticklabels=corr.columns.values,
    yticklabels=corr.columns.values,
    annot=True,
)
plt.show()



## === cell 7
print("toxic example (severe_toxic==1):")
print(train[train.severe_toxic == 1].iloc[3, 1])



## === cell 8
print("severe_toxic example:")
print(train[train.severe_toxic == 1].iloc[4, 1])



## === cell 9
print("Threat example:")
print(train[train.threat == 1].iloc[1, 1])



## === cell 10
print("Obscene example:")
print(train[train.obscene == 1].iloc[1, 1])



## === cell 11
print("identity_hate example:")
print(train[train.identity_hate == 1].iloc[4, 1])



## === cell 12
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values

train["comment_text"] = train["comment_text"].fillna("")
test["comment_text"] = test["comment_text"].fillna("")

list_sentences_train = train["comment_text"].astype(str)
list_sentences_test = test["comment_text"].astype(str)



## === cell 13
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)



## === cell 14
list_tokenized_train[:1]



## === cell 15
from tensorflow.keras.preprocessing.sequence import pad_sequences

maxlen = 200
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

print("X_t:", X_t.shape, "X_te:", X_te.shape, "y:", y.shape)



## === cell 16
totalNumWords = [len(one_comment) for one_comment in list_tokenized_train]
plt.hist(totalNumWords, bins=np.arange(0, 410, 10))
plt.show()



## === cell 17
from tensorflow.keras.layers import Input, LSTM, Dropout, Activation

inp = Input(shape=(maxlen,))



## === cell 18
from tensorflow.keras.layers import Embedding

embed_size = 128
x = Embedding(max_features, embed_size)(inp)



## === cell 19
x = LSTM(60, return_sequences=True, name="lstm_layer")(x)



## === cell 20
from tensorflow.keras.layers import GlobalMaxPool1D

x = GlobalMaxPool1D()(x)



## === cell 21
x = Dropout(0.1)(x)



## === cell 22
from tensorflow.keras.layers import Dense

x = Dense(50, activation="relu")(x)



## === cell 23
x = Dropout(0.1)(x)



## === cell 24
x = Dense(6, activation="sigmoid")(x)



## === cell 25
from tensorflow.keras.models import Model

model = Model(inputs=inp, outputs=x)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])



## === cell 26
batch_size = 32
epochs = 2
history = model.fit(X_t, y, batch_size=batch_size, epochs=epochs, validation_split=0.1)



## === cell 27
model.summary()



## === cell 28
predictions = model.predict(X_te, batch_size=1024, verbose=1)

submission = sample.copy()
submission[list_classes] = predictions
submission = submission[["id"] + list_classes]

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path, submission.shape)
print(submission.head())
