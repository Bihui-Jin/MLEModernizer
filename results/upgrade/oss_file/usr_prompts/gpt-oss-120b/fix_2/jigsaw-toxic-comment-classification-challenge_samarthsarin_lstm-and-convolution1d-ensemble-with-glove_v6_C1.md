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

3.7

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

# 5. Target score

0.9680068143258534

# 6. Current score

0.82472

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.82472) has done: 'The fix updates the imports to use `tensorflow.keras` (avoiding the protobuf error), ensures `tqdm` is correctly imported, adds a safe fallback when the GloVe file is missing (using random embeddings), replaces `CuDNNLSTM` with a standard `LSTM` if unavailable, and restores the full preprocessing‑training‑prediction pipeline so that a proper `submission.csv` is written.'

# 9. Code solution

## === cell 0
import os, numpy as np, pandas as pd
from tqdm import tqdm

print(os.listdir("../input"))



## === cell 1
train_path = "../input/jigsaw-toxic-comment-classification-challenge/train.csv"
df = pd.read_csv(train_path)



## === cell 2
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = df[list_classes].values



## === cell 3
token = pd.Series(df["comment_text"]).astype(str)  # ensure string type
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(token)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
seq = tokenizer.texts_to_sequences(token)
from tensorflow.keras.preprocessing.sequence import pad_sequences

pad_seq = pad_sequences(seq, maxlen=50)



## === cell 5
embedding_dim = 50
embedding_path = "../input/gloveembeddings/glove.6B.50d.txt"
embedding_values = {}
if os.path.exists(embedding_path):
    with open(embedding_path, "r", encoding="utf8") as f:
        for line in tqdm(f, desc="Loading GloVe"):
            parts = line.rstrip().split(" ")
            word = parts[0]
            vect = np.asarray(parts[1:], dtype="float32")
            embedding_values[word] = vect

if embedding_values:
    all_embs = np.stack(list(embedding_values.values()))
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
else:
    emb_mean, emb_std = 0.0, 0.01



## === cell 6
vocab_size = len(tokenizer.word_index) + 1
embedding_matrix = np.random.normal(emb_mean, emb_std, (vocab_size, embedding_dim))
for word, i in tqdm(tokenizer.word_index.items(), desc="Building embedding matrix"):
    vec = embedding_values.get(word)
    if vec is not None:
        embedding_matrix[i] = vec



## === cell 7
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    GlobalMaxPool1D,
    Dense,
    Dropout,
)

model = Sequential()
model.add(
    Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=50,
        weights=[embedding_matrix],
        trainable=False,
    )
)
model.add(Bidirectional(LSTM(50, return_sequences=True)))
model.add(GlobalMaxPool1D())
model.add(Dense(50, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(6, activation="sigmoid"))



## === cell 8
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(pad_seq, y, epochs=2, batch_size=32, validation_split=0.1, verbose=2)



## === cell 9
test_path = "../input/jigsaw-toxic-comment-classification-challenge/test.csv"
test = pd.read_csv(test_path)
x_test = test["comment_text"].astype(str)
test_seq = tokenizer.texts_to_sequences(x_test)
test_pad_seq = pad_sequences(test_seq, maxlen=50)
predict = model.predict(test_pad_seq, batch_size=256)



## === cell 10
sample_sub_path = (
    "../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv"
)
sample_submission = pd.read_csv(sample_sub_path)
sample_submission[list_classes] = predict
sample_submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
