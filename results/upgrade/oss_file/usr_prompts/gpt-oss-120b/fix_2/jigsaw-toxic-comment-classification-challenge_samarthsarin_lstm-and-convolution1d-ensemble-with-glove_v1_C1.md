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

0.7941958474264363

# 6. Current score

0.96516

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.96516) has done: 'I fixed the missing imports, removed the unavailable NLTK and GloVe parts, replaced them with simple text cleaning, used TensorFlow’s keras instead of the old keras package, and switched the model to a multi‑label setup (sigmoid + binary_crossentropy). The code now runs end‑to‑end, creates the required `submission.csv`, and keeps the original pipeline logic while being compatible with the provided environment.'

# 9. Code solution

## === cell 0
import os, re, numpy as np, pandas as pd
from tqdm import tqdm

print("Input folder contents:", os.listdir("../input"))



## === cell 1
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
train_path = "../input/jigsaw-toxic-comment-classification-challenge/train.csv"
df = pd.read_csv(train_path)



## === cell 3
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = df[list_classes].values.astype(np.float32)




## === cell 4
def simple_clean(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["clean_comment"] = df["comment_text"].apply(simple_clean)



## === cell 5
token = Tokenizer()
token.fit_on_texts(df["clean_comment"])
seq = token.texts_to_sequences(df["clean_comment"])
pad_seq = pad_sequences(seq, maxlen=300)

vocab_size = len(token.word_index) + 1
embedding_dim = 100  # small random embeddings are sufficient for a baseline



## === cell 6
model = Sequential(
    [
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=300,
            trainable=True,
        ),  # random init, trainable
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(6, activation="sigmoid"),  # multi‑label
    ]
)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])



## === cell 7
history = model.fit(
    pad_seq, y, epochs=2, batch_size=256, validation_split=0.1, verbose=2
)



## === cell 8
test_path = "../input/jigsaw-toxic-comment-classification-challenge/test.csv"
test = pd.read_csv(test_path)
test["clean_comment"] = test["comment_text"].apply(simple_clean)

test_seq = token.texts_to_sequences(test["clean_comment"])
test_pad_seq = pad_sequences(test_seq, maxlen=300)



## === cell 9
predict = model.predict(test_pad_seq, batch_size=256)



## === cell 10
sample_sub_path = (
    "../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv"
)
sample_submission = pd.read_csv(sample_sub_path)
sample_submission[list_classes] = predict
sample_submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
