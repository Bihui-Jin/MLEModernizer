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

3.9

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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

0.946

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPool1D, Dense
from keras.metrics import AUC

MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
VALIDATION_SPLIT = 0.2




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def read_csv_smart(relative_path: str) -> pd.DataFrame:
    """
    Try a list of possible locations for the CSV file.
    """
    candidates = [
        Path(relative_path),
        Path("/kaggle/input") / Path(relative_path).name,
        Path("/kaggle/input/jigsaw-toxic-comment-classification-challenge")
        / Path(relative_path).name,
        Path("../input") / Path(relative_path).name,
    ]
    for p in candidates:
        if p.is_file():
            return pd.read_csv(p)
    raise FileNotFoundError(
        f"Unable to locate {relative_path} in any of the expected locations."
    )


train_path = "data/jigsaw-toxic-comment-classification-challenge/train.csv"
test_path = "data/jigsaw-toxic-comment-classification-challenge/test.csv"
sample_sub_path = (
    "data/jigsaw-toxic-comment-classification-challenge/sample_submission.csv"
)

df_train = read_csv_smart(train_path)
df_test = read_csv_smart(test_path)
sample_submission = read_csv_smart(sample_sub_path)



## === cell 2
train_texts = df_train["comment_text"].values
test_texts = df_test["comment_text"].values
label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
train_labels = df_train[label_cols].values
print("First comment text in training set:\n", train_texts[0])



## === cell 3
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

train_data = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

print("Shape of padded training data:", train_data.shape)
print("Sample padded sequence (last 20 tokens):", train_data[0][-20:])



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/650131076.py in <cell line: 0>()
----> 1 tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
      2 tokenizer.fit_on_texts(train_texts)
      3 train_sequences = tokenizer.texts_to_sequences(train_texts)
      4 test_sequences = tokenizer.texts_to_sequences(test_texts)
      5 

NameError: name 'Tokenizer' is not defined

## === cell 4
model = Sequential()
model.add(
    Embedding(input_dim=MAX_NUM_WORDS, output_dim=128, input_length=MAX_SEQUENCE_LENGTH)
)
model.add(Conv1D(filters=128, kernel_size=5, activation="relu"))
model.add(MaxPooling1D(pool_size=5))
model.add(Conv1D(filters=128, kernel_size=5, activation="relu"))
model.add(MaxPooling1D(pool_size=5))
model.add(Conv1D(filters=128, kernel_size=5, activation="relu"))
model.add(GlobalMaxPool1D())
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=6, activation="sigmoid"))

model.summary()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/861230832.py in <cell line: 0>()
----> 1 model = Sequential()
      2 model.add(
      3     Embedding(input_dim=MAX_NUM_WORDS, output_dim=128, input_length=MAX_SEQUENCE_LENGTH)
      4 )
      5 model.add(Conv1D(filters=128, kernel_size=5, activation="relu"))

NameError: name 'Sequential' is not defined

## === cell 5
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=[AUC(name="auc")])

X_train, X_val, y_train, y_val = train_test_split(
    train_data, train_labels, test_size=VALIDATION_SPLIT, random_state=123, shuffle=True
)
print("Training shapes:", X_train.shape, y_train.shape, X_val.shape, y_val.shape)

history = model.fit(
    X_train,
    y_train,
    batch_size=128,
    epochs=3,
    validation_data=(X_val, y_val),
    verbose=2,
)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/430412209.py in <cell line: 0>()
----> 1 model.compile(loss="binary_crossentropy", optimizer="adam", metrics=[AUC(name="auc")])
      2 
      3 X_train, X_val, y_train, y_val = train_test_split(
      4     train_data, train_labels, test_size=VALIDATION_SPLIT, random_state=123, shuffle=True
      5 )

NameError: name 'model' is not defined

## === cell 6
y_pred = model.predict(test_data, batch_size=128)

submission_df = pd.DataFrame(y_pred, columns=label_cols)
submission_df.insert(0, "id", df_test["id"])

submission_path = "sample_submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path} with shape:", submission_df.shape)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3504878079.py in <cell line: 0>()
----> 1 y_pred = model.predict(test_data, batch_size=128)
      2 
      3 submission_df = pd.DataFrame(y_pred, columns=label_cols)
      4 submission_df.insert(0, "id", df_test["id"])
      5 

NameError: name 'model' is not defined
