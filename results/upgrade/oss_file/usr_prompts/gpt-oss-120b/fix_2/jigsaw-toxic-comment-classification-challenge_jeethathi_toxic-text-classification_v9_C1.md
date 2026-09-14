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
seaborn==0.12.2
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

0.82914

# 6. Current score

0.93592

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.93592) has done: 'I replace the outdated Keras imports with TensorFlow Keras equivalents, re‑instantiate the tokenizer, pad the sequences, rebuild the models using the same architecture, and ensure the script writes a single valid `submission.csv` containing the required columns. This fixes the AttributeError, restores the missing variables, and guarantees a proper submission file.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import seaborn as sns
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train = pd.read_csv(
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv"
)
test = pd.read_csv(
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv"
)



## === cell 2
print(train.head())
print(test.head())



## === cell 3
print("Missing values in train:", train.isnull().any())
print("Missing values in test:", test.isnull().any())



## === cell 4
x_train = train["comment_text"].astype(str)
y_train = train[
    ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
]
x_test = test["comment_text"].astype(str)



## === cell 5
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 6
x_tokenized_train = tokenizer.texts_to_sequences(x_train)
x_tokenized_test = tokenizer.texts_to_sequences(x_test)



## === cell 7
lengths = [len(seq) for seq in x_tokenized_train]
print(f"The longest comment is {max(lengths)} tokens.")
sns.histplot(lengths, kde=False, bins=50)



## === cell 8
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_length = 200
X_train = pad_sequences(
    x_tokenized_train, maxlen=max_length, padding="post", truncating="post"
)
X_test = pad_sequences(
    x_tokenized_test, maxlen=max_length, padding="post", truncating="post"
)



## === cell 9
num_features = len(tokenizer.word_index)
embed_size = 128

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    GlobalAveragePooling1D,
    Dropout,
    Dense,
    LeakyReLU,
    Activation,
)

models = []

for _ in range(3):
    model = Sequential()
    model.add(Embedding(num_features + 1, embed_size, input_length=max_length))
    model.add(LSTM(64, return_sequences=True))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.1))
    model.add(Dense(48))
    model.add(LeakyReLU())
    model.add(Dropout(0.1))
    model.add(Dense(6, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    models.append(model)



## === cell 10
batch_size = 4096
validation_split = 0.1
epochs = 5

histories = []
for idx, model in enumerate(models):
    print(f"\nTraining model {idx+1}/{len(models)}")
    history = model.fit(
        X_train,
        y_train,
        validation_split=validation_split,
        batch_size=batch_size,
        epochs=epochs,
        verbose=2,
    )
    histories.append(history)



## === cell 11
y_preds = []
for idx, model in enumerate(models):
    print(f"Predicting with model {idx+1}")
    pred = model.predict(X_test, batch_size=batch_size, verbose=0)
    y_preds.append(pred)

y_pred_avg = np.mean(np.stack(y_preds, axis=0), axis=0)



## === cell 12
submission = pd.DataFrame(
    data=y_pred_avg,
    columns=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
)
submission.insert(0, "id", test["id"])
print(submission.head())



## === cell 13
output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission file written to {output_path}")
