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

0.82325

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.96114) has done: 'I replace the faulty Keras imports with TensorFlow‑Keras equivalents, download the required NLTK resources, fix the seaborn barplot call, ensure the tokenizer and callbacks are defined, and adjust the model checkpoint/early‑stopping to monitor validation AUC (a small boost to the metric). The resulting script runs end‑to‑end and writes a proper `submission.csv` file.'

# 9. Code solution

## === cell 0
"""Import libraries, download NLTK resources, and set up TensorFlow‑Keras"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Conv1D,
    GlobalMaxPooling1D,
    Dense,
    LeakyReLU,
)
from tensorflow.keras.metrics import AUC




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
"""Define paths and load training data (with fallback to current directory)"""


def locate_file(rel_path):
    """Return the first existing path among several common locations."""
    candidates = [
        rel_path,
        os.path.join(
            "data",
            "jigsaw-toxic-comment-classification-challenge",
            os.path.basename(rel_path),
        ),
        os.path.join(
            "kaggle",
            "data",
            "jigsaw-toxic-comment-classification-challenge",
            os.path.basename(rel_path),
        ),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"Could not find {rel_path}")


train_path = locate_file("train.csv")
test_path = locate_file("test.csv")
sample_sub_path = locate_file("sample_submission.csv")

dataset = pd.read_csv(train_path)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/636954651.py in <cell line: 0>()
     24 
     25 
---> 26 train_path = locate_file("train.csv")
     27 test_path = locate_file("test.csv")
     28 sample_sub_path = locate_file("sample_submission.csv")

/tmp/ipykernel_55/636954651.py in locate_file(rel_path)
     21         if os.path.isfile(p):
     22             return p
---> 23     raise FileNotFoundError(f"Could not find {rel_path}")
     24 
     25 

FileNotFoundError: Could not find train.csv

## === cell 2
"""Basic EDA and label distribution barplot (fixed)"""
print("Number of rows in data =", dataset.shape[0])
print("Number of columns in data =", dataset.shape[1])
print("\n---Sample data---")
print(dataset.head())

labels = dataset.columns.values[2:]
labels_count = dataset.iloc[:, 2:].sum().values
sns.set(font_scale=2)
plt.figure(figsize=(15, 8))
sns.barplot(x=labels, y=labels_count)  # fixed positional arguments
plt.title("Comments vs. Label", fontsize=24)
plt.ylabel("Number of comments", fontsize=20)
plt.xlabel("Label", fontsize=20)
for idx, rect in enumerate(plt.gca().patches):
    height = rect.get_height()
    plt.gca().text(
        rect.get_x() + rect.get_width() / 2,
        height + 5,
        f"{int(labels_count[idx])}",
        ha="center",
        va="bottom",
        fontsize=18,
    )
plt.show()




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2324749945.py in <cell line: 0>()
      1 """Basic EDA and label distribution barplot (fixed)"""
----> 2 print("Number of rows in data =", dataset.shape[0])
      3 print("Number of columns in data =", dataset.shape[1])
      4 print("\n---Sample data---")
      5 print(dataset.head())

NameError: name 'dataset' is not defined

## === cell 3
"""Text preprocessing function"""
stop_words = set(stopwords.words("english"))
stop_words.update(
    [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "may",
        "also",
        "across",
        "among",
        "beside",
        "however",
        "yet",
        "within",
    ]
)
lemmatizer = WordNetLemmatizer()


def clean_text(text_series):
    processed = []
    for txt in text_series:
        txt = txt[0]  # because input is shape (n,1)
        txt = re.sub(r"[^\w\s]", "", txt, flags=re.UNICODE)
        txt = re.sub("\n", " ", txt, flags=re.UNICODE)
        txt = re.sub("<.*?>", "", txt)
        txt = txt.lower()
        tokens = [lemmatizer.lemmatize(tok) for tok in txt.split()]
        tokens = [lemmatizer.lemmatize(tok, "v") for tok in tokens]
        tokens = [w for w in tokens if w not in stop_words]
        processed.append(" ".join(tokens))
    return processed




## === cell 4
"""Extract features/labels and clean text"""
X_raw = dataset.iloc[:, 1:2].values  # comment_text column
y_train = dataset.iloc[:, 2:8].values  # six toxicity labels
X_clean = clean_text(X_raw)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2118467264.py in <cell line: 0>()
      1 """Extract features/labels and clean text"""
----> 2 X_raw = dataset.iloc[:, 1:2].values  # comment_text column
      3 y_train = dataset.iloc[:, 2:8].values  # six toxicity labels
      4 X_clean = clean_text(X_raw)
      5 

NameError: name 'dataset' is not defined

## === cell 5
"""Tokenize and pad sequences"""
vocab_size = 10000
maxlen = 300
embed_dim = 64  # modestly increased embedding dimension
batch_size = 64

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_clean)
seqs = tokenizer.texts_to_sequences(X_clean)
X_padded = pad_sequences(seqs, maxlen=maxlen, padding="post")




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/371014765.py in <cell line: 0>()
      6 
      7 tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
----> 8 tokenizer.fit_on_texts(X_clean)
      9 seqs = tokenizer.texts_to_sequences(X_clean)
     10 X_padded = pad_sequences(seqs, maxlen=maxlen, padding="post")

NameError: name 'X_clean' is not defined

## === cell 6
"""Define callbacks (monitoring validation AUC)"""
es = EarlyStopping(
    monitor="val_auc", mode="max", verbose=1, patience=2, restore_best_weights=True
)
mc = ModelCheckpoint(
    "model_best.keras", monitor="val_auc", mode="max", verbose=1, save_best_only=True
)




## === cell 7
"""Build a small TextCNN model"""
input_X = Input(shape=(maxlen,))
embed = Embedding(vocab_size, embed_dim, input_length=maxlen)(input_X)
conv = Conv1D(filters=64, kernel_size=3, activation="relu", padding="valid")(embed)
pool = GlobalMaxPooling1D()(conv)
dense = Dense(64)(pool)
dense = LeakyReLU(alpha=0.2)(dense)
output = Dense(6, activation="sigmoid", name="output_layer")(dense)

model = Model(inputs=input_X, outputs=output)
model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["accuracy", AUC(name="auc")]
)
model.summary()




## === cell 8
"""Train the model"""
model.fit(
    X_padded,
    y_train,
    epochs=15,  # a few more epochs for better AUC
    batch_size=batch_size,
    validation_split=0.2,
    callbacks=[es, mc],
    verbose=1,
)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1734591575.py in <cell line: 0>()
      1 """Train the model"""
      2 model.fit(
----> 3     X_padded,
      4     y_train,
      5     epochs=15,  # a few more epochs for better AUC

NameError: name 'X_padded' is not defined

## === cell 9
"""Prepare test data for inference"""
test_df = pd.read_csv(test_path)
X_test_raw = test_df.iloc[:, 1:2].values
X_test_clean = clean_text(X_test_raw)
test_seqs = tokenizer.texts_to_sequences(X_test_clean)
X_test_padded = pad_sequences(test_seqs, maxlen=maxlen, padding="post")




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/749365791.py in <cell line: 0>()
      1 """Prepare test data for inference"""
----> 2 test_df = pd.read_csv(test_path)
      3 X_test_raw = test_df.iloc[:, 1:2].values
      4 X_test_clean = clean_text(X_test_raw)
      5 test_seqs = tokenizer.texts_to_sequences(X_test_clean)

NameError: name 'test_path' is not defined

## === cell 10
"""Generate predictions and write a correctly‑sized submission file"""
y_pred = model.predict(X_test_padded, batch_size=512, verbose=1)

submission = pd.DataFrame(
    data=y_pred,
    columns=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
)
submission.insert(0, "id", test_df["id"])
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1856803726.py in <cell line: 0>()
      1 """Generate predictions and write a correctly‑sized submission file"""
----> 2 y_pred = model.predict(X_test_padded, batch_size=512, verbose=1)
      3 
      4 submission = pd.DataFrame(
      5     data=y_pred,

NameError: name 'X_test_padded' is not defined
