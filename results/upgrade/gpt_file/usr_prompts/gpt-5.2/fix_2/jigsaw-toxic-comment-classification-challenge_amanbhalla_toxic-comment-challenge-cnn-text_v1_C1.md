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

0.94161

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.94161) has done: 'I fix the environment-breaking imports by switching to `tf_keras` (the Kaggle image has Keras 3, where `keras.preprocessing` and some legacy APIs are missing/broken), and I ensure all needed symbols (Tokenizer, pad_sequences, layers, callbacks, load_model) are imported correctly. I also make the NLTK resources available in a Kaggle-offline-safe way (download if missing, otherwise proceed), and adjust the seaborn barplot call to the current API so EDA cells run. Finally, I fix the ModelCheckpoint monitor (`val_acc` doesn’t exist here) and make sure the script always writes a valid `submission.csv` with the exact required columns and row alignment with `sample_submission.csv`.'

# 9. Code solution

## === cell 0
"""Importing the required libraries (fixed for Kaggle Keras 3 environment via tf_keras)."""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import tf_keras as keras
from tf_keras.preprocessing.text import Tokenizer
from tf_keras.preprocessing.sequence import pad_sequences
from tf_keras.callbacks import EarlyStopping, ModelCheckpoint
from tf_keras.models import load_model
from tf_keras.layers import (
    Input,
    Embedding,
    Conv1D,
    GlobalMaxPooling1D,
    Dense,
    LeakyReLU,
)
from tf_keras import Model

np.random.seed(42)
try:
    import tensorflow as tf

    tf.random.set_seed(42)
except Exception:
    pass

for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
"""Reading the training dataset and performing the EDA ( Exploratory Data Analysis ) in the upcoming cells"""

path = "../input/"
if not os.path.exists(path):
    path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/"

train_path_candidates = [
    os.path.join("../input", "train.csv"),
    os.path.join("/kaggle/input", "train.csv"),
    os.path.join(
        "/kaggle/input/jigsaw-toxic-comment-classification-challenge", "train.csv"
    ),
]
train_path = next((p for p in train_path_candidates if os.path.exists(p)), None)
if train_path is None:
    raise FileNotFoundError("Could not find train.csv in expected Kaggle input paths.")

dataset = pd.read_csv(train_path)




## === cell 2
"""Performing EDA ( Exploratory Data Analysis ) """

print("Number of rows in data = ", dataset.shape[0])
print("Number of columns in data = ", dataset.shape[1])
print("\n")
print("---Sample data---")
dataset.head()




## === cell 3
"""Performing EDA ( Exploratory Data Analysis ) (fixed seaborn API usage)"""

labels = dataset.columns.values[2:]
labels_count = dataset.iloc[:, 2:].sum().values

sns.set(font_scale=1.6)
plt.figure(figsize=(15, 8))

fig = sns.barplot(x=labels, y=labels_count)
plt.title("Comments vs. Label", fontsize=20)
plt.ylabel("Number of comments", fontsize=16)
plt.xlabel("Label", fontsize=16)

rects = fig.patches
for rect, label_count in zip(rects, labels_count):
    height = rect.get_height()
    fig.text(
        rect.get_x() + rect.get_width() / 2,
        height + 5,
        int(label_count),
        ha="center",
        va="bottom",
        fontsize=12,
    )
plt.tight_layout()
plt.show()




## === cell 4
"""Function for Text Preprocessing"""

try:
    stop_words = set(stopwords.words("english"))
except LookupError as e:
    raise LookupError(
        "NLTK stopwords corpus not found. Ensure nltk.download('stopwords') succeeded."
    ) from e

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


def clean_text(X):
    processed = []
    for text in X:
        text = text[0] if isinstance(text, (list, tuple, np.ndarray)) else text
        text = re.sub(r"[^\w\s]", "", str(text), flags=re.UNICODE)
        text = re.sub(r"\n", " ", text, flags=re.UNICODE)
        text = re.sub(r"<.*?>", "", text)
        text = text.lower()
        text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
        text = [lemmatizer.lemmatize(token, "v") for token in text]
        text = [word for word in text if word and (word not in stop_words)]
        processed.append(text)
    return processed




## === cell 5
"""Getting the X and Y and preprocessing them"""

X = dataset.iloc[:, 1:2].values
y_train = dataset.iloc[:, 2:8].values
X_train = clean_text(X)




## === cell 6
vocab_size = 10000
maxlen = 300
embed_dim = 20
batch_size = 64

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
tokenized_word_list = tokenizer.texts_to_sequences(X_train)
X_train_padded = pad_sequences(tokenized_word_list, maxlen=maxlen, padding="post")




## === cell 7
"""EarlyStopping and ModelCheckpoint (fix monitor key so it actually saves best model)"""

es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=2)

mc = ModelCheckpoint(
    "model_best.h5", monitor="val_accuracy", mode="max", verbose=1, save_best_only=True
)




## === cell 8
"""Creating TextCNN model for comment classification"""

input_X = Input(shape=(maxlen,))
embed = Embedding(vocab_size, embed_dim)(input_X)
conv_1 = Conv1D(filters=2, kernel_size=3, activation="relu", padding="valid")(embed)
out_1 = GlobalMaxPooling1D()(conv_1)
dense1 = Dense(32)(out_1)
dense1 = LeakyReLU(alpha=0.2)(dense1)
out = Dense(6, activation="sigmoid", name="output_layer")(dense1)

model = Model(input_X, out)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()




## === cell 9
"""Fitting the model"""

model.fit(
    X_train_padded,
    y_train,
    epochs=10,
    batch_size=batch_size,
    verbose=1,
    validation_split=0.2,
    callbacks=[es, mc],
)




## === cell 10
"""Importing the Test Data and making it ready to be passed to the Model"""

test_path_candidates = [
    os.path.join("../input", "test.csv"),
    os.path.join("/kaggle/input", "test.csv"),
    os.path.join(
        "/kaggle/input/jigsaw-toxic-comment-classification-challenge", "test.csv"
    ),
]
test_path = next((p for p in test_path_candidates if os.path.exists(p)), None)
if test_path is None:
    raise FileNotFoundError("Could not find test.csv in expected Kaggle input paths.")

dataset2 = pd.read_csv(test_path)
X_test = dataset2.iloc[:, 1:2].values
X_test = clean_text(X_test)
tokenized_word_list = tokenizer.texts_to_sequences(X_test)
X_test_padded = pad_sequences(tokenized_word_list, maxlen=maxlen, padding="post")




## === cell 11
"""Testing and creating the test results (ensure valid submission format/columns)"""

labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

if os.path.exists("model_best.h5"):
    model = load_model("model_best.h5")

y_test = model.predict(X_test_padded, batch_size=512, verbose=1)

sample_sub_candidates = [
    os.path.join("../input", "sample_submission.csv"),
    os.path.join("/kaggle/input", "sample_submission.csv"),
    os.path.join(
        "/kaggle/input/jigsaw-toxic-comment-classification-challenge",
        "sample_submission.csv",
    ),
]
sample_path = next((p for p in sample_sub_candidates if os.path.exists(p)), None)
if sample_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected Kaggle input paths."
    )

sample_submission = pd.read_csv(sample_path)

if len(sample_submission) != y_test.shape[0]:
    submission = pd.DataFrame({"id": dataset2["id"].values})
    for i, col in enumerate(labels):
        submission[col] = y_test[:, i]
else:
    submission = sample_submission.copy()
    submission[labels] = y_test

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
