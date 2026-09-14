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

0.78659940707927

# 6. Current score

0.88231

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.88231) has done: 'I fix the import/runtime issues by removing the NLTK dependency (it’s not needed by your pipeline and is failing due to missing corpora and a protobuf-related error) and by using `tf.keras` APIs that are stable in Kaggle’s environment. I also replace the missing GloVe file load with a randomly-initialized embedding matrix (so the model can still run end-to-end without external files), while keeping your model structure (Embedding → BiLSTM → Dense → Dense) and training loop intact. To match the multilabel AUC metric, I change the final activation to `sigmoid` and the loss to `binary_crossentropy` (the current `softmax` + `categorical_crossentropy` is incorrect for six independent labels and would severely hurt AUC). Finally, I ensure the submission is written with the correct columns/order into `submission.csv`.'
- What this solution (achieved 0.88231) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing TensorFlow to use the pure-Python protobuf implementation before importing it; this is a known Kaggle environment issue and is score-neutral. Since your current score (0.88231) is well above the target (0.7866) and higher-is-better, I avoid any model/training changes that could further improve the score and focus strictly on making the notebook run end-to-end reliably. I also make the dataset path resolution robust (support both `../input/...` and `/kaggle/input/...`) without changing what files are read. The script still write a valid `submission.csv` with the required columns/order.'
- What this solution (achieved 0.88231) has done: 'We fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing protobuf to use the pure-Python implementation *and* ensuring the setting is applied before any protobuf/TensorFlow-related imports, then clearing any preloaded protobuf modules that can cause the conflict in Kaggle’s runtime. This change is score-neutral and only unblocks execution so the notebook can train and write `submission.csv`. We not change the model, loss, data pipeline, or training loop to avoid moving the score further away from your target (your current score is already higher than target). Finally, we keep the existing robust path detection and ensure the submission is written with the required columns/order.'
- What this solution (achieved 0.88231) has done: 'We fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by ensuring the protobuf implementation environment variable is set early and by also setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` in a way that reliably takes effect in Kaggle, then clearing any already-imported protobuf modules before importing TensorFlow. This is runtime-only and score-neutral (no model/training/prediction logic changes). We also keep the robust input path detection and ensure the submission is written as `submission.csv` with the exact required columns/order. Everything else (tokenization, padding, Embedding→BiLSTM→Dense→sigmoid, loss, training loop) is preserved.'
- What this solution (achieved 0.88231) has done: 'I fix the TensorFlow import crash by ensuring the protobuf implementation env var is applied before any protobuf/TensorFlow import and by clearing any already-loaded protobuf modules, then importing TensorFlow with a safe fallback to the bundled `tf_keras` package if needed. This is a runtime-only change and is score-neutral (no model/training/prediction logic changes), keeping your architecture and training loop intact. I also add a couple of small stability settings (disable TF log spam, keep seeds) without affecting predictions materially. The pipeline then run end-to-end and write a valid `submission.csv` with the required columns/order.'
- What this solution (achieved 0.88231) has done: 'We fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by setting the protobuf env var *before any protobuf/tensorflow import* and by proactively clearing any already-loaded `google.protobuf` modules, then importing TensorFlow with a safe fallback to `tf_keras` if the environment provides it. This change is runtime-only and keeps your model, training loop, and prediction logic identical, so it should be score-neutral (and avoids moving your score further away from the target). We also keep the robust input path resolution and ensure `submission.csv` is always written with the exact required columns and order.'

# 9. Code solution

## === cell 0
import os
import sys
import random

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        del sys.modules[m]

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

for p in ("../input", "/kaggle/input"):
    if os.path.isdir(p):
        print("Input dir:", p)
        print(os.listdir(p)[:50])
        break




## === cell 1
tf = None
try:
    import tensorflow as tf  # noqa: F401
except Exception as e:
    for m in list(sys.modules.keys()):
        if m.startswith("google.protobuf"):
            del sys.modules[m]
    try:
        import tensorflow as tf  # noqa: F401
    except Exception:
        import tf_keras as tf  # type: ignore

tf.random.set_seed(42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
base_candidates = [
    "../input/jigsaw-toxic-comment-classification-challenge",
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge",
]
base_path = None
for c in base_candidates:
    if os.path.isfile(os.path.join(c, "train.csv")):
        base_path = c
        break
if base_path is None:
    raise FileNotFoundError(
        "Could not find train.csv in expected Kaggle input paths: "
        + ", ".join(base_candidates)
    )

train_path = os.path.join(base_path, "train.csv")
test_path = os.path.join(base_path, "test.csv")
sample_path = os.path.join(base_path, "sample_submission.csv")

print("Using base_path:", base_path)




## === cell 3
df = pd.read_csv(train_path)




## === cell 4
df.head()




## === cell 5
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = df[list_classes].values.astype("float32")




## === cell 6
x = df["comment_text"].fillna("")




## === cell 7
token = Tokenizer()
token.fit_on_texts(x)




## === cell 8
seq = token.texts_to_sequences(x)




## === cell 9
pad_seq = pad_sequences(seq, maxlen=100)




## === cell 10
vocab_size = len(token.word_index) + 1
print(vocab_size)




## === cell 11
embedding_dim = 100
embedding_matrix = np.random.normal(
    loc=0.0, scale=0.05, size=(vocab_size, embedding_dim)
).astype("float32")




## === cell 12
model = Sequential()




## === cell 13
model.add(
    Embedding(
        vocab_size,
        embedding_dim,
        input_length=100,
        weights=[embedding_matrix],
        trainable=False,
    )
)




## === cell 14
model.add(Bidirectional(LSTM(50)))




## === cell 15
model.add(Dense(64, activation="relu"))




## === cell 16
model.add(Dense(6, activation="sigmoid"))




## === cell 17
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])




## === cell 18
history = model.fit(pad_seq, y, epochs=2, batch_size=32, validation_split=0.2)




## === cell 19
model.summary()




## === cell 20
test = pd.read_csv(test_path)




## === cell 21
test.head()




## === cell 22
x_test = test["comment_text"].fillna("")
test_seq = token.texts_to_sequences(x_test)
test_pad_seq = pad_sequences(test_seq, maxlen=100)




## === cell 23
predict = model.predict(test_pad_seq, batch_size=1024)




## === cell 24
predict[0]




## === cell 25
sample_submission = pd.read_csv(sample_path)
sample_submission[list_classes] = predict.astype("float32")
sample_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sample_submission.shape)
print(sample_submission.head())
