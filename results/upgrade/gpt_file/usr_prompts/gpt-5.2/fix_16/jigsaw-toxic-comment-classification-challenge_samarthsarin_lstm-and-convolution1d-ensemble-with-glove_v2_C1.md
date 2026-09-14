# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.779438091833117

# 6. Current score

0.92733

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.94843) has done: 'I fix the environment-breaking imports by removing the NLTK dependency (it isn’t used later and requires unavailable resources) and by switching to `tf.keras` to avoid the protobuf/Keras compatibility error. I also remove the missing GloVe file dependency and instead initialize the embedding matrix randomly while keeping the same Embedding→LSTM→Dense architecture and training loop intact. Finally, I correct the model’s output/loss to a proper multi-label setup (sigmoid + binary_crossentropy) so that predictions match the competition metric/format and ensure a valid `submission.csv` is written with the required columns.'
- What this solution (achieved 0.94843) has done: 'I fix the TensorFlow/protobuf crash happening at import time (`MessageFactory.GetPrototype`) by forcing TensorFlow to use the pure-Python protobuf implementation before it’s imported, which is the minimal stable workaround in Kaggle’s TF1/TF2 + protobuf mismatch situations. I also keep everything else (tokenization, Embedding→LSTM→Dense architecture, loss, training loop, and submission formatting) unchanged to avoid unnecessary score movement since your current score (0.94843) is already far above the target (0.7794). Finally, I add a safe fallback to locate Kaggle input paths under both `/kaggle/input/...` and the provided `/kaggle/data/...` structure so the notebook runs end-to-end regardless of which mount is present.'
- What this solution (achieved 0.94843) has done: 'I fix the TensorFlow/protobuf import crash that happens before your code can run by forcing a compatible protobuf version and Python implementation *before* importing TensorFlow (this is the direct cause of the `MessageFactory.GetPrototype` error). I keep the model, tokenization, training loop, and submission formatting unchanged to avoid unnecessary score movement (your current score is already well above the target). I also add a small, safe fallback to pin protobuf if it’s available, and make the directory listing non-fatal. The script then run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.92733) has done: 'Your current score (0.94843) is far above the target (0.77944), so the safest way to move toward the target is to make a minimal, controlled reduction in model capacity without changing the overall approach (Tokenizer → fixed Embedding → LSTM → Dense → 6 sigmoid outputs, same loss and training loop). I reduce the LSTM units and the hidden Dense width, which typically lower AUC while keeping training/prediction semantics identical and still producing valid probabilities for all six labels. I keep epochs, optimizer, loss, and submission formatting unchanged to avoid destabilizing runtime and to keep changes narrowly tied to score movement. Everything still run end-to-end and write `submission.csv` with the required columns.'
- What this solution (achieved 0.92733) has done: 'Your current AUC (0.92733) is well above the target (0.77944), so we should *decrease* performance in a controlled way while keeping the same overall pipeline (Tokenizer → pad → Embedding → LSTM → Dense → sigmoid, same loss/training loop). The smallest, most stable lever is prediction calibration/post-processing: we keep training identical, but compress predicted probabilities toward 0.5, which generally reduces ROC AUC without breaking submission validity. I’m adding a single tunable parameter `PRED_SHRINK_ALPHA` applied after `model.predict`, leaving data paths, model, and training unchanged. This should move the score downward toward the target band while still producing a valid `submission.csv` with correct columns and row alignment.'
- What this solution (achieved 0.92733) has done: 'Your current score (0.92733) is still well above the target (0.77944), so the safest way to move closer is to *further* reduce ranking performance without changing the model/training pipeline. We keep the exact same model, loss, tokenizer, training loop, and submission formatting, and only adjust the existing post-processing shrink so predictions are pulled closer to 0.5 more strongly (this typically lowers ROC AUC). To make this change stable and deterministic, we also apply the shrink in float64 then cast back to float32, and keep clipping to avoid invalid probabilities. This is a single-parameter tweak intended to reduce the score toward the target band while preserving end-to-end execution and a valid `submission.csv`.'
- What this solution (achieved 0.92733) has done: 'Your current score (0.92733) is well above the target (0.77944), so we should deliberately reduce AUC in the most controlled, minimal way. We keep the entire training/tokenization/model/loss loop unchanged and only adjust the existing post-processing calibration that shrinks predictions toward 0.5. Increasing the shrink strength (smaller `PRED_SHRINK_ALPHA`) typically reduces ROC AUC while still producing valid probabilities and a correct submission file. I also keep clipping and dtype handling as-is for stability.'
- What this solution (achieved 0.92733) has done: 'Your current score (0.92733) is well above the target (0.77944), so we should deliberately reduce AUC in the smallest, most controlled way while keeping the model/training/tokenization pipeline unchanged. The safest lever is the existing post-processing shrink toward 0.5; increasing the shrink (smaller alpha) typically lowers ROC AUC without breaking submission validity. I only adjust `PRED_SHRINK_ALPHA` downward and keep clipping/formatting the same so the script still runs end-to-end and writes a correct `submission.csv`. This should move the score closer to the target band (±10%) without touching core modeling logic.'
- What this solution (achieved 0.92726) has done: 'Your current AUC (0.92733) is far above the target (0.77944), so we should intentionally reduce performance in the most controlled, minimal way while keeping the training/model pipeline identical. The smallest lever is the existing post-processing that shrinks predictions toward 0.5; increasing the shrink (smaller alpha) generally lowers ROC AUC without breaking submission validity. I only adjust `PRED_SHRINK_ALPHA` to a smaller value and keep clipping/formatting unchanged so it still runs end-to-end and produces a correct `submission.csv`. This should move the score downward toward the target tolerance band with minimal risk.'
- What this solution (achieved 0.89341) has done: 'Your current AUC (0.92726) is well above the target (0.77944), so the safest way to move closer is to deliberately reduce ranking signal with a tiny, post-processing-only change that keeps the model/training pipeline identical. I keep everything the same and only increase the existing shrink-to-0.5 strength by lowering `PRED_SHRINK_ALPHA`, which generally decreases ROC AUC while preserving valid probabilities and submission format. This is the smallest, most controlled lever and avoids touching architecture, loss, tokenizer, or training loop. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'Your current score (0.89341) is above the target (0.77944), so we should intentionally reduce AUC in the smallest, most controlled way while keeping the model/training pipeline unchanged. The most reliable minimal lever is the existing post-processing that shrinks predictions toward 0.5; we strengthen that shrink by decreasing `PRED_SHRINK_ALPHA`, which should reduce ranking signal and move the score downward toward the target band. Everything else (tokenization, Embedding→LSTM→Dense architecture, loss, training loop, file paths, and submission formatting) stays identical to preserve core logic and ensure a valid `submission.csv`. This change is deterministic and should not affect runtime meaningfully.'
- What this solution (achieved 0.92733) has done: 'Your current 0.5 score is caused by the extremely strong post-processing that shrinks predictions to ~0.5 (AUC becomes random). To move upward toward the target 0.779, we keep the entire model/tokenization/training pipeline unchanged and only weaken that shrink so predictions retain some ranking signal. Specifically, we set `PRED_SHRINK_ALPHA` to a moderate value (0.25) instead of ~0, which should increase AUC substantially while still being a minimal, stable, single-parameter change. Submission writing, paths, and column order remain identical.'
- What this solution (achieved 0.92733) has done: 'Your current score (0.92733) is above the target (0.77944), so we should intentionally reduce AUC in the smallest, most controlled way while keeping the model/training pipeline identical. The most stable lever is the existing post-processing that shrinks predictions toward 0.5; we strengthen that shrink slightly by lowering `PRED_SHRINK_ALPHA` so ranking signal decreases but doesn’t collapse to randomness. Everything else (tokenizer, padding, Embedding→LSTM→Dense architecture, loss, epochs, and submission formatting) remains unchanged to preserve core logic and ensure a valid `submission.csv`. This should move the score downward toward the target band without risking runtime or format issues.'
- What this solution (achieved 0.92733) has done: 'Your current AUC (0.92733) is well above the target (0.77944), so we should deliberately reduce ranking performance in the smallest, most stable way while keeping the model/training pipeline identical. The least invasive lever is the existing post-processing that shrinks probabilities toward 0.5; we strengthen that shrink slightly by lowering `PRED_SHRINK_ALPHA`. Everything else (tokenizer, padding length, Embedding→LSTM→Dense architecture, loss, epochs, and submission formatting/paths) stays unchanged to preserve core logic and ensure a valid `submission.csv`. This should move the score downward toward the target tolerance band without risking runtime or format issues.'
- What this solution (achieved 0.92733) has done: 'Your current score (0.92733) is well above the target (0.77944), so we should deliberately reduce performance in the smallest, most controlled way. The least invasive lever that preserves the full training/tokenization/model pipeline is the existing post-processing shrink of predictions toward 0.5; we strengthen it slightly by lowering `PRED_SHRINK_ALPHA`. This should reduce ROC AUC while keeping valid probabilities and the correct submission format. Everything else (data paths, model architecture, loss, epochs, and submission writing) remains unchanged to avoid destabilizing runtime.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import google.protobuf  # noqa: F401
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.*"],
        check=False,
    )
except Exception:
    pass

import numpy as np
import pandas as pd
import random

import tensorflow as tf

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

for p in ["../input", "/kaggle/input", "/kaggle/data"]:
    if os.path.exists(p):
        print("Listing:", p)
        try:
            print(os.listdir(p)[:50])
        except Exception as e:
            print("Could not list directory:", e)
        break



## === cell 1
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



## === cell 2
TRAIN_PATHS = [
    "../input/jigsaw-toxic-comment-classification-challenge/train.csv",
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv",
    "../input/train.csv",
    "/kaggle/input/train.csv",
    "/kaggle/data/jigsaw-toxic-comment-classification-challenge/train.csv",
    "/kaggle/data/train.csv",
]
train_path = next((p for p in TRAIN_PATHS if os.path.exists(p)), None)
if train_path is None:
    raise FileNotFoundError("Could not find train.csv in expected Kaggle input paths.")
print("Using train_path:", train_path)
df = pd.read_csv(train_path)



## === cell 3
df.head()



## === cell 4
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = df[list_classes].values.astype("float32")



## === cell 5
df["comment_text"] = df["comment_text"].fillna("")
x = df["comment_text"].astype(str)



## === cell 6
df.head()



## === cell 7
EMBED_DIM = 100



## === cell 8
x = df["comment_text"].astype(str)



## === cell 9
token = Tokenizer()
token.fit_on_texts(x)



## === cell 10
seq = token.texts_to_sequences(x)



## === cell 11
pad_seq = pad_sequences(seq, maxlen=100)



## === cell 12
vocab_size = len(token.word_index) + 1
print("vocab_size:", vocab_size)



## === cell 13
rng = np.random.RandomState(SEED)
embedding_matrix = rng.normal(loc=0.0, scale=0.6, size=(vocab_size, EMBED_DIM)).astype(
    "float32"
)
embedding_matrix[0] = 0.0  # padding token



## === cell 14
model = Sequential()



## === cell 15
model.add(
    Embedding(
        input_dim=vocab_size,
        output_dim=EMBED_DIM,
        input_length=100,
        weights=[embedding_matrix],
        trainable=False,
    )
)



## === cell 16
model.add(LSTM(12))



## === cell 17
model.add(Dense(16, activation="relu"))



## === cell 18
model.add(Dense(6, activation="sigmoid"))



## === cell 19
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])



## === cell 20
history = model.fit(pad_seq, y, epochs=2, batch_size=32, validation_split=0.2)



## === cell 21
model.summary()



## === cell 22
TEST_PATHS = [
    "../input/jigsaw-toxic-comment-classification-challenge/test.csv",
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv",
    "../input/test.csv",
    "/kaggle/input/test.csv",
    "/kaggle/data/jigsaw-toxic-comment-classification-challenge/test.csv",
    "/kaggle/data/test.csv",
]
test_path = next((p for p in TEST_PATHS if os.path.exists(p)), None)
if test_path is None:
    raise FileNotFoundError("Could not find test.csv in expected Kaggle input paths.")
print("Using test_path:", test_path)
test = pd.read_csv(test_path)



## === cell 23
test.head()



## === cell 24
test["comment_text"] = test["comment_text"].fillna("")
x_test = test["comment_text"].astype(str)
test_seq = token.texts_to_sequences(x_test)
test_pad_seq = pad_sequences(test_seq, maxlen=100)



## === cell 25
predict = model.predict(test_pad_seq, batch_size=1024, verbose=1)



## === cell 26
predict[0]



## === cell 27
PRED_SHRINK_ALPHA = 0.055
predict = predict.astype("float64", copy=False)
predict = 0.5 + PRED_SHRINK_ALPHA * (predict - 0.5)
predict = np.clip(predict, 1e-6, 1.0 - 1e-6).astype("float32")



## === cell 28
SUB_PATHS = [
    "../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv",
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv",
    "../input/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/jigsaw-toxic-comment-classification-challenge/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
]
sub_path = next((p for p in SUB_PATHS if os.path.exists(p)), None)
if sub_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected Kaggle input paths."
    )
print("Using sub_path:", sub_path)
sample_submission = pd.read_csv(sub_path)



## === cell 29
sample_submission[list_classes] = predict
sample_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sample_submission.shape)
print(sample_submission.head())
