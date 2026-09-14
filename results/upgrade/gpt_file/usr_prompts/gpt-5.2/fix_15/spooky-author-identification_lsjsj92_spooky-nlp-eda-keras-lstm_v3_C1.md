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
Given some text, predict the author.

## Metric
Multi-class logarithmic loss. 

The submitted probabilities for a given sentences are not required to sum to one because they are rescaled prior to being scored (each row is divided by the row sum).

In order to avoid the extremes of the log function, predicted probabilities are replaced with \\(max(min(p,1-10^{-15}),10^{-15})\\).

## Submission Format
You must submit a csv file with the id, and a probability for each of the three classes. The order of the rows does not matter. The file must have a header and should look like the following:

```
id,EAP,HPL,MWS
id07943,0.33,0.33,0.33
...
```

## Dataset 
### File descriptions
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Data fields
- **id** - a unique identifier for each sentence
- **text** - some text written by one of the authors
- **author** - the author of the sentence (EAP: Edgar Allan Poe, HPL: HP Lovecraft; MWS: Mary Wollstonecraft Shelley)

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
wordcloud==1.9.4

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (98 lines)
            sample_submission.csv (1959 lines)
            sample_submission.csv.zip (7.4 kB)
            test.csv (1959 lines)
            test.csv.zip (133.3 kB)
            train.csv (17622 lines)
            train.csv.zip (1.2 MB)
            train.zip (1.2 MB)
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
        input/
            description.md (98 lines)
            sample_submission.csv (1959 lines)
            sample_submission.csv.zip (7.4 kB)
            test.csv (1959 lines)
            test.csv.zip (133.3 kB)
            train.csv (17622 lines)
            train.csv.zip (1.2 MB)
            train.zip (1.2 MB)
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
        working/
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
```

-> data/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> data/spooky-author-identification/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> data/spooky-author-identification/test.csv has 1958 rows and 2 columns.
The columns are: id, text

-> data/spooky-author-identification/train.csv has 17621 rows and 3 columns.
The columns are: id, text, author

-> data/test.csv has 1958 rows and 2 columns.
The columns are: id, text

-> data/train.csv has 17621 rows and 3 columns.
The columns are: id, text, author

-> input/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> (stopped after 10 files for performance)

# 5. Target score

0.60129

# 6. Current score

0.46193

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.46481) has done: 'I fix the import/runtime failures caused by the Kaggle environment using Keras 3 (where `keras.preprocessing.*` and some legacy symbols break) by switching to the compatible `tf_keras` API for Tokenizer, padding, callbacks, and `to_categorical`, while keeping the exact same model architecture and training loop. I also make the data paths robust to either `/kaggle/input/...` or the provided `../input/...` layout so the notebook runs end-to-end. Finally, I ensure the submission uses the required columns (`id,EAP,HPL,MWS`) and writes a valid `.csv` file.'
- What this solution (achieved 0.45845) has done: 'The run currently fails before training due to a known protobuf/TensorFlow–Keras interaction that surfaces as `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. I fix this by forcing the pure-Python protobuf implementation *before* importing any TF/Keras modules, which is the minimal, standard workaround in Kaggle-style environments. I also keep your model/training logic unchanged, only adding small safety checks to ensure the submission columns/order are correct and the output is a valid `.csv`. These changes are score-neutral (they just restore runtime correctness), and your existing score (0.46481, already better than the 0.60129 target for logloss) should remain in the target tolerance band.'
- What this solution (achieved 0.45807) has done: 'I fix the protobuf-related runtime crash by setting both `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` and `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2` before importing any TF/Keras-related modules, which is the standard minimal workaround for the `MessageFactory.GetPrototype` error. I also ensure the input directory resolution always finds the provided `/kaggle/data/...` layout and that the submission is written with the required header/column order to a `.csv` file. These changes are score-neutral (they don’t change the model, training loop, or preprocessing) and should restore end-to-end execution and a valid submission. I keep the original architecture/training settings unchanged to avoid moving the score away from your current better-than-target logloss.'
- What this solution (achieved 0.45276) has done: 'We need to fix the runtime crash `'MessageFactory' object has no attribute 'GetPrototype'` that happens when importing `tf_keras`; this is caused by an incompatible protobuf implementation being loaded too early, so we force the pure-Python protobuf and also proactively remove any already-imported `google.protobuf` modules before importing TF/Keras. This is a minimal, score-neutral runtime fix that preserves your model, preprocessing, and training loop exactly. I also rename your first cell from `cell 0` to `cell 1` to match the required cell format and add a small fallback to write the submission into `/kaggle/working/` if the current directory is not writable. No score-changing modifications be made since your current logloss (0.45807) is already better than the target (0.60129) for a lower-is-better metric.'
- What this solution (achieved 0.46314) has done: 'We fix the `MessageFactory.GetPrototype` crash by ensuring the protobuf runtime is forced to the pure-Python implementation before any TF/Keras-related imports and by explicitly importing `google.protobuf` once after the env vars are set (so TF doesn’t later load the incompatible C++ backend). This is a minimal runtime-only fix that preserves your model architecture, preprocessing, and training loop, so the score should stay essentially unchanged (and since your current logloss is already better than the target, we avoid any score-improving changes). We also keep your robust input directory resolution and ensure the submission is written as a valid `.csv` with the required columns and order.'
- What this solution (achieved 0.4732) has done: 'We fix the protobuf/Keras import crash by forcing the pure-Python protobuf implementation and (critically) ensuring it’s set before any TensorFlow/Keras-related import, plus importing `tensorflow` first to stabilize the TF↔protobuf initialization path. This is a runtime-only fix that preserves your preprocessing, model architecture, and training loop, so it should keep the score essentially unchanged (and your current logloss is already better than the target). We also keep the robust input-directory resolver and keep the submission columns/order exactly `id,EAP,HPL,MWS`, writing a `.csv` into a writable location. Finally, we rename `cell 0` to `cell 1` to match the required cell format without changing behavior.'
- What this solution (achieved 0.46146) has done: 'We fix the `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf *and* ensuring no TensorFlow/Keras is imported before that, then importing `tensorflow` before `tf_keras` to stabilize initialization in this environment. This is a runtime-only change that preserves your exact preprocessing, model architecture, and training loop, so it should keep performance essentially unchanged (and your current logloss is already better than the target for a lower-is-better metric). We also correct the cell numbering to start at `cell 1` as required, and keep the submission columns/order exactly `id,EAP,HPL,MWS` written to a `.csv` file in a writable location.'
- What this solution (achieved 0.4637) has done: 'We fix the protobuf/TensorFlow import crash by forcing the pure-Python protobuf implementation and clearing any already-loaded protobuf modules before importing TensorFlow, then import `tf_keras` after `tensorflow` to stabilize initialization. This is a runtime-only change and keeps your preprocessing, model definition, training loop, and prediction logic identical (so score should remain essentially unchanged and still safely within the target band since lower is better and you’re already better than target). We also keep the robust input directory resolver and ensure the submission is written as a valid `my_submission.csv` with the required columns/order.'
- What this solution (achieved 0.45606) has done: 'I fix the protobuf/TensorFlow import crash that happens in cell 1 by forcing the pure-Python protobuf implementation and also disabling the C++ protobuf backend via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` *before* importing TensorFlow, and by importing `tensorflow` first (then `tf_keras`) which is the most stable order in this Kaggle-style environment. I keep your model, preprocessing, training loop, and prediction logic identical to avoid changing your (already better-than-target) logloss. I also renumber the notebook cells to start at `cell 1` (required format) and keep the submission writing exactly `id,EAP,HPL,MWS` to a `.csv` file.'
- What this solution (achieved 0.45137) has done: 'I fix the protobuf/TensorFlow import crash by forcing the pure-Python protobuf implementation as early as possible and importing TensorFlow before `tf_keras`, which is the most reliable order in this Kaggle-style environment. This change is runtime-only and keeps your preprocessing, model architecture, training loop, and prediction logic the same, so it should not intentionally move the score away from your current (already better-than-target) logloss. I also renumber the cells to start at 1 (required format) and keep the submission writing exactly `id,EAP,HPL,MWS` to a `.csv` in a writable location.'
- What this solution (achieved 0.47366) has done: 'We fix the runtime crash in cell 1 (`MessageFactory` has no `GetPrototype`) by forcing the pure-Python protobuf implementation and also forcing TensorFlow to use the Python protobuf backend via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` *before* any TensorFlow/tf_keras import, plus clearing any preloaded protobuf modules. This is a minimal runtime-only change that preserves your exact preprocessing, model architecture, training loop, and prediction logic (so score behavior should stay essentially the same, i.e., still better than the target). We also keep your robust INPUT_DIR resolution and ensure the submission is written with the required columns (`id,EAP,HPL,MWS`) to a `.csv` in a writable location.'
- What this solution (achieved 0.46107) has done: 'I fix the runtime crash (`MessageFactory` has no `GetPrototype`) by forcing the pure-Python protobuf implementation before any TensorFlow/tf_keras import, and by additionally disabling the C++ protobuf backend via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` plus importing TensorFlow first in a stable order. This is a minimal, score-neutral change that preserves your exact preprocessing, model architecture, training loop, and prediction logic. I also renumber the notebook cells to start at 1 (required format) and keep the robust input directory resolver unchanged. Finally, I ensure the submission is always written as a valid `.csv` with the required `id,EAP,HPL,MWS` columns.'
- What this solution (achieved 0.47879) has done: 'We fix the current runtime crash (`MessageFactory` has no `GetPrototype`) by forcing the pure-Python protobuf implementation *and* preventing any TensorFlow/tf_keras import from seeing the incompatible C++ protobuf backend (including clearing already-loaded protobuf modules before importing TensorFlow). Then we keep your preprocessing, model architecture, training loop, and submission logic the same, only adding a small safety check to ensure the submission probabilities are finite and the CSV columns/order match Kaggle’s required format. These are execution/stability fixes and should keep performance essentially unchanged (your current logloss is already better than the target for a lower-is-better metric). The output be a valid `my_submission.csv` in a writable location.'
- What this solution (achieved 0.46193) has done: 'We fix the `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf runtime and ensuring TensorFlow is imported in a stable order before `tf_keras`, while also explicitly preventing Keras 3 from being imported as `keras` (which can trigger the same protobuf path). These are minimal runtime-only changes that do not alter your model architecture, preprocessing, training loop, or prediction logic, so they should keep your logloss in the same range (already better than the target for a lower-is-better metric). We also keep the robust input directory resolution and ensure the submission is written as a valid `my_submission.csv` with columns `id,EAP,HPL,MWS`.'

# 9. Code solution

## === cell 0
import os
import sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

for k in list(sys.modules.keys()):
    if k.startswith("google.protobuf"):
        del sys.modules[k]

import google.protobuf  # noqa: F401

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

CANDIDATE_INPUT_DIRS = [
    "../input/spooky-author-identification",
    "../input",
    "/kaggle/input/spooky-author-identification",
    "/kaggle/input",
    "/kaggle/data/spooky-author-identification",
    "/kaggle/data",
]
INPUT_DIR = None
for d in CANDIDATE_INPUT_DIRS:
    if os.path.exists(d) and os.path.exists(os.path.join(d, "train.csv")):
        INPUT_DIR = d
        break

print("Resolved INPUT_DIR:", INPUT_DIR)
if INPUT_DIR is None:
    raise FileNotFoundError(
        "Could not locate train.csv/test.csv under expected input directories."
    )




## === cell 1
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
import tf_keras

from tf_keras.models import Model
from tf_keras.layers import (
    LSTM,
    Dense,
    Input,
    Dropout,
    Bidirectional,
    GlobalMaxPool1D,
    Embedding,
)
from tf_keras.preprocessing import sequence
from tf_keras.preprocessing.text import Tokenizer
from tf_keras.callbacks import EarlyStopping
from tf_keras.utils import to_categorical

np.random.seed(42)
tf_keras.utils.set_random_seed(42)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
train_path = os.path.join(INPUT_DIR, "train.csv")
test_path = os.path.join(INPUT_DIR, "test.csv")

data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

print("train shape:", data.shape, "test shape:", test_data.shape)
data.head()




## === cell 3
data.author.value_counts().plot(kind="bar")
plt.show()




## === cell 4
data_length = data.text.apply(len)
data_length.head()




## === cell 5
plt.figure(figsize=(12, 5))
plt.hist(data_length, bins=20, range=[0, 500], color="r", alpha=0.3)
plt.show()




## === cell 6
data_split_length = data.text.apply(lambda x: len(x.split(" ")))
data_split_length.head()




## === cell 7
plt.figure(figsize=(12, 5))
plt.hist(data_split_length, bins=10, range=[0, 100], color="g", alpha=0.5)
plt.show()




## === cell 8
print("data_length max : ", np.max(data_length))
print("data_length min : ", np.min(data_length))
print("data_length mean : ", np.mean(data_length))
print("data_length 75% : ", np.percentile(data_length, 75))
print("data_length 90% : ", np.percentile(data_length, 90))




## === cell 9
print("data_split_length max : ", np.max(data_split_length))
print("data_split_length min : ", np.min(data_split_length))
print("data_split_length mean : ", np.mean(data_split_length))
print("data_split_length 75% : ", np.percentile(data_split_length, 75))
print("data_split_length 90% : ", np.percentile(data_split_length, 90))




## === cell 10
cloud = WordCloud(width=400, height=200).generate(" ".join(data.text))
plt.figure(figsize=(12, 5))
plt.imshow(cloud)
plt.axis("off")
plt.show()




## === cell 11
cloud = WordCloud(width=400, height=200).generate(
    " ".join(data[data["author"] == "HPL"]["text"])
)
plt.figure(figsize=(12, 5))
plt.imshow(cloud)
plt.axis("off")
plt.show()




## === cell 12
cloud = WordCloud(width=400, height=200).generate(
    " ".join(data[data["author"] == "MWS"]["text"])
)
plt.figure(figsize=(12, 5))
plt.imshow(cloud)
plt.axis("off")
plt.show()




## === cell 13
cloud = WordCloud(width=400, height=200).generate(
    " ".join(data[data["author"] == "EAP"]["text"])
)
plt.figure(figsize=(12, 5))
plt.imshow(cloud)
plt.axis("off")
plt.show()




## === cell 14
le = LabelEncoder()
le.fit(data.author)
y = le.transform(data.author)




## === cell 15
y[:10]




## === cell 16
y = to_categorical(y)




## === cell 17
y[:10]




## === cell 18
num_words = 20000
max_len = 70
emb_size = 64




## === cell 19
tok = Tokenizer(num_words=num_words)
tok.fit_on_texts(list(data.text))




## === cell 20
X = tok.texts_to_sequences(data.text)
X_test = tok.texts_to_sequences(test_data.text)




## === cell 21
X = sequence.pad_sequences(X, maxlen=max_len)
X_test = sequence.pad_sequences(X_test, maxlen=max_len)




## === cell 22
X[0]




## === cell 23
def model():
    inp = Input(shape=(max_len,))
    layer = Embedding(num_words, emb_size)(inp)
    layer = Bidirectional(LSTM(32, return_sequences=True, recurrent_dropout=0.2))(layer)
    layer = GlobalMaxPool1D()(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(16, activation="relu")(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(3, activation="softmax")(layer)
    m = Model(inputs=inp, outputs=layer)
    m.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return m




## === cell 24
model = model()
model.summary()




## === cell 25
early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)




## === cell 26
hist = model.fit(
    X,
    y,
    batch_size=32,
    epochs=5,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=2,
)




## === cell 27
vloss = hist.history["val_loss"]
loss = hist.history["loss"]

x_len = np.arange(len(loss))

plt.plot(x_len, vloss, marker=".", color="r", label="val_loss")
plt.plot(x_len, loss, marker=".", color="b", label="loss")
plt.legend()
plt.grid()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()




## === cell 28
pred = model.predict(X_test, batch_size=256, verbose=0)
ids = test_data["id"].values

pred = np.asarray(pred, dtype=np.float64)
pred = np.nan_to_num(pred, nan=1.0 / 3.0, posinf=1.0 / 3.0, neginf=1.0 / 3.0)

results = pd.DataFrame(pred, columns=["EAP", "HPL", "MWS"])
results.insert(0, "id", ids)

results = results[["id", "EAP", "HPL", "MWS"]]
results.head()




## === cell 29
sub_path = "my_submission.csv"
try:
    results.to_csv(sub_path, index=False)
except Exception:
    sub_path = "/kaggle/working/my_submission.csv"
    results.to_csv(sub_path, index=False)

print("Wrote submission to:", sub_path, "shape:", results.shape)
print(results.head())
