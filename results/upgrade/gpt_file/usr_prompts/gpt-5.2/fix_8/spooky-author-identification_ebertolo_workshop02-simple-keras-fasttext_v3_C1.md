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

0.4427

# 6. Current score

0.38145

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.38534) has done: 'I fix the import/runtime issues caused by mixing `keras` v3 with legacy `keras.preprocessing` and `keras.utils`, by switching to the compatible `tf_keras` equivalents so Tokenizer/pad_sequences/to_categorical and model training work again. I also fix pathing to use the provided Kaggle-style `../input/*.csv` files, ensure `input_dim` is computed from the padded integer sequences (not from raw strings), and replace the deprecated `predict_proba()` call with `predict()` to generate class probabilities. Finally, I ensure the submission file is written with the required columns (`id,EAP,HPL,MWS`) and a `.csv` suffix.'
- What this solution (achieved 0.39099) has done: 'I fix the crash in the very first cell by avoiding the incompatible `keras` (v3) import path that triggers the protobuf `MessageFactory.GetPrototype` error, switching to the Kaggle-available `tf_keras` API for Tokenizer/pad_sequences/to_categorical and model components. Because cell 0 currently errors, all later `NameError`s are cascading; once imports and path detection run, the remaining cells execute normally. I also ensure `train_test_split` is imported, keep the model/training logic unchanged, and guarantee a correctly formatted `submission.csv` (id,EAP,HPL,MWS) is written. No score-tuning changes are introduced beyond restoring the intended pipeline so you get a valid, scorable submission.'
- What this solution (achieved 0.37988) has done: 'I fix the runtime crash in the first cell caused by the protobuf/Keras incompatibility by ensuring we use the Kaggle-safe `tf_keras` backend *and* by forcing the pure-Python protobuf implementation before TensorFlow/Keras imports. I also make input path detection robust for both `/kaggle/input` and `/kaggle/data` layouts by searching for `train.csv` recursively inside candidate roots (this is score-neutral but prevents missing-file cascades). Finally, I keep the model/training/prediction logic unchanged and only add a small safety renormalization of predicted probabilities (row-wise) to match the competition’s rescaling semantics and avoid any all-zero/NaN edge cases, while still clipping to `[1e-15, 1-1e-15]`.'
- What this solution (achieved 0.38145) has done: 'I fix the runtime crash in the first cell caused by the protobuf/tf_keras incompatibility by forcing the pure-Python protobuf implementation earlier and also disabling the C++ protobuf backend explicitly. This is a minimal, score-neutral change that unblocks all downstream cells so the pipeline runs end-to-end. I also make the TensorFlow random seed deterministic (without changing the model/training logic) to stabilize results across runs. Finally, I keep the existing training and submission logic intact and ensure `submission.csv` is always written with the required columns.'
- What this solution (achieved 0.38145) has done: 'I fix the protobuf/MessageFactory crash by forcing the pure-Python protobuf backend *before* importing anything that pulls in TensorFlow/tf_keras, and by also disabling the C++ implementation via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` plus the commonly-needed `TF_ENABLE_ONEDNN_OPTS=0` stability tweak (score-neutral, runtime-fix). Then I keep your model/tokenization/training logic intact, only adding a safe fallback import path if `tf_keras` still fails in this environment (so the notebook runs end-to-end). Finally, I ensure the submission is always written as `submission.csv` with the required columns and clipped probabilities (already present), without changing training semantics (so score should remain close to your current 0.38145, i.e., already better than the 0.4427 target).'
- What this solution (achieved 0.38145) has done: 'You’re currently crashing immediately due to a protobuf runtime incompatibility (`MessageFactory.GetPrototype`) that happens when TensorFlow/Keras imports pick up an incompatible protobuf backend. I fix this by pinning the pure-Python protobuf implementation *and* forcing TensorFlow to use the Python protobuf code path before any TF/tf_keras import, which is the minimal change that unblocks execution. I not change the model, tokenization, training loop, or prediction logic (your current score is already better than the target, and we should avoid score-moving edits). Finally, I keep the same submission-writing code and ensure it always produces `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import google.protobuf.internal.api_implementation as _pb_api_impl

    _pb_api_impl._set_implementation_type("python")
except Exception:
    pass

import numpy as np
import pandas as pd
from collections import defaultdict

try:
    import tf_keras as keras
    from tf_keras.layers import Dense, GlobalAveragePooling1D, Embedding
    from tf_keras.callbacks import EarlyStopping
    from tf_keras.models import Sequential
    from tf_keras.preprocessing.sequence import pad_sequences
    from tf_keras.preprocessing.text import Tokenizer
    from tf_keras.utils import to_categorical
except Exception:
    import tensorflow as tf  # noqa: F401
    from tensorflow import keras
    from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, Embedding
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

np.random.seed(7)
try:
    keras.utils.set_random_seed(7)
except Exception:
    pass


def _find_competition_dir(candidates):
    """
    Prefer a directory that directly contains train.csv; otherwise search deeper.
    This prevents failures when the files are under /kaggle/*/spooky-author-identification/.
    """
    for d in candidates:
        if os.path.exists(d) and os.path.exists(os.path.join(d, "train.csv")):
            return d

    for root in candidates:
        if not os.path.exists(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if (
                "train.csv" in filenames
                and "test.csv" in filenames
                and "sample_submission.csv" in filenames
            ):
                return dirpath
            if dirpath.count(os.sep) - root.count(os.sep) >= 4:
                dirnames[:] = []
    return None


INPUT_DIR_CANDIDATES = [
    "../input",  # standard Kaggle notebooks
    "/kaggle/input",
    "/kaggle/data",
    "/kaggle/input/spooky-author-identification",
    "/kaggle/data/spooky-author-identification",
]

INPUT_DIR = _find_competition_dir(INPUT_DIR_CANDIDATES)
if INPUT_DIR is None:
    raise FileNotFoundError(
        "Could not locate directory containing train.csv/test.csv/sample_submission.csv"
    )

TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

print("Using INPUT_DIR:", INPUT_DIR)
print("TRAIN_PATH exists:", os.path.exists(TRAIN_PATH))
print("TEST_PATH exists:", os.path.exists(TEST_PATH))
print("SAMPLE_SUB_PATH exists:", os.path.exists(SAMPLE_SUB_PATH))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
df = pd.read_csv(TRAIN_PATH)

a2c = {"EAP": 0, "HPL": 1, "MWS": 2}
y = np.array([a2c[a] for a in df.author.values], dtype=np.int64)
y = to_categorical(y, num_classes=3)
y[:3]



## === cell 2
counter = {name: defaultdict(int) for name in set(df.author)}
for text, author in zip(df.text, df.author):
    text = text.replace(" ", "")
    for c in text:
        counter[author][c] += 1

chars = set()
for v in counter.values():
    chars |= v.keys()

names = [author for author in counter.keys()]

print("c ", end="")
for n in names:
    print(n, end="   ")
print()
for c in chars:
    print(c, end=" ")
    for n in names:
        print(counter[n][c], end=" ")
    print()




## === cell 3
def preprocess(text):
    text = text.replace("' ", " ' ")
    signs = set(',.:;"?!')
    prods = set(text) & signs
    if not prods:
        return text

    for sign in prods:
        text = text.replace(sign, " {} ".format(sign))
    return text




## === cell 4
def create_docs(df, n_gram_max=2):
    def add_ngram(q, n_gram_max):
        ngrams = []
        for n in range(2, n_gram_max + 1):
            for w_index in range(len(q) - n + 1):
                ngrams.append("--".join(q[w_index : w_index + n]))
        return q + ngrams

    docs = []
    for doc in df.text:
        doc = preprocess(doc).split()
        docs.append(" ".join(add_ngram(doc, n_gram_max)))

    return docs




## === cell 5
min_count = 2

docs = create_docs(df)
tokenizer = Tokenizer(lower=False, filters="")
tokenizer.fit_on_texts(docs)
num_words = sum([1 for _, v in tokenizer.word_counts.items() if v >= min_count])

tokenizer = Tokenizer(num_words=num_words, lower=False, filters="")
tokenizer.fit_on_texts(docs)
docs = tokenizer.texts_to_sequences(docs)
print("n_docs:", len(docs))

maxlen = 256
docs = pad_sequences(sequences=docs, maxlen=maxlen)
print("padded shape:", docs.shape, "dtype:", docs.dtype)



## === cell 6
docs_origin = create_docs(df)
docs_origin[0]



## === cell 7
input_dim = int(np.max(docs)) + 1
embedding_dims = 20
input_dim




## === cell 8
def create_model(embedding_dims=20, optimizer="adam"):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=embedding_dims))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(20, activation="tanh"))
    model.add(Dense(3, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    return model




## === cell 9
model = create_model()
model.summary()



## === cell 10
epochs = 25
x_train, x_val, y_train, y_val = train_test_split(
    docs, y, test_size=0.25, random_state=7, stratify=y.argmax(1)
)

hist = model.fit(
    x_train,
    y_train,
    batch_size=32,
    validation_data=(x_val, y_val),
    epochs=epochs,
    callbacks=[
        EarlyStopping(patience=5, monitor="val_loss", restore_best_weights=True)
    ],
    verbose=2,
)



## === cell 11
docs = create_docs(df)
tokenizer = Tokenizer(lower=True, filters="")
tokenizer.fit_on_texts(docs)
num_words = sum([1 for _, v in tokenizer.word_counts.items() if v >= min_count])

tokenizer = Tokenizer(num_words=num_words, lower=True, filters="")
tokenizer.fit_on_texts(docs)
docs = tokenizer.texts_to_sequences(docs)

maxlen = 256
docs = pad_sequences(sequences=docs, maxlen=maxlen)

input_dim = int(np.max(docs)) + 1
input_dim



## === cell 12
model = create_model()
model.summary()



## === cell 13
epochs = 25
x_train, x_val, y_train, y_val = train_test_split(
    docs, y, test_size=0.25, random_state=7, stratify=y.argmax(1)
)

hist = model.fit(
    x_train,
    y_train,
    batch_size=32,
    validation_data=(x_val, y_val),
    epochs=epochs,
    callbacks=[
        EarlyStopping(patience=5, monitor="val_loss", restore_best_weights=True)
    ],
    verbose=2,
)



## === cell 14
test_df = pd.read_csv(TEST_PATH)
test_docs = create_docs(test_df)
test_docs = tokenizer.texts_to_sequences(test_docs)
test_docs = pad_sequences(sequences=test_docs, maxlen=maxlen)

y_pred = model.predict(test_docs, batch_size=256, verbose=0)

row_sums = y_pred.sum(axis=1, keepdims=True)
row_sums[row_sums == 0.0] = 1.0
y_pred = y_pred / row_sums

result = pd.read_csv(SAMPLE_SUB_PATH)
for a, i in a2c.items():
    result[a] = y_pred[:, i]

eps = 1e-15
for col in ["EAP", "HPL", "MWS"]:
    result[col] = np.clip(result[col].astype(float), eps, 1.0 - eps)

result.head()



## === cell 15
out_path = "submission.csv"
result.to_csv(out_path, index=False)
print(
    "Wrote submission:",
    out_path,
    "shape:",
    result.shape,
    "columns:",
    list(result.columns),
)
print("Submission preview:")
print(result.head(3).to_string(index=False))
