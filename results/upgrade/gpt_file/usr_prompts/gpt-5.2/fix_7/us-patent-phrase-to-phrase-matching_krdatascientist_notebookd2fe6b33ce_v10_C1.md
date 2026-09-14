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
Given pairs of phrases (an `anchor` and a `target` phrase), build a model to rate how similar they are.  

## Metric
Pearson correlation coefficient.

## Submission Format
For each `id` (representing a pair of phrases) in the test set, you must predict the similarity `score`. The file should contain a header and have the following format:

```
id,score
4112d61851461f60,0
09e418c93a776564,0.25
36baf228038e314b,1
etc.

```

## Dataset
The scores are in the 0-1 range with increments of 0.25 with the following meanings:

- **1.0** - Very close match. This is typically an exact match except possibly for differences in conjugation, quantity (e.g. singular vs. plural), and addition or removal of stopwords (e.g. "the", "and", "or").
- **0.75** - Close synonym, e.g. "mobile phone" vs. "cellphone". This also includes abbreviations, e.g. "TCP" -> "transmission control protocol".
- **0.5** - Synonyms which don't have the same meaning (same function, same properties). This includes broad-narrow (hyponym) and narrow-broad (hypernym) matches.
- **0.25** - Somewhat related, e.g. the two phrases are in the same high level domain but are not synonyms. This also includes antonyms.
- **0.0** - Unrelated.

Files
-----

- **train.csv** - the training set, containing phrases, contexts, and their similarity scores
- **test.csv** - the test set set, identical in structure to the training set but without the score
- **sample_submission.csv** - a sample submission file in the correct format

Columns
-------

- `id` - a unique identifier for a pair of phrases
- `anchor` - the first phrase
- `target` - the second phrase
- `context` - the [CPC classification (version 2021.05)](https://en.wikipedia.org/wiki/Cooperative_Patent_Classification), which indicates the subject within which the similarity is to be scored
- `score` - the similarity. This is sourced from a combination of one or more manual expert ratings.

# 2. Python version

3.10

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        input/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        working/
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
```

-> data/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> data/us-patent-phrase-to-phrase-matching/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/us-patent-phrase-to-phrase-matching/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/us-patent-phrase-to-phrase-matching/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> input/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> (stopped after 10 files for performance)

# 5. Target score

0.1399

# 6. Current score

0.23297

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.20262) has done: 'I fix the crash caused by an incompatibility between TensorFlow 2.18 and the installed `protobuf` (it triggers `MessageFactory.GetPrototype` errors) by forcing the pure-Python protobuf implementation before importing TensorFlow. I also make the notebook cells consistent (start at cell 1) and keep the exact same modeling/training logic so the score behavior stays essentially the same (your current score is already above target, so we avoid changes that would move it further away). Finally, I ensure the submission file is written as `submission.csv` with the required `id,score` columns.'
- What this solution (achieved 0.2387) has done: 'I fix the TensorFlow import crash by ensuring the protobuf environment variables are set before any TensorFlow-related import and by restarting the import path cleanly in the first cell. I also make the inference text construction match the training text construction (it currently omits `id` at test time), which is a small correctness fix that typically stabilizes predictions without changing the model architecture or training loop. Finally, I keep the output exactly in the required `id,score` format and ensure `submission.csv` is always written.'
- What this solution (achieved 0.24264) has done: 'You’re hitting the known TensorFlow 2.18 + protobuf 6.x incompatibility (`MessageFactory.GetPrototype`), but the environment variables must be set before any TensorFlow/protobuf-related import and it’s safest to also force the pure-Python protobuf backend at runtime. I move the protobuf environment setup to the very top, add a small safety import of `google.protobuf` after setting those vars (before importing TensorFlow), and keep the model/tokenization/training exactly the same so score behavior remains essentially unchanged (your current score is already above the target). I also renumber cells to start at 1 and ensure the submission is always written as `submission.csv` with `id,score`.'
- What this solution (achieved 0.18458) has done: 'I fix the TensorFlow import crash caused by the protobuf 6.x incompatibility by forcing the pure-Python protobuf runtime and (critically) also disabling the C++ protobuf implementation via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` *and* `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2` before any TensorFlow/protobuf-related import, then importing `google.protobuf` as a sanity check. I keep your model architecture, tokenization, training loop, and inference logic unchanged to avoid unnecessary score drift (your current score is already above the target). I also renumber cells to start at 1 and ensure the notebook always writes a valid `submission.csv` with exactly `id,score`.'
- What this solution (achieved 0.23312) has done: 'You’re still hitting the TensorFlow 2.18 + protobuf 6.x incompatibility because setting the environment variables inside the notebook is not always sufficient once protobuf/TensorFlow have partially initialized. I make the protobuf “python” runtime enforcement more robust by (1) setting the env vars at the very top, (2) explicitly forcing the python protobuf implementation via `google.protobuf.internal.api_implementation`, and (3) purging any preloaded protobuf modules before importing TensorFlow. This is a correctness/stability fix only; the model, tokenization, training loop, and submission writing remain identical so the score behavior should stay essentially the same (and not drift further away from your target). The script then run end-to-end and write a valid `submission.csv` with `id,score`.'
- What this solution (achieved 0.23297) has done: 'I fix the TensorFlow import crash caused by the TensorFlow 2.18 + protobuf 6.x incompatibility by forcing the pure-Python protobuf runtime before any TensorFlow-related import and (critically) also pinning protobuf to a compatible major version at runtime (Kaggle allows `pip` installs). This is a stability/correctness fix only and keeps your model architecture, tokenization, training loop, and inference logic unchanged so the score should remain essentially the same (and not drift further away from your already-above-target score). I also renumber the notebook cells to start at 1 and ensure the script always writes a valid `submission.csv` with exactly `id,score`.'

# 9. Code solution

## === cell 0
import os
import sys
import importlib
import subprocess

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"],
    check=False,
)

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        del sys.modules[m]

import numpy as np
import pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
df_sub = pd.read_csv(
    "/kaggle/input/us-patent-phrase-to-phrase-matching/sample_submission.csv"
)
df_train = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/train.csv")
df_test = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv")

print("train:", df_train.shape, "test:", df_test.shape, "sample_sub:", df_sub.shape)



## === cell 2
df_train.head(3)



## === cell 3
import google.protobuf  # noqa: F401
from google.protobuf.internal import api_implementation

try:
    api_implementation._SetImplementationType("python")
except Exception:
    pass

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

print("TensorFlow:", tf.__version__)
print("protobuf:", google.protobuf.__version__)



## === cell 4
input_text = (
    df_train.id + " " + df_train.anchor + " " + df_train.context + " " + df_train.target
).str.lower()



## === cell 5
token = Tokenizer()



## === cell 6
token.fit_on_texts(input_text)



## === cell 7
vocab_size = len(token.word_index) + 1
print("vocab_size:", vocab_size)



## === cell 8
embedding_doc = token.texts_to_sequences(input_text)



## === cell 9
max([len(i) for i in embedding_doc])



## === cell 10
max_length = 18
padded_docs = pad_sequences(embedding_doc, maxlen=max_length, padding="post")



## === cell 11
e = tf.keras.layers.Embedding(vocab_size, 50, input_length=max_length)



## === cell 12
model = Sequential()
model.add(e)
model.add(Flatten())
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse", metrics=["mse"])
print(model.summary())



## === cell 13
df_train.head(3)



## === cell 14
model.fit(padded_docs, df_train.score, epochs=5, verbose=1, validation_split=0.3)



## === cell 15
test_text = (
    df_test.id + " " + df_test.anchor + " " + df_test.context + " " + df_test.target
).str.lower()

res = model.predict(
    pad_sequences(
        token.texts_to_sequences(test_text),
        maxlen=max_length,
        padding="post",
    ),
    verbose=0,
).ravel()



## === cell 16
res_train = model.predict(padded_docs, verbose=0).ravel()
print("Train corrcoef matrix:\n", np.corrcoef(res_train, df_train.score.to_numpy()))



## === cell 17
df_test["score"] = res.astype(np.float64)

sub = df_test[["id", "score"]]
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
print(sub.dtypes)
