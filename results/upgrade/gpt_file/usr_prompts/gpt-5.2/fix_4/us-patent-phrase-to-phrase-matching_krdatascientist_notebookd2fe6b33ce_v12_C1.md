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

0.1947

# 6. Current score

0.42421

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.42288) has done: 'I fix the runtime crash caused by an incompatibility between TensorFlow and the installed `protobuf` version by forcing Python protobuf parsing before importing TensorFlow. Then I keep the same preprocessing, model, training loop, and output logic, but correct one clear inference-time logic issue: the test text mistakenly includes `id` even though it wasn’t used during training, which can hurt correlation. Finally, I ensure the submission is written as a proper `submission.csv` with `id,score` and that predictions are clipped to `[0,1]` (score-neutral safety).'
- What this solution (achieved 0.42421) has done: 'I fix the TensorFlow import crash caused by the `protobuf` 6.x incompatibility by forcing the pure-Python protobuf implementation and (for extra safety) pinning the protobuf API version before TensorFlow is imported. I keep the same preprocessing, model architecture, training loop, and inference logic, only adding minimal determinism settings so it runs reliably end-to-end. Since your current score (0.42288) is already well above the target (0.1947), I not make any changes intended to improve performance; the patch is strictly to make it run and produce a valid `submission.csv`. The script still write `submission.csv` with exactly `id,score` and clip predictions to `[0,1]`.'
- What this solution (achieved 0.42421) has done: 'I fix the TensorFlow import crash caused by the protobuf 6.x incompatibility by forcing the pure-Python protobuf implementation and pinning the protobuf API version *before* importing TensorFlow. I keep your preprocessing, model architecture, training, and inference logic unchanged (no score-tuning), only removing an unused import that can trigger extra TF/protobuf initialization. Finally, I ensure the notebook runs end-to-end and writes a valid `submission.csv` with exactly `id,score` and the correct row count.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONHASHSEED", "0")

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



## === cell 2
df_train.head(3)



## === cell 3
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten

tf.keras.utils.set_random_seed(0)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
input_text = (
    df_train.anchor + " " + df_train.context + " " + df_train.target
).str.lower()



## === cell 5
token = Tokenizer()



## === cell 6
token.fit_on_texts(input_text)



## === cell 7
vocab_size = len(token.word_index) + 1



## === cell 8
embedding_doc = token.texts_to_sequences(input_text)



## === cell 9
max([len(i) for i in embedding_doc])



## === cell 10
max_length = 20
padded_docs = pad_sequences(embedding_doc, maxlen=max_length, padding="post")



## === cell 11
y_train = pd.get_dummies(df_train.score).values



## === cell 12
e = tf.keras.layers.Embedding(vocab_size, 50, input_length=max_length)



## === cell 13
model = Sequential()
model.add(e)
model.add(Conv1D(32, 3, activation="relu"))
model.add(Conv1D(64, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(y_train.shape[1], activation="sigmoid"))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
print(model.summary())



## === cell 14
model.fit(padded_docs, y_train, epochs=5, verbose=1, validation_split=0.3)



## === cell 15
test_text = (df_test.anchor + " " + df_test.context + " " + df_test.target).str.lower()
res = model.predict(
    pad_sequences(
        token.texts_to_sequences(test_text), maxlen=max_length, padding="post"
    ),
    verbose=0,
)



## === cell 16
res = pd.Series(res.argmax(1)).replace(
    dict(zip(range(0, 5), pd.get_dummies(df_train.score).columns.tolist()))
)



## === cell 17
res_train = pd.Series(model.predict(padded_docs, verbose=0).argmax(1)).replace(
    dict(zip(range(0, 5), pd.get_dummies(df_train.score).columns.tolist()))
)
np.corrcoef(res_train, df_train.score.tolist())



## === cell 18
df_test["score"] = np.clip(res.astype(float), 0.0, 1.0)



## === cell 19
df_test[["id", "score"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_test[["id", "score"]].shape)
print(df_test[["id", "score"]].head())
