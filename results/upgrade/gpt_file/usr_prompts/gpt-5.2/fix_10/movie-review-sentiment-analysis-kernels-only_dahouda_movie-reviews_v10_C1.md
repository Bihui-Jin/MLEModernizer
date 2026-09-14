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
Predict the sentiment of phrases.

## Metric
Classification accuracy.

## Submission Format
For each phrase in the test set, predict a label for the sentiment. Your submission should have a header and look like the following:

```
PhraseId,Sentiment
156061,2
156062,2
156063,2
...
```

## Dataset
The dataset is comprised of tab-separated files with phrases. Each phrase has a PhraseId. Each sentence has a SentenceId.

The sentiment labels are:

0 - negative

1 - somewhat negative

2 - neutral

3 - somewhat positive

4 - positive

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (72 lines)
            sampleSubmission.csv (46819 lines)
            sampleSubmission.csv.zip (146.0 kB)
            test.tsv (46819 lines)
            test.tsv.zip (1.1 MB)
            train.tsv (109243 lines)
            train.tsv.zip (2.7 MB)
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
        input/
            description.md (72 lines)
            sampleSubmission.csv (46819 lines)
            sampleSubmission.csv.zip (146.0 kB)
            test.tsv (46819 lines)
            test.tsv.zip (1.1 MB)
            train.tsv (109243 lines)
            train.tsv.zip (2.7 MB)
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
        working/
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
```

-> data/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv has 46818 rows and 2 columns.
Here is some information about the columns:
PhraseId (int64) has range: 29.00 - 156030.00, 0 nan values
Sentiment (int64) has 1 unique values: [2]

-> data/sampleSubmission.csv has 46818 rows and 2 columns.
Here is some information about the columns:
PhraseId (int64) has range: 29.00 - 156030.00, 0 nan values
Sentiment (int64) has 1 unique values: [2]

-> (stopped after 10 files for performance)

# 5. Target score

0.6404090991371508

# 6. Current score

0.22318

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.22318) has done: 'The timeout is almost certainly dominated by the LSTM training cost (25 epochs over ~87k sequences), plus some avoidable overhead from passing NumPy through `from_tensor_slices`. To keep identical model/loop semantics while speeding up, I (1) enable XLA JIT (same math, faster kernels), (2) replace the `from_tensor_slices(...).shuffle().batch()` pipeline with a deterministic Keras `Sequence` that does the same shuffling once and feeds contiguous batches without per-element tf.data overhead, and (3) remove redundant dataset `.cache()` usage that provides little benefit for in-memory arrays. Tokenization/feature extraction and the model architecture/loss/training epochs remain unchanged.'

# 9. Code solution

## === cell 0
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", None)

import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

seed = 101
np.random.seed(seed)
tf.random.set_seed(seed)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

print("TensorFlow:", tf.__version__)
print("Devices:", tf.config.list_physical_devices())



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_PATH = "/kaggle/input/movie-review-sentiment-analysis-kernels-only"

train_path = os.path.join(BASE_PATH, "train.tsv.zip")
test_path = os.path.join(BASE_PATH, "test.tsv.zip")
sub_path = os.path.join(BASE_PATH, "sampleSubmission.csv")

train = pd.read_csv(
    train_path, sep="\t", encoding="utf-8", usecols=["Phrase", "Sentiment"]
)
test = pd.read_csv(
    test_path, sep="\t", encoding="utf-8", usecols=["PhraseId", "Phrase"]
)
sub = pd.read_csv(sub_path)

print(train.shape)
print(test.shape)



## === cell 2
pass



## === cell 3
X = train["Phrase"].astype(str).tolist()
temp = test["Phrase"].astype(str).tolist()

num_classes = train["Sentiment"].nunique()
y = to_categorical(train["Sentiment"].values, num_classes=num_classes).astype("float32")

print("Number of classes:", num_classes)
print("X len:", len(X), "y shape:", y.shape)



## === cell 4
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=train["Sentiment"].values, random_state=seed
)
print(len(X_train), len(X_test), y_train.shape, y_test.shape)



## === cell 5
pass



## === cell 6
max_features = 15000

tokenizer = Tokenizer(num_words=max_features, oov_token="<OOV>")

tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
temp_seq = tokenizer.texts_to_sequences(temp)

max_words = 50

X_train_pad = sequence.pad_sequences(X_train_seq, maxlen=max_words).astype(
    np.int32, copy=False
)
X_test_pad = sequence.pad_sequences(X_test_seq, maxlen=max_words).astype(
    np.int32, copy=False
)
temp_pad = sequence.pad_sequences(temp_seq, maxlen=max_words).astype(
    np.int32, copy=False
)

X_train_pad = np.ascontiguousarray(X_train_pad)
X_test_pad = np.ascontiguousarray(X_test_pad)
temp_pad = np.ascontiguousarray(temp_pad)

print(X_train_pad.shape, X_test_pad.shape, temp_pad.shape)



## === cell 7
print("X_train_pad dtype/shape:", X_train_pad.dtype, X_train_pad.shape)



## === cell 8
print("X_test_pad dtype/shape:", X_test_pad.dtype, X_test_pad.shape)



## === cell 9
batch_size = 64
epochs = 25


def get_model(max_features_local, embed_dim, embedding_matrix):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    model = Sequential()
    model.add(
        Embedding(
            input_dim=max_features_local,
            output_dim=embed_dim,
            input_length=X_train_pad.shape[1],
            weights=[embedding_matrix],
        )
    )

    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.0))

    model.add(Dense(100, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
        run_eagerly=False,
        steps_per_execution=32,
    )
    return model




## === cell 10
embed_dim = 100

word_index = tokenizer.word_index
num_words = min(max_features, len(word_index) + 1)

rng = np.random.default_rng(seed)
embedding_matrix = rng.normal(loc=0.0, scale=0.05, size=(num_words, embed_dim)).astype(
    "float32"
)

max_features_effective = embedding_matrix.shape[0]
print("Using random embedding matrix with shape:", embedding_matrix.shape)




## === cell 11
class NumpyBatchSequence(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size, shuffle=False, seed=101):
        self.X = X
        self.y = y
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.indices = np.arange(len(self.X), dtype=np.int32)
        if self.shuffle:
            rs = np.random.RandomState(self.seed)
            rs.shuffle(self.indices)

    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        sl = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        batch_idx = self.indices[sl]
        xb = self.X[batch_idx]
        yb = self.y[batch_idx] if self.y is not None else None
        return (xb, yb) if yb is not None else xb


model = get_model(max_features_effective, embed_dim, embedding_matrix)

train_seq = NumpyBatchSequence(
    X_train_pad, y_train, batch_size=batch_size, shuffle=True, seed=seed
)
val_seq = NumpyBatchSequence(
    X_test_pad, y_test, batch_size=batch_size, shuffle=False, seed=seed
)

model.fit(
    train_seq,
    epochs=epochs,
    verbose=2,
    validation_data=None,
    workers=0,
    use_multiprocessing=False,
)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3527122011.py in <cell line: 0>()
     38 )
     39 
---> 40 model.fit(
     41     train_seq,
     42     epochs=epochs,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    117             return fn(*args, **kwargs)
    118         except Exception as e:
--> 119             filtered_tb = _process_traceback_frames(e.__traceback__)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`

TypeError: TensorFlowTrainer.fit() got an unexpected keyword argument 'workers'

## === cell 12
test_seq = NumpyBatchSequence(
    temp_pad, None, batch_size=batch_size, shuffle=False, seed=seed
)

probs = model.predict(test_seq, verbose=0)
preds = np.argmax(probs, axis=1).astype(int)

submission = pd.DataFrame({"PhraseId": test["PhraseId"].values, "Sentiment": preds})
submission.to_csv("submission.csv", index=False)

print(submission.head())

print("submission.csv rows:", submission.shape[0])
print("Unique sentiments predicted:", sorted(submission["Sentiment"].unique().tolist()))
print(submission.head())
