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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
tf_keras==2.18.0

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

-0.3844

# 6. Current score

0.36541

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.53687) has done: 'I fix the environment/runtime issues caused by incompatible `tensorflow.python.keras` imports and the Keras 3 `Lambda` output-shape inference error, so the Siamese model can be built and trained. I keep the same architecture and training loop, but use stable public Keras APIs (`tensorflow.keras.utils.plot_model` only if available) and explicitly provide `output_shape` for the distance `Lambda`. Then I ensure the test predictions are converted into a valid `score` column and written to `submission.csv` with exactly `['id','score']`. These changes are primarily correctness/stability fixes so you get a valid submission file end-to-end.'
- What this solution (achieved 0.52751) has done: 'The current notebook fails immediately when importing TensorFlow due to a protobuf/Keras/TensorFlow incompatibility showing up as `MessageFactory.GetPrototype`. To make it run end-to-end in this Kaggle environment, I force the Python protobuf implementation (a common fix for this exact error) before TensorFlow is imported, and I also make sure paths work whether the data is in `/kaggle/input/...` or `../input/...`. These changes are runtime/stability fixes and should keep the model logic and score behavior essentially the same. The rest of the code (tokenization, Siamese model, training loop, and submission writing) is preserved.'
- What this solution (achieved 0.35992) has done: 'I make the code produce a valid submission reliably by removing the forced Python re-exec (which often prevents Kaggle notebooks from completing and thus “Not yielded”), while keeping the protobuf environment fix in place before importing TensorFlow. I also fix a key train/validation leakage issue: the tokenizer is currently fit on the holdout split labels’ texts (X_test), which can inflate the local correlation and hurt generalization; fitting only on training texts (plus unlabeled Kaggle test texts, which is allowed) usually improves leaderboard Pearson without changing the model itself. Finally, I ensure the submission rows are aligned to `sample_submission.csv` by merging on `id` so the output order always matches the expected format.'
- What this solution (achieved 0.36976) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation *before* any TensorFlow/keras import in the same process, and by clearing any preloaded `google.protobuf` modules to ensure the setting takes effect. I keep the rest of your pipeline (tokenizer, Siamese network, contrastive loss, training loop, calibration, and submission merge-on-id) unchanged to preserve evaluation semantics and score behavior. I also make the data path detection a bit more robust (still using the same canonical Kaggle paths) and keep the output as a valid `submission.csv` with exactly `id,score`. These changes are runtime/stability fixes and should bring the run back to producing a valid submission with similar expected score.'
- What this solution (achieved 0.35633) has done: 'The crash happens before training because TensorFlow is imported in cell 1, but the protobuf environment fix is applied in a different earlier cell; in Kaggle, running cell 1 standalone can still import TF with the incompatible C++ protobuf backend. I make the protobuf environment setup happen at the very top of the script (cell 1) before any TensorFlow/Keras import, and also set `TF_CPP_MIN_LOG_LEVEL` to reduce noisy logs (score-neutral). I also make the data-path selection deterministic and keep the rest of your pipeline (tokenizer fitting strategy, Siamese network, contrastive loss, calibration, and submission merge-on-id) unchanged to preserve semantics and keep score behavior consistent. Finally, I ensure the submission is always written as `submission.csv` with exactly `id,score`.'
- What this solution (achieved 0.36905) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf runtime at the very top of the script and (critically) doing it before any TensorFlow/Keras import, while also clearing any already-imported protobuf modules so the setting actually takes effect. This is a runtime/stability fix and keeps your model/training/calibration logic unchanged, so the score behavior should remain essentially the same (already within the target tolerance band). I also remove the extra directory-walk printing to reduce overhead/noise but keep all I/O paths and submission-writing logic intact. Finally, I ensure the notebook starts at cell 1 (Kaggle-friendly) and still writes a valid `submission.csv` with exactly `id,score`.'
- What this solution (achieved 0.36219) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf runtime *and* fully preventing any C++/upb protobuf backend from being used, which is the direct cause of this error in TF 2.18 + protobuf 6.x environments. This is done at the very top before any TensorFlow/Keras import, and by setting additional safe environment flags plus clearing any pre-imported protobuf modules. I keep the model/tokenization/training/calibration logic unchanged to preserve score behavior, and only touch the import/runtime stability pieces so the notebook runs end-to-end. Finally, the script still write a valid `submission.csv` with exactly `id,score`.'
- What this solution (achieved 0.36144) has done: 'I fix the immediate TensorFlow import crash (`MessageFactory.GetPrototype`) by moving the protobuf environment setup to the very top and also forcing the Python protobuf runtime via `google.protobuf.internal.api_implementation` *before* importing TensorFlow. This is a runtime-only stability change and preserves your model/tokenization/training/calibration logic as-is, so expected score behavior should remain essentially unchanged (still in the same range as your current 0.36219). I also keep the Kaggle input path selection and submission-writing logic intact to ensure a valid `submission.csv` with exactly `id,score` is always produced. No architecture, loss, training loop, or feature-extraction changes are introduced.'
- What this solution (achieved 0.36334) has done: 'I fix the TensorFlow import crash (`AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`) by forcing the pure-Python protobuf runtime *before any TensorFlow/Keras import* and doing a safe process-level fallback: if TF import still fails, the script generate a valid `submission.csv` using the sample submission format (so you always “yield” a file). This is a runtime/stability fix; it preserves your existing model/tokenizer/training/calibration logic unchanged when TensorFlow imports successfully. I also keep the Kaggle input path detection as-is and ensure the submission always has exactly `id,score` and is written to `submission.csv`.'
- What this solution (achieved 0.36541) has done: 'I fix the TensorFlow import crash at its root by forcing TensorFlow to use the pure-Python protobuf implementation and applying the environment variables *before* any protobuf/TensorFlow modules are imported. This removes the current failure (`MessageFactory.GetPrototype`) so the original Siamese network, contrastive loss, training loop, calibration, and submission-writing logic can run unchanged end-to-end. I also keep the existing safe fallback (write a valid `submission.csv`) in case TensorFlow still cannot import for any unexpected reason. These changes are runtime/stability-only and should preserve (or slightly improve) your achieved score while ensuring a valid CSV is always produced.'

# 9. Code solution

## === cell 0
import os
import sys

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_UPB", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_DISABLE_UPB", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

if "TF_USE_LEGACY_KERAS" not in os.environ:
    pass

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        sys.modules.pop(m, None)

import numpy as np
import pandas as pd



## === cell 1
import matplotlib.pyplot as plt

TF_AVAILABLE = True
TF_IMPORT_ERROR = None
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from sklearn.model_selection import train_test_split

    tf.random.set_seed(42)
    np.random.seed(42)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
except Exception as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = e
    print(
        "TensorFlow import failed; will fall back to baseline submission. Error:",
        repr(e),
    )



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
import os

TRAIN_PATH = "/kaggle/input/us-patent-phrase-to-phrase-matching/train.csv"
TEST_PATH = "/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv"
SAMPLE_SUB_PATH = (
    "/kaggle/input/us-patent-phrase-to-phrase-matching/sample_submission.csv"
)

if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = (
        "/kaggle/input/train.csv"
        if os.path.exists("/kaggle/input/train.csv")
        else "../input/us-patent-phrase-to-phrase-matching/train.csv"
    )
if not os.path.exists(TEST_PATH):
    TEST_PATH = (
        "/kaggle/input/test.csv"
        if os.path.exists("/kaggle/input/test.csv")
        else "../input/us-patent-phrase-to-phrase-matching/test.csv"
    )
if not os.path.exists(SAMPLE_SUB_PATH):
    SAMPLE_SUB_PATH = (
        "/kaggle/input/sample_submission.csv"
        if os.path.exists("/kaggle/input/sample_submission.csv")
        else "../input/us-patent-phrase-to-phrase-matching/sample_submission.csv"
    )

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

if not TF_AVAILABLE:
    submission = sample_sub[["id"]].copy()
    submission["score"] = 0.0
    submission.to_csv("submission.csv", index=False)
    print(
        "TF unavailable -> wrote fallback submission.csv with shape:", submission.shape
    )
    print("Columns:", submission.columns.tolist())
    raise SystemExit(0)



## === cell 3
train.head()



## === cell 4
X_train, X_test, y_train, y_test = train_test_split(
    train[["anchor", "target", "context"]],
    train["score"],
    test_size=0.25,
    random_state=42,
)



## === cell 5
print(X_train.head())
print(X_test.head())
print(y_train.head())
print(y_test.head())



## === cell 6
X_train = X_train.copy()
X_test = X_test.copy()

X_train["text"] = X_train[["anchor", "target", "context"]].apply(
    lambda x: f"{str(x[0])} {str(x[1])} {str(x[2])}", axis=1
)



## === cell 7
X_train.head()



## === cell 8
X_train["anchor"].str.len().plot(kind="hist")



## === cell 9
X_train["target"].str.len().plot(kind="hist")



## === cell 10
print(max(X_train["anchor"].str.len()))
print(max(X_train["target"].str.len()))



## === cell 11
X_train.head()



## === cell 12
vocab_size = 7905
embedding_dim = 16
max_length = 4
trunc_type = "post"
oov_tok = ""



## === cell 13
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

fit_texts = pd.concat(
    [
        X_train["text"],
        test[["anchor", "target", "context"]].apply(
            lambda x: f"{str(x[0])} {str(x[1])} {str(x[2])}", axis=1
        ),
    ],
    axis=0,
).values

tokenizer.fit_on_texts(fit_texts)



## === cell 14
word_index = tokenizer.word_index
len(word_index), list(word_index.items())[:10]



## === cell 15
anchor_sequences = tokenizer.texts_to_sequences(X_train["anchor"].values)
target_sequences = tokenizer.texts_to_sequences(X_train["target"].values)

padded_anchor_sequences = pad_sequences(
    anchor_sequences, maxlen=max_length, truncating=trunc_type, padding="post"
)
padded_target_sequences = pad_sequences(
    target_sequences, maxlen=max_length, truncating=trunc_type, padding="post"
)



## === cell 16
padded_anchor_sequences.shape[1]



## === cell 17
val_anchor_sequences = tokenizer.texts_to_sequences(X_test["anchor"].values)
val_target_sequences = tokenizer.texts_to_sequences(X_test["target"].values)

val_padded_anchor_sequences = pad_sequences(
    val_anchor_sequences, maxlen=max_length, truncating=trunc_type, padding="post"
)
val_padded_target_sequences = pad_sequences(
    val_target_sequences, maxlen=max_length, truncating=trunc_type, padding="post"
)



## === cell 18
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Lambda
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K

try:
    from tensorflow.keras.utils import plot_model
except Exception:
    plot_model = None




## === cell 19
def initialize_base_network():
    inp = Input(shape=(padded_anchor_sequences.shape[1],))
    common_embedding = tf.keras.layers.Embedding(
        vocab_size, embedding_dim, input_length=max_length
    )(inp)
    common_lstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(32, return_sequences=True)
    )(common_embedding)
    flatten_layer = Flatten()(common_lstm)
    return Model(inputs=inp, outputs=flatten_layer)




## === cell 20
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)




## === cell 21
base_network = initialize_base_network()

if plot_model is not None:
    try:
        plot_model(base_network, show_shapes=True)
    except Exception as e:
        print("plot_model skipped:", repr(e))



## === cell 22
input_1 = Input(shape=(padded_anchor_sequences.shape[1],), name="input_1")
input_2 = Input(shape=(padded_target_sequences.shape[1],), name="input_2")

vec_1 = base_network(input_1)
vec_2 = base_network(input_2)

output = Lambda(
    euclidean_distance, name="output_layer", output_shape=eucl_dist_output_shape
)([vec_1, vec_2])

siamese_model = Model([input_1, input_2], output)

if plot_model is not None:
    try:
        plot_model(
            siamese_model,
            show_shapes=True,
            show_layer_names=True,
            to_file="outer-model.png",
        )
    except Exception as e:
        print("plot_model skipped:", repr(e))




## === cell 23
def contrastive_loss_with_margin(margin):
    def contrastive_loss(y_true, y_pred):
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

    return contrastive_loss




## === cell 24
rms = RMSprop()
siamese_model.compile(loss=contrastive_loss_with_margin(margin=1), optimizer=rms)
siamese_model.summary()



## === cell 25
y_train_bin = (y_train.values.astype(np.float32) >= 0.75).astype(np.float32)
y_test_bin = (y_test.values.astype(np.float32) >= 0.75).astype(np.float32)

history = siamese_model.fit(
    [padded_anchor_sequences, padded_target_sequences],
    y_train_bin,
    epochs=10,
    batch_size=64,
    validation_data=(
        [val_padded_anchor_sequences, val_padded_target_sequences],
        y_test_bin,
    ),
    verbose=2,
)



## === cell 26
val_distance = siamese_model.predict(
    [val_padded_anchor_sequences, val_padded_target_sequences], verbose=0
)



## === cell 27
print(val_distance.shape)
val_distance = val_distance.reshape((len(val_distance),))
print(val_distance.shape)



## === cell 28
val_similarity = 1.0 / (1.0 + val_distance.astype(np.float32))
val_similarity = np.clip(val_similarity, 0.0, 1.0)

try:
    corr = np.corrcoef(y_test.values.astype(float), val_similarity.astype(float))[0, 1]
    print("Holdout Pearson (1/(1+distance) vs score):", corr)
except Exception as e:
    print("Correlation check skipped:", repr(e))



## === cell 29
from sklearn.linear_model import LinearRegression

_cal = LinearRegression()
_cal.fit(val_similarity.reshape(-1, 1), y_test.values.astype(np.float32))
a = float(_cal.coef_[0])
b = float(_cal.intercept_)
print("Calibration: y = a*x + b with a=", a, "b=", b)

val_similarity_cal = np.clip(a * val_similarity + b, 0.0, 1.0)
try:
    corr_cal = np.corrcoef(
        y_test.values.astype(float), val_similarity_cal.astype(float)
    )[0, 1]
    print("Holdout Pearson (calibrated):", corr_cal)
except Exception as e:
    print("Correlation check skipped:", repr(e))



## === cell 30
test_anchor_sequences = tokenizer.texts_to_sequences(test["anchor"].values)
test_target_sequences = tokenizer.texts_to_sequences(test["target"].values)

test_padded_anchor_sequences = pad_sequences(
    test_anchor_sequences, maxlen=max_length, truncating=trunc_type, padding="post"
)
test_padded_target_sequences = pad_sequences(
    test_target_sequences, maxlen=max_length, truncating=trunc_type, padding="post"
)



## === cell 31
test_distance = siamese_model.predict(
    [test_padded_anchor_sequences, test_padded_target_sequences], verbose=0
)



## === cell 32
test_distance = test_distance.reshape((len(test_distance),))

test = test.copy()
test_similarity = 1.0 / (1.0 + test_distance.astype(np.float32))
test_similarity = np.clip(test_similarity, 0.0, 1.0)

test["score"] = np.clip(a * test_similarity + b, 0.0, 1.0).astype(np.float32)



## === cell 33
test.head(5)



## === cell 34
submission = sample_sub[["id"]].merge(test[["id", "score"]], on="id", how="left")
submission["score"] = submission["score"].fillna(0.0).astype(np.float32)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print("Columns:", submission.columns.tolist())
print(submission.head())
