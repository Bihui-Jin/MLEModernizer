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

gensim==4.4.0
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

0.53147

# 6. Current score

0.61321

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.61768) has done: 'I fix the runtime errors caused by gensim 4 API changes (`size`→`vector_size`, `wv.vocab` removal, and indexing through `wv`) so the Word2Vec embedding trains and can be used for averaging. I also fix the Keras crash in this Kaggle environment by switching the imports to `tf_keras` (keeping the same Sequential/Dense architecture, loss, and training loops) so model training/inference runs end-to-end. Finally, I make the text-to-vector averaging robust to empty/unknown tokens to avoid NaNs that can hurt logloss, and I keep the submission formatting exactly as required (`id,EAP,HPL,MWS`) writing `submission.csv`.'
- What this solution (achieved 0.62344) has done: 'I fix the crash at model creation by resolving the protobuf/Keras incompatibility that triggers `MessageFactory.GetPrototype` in this Kaggle image; this is a runtime blocker and score-neutral. I keep the same data prep, Word2Vec embedding training, averaging, and the same two Sequential model definitions/training loops, only changing the Keras import backend to one that works reliably here. I also make the train/dev split stratified (same split size and random_state) to stabilize validation behavior without changing the final full-train model used for submission. Finally, I keep the submission formatting identical and ensure `submission.csv` is written.'
- What this solution (achieved 0.61609) has done: 'I fix the runtime crash coming from the protobuf/TensorFlow/Keras stack by forcing the compatible pure-Python protobuf implementation *before* any library that can import protobuf is loaded, and by using the stable `tf_keras` backend consistently. This is required to make model creation/training run end-to-end and produce `submission.csv`. I also keep the model architecture, training loops, and feature extraction identical to preserve evaluation semantics and avoid unintended score shifts. Finally, I make the label dummy-column order deterministic (`EAP,HPL,MWS`) so training targets always align with the submission columns, which can otherwise silently hurt logloss.'
- What this solution (achieved 0.61983) has done: 'I fix the protobuf/Keras crash by forcing a compatible protobuf version early (and falling back to `tensorflow.keras` if `tf_keras` still triggers the `MessageFactory.GetPrototype` error), without changing your model architectures, losses, or training loops. I also make sure no heavy plotting blocks execution (use a non-interactive backend) and keep all paths and feature extraction identical. These changes are primarily runtime/stability fixes and should be score-neutral; your existing label/column ordering and submission formatting are preserved to avoid silent logloss regressions. The script run end-to-end and always write a valid `submission.csv` with columns `id,EAP,HPL,MWS`.'
- What this solution (achieved 0.62773) has done: 'I fix the protobuf/Keras crash that prevents the model from being created by pinning the compatible pure-Python protobuf behavior and (more importantly) forcing the known-stable `tf_keras` backend while preventing accidental import of standalone `keras` (which triggers the `MessageFactory.GetPrototype` error in this environment). This change is runtime/stability focused and keeps your exact model architectures, losses, and training loops intact. I also make the input path robust to both `/kaggle/input/*.csv` and the nested `spooky-author-identification` folder without changing which data is used. Finally, I ensure the submission CSV is always written with the required columns `id,EAP,HPL,MWS`.'
- What this solution (achieved 0.61388) has done: 'I fix the protobuf/Keras crash causing `MessageFactory.GetPrototype` by forcing a compatible protobuf runtime early and then using `tf_keras` with a safe fallback to `tensorflow.keras` if needed; this is a runtime blocker and should be score-neutral. I also update the data path candidates to match your actual environment (`/kaggle/input/...` doesn’t exist here; the files are under `/kaggle/data` and `/kaggle/input` in your listing) so the notebook reliably finds the CSVs. Finally, I keep your exact feature extraction, Word2Vec settings, model architectures, losses, and training loops intact, and ensure the submission is written as `submission.csv` with columns `id,EAP,HPL,MWS`.'
- What this solution (achieved 0.62304) has done: 'I fix the Keras/protobuf crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation early and by preferring the stable `tf_keras` backend while actively preventing the standalone `keras` import path from being used in this environment. This is a runtime-blocking issue; the model architecture, losses, training loops, Word2Vec feature extraction, and submission formatting remain unchanged. I also add a safe final fallback to `tensorflow.keras` if `tf_keras` still fails to initialize, so the notebook always runs end-to-end and writes `submission.csv` with `id,EAP,HPL,MWS`. These changes are intended to be score-neutral (they should not materially change predictions), focusing on correctness and stability.'
- What this solution (achieved 0.60664) has done: 'The crash comes from a protobuf/Keras incompatibility where TensorFlow’s Keras tries to use a `MessageFactory.GetPrototype` method that doesn’t exist in the protobuf version in this environment. To keep your model architecture/training logic intact, I add an early, safe monkey-patch that provides `GetPrototype` as an alias to `GetMessageClass` when missing, which unblocks model creation/fit/predict. I also keep your existing `tf_keras`→`tensorflow.keras` fallback, but force-import TensorFlow after the patch is applied so it takes effect before Keras initializes. These changes are runtime/stability focused and should be score-neutral (no intentional score tuning), while ensuring a valid `submission.csv` is always produced.'
- What this solution (achieved 0.62523) has done: 'I fix the protobuf/Keras crash by replacing the fragile `MessageFactory.GetPrototype` aliasing with a safer patch that adds `GetPrototype` to the *instance* returned by protobuf when it’s missing, which prevents the `AttributeError` shown in cell 0. This change is runtime-only and does not alter your model architecture, training loops, Word2Vec settings, or feature extraction, so it should be score-neutral while ensuring the notebook runs end-to-end. I also keep the existing `tf_keras` → `tensorflow.keras` fallback intact and preserve the exact submission formatting (`id,EAP,HPL,MWS`) to avoid silent logloss regressions. No score-tuning changes are introduced because your current score is already reasonably close to the target band and the priority is correctness/stability.'
- What this solution (achieved 0.61512) has done: 'The immediate blocker is the protobuf patch: the current code still triggers `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` during import-time initialization. I replace it with a safer early patch that adds `GetPrototype` to both the `MessageFactory` class and to the generated global factory instance if missing, before importing TensorFlow/Keras, which fixes the crash without changing your modeling logic. I also keep the rest of the pipeline (Word2Vec training, averaging, and the same two Keras models/training loops) intact, only adding deterministic seeds for TensorFlow to stabilize runs (generally score-neutral). The script still write a valid `submission.csv` with columns `id,EAP,HPL,MWS`.'
- What this solution (achieved 0.61541) has done: 'I fix the import-time crash by applying the protobuf `MessageFactory.GetPrototype` patch even earlier and more comprehensively (including the C++/python factory fallback points) before importing TensorFlow/Keras, which is what currently stops the notebook at cell 0. This is a runtime/stability fix and keeps your Word2Vec averaging, model architectures, training loops, and submission formatting unchanged (so score behavior should remain essentially the same, just runnable). I also add a safe TensorFlow import guard so it won’t re-trigger the crash, and keep deterministic seeding as you already intended. The rest of the pipeline is kept intact to avoid unnecessary score drift away from your current ~0.615 toward the target 0.531.'
- What this solution (achieved 0.61321) has done: 'You’re crashing before any modeling because TensorFlow/Keras import triggers protobuf’s `MessageFactory.GetPrototype` on a factory instance that still lacks that attribute; your current patch only conditionally aliases it and doesn’t reliably cover the runtime factory objects protobuf actually uses. I replace the protobuf patch with a minimal, deterministic one that adds `GetPrototype` to the `MessageFactory` class and also patches both `_DEFAULT_FACTORY` and `_GENERATED_MESSAGE_FACTORY` instances when present, and I do it strictly before importing TensorFlow/keras. This is a runtime/stability fix and is score-neutral; I won’t change your Word2Vec settings, averaging, model architectures, training loops, or submission formatting. The rest of the pipeline remains the same and write a valid `submission.csv` with columns `id,EAP,HPL,MWS`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

import re
import gensim
from gensim.models import Word2Vec

np.random.seed(123)

try:
    from google.protobuf import message_factory as _message_factory

    def _patch_factory_obj(factory_obj):
        if factory_obj is None:
            return
        if (not hasattr(factory_obj, "GetPrototype")) and hasattr(
            factory_obj, "GetMessageClass"
        ):
            try:
                setattr(factory_obj, "GetPrototype", factory_obj.GetMessageClass)
            except Exception:
                pass

    if hasattr(_message_factory, "MessageFactory"):
        _mf_cls = _message_factory.MessageFactory
        if (not hasattr(_mf_cls, "GetPrototype")) and hasattr(
            _mf_cls, "GetMessageClass"
        ):
            try:
                _mf_cls.GetPrototype = _mf_cls.GetMessageClass
            except Exception:
                pass

    for _name in ["_DEFAULT_FACTORY", "_GENERATED_MESSAGE_FACTORY"]:
        _patch_factory_obj(getattr(_message_factory, _name, None))

except Exception as _e:
    print("Warning: protobuf patch skipped due to:", repr(_e))

try:
    import tensorflow as tf  # noqa: F401

    try:
        tf.random.set_seed(123)
    except Exception:
        pass
except Exception as _e:
    print("Warning: tensorflow import failed (may still work via tf_keras):", repr(_e))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
TRAIN_CANDIDATES = [
    "/kaggle/input/train.csv",
    "/kaggle/input/spooky-author-identification/train.csv",
    "/kaggle/data/train.csv",
    "/kaggle/data/spooky-author-identification/train.csv",
]
TEST_CANDIDATES = [
    "/kaggle/input/test.csv",
    "/kaggle/input/spooky-author-identification/test.csv",
    "/kaggle/data/test.csv",
    "/kaggle/data/spooky-author-identification/test.csv",
]


def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of these paths exist: {paths}")


train_path = _first_existing(TRAIN_CANDIDATES)
test_path = _first_existing(TEST_CANDIDATES)

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
print("Using:", train_path, "and", test_path)
print(train_df.shape, test_df.shape)




## === cell 2
def clean_text(text):
    """
    Convert all to lowercase and remove punctuations
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # remove everything that isn't word or space
    text = re.sub(r"\_", "", text)  # remove underscore
    return text




## === cell 3
train_df["text"] = train_df["text"].map(lambda x: clean_text(x))
train_df["text"] = train_df["text"].map(lambda x: x.strip().split())
train_df.head()



## === cell 4
test_df["text"] = test_df["text"].map(lambda x: clean_text(x))
test_df["text"] = test_df["text"].map(lambda x: x.strip().split())
test_df.head()



## === cell 5
data = []
for i in range(len(train_df)):
    data.append(train_df["text"].iloc[i])
for j in range(len(test_df)):
    data.append(test_df["text"].iloc[j])



## === cell 6
print(len(data))



## === cell 7
embedding = gensim.models.Word2Vec(
    sentences=data, vector_size=50, window=10, min_count=1, sg=0, workers=1, seed=123
)



## === cell 8
print(embedding)



## === cell 9
embedding.train(data, total_examples=len(data), epochs=30)



## === cell 10
words = list(embedding.wv.key_to_index.keys())
print(len(words))



## === cell 11
if "capered" in embedding.wv:
    print(embedding.wv["capered"])
else:
    print("Word 'capered' not in vocabulary")



## === cell 12
if "dark" in embedding.wv:
    print(embedding.wv.most_similar("dark", topn=5))
else:
    print("Word 'dark' not in vocabulary")



## === cell 13
if "shocked" in embedding.wv:
    print(embedding.wv.most_similar("shocked", topn=5))
else:
    print("Word 'shocked' not in vocabulary")



## === cell 14
if "sprang" in embedding.wv:
    print(embedding.wv.most_similar("sprang", topn=5))
else:
    print("Word 'sprang' not in vocabulary")



## === cell 15
if "pride" in embedding.wv:
    print(embedding.wv.most_similar("pride", topn=5))
else:
    print("Word 'pride' not in vocabulary")



## === cell 16
train_df["author"] = pd.Categorical(
    train_df["author"], categories=["EAP", "HPL", "MWS"]
)
df_Dummies = pd.get_dummies(train_df["author"], prefix="author")
for col in ["author_EAP", "author_HPL", "author_MWS"]:
    if col not in df_Dummies.columns:
        df_Dummies[col] = 0
df_Dummies = df_Dummies[["author_EAP", "author_HPL", "author_MWS"]]
train_df = pd.concat([train_df, df_Dummies], axis=1)
train_df.head()



## === cell 17
X = train_df["text"]
Y = train_df[["author_EAP", "author_HPL", "author_MWS"]].values
print(X.shape, X.iloc[0], Y.shape, Y[0])



## === cell 18
X_test = test_df["text"]
print(X_test.shape, X_test.iloc[0])




## === cell 19
def text_to_avg(text):
    """Average the Word2Vec representations of tokens in `text`.
    Robust to unknown tokens and empty texts to avoid NaNs harming logloss."""
    avg = np.zeros((50,), dtype=np.float32)
    cnt = 0
    for w in text:
        if w in embedding.wv:
            avg += embedding.wv[w]
            cnt += 1
    if cnt == 0:
        return avg
    return avg / cnt




## === cell 20
X_avg = np.zeros((X.shape[0], 50), dtype=np.float32)  # initialize X_avg
for i in range(X.shape[0]):
    X_avg[i] = text_to_avg(X.iloc[i])



## === cell 21
print(X_avg.shape)
print(X_avg[0])



## === cell 22
X_test_avg = np.zeros((X_test.shape[0], 50), dtype=np.float32)  # initialize X_test_avg
for i in range(X_test.shape[0]):
    X_test_avg[i] = text_to_avg(X_test.iloc[i])



## === cell 23
print(X_test_avg.shape)
print(X_test_avg[0])



## === cell 24
from sklearn.model_selection import train_test_split

y_class = np.argmax(Y, axis=1)
X_train, X_dev, Y_train, Y_dev = train_test_split(
    X_avg, Y, test_size=0.2, random_state=123, stratify=y_class
)
print(X_train.shape, Y_train.shape, X_dev.shape, Y_dev.shape)



## === cell 25
_KERAS_BACKEND_USED = None
models = None
layers = None

try:
    from tf_keras import models as _models, layers as _layers

    models, layers = _models, _layers
    _KERAS_BACKEND_USED = "tf_keras"
except Exception as e1:
    print("tf_keras import failed, falling back to tensorflow.keras. Error:", repr(e1))
    try:
        from tensorflow.keras import models as _models, layers as _layers

        models, layers = _models, _layers
        _KERAS_BACKEND_USED = "tensorflow.keras"
    except Exception as e2:
        raise RuntimeError(
            "Failed to import both tf_keras and tensorflow.keras backends."
        ) from e2

print("Using Keras backend:", _KERAS_BACKEND_USED)

model = models.Sequential()
model.add(layers.Dense(64, activation="relu", input_shape=(50,)))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(3, activation="softmax"))

model.summary()



## === cell 26
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])



## === cell 27
history = model.fit(
    X_train,
    Y_train,
    epochs=30,
    batch_size=128,
    validation_data=(X_dev, Y_dev),
    verbose=2,
)



## === cell 28
loss = history.history["loss"]
dev_loss = history.history["val_loss"]
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, "bo", label="training loss")
plt.plot(epochs, dev_loss, "b", label="validation loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("train_val_loss.png")
print("Saved plot to train_val_loss.png")



## === cell 29
model = models.Sequential()
model.add(layers.Dense(50, activation="relu", input_shape=(50,)))
model.add(layers.Dense(3, activation="softmax"))
model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)



## === cell 30
model.fit(X_avg, Y, epochs=20, batch_size=128, verbose=2)



## === cell 31
preds = model.predict(X_test_avg, verbose=0)
print(preds.shape)
print(preds[7])



## === cell 32
pred_labels = []
for i in range(len(X_test_avg)):
    pred_label = np.argmax(preds[i])
    pred_labels.append(pred_label)



## === cell 33
print(pred_labels[7])



## === cell 34
result = pd.DataFrame(preds, columns=["EAP", "HPL", "MWS"])
result.insert(0, "id", test_df["id"].values)
result.head()



## === cell 35
result.to_csv("submission.csv", index=False, float_format="%.20f")
print("Wrote submission.csv with shape:", result.shape)
print(result.head())
