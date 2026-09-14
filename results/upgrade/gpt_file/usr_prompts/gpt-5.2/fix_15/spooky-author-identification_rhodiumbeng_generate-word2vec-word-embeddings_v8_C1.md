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

0.5357

# 6. Current score

0.87539

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.88794) has done: 'I fix the gensim Word2Vec API breakages (gensim 4 uses `vector_size` and the vocabulary API changed) so embeddings train and can be indexed reliably. I also adjust the Keras import to use `tf_keras` in this environment to avoid the known protobuf/MessageFactory crash seen with `keras==3.x` here, while keeping the exact same network architecture and training loop. To improve logloss toward your target, I keep predicting calibrated class probabilities (as required) and ensure the author one-hot column order matches `EAP,HPL,MWS` exactly. Finally, I keep the submission format unchanged and guarantee `submission.csv` is written successfully.'
- What this solution (achieved 0.87159) has done: 'I fix the protobuf/MessageFactory crash by switching the Keras imports to `tensorflow.keras`, which is the most stable option in Kaggle TF environments and keeps your exact same model architecture/training loop intact. I also make the random seeds more deterministic across NumPy and TensorFlow to stabilize results without changing the approach. Finally, I keep the submission format identical (id + EAP/HPL/MWS) and ensure `submission.csv` is always written successfully. No score-targeting changes beyond runtime stability are introduced, since your current score is already worse than the target and the main blocker is execution.'
- What this solution (achieved 0.87618) has done: 'I fix the TensorFlow/Keras protobuf crash causing the `MessageFactory` error by avoiding `tensorflow.keras`/`keras==3` imports and using the already-installed `tf_keras` backend instead, while keeping your exact same model architecture and training loop. I also make the data loading path robust to both `/kaggle/input/train.csv` and the nested competition folder, since Kaggle datasets can be mounted either way. Finally, I ensure the submission is always written as `submission.csv` with the required columns `id,EAP,HPL,MWS` and stable float formatting; no score-tuning changes are introduced beyond making the pipeline run correctly end-to-end.'
- What this solution (achieved 0.87558) has done: 'I fix the runtime crash coming from the protobuf/TF-Keras mismatch by forcing TensorFlow to use the pure-Python protobuf implementation before importing TensorFlow/tf_keras, which avoids the `MessageFactory.GetPrototype` error in this Kaggle environment. I also keep your exact model architecture/training loop intact, only adjusting imports/ordering and adding a small defensive probability clip before writing the submission (score-neutral but prevents logloss edge-case issues). Finally, I keep the same input paths and ensure `submission.csv` is always produced with columns `id,EAP,HPL,MWS`.'
- What this solution (achieved 0.87558) has done: 'I fix the TensorFlow/Keras protobuf `MessageFactory.GetPrototype` crash by importing `tf_keras` before `tensorflow` and ensuring the protobuf env vars are set before either import, which is the minimal change to unblock training/inference. I also remove an unnecessary standalone `import tensorflow as tf` (while still seeding deterministically) to avoid triggering the problematic code path. These changes preserve your exact model architecture, training loops, and feature extraction, and should be score-neutral aside from improved runtime stability. The pipeline still write `submission.csv` with the required `id,EAP,HPL,MWS` columns.'
- What this solution (achieved 0.87558) has done: 'I fix the TensorFlow/Keras protobuf `MessageFactory.GetPrototype` crash by ensuring the protobuf env vars are set before *any* TensorFlow/tf_keras import and by forcing the pure-Python protobuf runtime early in the notebook (this is the direct cause of your current runtime error in cell 24). I also avoid importing TensorFlow alongside `tf_keras` until after that environment setup, while keeping your exact model architecture, training loop, and feature extraction unchanged. Finally, I keep the submission generation identical (id + `EAP,HPL,MWS`) and ensure `submission.csv` is always written.'
- What this solution (achieved 0.87558) has done: 'I fix the protobuf/TensorFlow crash causing `MessageFactory.GetPrototype` by forcing the pure-Python protobuf implementation *before any TF/tf_keras import* and by avoiding the extra `import tensorflow as tf` (we can seed via `tf_keras.utils.set_random_seed`). This is the minimal change needed to make the notebook run end-to-end and write `submission.csv`. I also keep the model/feature logic unchanged, but make the softmax output numerically safe and row-normalized right before saving (this matches the metric’s rescaling rule and is effectively score-neutral). No architecture, training loop, or feature extraction changes are introduced.'
- What this solution (achieved 0.87558) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by setting the protobuf environment variables before any TensorFlow/tf_keras-related import and by importing `tensorflow` first (then `tf_keras`), which is the most reliable ordering in this Kaggle environment. I keep your exact model architecture, training loop, and feature extraction the same, only adjusting imports/initialization to ensure the notebook runs end-to-end. I also keep the submission generation identical (id + `EAP,HPL,MWS`) and ensure predictions are safely clipped and row-normalized before saving `submission.csv`. No score-tuning changes are introduced beyond making the pipeline execute correctly and deterministically.'
- What this solution (achieved 0.87569) has done: 'I fix the `MessageFactory.GetPrototype` crash by setting protobuf environment variables before any TensorFlow import and by avoiding the direct `import tensorflow as tf` in the failing cell, using `tf_keras` only (same model/fit logic). I also make the train/dev split stratified to better match class balance (same data and model, just a safer split), which should improve logloss toward your target without changing the core approach. Finally, I keep the submission format identical and ensure probabilities are clipped and row-normalized before writing `submission.csv`.'
- What this solution (achieved 0.87558) has done: 'I fix the runtime crash in the model cell by forcing the pure-Python protobuf implementation *before* any TensorFlow/tf_keras import and by importing `tensorflow` first, then `tf_keras`, which avoids the `MessageFactory.GetPrototype` failure in this environment. I keep your exact model architecture, training loops, and feature extraction unchanged. I also add a small safety check to ensure predictions always have the expected `(n_test, 3)` shape before writing the submission, and keep the required `id,EAP,HPL,MWS` column order with clipping + row-normalization (score-neutral but prevents metric edge cases). No other score-changing modifications are introduced.'
- What this solution (achieved 0.87558) has done: 'I fix the runtime crash in the model-building cell caused by an incompatibility between TensorFlow/protobuf and the current import order by ensuring the protobuf environment variables are set before any TF import and by using `tf_keras` without importing `tensorflow` separately. This is the smallest change that unblocks training/inference while preserving your exact network architecture, training loops, and feature extraction. I also keep the deterministic seeding intact, and leave the probability clipping + row-normalization (score-neutral and aligned with the metric’s rescaling behavior). The notebook then run end-to-end and always write a valid `submission.csv` with `id,EAP,HPL,MWS`.'
- What this solution (achieved 0.87558) has done: 'I fix the protobuf/TensorFlow incompatibility that triggers `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` by forcing the pure-Python protobuf runtime and disabling the C++ implementation *before any TensorFlow/tf_keras import*, and by importing TensorFlow once in a safe order. I keep your model architecture, training loop, and Word2Vec feature pipeline unchanged, only adjusting the environment setup/import sequence to make the notebook run end-to-end reliably. I also ensure the submission is always written as `submission.csv` with the required columns `id,EAP,HPL,MWS` and numerically safe, row-normalized probabilities (score-neutral with respect to the competition’s rescaling). No other score-tuning changes are introduced.'
- What this solution (achieved 0.87539) has done: 'I fix the TensorFlow/protobuf crash (`MessageFactory.GetPrototype`) by setting the protobuf env vars before any TF import and avoiding the extra `import tensorflow as tf` altogether, using `tf_keras` only (this keeps the same model architecture and training loop). I also make the TF execution deterministic and CPU-safe by forcing CPU visibility (prevents GPU/driver-related TF init paths that can re-trigger protobuf issues in some Kaggle images). Finally, I keep your feature pipeline and submission generation identical, ensuring `submission.csv` is always written with `id,EAP,HPL,MWS` and numerically safe probabilities. This is score-neutral except that it allows the notebook to actually train/infer end-to-end.'
- What this solution (achieved 0.87539) has done: 'I fix the `MessageFactory.GetPrototype` crash by setting the protobuf environment variables *before* any TensorFlow/tf_keras import and by importing TensorFlow once in a safe order, then using `tf_keras` (this is the minimal change that unblocks model creation/training). I keep your exact feature pipeline (Word2Vec averaging) and the exact dense network architecture/training loops unchanged. I also keep the submission generation identical but add a small defensive import-time setup to prevent the same crash from reappearing in later cells. No score-tuning changes are introduced beyond making the notebook run end-to-end reliably and produce `submission.csv` in the required format.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PYTHONHASHSEED", "123")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re
import gensim
from gensim.models import Word2Vec

np.random.seed(123)
random.seed(123)




## === cell 1
def _resolve_input_path(filename: str) -> str:
    candidates = [
        f"/kaggle/input/{filename}",
        f"/kaggle/input/spooky-author-identification/{filename}",
        f"/kaggle/data/{filename}",
        f"/kaggle/data/spooky-author-identification/{filename}",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Could not find {filename} in expected Kaggle locations: {candidates}"
    )


train_path = _resolve_input_path("train.csv")
test_path = _resolve_input_path("test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
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
    data.append(train_df["text"][i])
for j in range(len(test_df)):
    data.append(test_df["text"][j])



## === cell 6
print(len(data))



## === cell 7
embedding = gensim.models.Word2Vec(
    sentences=data,
    vector_size=50,
    window=10,
    min_count=1,
    sg=0,
    workers=1,  # deterministic-ish / stable in constrained env
    seed=123,
)



## === cell 8
print(embedding)



## === cell 9
words = list(embedding.wv.index_to_key)
print(len(words))



## === cell 10
if "capered" in embedding.wv:
    print(embedding.wv["capered"])
else:
    print("Token 'capered' not in vocabulary")



## === cell 11
if "dark" in embedding.wv:
    print(embedding.wv.most_similar("dark", topn=5))
else:
    print("Token 'dark' not in vocabulary")



## === cell 12
if "shocked" in embedding.wv:
    print(embedding.wv.most_similar("shocked", topn=5))
else:
    print("Token 'shocked' not in vocabulary")



## === cell 13
if "sprang" in embedding.wv:
    print(embedding.wv.most_similar("sprang", topn=5))
else:
    print("Token 'sprang' not in vocabulary")



## === cell 14
if "pride" in embedding.wv:
    print(embedding.wv.most_similar("pride", topn=5))
else:
    print("Token 'pride' not in vocabulary")



## === cell 15
train_df["author"] = pd.Categorical(train_df["author"])

df_Dummies = pd.get_dummies(train_df["author"], prefix="author")
for col in ["author_EAP", "author_HPL", "author_MWS"]:
    if col not in df_Dummies.columns:
        df_Dummies[col] = 0
df_Dummies = df_Dummies[["author_EAP", "author_HPL", "author_MWS"]]

train_df = pd.concat([train_df, df_Dummies], axis=1)
train_df.head()



## === cell 16
X = train_df["text"].str[:50]
Y = train_df[["author_EAP", "author_HPL", "author_MWS"]].values
print(X.shape, X[0], Y.shape, Y[0])



## === cell 17
X_test = test_df["text"].str[:50]
print(X_test.shape, X_test[0])




## === cell 18
def text_to_avg(text):
    """
    Given a list of words, average the Word2Vec vectors into a single vector.
    """
    avg = np.zeros((50,), dtype=np.float32)
    if text is None or len(text) == 0:
        return avg

    count = 0
    for w in text:
        if w in embedding.wv:
            avg += embedding.wv[w]
            count += 1
    if count == 0:
        return avg
    avg = avg / float(count)
    return avg




## === cell 19
X_avg = np.zeros((X.shape[0], 50), dtype=np.float32)  # initialize X_avg
for i in range(X.shape[0]):
    X_avg[i] = text_to_avg(X[i])



## === cell 20
print(X_avg.shape)
print(X_avg[0])



## === cell 21
X_test_avg = np.zeros((X_test.shape[0], 50), dtype=np.float32)  # initialize X_test_avg
for i in range(X_test.shape[0]):
    X_test_avg[i] = text_to_avg(X_test[i])



## === cell 22
print(X_test_avg.shape)
print(X_test_avg[0])



## === cell 23
from sklearn.model_selection import train_test_split

y_strat = train_df["author"].astype(str).values
X_train, X_dev, Y_train, Y_dev = train_test_split(
    X_avg, Y, test_size=0.2, random_state=123, stratify=y_strat
)
print(X_train.shape, Y_train.shape, X_dev.shape, Y_dev.shape)



## === cell 24
import tensorflow as tf  # noqa: F401
import tf_keras as keras

keras.utils.set_random_seed(123)

models = keras.models
layers = keras.layers

model = models.Sequential()
model.add(layers.Dense(64, activation="relu", input_shape=(50,)))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(3, activation="softmax"))

model.summary()



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 25
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])



## === cell 26
epochs = 50
history = model.fit(
    X_train,
    Y_train,
    epochs=epochs,
    batch_size=128,
    validation_data=(X_dev, Y_dev),
    verbose=2,
)



## === cell 27
loss = history.history["loss"]
dev_loss = history.history["val_loss"]
epochs_range = range(1, len(loss) + 1)
plt.plot(epochs_range, loss, "bo", label="training loss")
plt.plot(epochs_range, dev_loss, "b", label="validation loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()



## === cell 28
model = models.Sequential()
model.add(layers.Dense(64, activation="relu", input_shape=(50,)))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(3, activation="softmax"))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])



## === cell 29
epochs = 10
model.fit(X_avg, Y, epochs=epochs, batch_size=128, verbose=2)



## === cell 30
preds = model.predict(X_test_avg, verbose=0)
print(preds.shape)
print(preds[7])



## === cell 31
pred_labels = []
for i in range(len(X_test_avg)):
    pred_label = int(np.argmax(preds[i]))
    pred_labels.append(pred_label)



## === cell 32
print(pred_labels[7])



## === cell 33
preds = np.asarray(preds)
if preds.ndim != 2 or preds.shape[1] != 3:
    raise ValueError(f"Unexpected preds shape: {preds.shape}, expected (n_test, 3)")

preds = np.clip(preds, 1e-15, 1.0 - 1e-15)
preds = preds / preds.sum(axis=1, keepdims=True)

result = pd.DataFrame(preds, columns=["EAP", "HPL", "MWS"])
result.insert(0, "id", test_df["id"].values)
result.head()



## === cell 34
result.to_csv("submission.csv", index=False, float_format="%.20f")
print("Wrote submission.csv with shape:", result.shape)
print(result.head())
