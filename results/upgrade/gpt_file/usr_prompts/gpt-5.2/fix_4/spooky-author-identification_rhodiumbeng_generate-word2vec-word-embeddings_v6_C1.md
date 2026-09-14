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

0.5325

# 6. Current score

0.62284

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.62529) has done: 'I fix the gensim Word2Vec API breakage (gensim 4 uses `vector_size` instead of `size`, and `wv.key_to_index` instead of `wv.vocab`) so the embedding model is actually created and the downstream averaging code runs. I also fix the Keras import/runtime issue by using `tf_keras` (available in your environment) consistently, which avoids the protobuf-related `MessageFactory` error you hit with `keras==3.x`. To move the logloss score toward your target (0.5325 from 1.0847; lower is better), I make a minimal, metric-aligned change to the final training: switch the final optimizer back to `adam` (matching the tuned/validated compile) and train for the same 30 epochs you already validated, without altering the architecture or feature extraction. The script then write a valid `submission.csv` with `id,EAP,HPL,MWS`.'
- What this solution (achieved 0.61754) has done: 'I fix the runtime crash in the `tf_keras` import/model creation by forcing a compatible protobuf Python implementation before any TensorFlow/tf_keras code is loaded, which resolves the `MessageFactory.GetPrototype` attribute error seen in this environment. I also keep the model/training logic identical, but ensure class/probability column order matches the required `EAP,HPL,MWS` header regardless of how pandas orders dummy columns, avoiding subtle label/probability mismatches that can hurt logloss. Paths and feature extraction remain unchanged, and the script still write a valid `submission.csv` with the correct columns and shape.'
- What this solution (achieved 0.62284) has done: 'I fix the runtime crash in model creation by forcing a protobuf version compatible with `tf_keras` before any TensorFlow-related import happens, and by importing `tf_keras` only after that environment setup. I also make the train/dev split stratified (same model/feature pipeline) to reduce variance and typically improve logloss a bit, moving your 0.61754 toward the 0.5325 target without changing the architecture or training regimen. Finally, I keep the required probability column order (`EAP,HPL,MWS`) and ensure the submission file is written as a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

try:
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass

import re
import gensim
from gensim.models import Word2Vec

np.random.seed(123)



## === cell 1
train_df = pd.read_csv("/kaggle/input/train.csv")
test_df = pd.read_csv("/kaggle/input/test.csv")
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
    sentences=data, vector_size=50, window=10, min_count=1, sg=0, workers=1, seed=123
)



## === cell 8
print(embedding)



## === cell 9
embedding.train(data, total_examples=len(data), epochs=30)



## === cell 10
words = list(embedding.wv.key_to_index)
print(len(words))



## === cell 11
print(embedding.wv["capered"])



## === cell 12
embedding.wv.most_similar("dark", topn=5)



## === cell 13
embedding.wv.most_similar("shocked", topn=5)



## === cell 14
embedding.wv.most_similar("sprang", topn=5)



## === cell 15
embedding.wv.most_similar("pride", topn=5)



## === cell 16
train_df["author"] = pd.Categorical(
    train_df["author"], categories=["EAP", "HPL", "MWS"]
)
df_Dummies = pd.get_dummies(train_df["author"], prefix="author")
df_Dummies = df_Dummies.reindex(
    columns=["author_EAP", "author_HPL", "author_MWS"], fill_value=0
)
train_df = pd.concat([train_df, df_Dummies], axis=1)
train_df.head()



## === cell 17
X = train_df["text"]
Y = train_df[["author_EAP", "author_HPL", "author_MWS"]].values
print(X.shape, X[0], Y.shape, Y[0])



## === cell 18
X_test = test_df["text"]
print(X_test.shape, X_test[0])




## === cell 19
def text_to_avg(text):
    """Given a list of words, extract Word2Vec representations and average them."""
    avg = np.zeros((50,), dtype=np.float32)
    if len(text) == 0:
        return avg
    for w in text:
        if w in embedding.wv:
            avg += embedding.wv[w]
    avg = avg / max(len(text), 1)
    return avg




## === cell 20
X_avg = np.zeros((X.shape[0], 50), dtype=np.float32)  # initialize X_avg
for i in range(X.shape[0]):
    X_avg[i] = text_to_avg(X[i])



## === cell 21
print(X_avg.shape)
print(X_avg[0])



## === cell 22
X_test_avg = np.zeros((X_test.shape[0], 50), dtype=np.float32)  # initialize X_test_avg
for i in range(X_test.shape[0]):
    X_test_avg[i] = text_to_avg(X_test[i])



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
from tf_keras import models, layers

model = models.Sequential()
model.add(layers.Dense(50, activation="relu", input_shape=(50,)))
model.add(layers.Dense(3, activation="softmax"))

model.summary()



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
plt.show()



## === cell 29
model = models.Sequential()
model.add(layers.Dense(50, activation="relu", input_shape=(50,)))
model.add(layers.Dense(3, activation="softmax"))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])



## === cell 30
model.fit(X_avg, Y, epochs=30, batch_size=128, verbose=2)



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
result.insert(0, "id", test_df["id"])
result.head()



## === cell 35
result.to_csv("submission.csv", index=False, float_format="%.20f")
print("Wrote submission.csv with shape:", result.shape)
print(result.head())
