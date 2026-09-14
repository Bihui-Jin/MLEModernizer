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
nltk==3.9.2
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

0.58249

# 6. Current score

0.64951

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 0.64951) has done: 'I fix the Word2Vec API breakage from gensim 4 by replacing the deprecated `size` argument with `vector_size`, and I update vector lookup to use `embedding.wv[...]` so embeddings are actually accessible. I also remove notebook-only magic (`%matplotlib inline`) and switch Keras imports to `tf_keras` to avoid the protobuf-related `MessageFactory` crash while keeping the same model architecture and training loop. To improve logloss toward your target, I correct the average-embedding extraction to be robust to OOV/empty texts and use a stratified split for validation (score-improving but still the same approach). Finally, the script reliably write a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os

print(os.listdir("../input"))



## === cell 1
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re
import gensim
from gensim.models import Word2Vec

SEED = 123
np.random.seed(SEED)



## === cell 2
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print(train_df.shape, test_df.shape)




## === cell 3
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\_", "", text)  # remove underscore
    return text




## === cell 4
train_df["text"] = train_df["text"].map(lambda x: clean_text(x))
train_df["text"] = train_df["text"].map(lambda x: x.strip().split())
train_df.head()



## === cell 5
test_df["text"] = test_df["text"].map(lambda x: clean_text(x))
test_df["text"] = test_df["text"].map(lambda x: x.strip().split())
test_df.head()



## === cell 6
data = []
for i in range(len(train_df)):
    data.append(train_df["text"].iloc[i])
for j in range(len(test_df)):
    data.append(test_df["text"].iloc[j])



## === cell 7
print(len(data))



## === cell 8
embedding = Word2Vec(
    sentences=data, vector_size=50, window=5, min_count=1, workers=4, seed=SEED
)



## === cell 9
embedding.train(data, total_examples=len(data), epochs=30)



## === cell 10
train_df["author"] = pd.Categorical(train_df["author"])
df_Dummies = pd.get_dummies(train_df["author"], prefix="author")
train_df = pd.concat([train_df, df_Dummies], axis=1)
train_df.head()



## === cell 11
X = train_df["text"]
Y = train_df[["author_EAP", "author_HPL", "author_MWS"]].values



## === cell 12
print(X.shape, X.iloc[0])
print(Y.shape, Y[0])



## === cell 13
X_test = test_df["text"]
print(X_test.shape, X_test.iloc[0])




## === cell 14
def text_to_avg(text):
    """
    Average Word2Vec vectors for tokens in `text`.
    Robust to empty texts and (future) OOV tokens.
    """
    vec_size = embedding.vector_size
    avg = np.zeros((vec_size,), dtype=np.float32)

    if text is None or len(text) == 0:
        return avg

    n = 0
    for w in text:
        if w in embedding.wv:
            avg += embedding.wv[w]
            n += 1
    if n == 0:
        return avg
    return avg / n




## === cell 15
X_avg = np.zeros((X.shape[0], 50), dtype=np.float32)
for i in range(X.shape[0]):
    X_avg[i] = text_to_avg(X.iloc[i])



## === cell 16
print(X_avg.shape)
print(X_avg[0])



## === cell 17
X_test_avg = np.zeros((X_test.shape[0], 50), dtype=np.float32)
for i in range(X_test.shape[0]):
    X_test_avg[i] = text_to_avg(X_test.iloc[i])



## === cell 18
print(X_test_avg.shape)
print(X_test_avg[0])



## === cell 19
from sklearn.model_selection import train_test_split

y_class = np.argmax(Y, axis=1)
X_train, X_dev, Y_train, Y_dev = train_test_split(
    X_avg, Y, test_size=0.2, random_state=SEED, stratify=y_class
)
print(X_train.shape, Y_train.shape, X_dev.shape, Y_dev.shape)



## === cell 20
from tf_keras import models
from tf_keras import layers

model = models.Sequential()
model.add(layers.Dense(50, activation="relu", input_shape=(50,)))
model.add(layers.Dense(3, activation="softmax"))

model.summary()



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 21
model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)



## === cell 22
history = model.fit(
    X_train, Y_train, epochs=30, batch_size=None, validation_data=(X_dev, Y_dev)
)



## === cell 23
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



## === cell 24
model = models.Sequential()
model.add(layers.Dense(50, activation="relu", input_shape=(50,)))
model.add(layers.Dense(3, activation="softmax"))
model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)



## === cell 25
model.fit(X_avg, Y, epochs=10, batch_size=None)



## === cell 26
preds = model.predict(X_test_avg, verbose=0)
print(preds.shape)
print(preds[7])



## === cell 27
pred_labels = []
for i in range(len(X_test_avg)):
    pred_label = int(np.argmax(preds[i]))
    pred_labels.append(pred_label)



## === cell 28
print(pred_labels[7])



## === cell 29
result = pd.DataFrame(preds, columns=["EAP", "HPL", "MWS"])
result.insert(0, "id", test_df["id"].values)
result.head()



## === cell 30
result.to_csv("submission.csv", index=False, float_format="%.20f")
print("Wrote submission.csv with shape:", result.shape)
print(result.head())
