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

1.53812

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.70298) has done: 'The fix updates the Word2Vec initialization to the current gensim API, accesses word vectors via `embedding.wv`, replaces the outdated standalone Keras imports with `tensorflow.keras`, and adjusts the neural network architecture slightly (larger hidden layers) to improve predictive power while keeping the original workflow. Unused buggy cells are turned into comments, ensuring the notebook runs from start to finish and writes a proper `submission.csv` file with the required columns.'
- What this solution (achieved 0.64499) has done: 'The fix updates the imports to use the standalone keras package (avoiding the protobuf error), corrects the stratification argument, makes the embedding dimension a shared constant, adjusts the averaging function, and adds a dropout layer while increasing the vector size and training epochs to improve validation loss and move the log‑loss closer to the target. These changes keep the original workflow intact and ensure a proper `submission.csv` is written.'
- What this solution (achieved 1.06259) has done: 'I fix the import error by using TensorFlow’s Keras (`tensorflow.keras`) which is compatible with the installed packages, and I slightly increase model capacity (larger hidden layers) and training epochs to modestly improve the validation log‑loss, moving the score closer to the target while keeping the original workflow intact.'
- What this solution (achieved 1.53376) has done: 'I replace the TensorFlow‑Keras import with the standalone keras import to eliminate the protobuf error, reduce dropout slightly (0.2 → 0.3) to improve learning, and increase the training epochs to give the model more opportunity to converge. These changes keep the original architecture and workflow intact while fixing the runtime failure and nudging the validation log‑loss toward the target.'
- What this solution (achieved 1.53812) has done: 'The fix switches to the compatible `tensorflow.keras` import to resolve the protobuf error, adds a small extra dense layer to give the model a bit more capacity (helping lower log‑loss), and keeps the rest of the workflow unchanged. The script now runs end‑to‑end and writes a proper `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from gensim.models import Word2Vec

VECTOR_DIM = 100



## === cell 1
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print(train_df.shape, test_df.shape)




## === cell 2
def clean_text(text):
    """Convert all to lowercase and remove punctuations"""
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
embedding = Word2Vec(
    sentences=data,
    vector_size=VECTOR_DIM,
    window=10,
    min_count=1,
    sg=0,  # CBOW
    workers=4,
    seed=42,
)



## === cell 7
print(f"Vocabulary size: {len(embedding.wv)}")



## === cell 8
embedding.train(data, total_examples=len(data), epochs=30)



## === cell 9
train_df["author"] = pd.Categorical(train_df["author"])
df_Dummies = pd.get_dummies(train_df["author"], prefix="author")
train_df = pd.concat([train_df, df_Dummies], axis=1)
train_df.head()



## === cell 10
X = train_df["text"]
Y = train_df[["author_EAP", "author_HPL", "author_MWS"]].values
print(X.shape, X.iloc[0], Y.shape, Y[0])



## === cell 11
X_test = test_df["text"]
print(X_test.shape, X_test.iloc[0])




## === cell 12
def text_to_avg(text):
    """Average Word2Vec vectors for a list of words."""
    avg = np.zeros((VECTOR_DIM,))
    for w in text:
        if w in embedding.wv:
            avg += embedding.wv[w]
    return avg / max(len(text), 1)




## === cell 13
X_avg = np.zeros((X.shape[0], VECTOR_DIM))
for i in range(X.shape[0]):
    X_avg[i] = text_to_avg(X.iloc[i])



## === cell 14
print(X_avg.shape)
print(X_avg[0])



## === cell 15
X_test_avg = np.zeros((X_test.shape[0], VECTOR_DIM))
for i in range(X_test.shape[0]):
    X_test_avg[i] = text_to_avg(X_test.iloc[i])



## === cell 16
print(X_test_avg.shape)
print(X_test_avg[0])



## === cell 17
from sklearn.model_selection import train_test_split

X_train, X_dev, Y_train, Y_dev = train_test_split(
    X_avg,
    Y,
    test_size=0.2,
    random_state=123,
    stratify=train_df["author"],  # use categorical author for stratification
)
print(X_train.shape, Y_train.shape, X_dev.shape, Y_dev.shape)



## === cell 18
from tensorflow.keras import models, layers

model = models.Sequential()
model.add(layers.Dense(256, activation="relu", input_shape=(VECTOR_DIM,)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(3, activation="softmax"))
model.summary()



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 19
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])



## === cell 20
history = model.fit(
    X_train,
    Y_train,
    epochs=120,
    batch_size=128,
    validation_data=(X_dev, Y_dev),
    verbose=2,
)



## === cell 21
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



## === cell 22
preds = model.predict(X_test_avg, batch_size=128)
print(preds.shape)
print(preds[7])



## === cell 23
result = pd.DataFrame(preds, columns=["EAP", "HPL", "MWS"])
result.insert(0, "id", test_df["id"])
result.head()



## === cell 24
result.to_csv("submission.csv", index=False, float_format="%.20f")
