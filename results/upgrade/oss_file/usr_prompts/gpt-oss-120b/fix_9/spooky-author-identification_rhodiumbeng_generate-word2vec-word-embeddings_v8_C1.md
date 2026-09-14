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
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 2 other files
        input/
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 2 other files
        working/
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 2 other files
```

-> data/spooky-author-identification/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> data/spooky-author-identification/test.csv has 1958 rows and 2 columns.
The columns are: id, text

-> data/spooky-author-identification/train.csv has 17621 rows and 3 columns.
The columns are: id, text, author

-> input/spooky-author-identification/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> input/spooky-author-identification/test.csv has 1958 rows and 2 columns.
The columns are: id, text

-> input/spooky-author-identification/train.csv has 17621 rows and 3 columns.
The columns are: id, text, author

-> working/spooky-author-identification/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> (stopped after 10 files for performance)

# 5. Target score

0.5357

# 6. Current score

3.09857

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.87389) has done: 'I remove the invalid `%matplotlib inline` line, update the Word2Vec constructor to use the correct `vector_size` argument, and switch the Keras imports to TensorFlow’s Keras to avoid the protobuf error. These fixes let the pipeline run, produce embeddings, train the neural network, and finally write a proper `submission.csv` file.'
- What this solution (achieved 0.84613) has done: 'I fixed the import that caused the protobuf error, corrected the data paths to the actual Kaggle input location, added safety handling for missing words in the embeddings, and ensured the script creates a proper `submission.csv` in the working directory. These changes let the notebook run end‑to‑end, generate predictions, and write a valid submission file while preserving the original model architecture and training logic.'
- What this solution (achieved 0.86008) has done: 'I fixed the protobuf import error by switching to the tf_keras module, corrected the stratified split to use a 1‑dim label array, and increased training epochs to give the model more capacity to learn, which should lower the log‑loss toward the target. The rest of the pipeline is unchanged and the script now writes a proper submission.csv file.'
- What this solution (achieved 2.69837) has done: 'I replace the problematic `tf_keras` import with the regular `keras` package, enlarge the Word2Vec embeddings and training epochs, and modestly increase the neural‑network capacity and training epochs. These changes fix the import error, give the model richer representations, and should lower the log‑loss toward the target while preserving the overall pipeline.'
- What this solution (achieved 2.8623) has done: 'I fixed the import error that prevented the notebook from running by replacing the standalone keras imports with TensorFlow’s tf.keras modules, which are compatible with the installed protobuf version. This change restores the full training and prediction pipeline so the model can be trained properly and a valid submission.csv file is written.'
- What this solution (achieved 3.09857) has done: 'The fix replaces the TensorFlow import that caused the protobuf failure with the compatible tf_keras package, and enlarges the Word2Vec embeddings and training epochs (both for Word2Vec and the neural network) to give the model more capacity to learn, which should lower the log‑loss toward the target while keeping the original architecture.'

# 9. Code solution

## === cell 0
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec

from tf_keras import models, layers
from sklearn.model_selection import train_test_split



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
base_path = "/kaggle/input/spooky-author-identification"
train_path = os.path.join(base_path, "train.csv")
test_path = os.path.join(base_path, "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
print("train shape:", train_df.shape, "test shape:", test_df.shape)




## === cell 2
def clean_text(text):
    """lower‑case and remove punctuation / underscores."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # keep words and spaces
    text = re.sub(r"_", "", text)  # remove underscores
    return text




## === cell 3
train_df["text"] = train_df["text"].apply(lambda x: clean_text(x))
train_df["text"] = train_df["text"].apply(lambda x: x.strip().split())

test_df["text"] = test_df["text"].apply(lambda x: clean_text(x))
test_df["text"] = test_df["text"].apply(lambda x: x.strip().split())



## === cell 4
data = []
for tokens in train_df["text"]:
    data.append(tokens)
for tokens in test_df["text"]:
    data.append(tokens)



## === cell 5
embedding = Word2Vec(
    sentences=data,
    vector_size=200,  # larger embedding dimension for richer representations
    window=10,
    min_count=1,
    sg=0,  # CBOW
    workers=4,
    epochs=50,  # more training passes for better word vectors
)
print(f"Vocabulary size: {len(embedding.wv)}")



## === cell 6
train_df["author"] = pd.Categorical(train_df["author"])
author_dummies = pd.get_dummies(train_df["author"], prefix="author")
train_df = pd.concat([train_df, author_dummies], axis=1)



## === cell 7
X = train_df["text"].apply(lambda lst: lst[:50])
Y = train_df[["author_EAP", "author_HPL", "author_MWS"]].values
print("X shape:", X.shape, "Y shape:", Y.shape)



## === cell 8
X_test = test_df["text"].apply(lambda lst: lst[:50])
print("X_test shape:", X_test.shape)




## === cell 9
def text_to_avg(tokens):
    """Average Word2Vec vectors for a list of tokens."""
    if not tokens:
        return np.zeros(200, dtype=np.float32)  # match new dim
    vec_sum = np.zeros(200, dtype=np.float32)
    count = 0
    for w in tokens:
        if w in embedding.wv:
            vec_sum += embedding.wv[w]
            count += 1
    if count == 0:  # none of the tokens were in vocab
        return np.zeros(200, dtype=np.float32)
    return vec_sum / count




## === cell 10
X_avg = np.zeros((X.shape[0], 200), dtype=np.float32)
for i, tokens in enumerate(X):
    X_avg[i] = text_to_avg(tokens)
print("X_avg shape:", X_avg.shape)



## === cell 11
X_test_avg = np.zeros((X_test.shape[0], 200), dtype=np.float32)
for i, tokens in enumerate(X_test):
    X_test_avg[i] = text_to_avg(tokens)
print("X_test_avg shape:", X_test_avg.shape)



## === cell 12
X_train, X_dev, Y_train, Y_dev = train_test_split(
    X_avg, Y, test_size=0.2, random_state=123, stratify=Y.argmax(axis=1)
)
print("Train/Dev shapes:", X_train.shape, Y_train.shape, X_dev.shape, Y_dev.shape)



## === cell 13
model = models.Sequential(
    [
        layers.Dense(128, activation="relu", input_shape=(200,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(3, activation="softmax"),
    ]
)
model.summary()



## === cell 14
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])



## === cell 15
epochs = 300  # increased epochs for better learning
history = model.fit(
    X_train,
    Y_train,
    epochs=epochs,
    batch_size=128,
    validation_data=(X_dev, Y_dev),
    verbose=0,
)



## === cell 16
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs_range = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs_range, loss, "bo", label="training loss")
plt.plot(epochs_range, val_loss, "b", label="validation loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()



## === cell 17
preds = model.predict(X_test_avg, verbose=0)
print("Preds shape:", preds.shape)
print("Sample prediction:", preds[7])



## === cell 18
result = pd.DataFrame(preds, columns=["EAP", "HPL", "MWS"])
result.insert(0, "id", test_df["id"])
result.head()



## === cell 19
submission_path = "/kaggle/working/submission.csv"
result.to_csv(submission_path, index=False, float_format="%.20f")
print(f"Submission written to {submission_path}")
