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

0.90907

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.91232) has done: 'I updated the script to match the current gensim API, fixed the Keras import to avoid the protobuf error, corrected the word‑vector lookup, and removed the redundant train call. These changes let the notebook run end‑to‑end, generate a proper `submission.csv`, and should improve the log‑loss toward the target score.'
- What this solution (achieved 0.86775) has done: 'I replace the TensorFlow‑based Keras imports with the standalone keras package to fix the protobuf‑related AttributeError, and I slightly improve the final model (add an extra hidden layer and use the Adam optimizer with more training epochs) to bring the log‑loss down toward the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.88015) has done: 'The fix replaces the problematic standalone keras import with TensorFlow’s keras (which avoids the protobuf `MessageFactory` error) and slightly enlarges the final neural network while training a few more epochs. These modest changes keep the original pipeline intact but improve model capacity, nudging the log‑loss toward the target score and ensuring a valid `submission.csv` is written.'
- What this solution (achieved 0.97378) has done: 'I replace the TensorFlow/Keras imports that cause a protobuf error with the standalone keras package, skip the unnecessary dev‑model training, and modestly enlarge the final network and train it a few more epochs to nudge the log‑loss toward the target while keeping the core pipeline unchanged.'
- What this solution (achieved 1.67663) has done: 'We replace the failing `keras` import with TensorFlow’s Keras to avoid the protobuf error, expand the Word2Vec vectors to 100 dimensions for richer text features, and enlarge the neural network (256‑128‑64 units) with a few more training epochs. Finally, we clip the predicted probabilities to the range required by the competition before writing the CSV, ensuring a valid submission file.'
- What this solution (achieved 0.90171) has done: 'I fix the protobuf import error by using the standalone keras package, increase the Word2Vec embedding size to 200 (and adjust all related dimensions), train the model on the proper train/dev split with early‑stopping, and update the model layers to match the larger input size. These changes resolve the runtime crash and should improve the log‑loss toward the target while keeping the overall pipeline intact.'
- What this solution (achieved 0.95409) has done: 'I replaced the failing standalone keras import with TensorFlow’s tf.keras to avoid the protobuf MessageFactory error, updated the EarlyStopping import accordingly, and added Dropout layers to the network to improve generalisation and lower log‑loss while keeping the overall pipeline unchanged. The script now runs end‑to‑end and writes a correctly‑formatted `submission.csv`.'
- What this solution (achieved 0.94981) has done: 'I replace the failing TensorFlow/Keras imports with the standalone keras package to eliminate the protobuf `MessageFactory` error. This only changes the import statements (cells 19 and 21) and leaves the model architecture, training loop, and all other logic untouched, ensuring the script runs end‑to‑end and writes a valid `submission.csv`.'
- What this solution (achieved 0.92316) has done: 'I replace the failing standalone keras import with tensorflow.keras to avoid the protobuf error, adjust the early‑stopping patience to allow longer training, and increase the maximum epochs so the model can converge better—these minimal changes fix the runtime crash and help lower the log‑loss toward the target while keeping the original pipeline intact.'
- What this solution (achieved 0.88776) has done: 'The fix replaces the TensorFlow‑Keras imports with the standalone keras package to avoid the protobuf `MessageFactory` error, and modestly enlarges the neural network (adding a larger first layer and reducing dropout) while extending the early‑stopping patience so the model can train longer. These changes resolve the runtime crash and improve the log‑loss toward the target without altering the overall pipeline.'
- What this solution (achieved 0.92927) has done: 'I replace the failing standalone keras imports with tf.keras to fix the protobuf error, add a simple text‑length feature to the averaged Word2Vec vectors (giving the model a bit more information), adjust the model’s input shape accordingly, and reduce dropout slightly for better learning. These changes keep the overall pipeline intact while fixing the runtime crash and should lower the log‑loss toward the target.'
- What this solution (achieved 1.60135) has done: 'I replace the failing TensorFlow‑Keras imports with the standalone keras package, add feature standardization, increase early‑stopping patience, and slightly reduce dropout to let the model train a bit longer and improve generalisation. These minimal changes fix the runtime error, ensure a valid `submission.csv`, and are expected to lower the log‑loss toward the target while preserving the overall pipeline.'
- What this solution (achieved 0.93169) has done: 'I replace the failing TensorFlow‑Keras imports with the TensorFlow tf.keras equivalents to fix the protobuf error, and swap the neural network for a multinomial Logistic Regression model (which keeps the same feature engineering but can achieve a lower log‑loss). The changes are minimal, preserve the overall pipeline, and ensure a valid `submission.csv` is written.'
- What this solution (achieved 0.90907) has done: 'Implemented fixes to resolve the protobuf import error by switching to the standalone keras package (with a safe fallback) and adjusted the Logistic Regression regularization strength for better calibration. Added clear comments describing each modification. The script now runs end‑to‑end, produces a properly formatted `submission.csv`, and the model tweaks are expected to move the log‑loss closer to the target.'

# 9. Code solution

## === cell 0
import os

print(os.listdir("../input"))



## === cell 1
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec



## === cell 2
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print(train_df.shape, test_df.shape)




## === cell 3
def clean_text(text):
    text = text.lower()
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
    data.append(train_df["text"][i])
for j in range(len(test_df)):
    data.append(test_df["text"][j])



## === cell 7
print(len(data))



## === cell 8
embedding = Word2Vec(
    sentences=data,
    vector_size=200,  # upgraded dimension
    window=5,
    min_count=1,
    workers=4,
    sg=0,  # CBOW (default)
)



## === cell 9
train_df["author"] = pd.Categorical(train_df["author"])
df_Dummies = pd.get_dummies(train_df["author"], prefix="author")
train_df = pd.concat([train_df, df_Dummies], axis=1)
train_df.head()



## === cell 10
X = train_df["text"]
Y = train_df[["author_EAP", "author_HPL", "author_MWS"]].values



## === cell 11
print(X.shape, X[0])
print(Y.shape, Y[0])



## === cell 12
X_test = test_df["text"]
print(X_test.shape, X_test[0])




## === cell 13
def text_to_avg(text):
    """Average Word2Vec vectors for a list of tokens."""
    if len(text) == 0:
        return np.zeros((200,))  # match new vector size
    avg = np.zeros((200,))
    for w in text:
        avg += embedding.wv[w]
    avg = avg / len(text)
    return avg




## === cell 14
X_avg = np.zeros((X.shape[0], 200))
X_len = np.zeros((X.shape[0], 1))
for i in range(X.shape[0]):
    X_avg[i] = text_to_avg(X[i])
    X_len[i] = len(X[i])
X_len = X_len / X_len.max()
X_feat = np.hstack([X_avg, X_len])
print("Train feature shape:", X_feat.shape)



## === cell 15
print(X_feat.shape)
print(X_feat[0])



## === cell 16
X_test_avg = np.zeros((X_test.shape[0], 200))
X_test_len = np.zeros((X_test.shape[0], 1))
for i in range(X_test.shape[0]):
    X_test_avg[i] = text_to_avg(X_test[i])
    X_test_len[i] = len(X_test[i])
X_test_len = X_test_len / X_test_len.max()
X_test_feat = np.hstack([X_test_avg, X_test_len])
print("Test feature shape:", X_test_feat.shape)



## === cell 17
print(X_test_feat.shape)
print(X_test_feat[0])



## === cell 18
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_feat = scaler.fit_transform(X_feat)
X_test_feat = scaler.transform(X_test_feat)

X_train, X_dev, Y_train, Y_dev = train_test_split(
    X_feat, Y, test_size=0.2, random_state=123
)
print(X_train.shape, Y_train.shape, X_dev.shape, Y_dev.shape)



## === cell 19
try:
    import keras
    from keras import models, layers
except Exception as e:
    models = layers = None
    print("Keras import failed, proceeding without it:", e)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 20
dev_history = None



## === cell 21
try:
    from keras.callbacks import EarlyStopping

    early_stop = EarlyStopping(
        monitor="val_loss", patience=50, restore_best_weights=True
    )
except Exception as e:
    early_stop = None
    print("EarlyStopping import failed, continuing without it:", e)



## === cell 22
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=2000,
    C=10.0,  # increased from 2.0
    n_jobs=-1,
)



## === cell 23
y_train_int = np.argmax(Y_train, axis=1)
y_dev_int = np.argmax(Y_dev, axis=1)

logreg.fit(X_train, y_train_int)



## === cell 24
preds = logreg.predict_proba(X_test_feat)  # shape (n_test, 3)
print(preds.shape)
print(preds[7])



## === cell 25
preds_clipped = np.clip(preds, 1e-15, 1 - 1e-15)

result = pd.DataFrame(preds_clipped, columns=["EAP", "HPL", "MWS"])
result.insert(0, "id", test_df["id"])
result.head()



## === cell 26
result.to_csv("submission.csv", index=False, float_format="%.20f")
