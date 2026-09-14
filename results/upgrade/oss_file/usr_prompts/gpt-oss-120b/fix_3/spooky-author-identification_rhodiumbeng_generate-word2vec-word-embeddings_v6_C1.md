# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.5325

# 6. Current score

0.89312

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 0.89312) has done: 'I fix the logistic regression target handling (use 1‑dim integer labels instead of one‑hot), keep the author order for column names, and ensure the pipeline runs through to write a proper `submission.csv`. The changes are minimal, preserving the original preprocessing and embedding logic while correcting the model fitting and submission creation.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss



## === cell 1
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print(train_df.shape, test_df.shape)




## === cell 2
def clean_text(text):
    """lower‑case and remove punctuation/underscores"""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"_", "", text)
    return text




## === cell 3
train_df["text"] = (
    train_df["text"].map(lambda x: clean_text(x)).map(lambda x: x.strip().split())
)
train_df.head()



## === cell 4
test_df["text"] = (
    test_df["text"].map(lambda x: clean_text(x)).map(lambda x: x.strip().split())
)
test_df.head()



## === cell 5
data = list(train_df["text"]) + list(test_df["text"])
print(f"Total sentences for embedding: {len(data)}")



## === cell 6
embedding = Word2Vec(
    sentences=data, vector_size=50, window=10, min_count=1, sg=0  # CBOW
)
print(f"Embedding vocab size: {len(embedding.wv)}")



## === cell 7
train_df["author"] = pd.Categorical(train_df["author"])
author_categories = train_df[
    "author"
].cat.categories.tolist()  # e.g., ['EAP','HPL','MWS']
y = train_df["author"].cat.codes.values  # 1‑d label array



## === cell 8
X = train_df["text"].values
Y = y
print(f"X shape: {X.shape}, Y shape: {Y.shape}")



## === cell 9
X_test = test_df["text"].values
print(f"X_test shape: {X_test.shape}")




## === cell 10
def text_to_avg(words):
    """Average Word2Vec vectors for a list of tokens."""
    avg = np.zeros(embedding.vector_size)
    for w in words:
        avg += embedding.wv[w]
    if len(words) > 0:
        avg /= len(words)
    return avg




## === cell 11
X_avg = np.vstack([text_to_avg(txt) for txt in X])
print(f"X_avg shape: {X_avg.shape}")



## === cell 12
X_test_avg = np.vstack([text_to_avg(txt) for txt in X_test])
print(f"X_test_avg shape: {X_test_avg.shape}")



## === cell 13
X_train, X_dev, Y_train, Y_dev = train_test_split(
    X_avg, Y, test_size=0.2, random_state=123, stratify=Y
)
print(f"Train/Dev shapes: {X_train.shape}/{X_dev.shape}")



## === cell 14
clf = LogisticRegression(
    multi_class="multinomial", solver="lbfgs", C=4.0, max_iter=300, n_jobs=-1
)
clf.fit(X_train, Y_train)



## === cell 15
dev_pred = clf.predict_proba(X_dev)
dev_logloss = log_loss(Y_dev, dev_pred)
print(f"Dev LogLoss: {dev_logloss:.5f}")



## === cell 16
test_preds = clf.predict_proba(X_test_avg)
print(f"Test predictions shape: {test_preds.shape}")



## === cell 17
result = pd.DataFrame(test_preds, columns=author_categories)
result.insert(0, "id", test_df["id"])
result.head()



## === cell 18
result.to_csv("submission.csv", index=False, float_format="%.20f")
print("Submission file written to submission.csv")
