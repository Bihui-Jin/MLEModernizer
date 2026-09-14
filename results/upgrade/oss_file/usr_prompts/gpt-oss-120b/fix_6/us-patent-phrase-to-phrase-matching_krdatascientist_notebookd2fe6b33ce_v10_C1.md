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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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

0.1399

# 6. Current score

0.49986

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.49986) has done: 'The optimization replaces the dense Keras Tokenizer TF‑IDF matrix with a sparse TfidfVectorizer, eliminating huge dense matrices and dramatically reducing memory and computation.  The least‑squares solution is obtained with SciPy’s sparse lsqr, which works directly on the sparse matrix plus a bias column.  All other logic—including data preparation, evaluation, and CSV output—remains unchanged, preserving identical predictions up to negligible floating‑point differences.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
df_sub = pd.read_csv(
    "/kaggle/input/us-patent-phrase-to-phrase-matching/sample_submission.csv"
)
df_train = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/train.csv")
df_test = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv")



## === cell 2
df_train.head(3)



## === cell 3
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
from scipy import sparse
from scipy.sparse.linalg import lsqr

warnings.filterwarnings("ignore")



## === cell 4
input_text = (
    df_train.id + " " + df_train.anchor + " " + df_train.context + " " + df_train.target
).str.lower()



## === cell 5
token = TfidfVectorizer(lowercase=True)



## === cell 6
token.fit(input_text)



## === cell 7
vocab_size = len(token.vocabulary_) + 1  # includes index 0
token.num_words = vocab_size  # mimic original attribute



## === cell 8
X_train = token.transform(input_text).astype(np.float32)
y_train = df_train["score"].values



## === cell 9
print("Training matrix shape:", X_train.shape)



## === cell 10
max_length = None



## === cell 11
bias_column = np.ones((X_train.shape[0], 1), dtype=np.float32)
X_train_bias = sparse.hstack([X_train, bias_column])
coef = lsqr(X_train_bias, y_train)[0]  # solution vector

train_pred = X_train_bias @ coef
print("Model trained on", X_train.shape[0], "samples.")



## === cell 12
df_train.head(3)



## === cell 13
train_corr = np.corrcoef(train_pred, y_train)[0, 1]
print("Pearson correlation on training data:", train_corr)



## === cell 14
test_text = (
    df_test.id + " " + df_test.anchor + " " + df_test.context + " " + df_test.target
).str.lower()
X_test = token.transform(test_text).astype(np.float32)

bias_column_test = np.ones((X_test.shape[0], 1), dtype=np.float32)
X_test_bias = sparse.hstack([X_test, bias_column_test])
test_pred = X_test_bias @ coef



## === cell 15
np.corrcoef(train_pred, y_train)



## === cell 16
df_test["score"] = test_pred.ravel()



## === cell 17
df_test[["id", "score"]].to_csv("submission.csv", index=False)
