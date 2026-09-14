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
Predict the sentiment of phrases.

## Metric
Classification accuracy.

## Submission Format
For each phrase in the test set, predict a label for the sentiment. Your submission should have a header and look like the following:

```
PhraseId,Sentiment
156061,2
156062,2
156063,2
...
```

## Dataset
The dataset is comprised of tab-separated files with phrases. Each phrase has a PhraseId. Each sentence has a SentenceId.

The sentiment labels are:

0 - negative

1 - somewhat negative

2 - neutral

3 - somewhat positive

4 - positive

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (72 lines)
            sampleSubmission.csv (46819 lines)
            sampleSubmission.csv.zip (146.0 kB)
            test.tsv (46819 lines)
            test.tsv.zip (1.1 MB)
            train.tsv (109243 lines)
            train.tsv.zip (2.7 MB)
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
        input/
            description.md (72 lines)
            sampleSubmission.csv (46819 lines)
            sampleSubmission.csv.zip (146.0 kB)
            test.tsv (46819 lines)
            test.tsv.zip (1.1 MB)
            train.tsv (109243 lines)
            train.tsv.zip (2.7 MB)
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
        working/
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
```

-> data/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv has 46818 rows and 2 columns.
Here is some information about the columns:
PhraseId (int64) has range: 29.00 - 156030.00, 0 nan values
Sentiment (int64) has 1 unique values: [2]

-> data/sampleSubmission.csv has 46818 rows and 2 columns.
Here is some information about the columns:
PhraseId (int64) has range: 29.00 - 156030.00, 0 nan values
Sentiment (int64) has 1 unique values: [2]

-> (stopped after 10 files for performance)

# 5. Target score

0.6511796295178905

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.0588) has done: 'The changes focus on eliminating unnecessary Python‑level overhead during training and freeing memory after loading the large GloVe file.  After building the embedding matrix we delete the original dictionary and run garbage collection.  Training now uses a `tf.data.Dataset` pipeline, which streams batches efficiently and matches the original shuffling behavior, keeping the model architecture, optimizer, epochs and other hyper‑parameters unchanged.  These tweaks speed up the fit step enough to stay under the 600‑second limit while preserving exact results.'

# 9. Code solution

## === cell 0
import os, re, gc, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", None)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


## === cell 1
DATA_DIR = "/kaggle/input/movie-review-sentiment-analysis-kernels-only"
train_file = os.path.join(DATA_DIR, "train.tsv")
test_file = os.path.join(DATA_DIR, "test.tsv")
df_train = pd.read_csv(train_file, sep="\t")
df_test = pd.read_csv(test_file, sep="\t")
sub = pd.read_csv(os.path.join(DATA_DIR, "sampleSubmission.csv"))


## === cell 2
print(df_train.head())
print(df_test.head())


## === cell 3
df_train["Phrase"] = df_train["Phrase"].str.replace("n't", "not")
df_test["Phrase"] = df_test["Phrase"].str.replace("n't", "not")


## === cell 4
df_train["Phrase"] = df_train["Phrase"].apply(lambda x: re.sub(r"[0-9]+", "0", x))
df_test["Phrase"] = df_test["Phrase"].apply(lambda x: re.sub(r"[0-9]+", "0", x))


## === cell 5
seed = 101
np.random.seed(seed)

X = df_train["Phrase"].astype(str)
X_test_raw = df_test["Phrase"].astype(str)  # keep for final prediction
y = df_train["Sentiment"].values
num_classes = df_train["Sentiment"].nunique()
print("Number of classes:", num_classes)


## === cell 6
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=seed
)
print("Split sizes:", X_train.shape, X_val.shape, y_train.shape, y_val.shape)


## === cell 7
max_features = 15000
vectorizer = TfidfVectorizer(max_features=max_features, oov_token="<OOV>")
vectorizer.fit(X_train)

X_train_vec = vectorizer.transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test_raw)
print("Vector shapes:", X_train_vec.shape, X_val_vec.shape, X_test_vec.shape)


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3589811662.py in <cell line: 0>()
      1 max_features = 15000
----> 2 vectorizer = TfidfVectorizer(max_features=max_features, oov_token="<OOV>")
      3 vectorizer.fit(X_train)
      4 
      5 X_train_vec = vectorizer.transform(X_train)

TypeError: TfidfVectorizer.__init__() got an unexpected keyword argument 'oov_token'

## === cell 8
model = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=200,
    n_jobs=-1,
    random_state=seed,
)
model.fit(X_train_vec, y_train)
val_preds = model.predict(X_val_vec)
val_acc = accuracy_score(y_val, val_preds)
print(f"Validation accuracy: {val_acc:.4f}")


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1252739021.py in <cell line: 0>()
      6     random_state=seed,
      7 )
----> 8 model.fit(X_train_vec, y_train)
      9 val_preds = model.predict(X_val_vec)
     10 val_acc = accuracy_score(y_val, val_preds)

NameError: name 'X_train_vec' is not defined

## === cell 9
test_pred = model.predict(X_test_vec)
sub["Sentiment"] = test_pred
submission_path = "submission.csv"
sub.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print(sub.head())

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/645115892.py in <cell line: 0>()
----> 1 test_pred = model.predict(X_test_vec)
      2 sub["Sentiment"] = test_pred
      3 submission_path = "submission.csv"
      4 sub.to_csv(submission_path, index=False)
      5 print(f"Submission saved to {submission_path}")

NameError: name 'X_test_vec' is not defined
