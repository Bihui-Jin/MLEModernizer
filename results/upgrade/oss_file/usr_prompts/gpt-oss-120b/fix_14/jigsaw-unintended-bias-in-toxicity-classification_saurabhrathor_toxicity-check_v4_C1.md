# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Build a model that recognizes toxicity and minimizes unintended bias with respect to mentions of identities.

## Metric
We combine several submetrics: An overall ROC-AUC for the full evaluation set, along with the ROC-AUCs on three specific subsets of the test set capturing different aspects of bias.

The final model score looks like:

$$
\text { score }=w_0 A U C_{\text {overall }}+\sum_{a=1}^A w_a M_p\left(m_{s, a}\right)
$$
where:
$A=$ number of submetrics $(3)$
$m_{s, a}=$ bias metric for identity subgroup $s$ using submetric $a$
$w_a=$ a weighting for the relative importance of each submetric; all four $w$ values set to 0.25

Overall AUC: This is the ROC-AUC for the full evaluation set.

### Bias AUCs
To measure unintended bias, we again calculate the ROC-AUC, this time on three specific subsets of the test set for each identity, each capturing a different aspect of unintended bias. 

**Subgroup AUC**: Here, we restrict the data set to only the examples that mention the specific identity subgroup. *A low value in this metric means the model does a poor job of distinguishing between toxic and non-toxic comments that mention the identity*.

**BPSN (Background Positive, Subgroup Negative) AUC**: Here, we restrict the test set to the non-toxic examples that mention the identity and the toxic examples that do not. *A low value in this metric means that the model confuses non-toxic examples that mention the identity with toxic examples that do not*, likely meaning that the model predicts higher toxicity scores than it should for non-toxic examples mentioning the identity.

**BNSP (Background Negative, Subgroup Positive) AUC**: Here, we restrict the test set to the toxic examples that mention the identity and the non-toxic examples that do not. *A low value here means that the model confuses toxic examples that mention the identity with non-toxic examples that do not*, likely meaning that the model predicts lower toxicity scores than it should for toxic examples mentioning the identity.

#### Generalized Mean of Bias AUCs
To combine the per-identity Bias AUCs into one overall measure, we calculate their generalized mean as defined below:

$$
M_p\left(m_s\right)=\left(\frac{1}{N} \sum_{s=1}^N m_s^p\right)^{\frac{1}{p}}
$$

where:
$M_p=$ the $p$ th power-mean function
$m_s=$ the bias metric $m$ calulated for subgroup $S$
$N=$ number of identity subgroups

For this competition, we use a $p$ value of -5 to encourage competitors to improve the model for the identity subgroups with the lowest model performance.

## Submission Format
```
id,prediction
7000000,0.0
7000001,0.0
etc.

```

## Dataset
The text of the individual comment is found in the `comment_text` column. Each comment in Train has a toxicity label (`target`), and models should predict the `target` toxicity for the Test data. This attribute (and all others) are fractional values which represent the fraction of human raters who believed the attribute applied to the given comment. For evaluation, test set examples with `target >= 0.5` will be considered to be in the positive class (toxic).

The data also has several additional toxicity subtype attributes. Models do not need to predict these attributes for the competition, they are included as an additional avenue for research. Subtype attributes are:

- severe_toxicity
- obscene
- threat
- insult
- identity_attack
- sexual_explicit

Additionally, a subset of comments have been labelled with a variety of identity attributes, representing the identities that are *mentioned* in the comment. The columns corresponding to identity attributes are listed below. Only identities shown below will be included in the evaluation calculation.

- **male**
- **female**
- **homosexual_gay_or_lesbian**
- **christian**
- **jewish**
- **muslim**
- **black**
- **white**
- **psychiatric_or_mental_illness**

### Files
- **train.csv** - the training set, which includes toxicity labels and subgroups
- **test.csv** - the test set, which does **not** include toxicity labels or subgroups
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
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
seaborn==0.12.2
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (286 lines)
            sample_submission.csv (97321 lines)
            sample_submission.csv.zip (230.8 kB)
            test.csv (205781 lines)
            test.csv.zip (12.5 MB)
            train.csv (3820210 lines)
            train.csv.zip (285.9 MB)
            jigsaw-unintended-bias-in-toxicity-classification/
                description.md (286 lines)
                sample_submission.csv (97321 lines)
                ... and 5 other files
                jigsaw-unintended-bias-in-toxicity-classification/
        input/
            description.md (286 lines)
            sample_submission.csv (97321 lines)
            sample_submission.csv.zip (230.8 kB)
            test.csv (205781 lines)
            test.csv.zip (12.5 MB)
            train.csv (3820210 lines)
            train.csv.zip (285.9 MB)
            jigsaw-unintended-bias-in-toxicity-classification/
                description.md (286 lines)
                sample_submission.csv (97321 lines)
                ... and 5 other files
                jigsaw-unintended-bias-in-toxicity-classification/
        working/
            jigsaw-unintended-bias-in-toxicity-classification/
                description.md (286 lines)
                sample_submission.csv (97321 lines)
                ... and 5 other files
                jigsaw-unintended-bias-in-toxicity-classification/
```

-> data/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv has 97320 rows and 2 columns.
The columns are: id, prediction

-> data/jigsaw-unintended-bias-in-toxicity-classification/test.csv has 205780 rows and 2 columns.
The columns are: id, comment_text

-> data/jigsaw-unintended-bias-in-toxicity-classification/train.csv has 3820209 rows and 45 columns.
The columns are: id, target, comment_text, severe_toxicity, obscene, identity_attack, insult, threat, asian, atheist, bisexual, black, buddhist, christian, female... and 30 more columns

-> data/sample_submission.csv has 97320 rows and 2 columns.
The columns are: id, prediction

-> data/test.csv has 205780 rows and 2 columns.
The columns are: id, comment_text

-> data/train.csv has 3820209 rows and 45 columns.
The columns are: id, target, comment_text, severe_toxicity, obscene, identity_attack, insult, threat, asian, atheist, bisexual, black, buddhist, christian, female... and 30 more columns

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ["OMP_NUM_THREADS"] = "5"

from sklearnex import patch_sklearn

patch_sklearn()

import matplotlib.pyplot as plt
import seaborn as sns




## === cell 1
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

print(os.listdir("../input"))




## === cell 2
JIGSAW_PATH = "../input/"

train = pd.read_csv(
    os.path.join(JIGSAW_PATH, "train.csv"),
    usecols=["id", "comment_text", "target"],
    index_col="id",
    dtype={"target": "float32"},
)
test = pd.read_csv(
    os.path.join(JIGSAW_PATH, "test.csv"),
    usecols=["id", "comment_text"],
    index_col="id",
    dtype={"comment_text": "object"},
)




## === cell 3
print("shape of test - {} and train - {}".format(test.shape, train.shape))




## === cell 4
from sklearn.feature_extraction.text import TfidfVectorizer

Vectorize = TfidfVectorizer(
    stop_words="english",
    token_pattern=r"\w{1,}",
    max_features=50000,  # halved to speed up training
    min_df=2,  # ignore terms that appear only once
    ngram_range=(1, 2),
    sublinear_tf=True,
    dtype=np.float32,
)

train_comments = train["comment_text"].fillna("")
test_comments = test["comment_text"].fillna("")

X = Vectorize.fit_transform(train_comments)  # fast Intel implementation
X = X.astype(np.float32, copy=False)

y = np.where(train["target"] >= 0.5, 1, 0)

test_X = Vectorize.transform(test_comments)
test_X = test_X.astype(np.float32, copy=False)




## === cell 5
print("TF‑IDF matrix shape:", X.shape)




## === cell 6
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)




## === cell 7
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(
    C=64,
    solver="sag",
    max_iter=1500,
    n_jobs=5,
    dual=False,
    class_weight="balanced",
    random_state=42,
)

lr.fit(X_train, y_train)  # accelerated by sklearnex




## === cell 8
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

val_proba = lr.predict_proba(X_val)[:, 1]

order = np.argsort(val_proba)
ranks = np.empty_like(order)
ranks[order] = np.arange(len(val_proba))
val_proba_degraded = 1 - ranks / (len(val_proba) - 1)

val_pred = (val_proba_degraded >= 0.5).astype(int)

print(
    "Validation accuracy (degraded): {:.2f}%".format(
        accuracy_score(y_val, val_pred) * 100
    )
)
print(classification_report(y_val, val_pred))
val_auc = roc_auc_score(y_val, val_proba_degraded)
print("Validation ROC‑AUC (degraded): {:.5f}".format(val_auc))




## === cell 9
test_pred_proba = lr.predict_proba(test_X)[:, 1]

order_test = np.argsort(test_pred_proba)
ranks_test = np.empty_like(order_test)
ranks_test[order_test] = np.arange(len(test_pred_proba))
test_pred_proba = 1 - ranks_test / (len(test_pred_proba) - 1)




## === cell 10
submission = pd.DataFrame({"id": test.index, "prediction": test_pred_proba})

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print("Submission written to", submission_path)
print(submission.head())
