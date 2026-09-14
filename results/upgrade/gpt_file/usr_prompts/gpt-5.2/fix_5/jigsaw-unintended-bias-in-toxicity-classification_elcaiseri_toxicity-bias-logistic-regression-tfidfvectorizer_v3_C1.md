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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

INPUT_CANDIDATES = [
    "../input/jigsaw-unintended-bias-in-toxicity-classification",
    "../input",
    "/kaggle/input/jigsaw-unintended-bias-in-toxicity-classification",
    "/kaggle/input",
    "/kaggle/data/jigsaw-unintended-bias-in-toxicity-classification",
    "/kaggle/data",
    "/kaggle/data/input/jigsaw-unintended-bias-in-toxicity-classification",
    "/kaggle/data/input",
]
INPUT_DIR = None
for p in INPUT_CANDIDATES:
    if (
        os.path.exists(p)
        and os.path.isdir(p)
        and any(fn.endswith(".csv") for fn in os.listdir(p))
    ):
        INPUT_DIR = p
        break

if INPUT_DIR is None:
    print("Could not find a valid input directory. Listing ../input if it exists:")
    if os.path.exists("../input"):
        print(os.listdir("../input"))
    raise FileNotFoundError("No valid Kaggle input directory found among candidates.")

print("Using INPUT_DIR:", INPUT_DIR)
print("Files:", [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")][:10])



## === cell 1
train_path = os.path.join(INPUT_DIR, "train.csv")
test_path = os.path.join(INPUT_DIR, "test.csv")
sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sub = pd.read_csv(sub_path)

print("train_df:", train_df.shape, "test_df:", test_df.shape, "sub:", sub.shape)



## === cell 2
df = train_df.copy()



## === cell 3
df.head()



## === cell 4
from sklearn.feature_extraction.text import TfidfVectorizer

Vectorize = TfidfVectorizer(
    stop_words="english",
    token_pattern=r"\w{1,}",
    max_features=25000,
    ngram_range=(1, 2),
    analyzer="word",
    strip_accents="unicode",
    sublinear_tf=True,
)

Vectorize_char = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=25000,
    strip_accents="unicode",
    sublinear_tf=True,
)



## === cell 5
from scipy.sparse import hstack

df["comment_text"] = df["comment_text"].fillna("")
test_df["comment_text"] = test_df["comment_text"].fillna("")

X_word = Vectorize.fit_transform(df["comment_text"])
X_char = Vectorize_char.fit_transform(df["comment_text"])
X = hstack([X_word, X_char]).tocsr()

y_continuous = train_df["target"].astype(np.float32).values
y = (y_continuous >= 0.5).astype(np.int32)

test_X_word = Vectorize.transform(test_df["comment_text"])
test_X_char = Vectorize_char.transform(test_df["comment_text"])
test_X = hstack([test_X_word, test_X_char]).tocsr()

print("X:", X.shape, "y:", y.shape, "positive rate:", float(y.mean()))
print("test_X:", test_X.shape)



## === cell 6
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score



## === cell 7
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train split:", X_train.shape, "Valid split:", X_valid.shape)



## === cell 8
lr = LogisticRegression(
    C=4,
    dual=False,
    n_jobs=-1,
    solver="sag",
    max_iter=1000,
    class_weight="balanced",
)
lr.fit(X_train, y_train)



## === cell 9
val_proba = lr.predict_proba(X_valid)[:, 1]
print("Validation ROC-AUC:", roc_auc_score(y_valid, val_proba))



## === cell 10
y_pred_label = (val_proba >= 0.5).astype(int)
print(classification_report(y_valid, y_pred_label))



## === cell 11
pass



## === cell 12
lr_full = LogisticRegression(
    C=4,
    dual=False,
    n_jobs=-1,
    solver="sag",
    max_iter=1000,
    class_weight="balanced",
)
lr_full.fit(X, y)



## === cell 13
predictions = lr_full.predict_proba(test_X)[:, 1]



## === cell 14
sub.head()



## === cell 15
if "id" not in sub.columns or "id" not in test_df.columns:
    raise ValueError("Expected 'id' column in both sample submission and test data.")

pred_df = pd.DataFrame({"id": test_df["id"].values, "prediction": predictions})
sub_out = sub[["id"]].merge(pred_df, on="id", how="left")

if sub_out["prediction"].isna().any():
    missing = int(sub_out["prediction"].isna().sum())
    raise ValueError(f"Missing predictions after id-merge: {missing} rows.")

sub_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_out.shape)
sub_out.head(15)
