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

3.8

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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
import numpy as np
import pandas as pd

INPUT_DIR = "/kaggle/input/jigsaw-unintended-bias-in-toxicity-classification"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "/kaggle/input"

print("Listing /kaggle/input:")
print(os.listdir("/kaggle/input")[:50])



## === cell 1
train_path = os.path.join(INPUT_DIR, "train.csv")
test_path = os.path.join(INPUT_DIR, "test.csv")
sample_path = os.path.join(INPUT_DIR, "sample_submission.csv")

if not (
    os.path.exists(train_path)
    and os.path.exists(test_path)
    and os.path.exists(sample_path)
):
    nested = os.path.join(
        INPUT_DIR, "jigsaw-unintended-bias-in-toxicity-classification"
    )
    train_path = os.path.join(nested, "train.csv")
    test_path = os.path.join(nested, "test.csv")
    sample_path = os.path.join(nested, "sample_submission.csv")

assert os.path.exists(train_path), f"train.csv not found at {train_path}"
assert os.path.exists(test_path), f"test.csv not found at {test_path}"
assert os.path.exists(sample_path), f"sample_submission.csv not found at {sample_path}"

print("Using paths:")
print(train_path)
print(test_path)
print(sample_path)



## === cell 2
identity_cols = [
    "male",
    "female",
    "homosexual_gay_or_lesbian",
    "christian",
    "jewish",
    "muslim",
    "black",
    "white",
    "psychiatric_or_mental_illness",
]
usecols_train = ["comment_text", "target"] + identity_cols
usecols_test = ["id", "comment_text"]

train = pd.read_csv(train_path, usecols=usecols_train)
test = pd.read_csv(test_path, usecols=usecols_test)
sample_sub = pd.read_csv(sample_path)

print(train.shape, test.shape, sample_sub.shape)
print("Train target NA:", train["target"].isna().mean())
print("Train text NA:", train["comment_text"].isna().mean())
print("Test text NA:", test["comment_text"].isna().mean())



## === cell 3
train["comment_text"] = train["comment_text"].fillna("")
test["comment_text"] = test["comment_text"].fillna("")

X_text_all = train["comment_text"].values
X_test_text = test["comment_text"].values

y_cont_all = train["target"].astype(np.float32).values
y = (y_cont_all >= 0.5).astype(np.int8)

idn_mat_all = train[identity_cols].fillna(0.0).to_numpy(dtype=np.float32)

MAX_TRAIN = 1200000
if len(train) > MAX_TRAIN:
    rng = np.random.RandomState(42)
    idx = rng.choice(len(train), size=MAX_TRAIN, replace=False)
    X_text = X_text_all[idx]
    y = y[idx]
    idn_mat = idn_mat_all[idx]
    print(f"Downsampled train to {MAX_TRAIN} rows for runtime.")
else:
    X_text = X_text_all
    idn_mat = idn_mat_all
    print("Using full training set.")

subgroup_mentioned = (idn_mat >= 0.5).any(axis=1).astype(np.float32)
sample_weight = np.where(subgroup_mentioned > 0, 1.5, 1.0).astype(np.float32)

print("y (binary) stats:", int(np.min(y)), float(np.mean(y)), int(np.max(y)))
print(
    "subgroup_mentioned rate (in used train):", float(np.mean(subgroup_mentioned > 0))
)



## === cell 4
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

word_tfidf = TfidfVectorizer(
    strip_accents="unicode",
    lowercase=True,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    max_features=200000,
    sublinear_tf=True,
)

char_tfidf = TfidfVectorizer(
    strip_accents="unicode",
    lowercase=True,
    analyzer="char",
    ngram_range=(3, 5),
    min_df=3,
    max_df=0.9,
    max_features=150000,
    sublinear_tf=True,
)

model = Pipeline(
    steps=[
        (
            "feats",
            FeatureUnion(
                transformer_list=[
                    ("word", word_tfidf),
                    ("char", char_tfidf),
                ],
                n_jobs=2,
            ),
        ),
        (
            "clf",
            LogisticRegression(
                solver="saga",
                penalty="l2",
                max_iter=300,
                C=4.0,
                n_jobs=2,
                random_state=42,
                class_weight="balanced",
            ),
        ),
    ]
)

print(model)



## === cell 5
model.fit(X_text, y, clf__sample_weight=sample_weight)



## === cell 6
pred = model.predict_proba(X_test_text)[:, 1].astype(np.float64)
pred = np.clip(pred, 0.0, 1.0)



## === cell 7
sub = pd.DataFrame({"id": test["id"].values, "prediction": pred})

assert list(sub.columns) == list(
    sample_sub.columns
), f"Submission columns {sub.columns.tolist()} != {sample_sub.columns.tolist()}"
assert len(sub) == len(
    sample_sub
), f"Submission rows {len(sub)} != sample rows {len(sample_sub)}"
assert sub["id"].isna().sum() == 0
assert sub["prediction"].isna().sum() == 0

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv:", sub.shape)
print(sub.head())



## === cell 8
submission = pd.read_csv("/kaggle/working/submission.csv")
print(submission.shape)
print(submission.head())
print(
    "prediction range:", submission["prediction"].min(), submission["prediction"].max()
)
