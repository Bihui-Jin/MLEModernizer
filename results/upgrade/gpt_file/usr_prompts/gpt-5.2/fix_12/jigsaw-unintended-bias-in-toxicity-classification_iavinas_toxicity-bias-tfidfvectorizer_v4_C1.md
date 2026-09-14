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
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

print(os.listdir("../input"))



## === cell 1
identity_cols_eval = [
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
usecols_train = ["id", "target", "comment_text"] + identity_cols_eval
usecols_test = ["id", "comment_text"]

dtype_train = {"id": np.int64, "target": np.float32, "comment_text": object}
for c in identity_cols_eval:
    dtype_train[c] = np.float32
dtype_test = {"id": np.int64, "comment_text": object}

train_df = pd.read_csv(
    "../input/train.csv", usecols=usecols_train, dtype=dtype_train
).set_index("id")
test_df = pd.read_csv(
    "../input/test.csv", usecols=usecols_test, dtype=dtype_test
).set_index("id")
train_df.head()



## === cell 2
train_text = train_df["comment_text"].fillna("").astype(str)
test_text = test_df["comment_text"].fillna("").astype(str)

t = train_df["target"].to_numpy(dtype=np.float32, copy=False)
t = np.clip(t, 0.0, 1.0)

identity_cols_eval = [c for c in identity_cols_eval if c in train_df.columns]

if len(identity_cols_eval) == 0:
    identity_mentioned = np.zeros(len(train_df), dtype=np.float32)
else:
    identity_mentioned = (
        (
            train_df[identity_cols_eval]
            .fillna(0.0)
            .to_numpy(dtype=np.float32, copy=False)
            >= 0.5
        )
        .any(axis=1)
        .astype(np.float32)
    )

y_bin = (t >= 0.5).astype(np.float32)

w = 1.0 + 3.0 * identity_mentioned + 3.0 * y_bin
w = (
    w
    + 2.0 * (identity_mentioned * (1.0 - y_bin))
    + 2.0 * ((1.0 - identity_mentioned) * y_bin)
)
sample_weight = w.astype(np.float64, copy=False)

Vect = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    max_features=250000,
    strip_accents="unicode",
    lowercase=True,
    sublinear_tf=True,
    stop_words="english",
    token_pattern=r"(?u)\b\w+\b",
    dtype=np.float32,
)

X_train = Vect.fit_transform(train_text)
X_test = Vect.transform(test_text)



## === cell 3
print(X_train.shape, X_test.shape, t.shape, sample_weight.shape)
print("X_train dtype:", X_train.dtype, "X_test dtype:", X_test.dtype)



## === cell 4

t64 = t.astype(np.float64, copy=False)
sw = sample_weight.astype(np.float64, copy=False)


from scipy import sparse

pos_w = sw * t64
neg_w = sw * (1.0 - t64)

X_train_dup = sparse.vstack((X_train, X_train), format="csr")
y_dup = np.empty(X_train.shape[0] * 2, dtype=np.int32)
y_dup[: X_train.shape[0]] = 1
y_dup[X_train.shape[0] :] = 0
sw_dup = np.concatenate((pos_w, neg_w), axis=0)

print("Training with duplicated X for soft targets.")
print(
    "Weights stats (pos min/mean/max):",
    float(pos_w.min()),
    float(pos_w.mean()),
    float(pos_w.max()),
)
print(
    "Weights stats (neg min/mean/max):",
    float(neg_w.min()),
    float(neg_w.mean()),
    float(neg_w.max()),
)

clf = LogisticRegression(
    solver="saga",
    penalty="l2",
    C=4.0,
    max_iter=400,
    n_jobs=-1,
    random_state=42,
)

clf.fit(X_train_dup, y_dup, sample_weight=sw_dup)

pre_pos = clf.predict_proba(X_test)[:, 1].astype(np.float32, copy=False)



## === cell 5
sub = pd.read_csv("../input/sample_submission.csv")

pred_df = pd.DataFrame(
    {
        "id": test_df.index.to_numpy(copy=False),
        "prediction": np.asarray(pre_pos, dtype=np.float32),
    }
)

sub = sub[["id"]].merge(pred_df, on="id", how="left")
sub["prediction"] = sub["prediction"].fillna(0.0).astype(np.float32, copy=False)
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
sub.head()
