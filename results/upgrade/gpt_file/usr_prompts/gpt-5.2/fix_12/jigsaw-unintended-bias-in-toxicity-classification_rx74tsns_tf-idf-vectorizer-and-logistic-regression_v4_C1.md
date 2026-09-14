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
import random
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

try:
    from sklearnex import patch_sklearn  # type: ignore

    patch_sklearn()
except Exception:
    pass

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)



## === cell 1
BASE_DIR = "/kaggle/input/jigsaw-unintended-bias-in-toxicity-classification"
if not os.path.exists(BASE_DIR):
    BASE_DIR = "/kaggle/input"

TRAIN_PATH = os.path.join(BASE_DIR, "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "test.csv")
SUB_PATH = os.path.join(BASE_DIR, "sample_submission.csv")

identity_columns_all = [
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
train_usecols = ["target", "comment_text"] + identity_columns_all
usecols_set = set(train_usecols)

dtype_train = {c: "float32" for c in identity_columns_all}
dtype_train.update({"target": "float32", "comment_text": "string"})
dtype_test = {"id": "int64", "comment_text": "string"}

df_train = pd.read_csv(
    TRAIN_PATH, usecols=lambda c: c in usecols_set, dtype=dtype_train
)
df_test = pd.read_csv(TEST_PATH, usecols=["id", "comment_text"], dtype=dtype_test)
sub_schema = pd.read_csv(SUB_PATH, usecols=["id", "prediction"])

df_train["comment_text"] = df_train["comment_text"].fillna("")
df_test["comment_text"] = df_test["comment_text"].fillna("")



## === cell 2
Vectorize = TfidfVectorizer(
    stop_words=None,
    analyzer="char_wb",
    ngram_range=(3, 5),
    max_features=300000,
    min_df=2,
    dtype=np.float32,
)

X = Vectorize.fit_transform(df_train["comment_text"])
test_X = Vectorize.transform(df_test["comment_text"])

X = X.tocsr()
test_X = test_X.tocsr()

y = (df_train["target"].to_numpy(dtype=np.float32, copy=False) >= 0.5).astype(
    np.int8, copy=False
)



## === cell 3
X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
    X, y, df_train.index.values, test_size=0.2, random_state=SEED, stratify=y
)



## === cell 4
identity_columns = [c for c in identity_columns_all if c in df_train.columns]

target = df_train["target"].to_numpy(dtype=np.float32, copy=False)
is_toxic = target >= 0.5

if len(identity_columns) > 0:
    ident_mat = df_train[identity_columns].to_numpy(dtype=np.float32, copy=False)
    subgroup = ident_mat.max(axis=1) >= 0.5
else:
    subgroup = np.zeros(df_train.shape[0], dtype=bool)

w_subgroup = 2.0
w_bg = 1.0
w_subgroup_toxic = 2.0
w_subgroup_nontoxic = 2.0

sample_weight = np.full(df_train.shape[0], w_bg, dtype=np.float64)
mask_sub = subgroup
mask_tox = is_toxic
sample_weight[mask_sub & mask_tox] = w_subgroup * w_subgroup_toxic
sample_weight[mask_sub & (~mask_tox)] = w_subgroup * w_subgroup_nontoxic

sample_weight_train = sample_weight[idx_train]
sample_weight_val = sample_weight[idx_val]



## === cell 5
X_train = X_train.tocsr()
X_val = X_val.tocsr()

clf = SGDClassifier(
    loss="log_loss",
    penalty="l2",
    alpha=1e-6,
    fit_intercept=True,
    max_iter=200,
    tol=1e-5,
    shuffle=True,
    random_state=SEED,
    n_jobs=-1,
    average=False,
)
clf.fit(X_train, y_train, sample_weight=sample_weight_train)



## === cell 6
y_val_pred = clf.predict(X_val)
print("Validation Accuracy is {0:.2f}%".format(accuracy_score(y_val, y_val_pred) * 100))



## === cell 7
z = clf.decision_function(X_val)
val_proba = 1.0 / (1.0 + np.exp(-z))

fpr, tpr, thr = roc_curve(y_val, val_proba)
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic Plot (Validation)")
auc_val = auc(fpr, tpr) * 100
plt.legend(["AUC {0:.3f}".format(auc_val)])

print("Val proba mean/std:", float(np.mean(val_proba)), float(np.std(val_proba)))



## === cell 8
zt = clf.decision_function(test_X)
predictions = 1.0 / (1.0 + np.exp(-zt))
predictions = np.clip(predictions, 0.0, 1.0)

sub = pd.DataFrame(
    {
        "id": df_test["id"].to_numpy(copy=False),
        "prediction": predictions.astype(np.float32, copy=False),
    }
)

assert (
    sub.shape[0] == df_test.shape[0]
), f"Submission rows {sub.shape[0]} != test rows {df_test.shape[0]}"
assert sub["id"].is_unique, "Test ids are not unique; cannot create a valid submission."

pred_std = float(np.std(sub["prediction"].to_numpy(copy=False)))
print("Test prediction mean/std:", float(sub["prediction"].mean()), pred_std)
assert pred_std > 1e-6, "Predictions are (near-)constant; likely to score ~0.5."

sub[["id", "prediction"]].to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub[["id", "prediction"]].shape)
