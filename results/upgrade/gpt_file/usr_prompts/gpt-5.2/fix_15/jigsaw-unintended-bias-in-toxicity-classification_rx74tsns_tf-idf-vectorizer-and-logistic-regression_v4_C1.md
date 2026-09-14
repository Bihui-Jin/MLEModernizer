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

# 5. Target score

0.88424

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.71715) has done: 'I fix the pipeline so it runs end-to-end and writes a valid `submission.csv`. The immediate runtime failure is caused by `NaN` values in `comment_text`, which `TfidfVectorizer` cannot process; I fill missing text with empty strings in both train and test before vectorizing. I also correct the input paths to match the provided Kaggle directory layout and keep the model/training logic unchanged (TF‑IDF + LogisticRegression). Finally, I ensure the submission uses the required `id,prediction` columns and aligns predictions to the test rows.'
- What this solution (achieved 0.75747) has done: 'Your current score is far below the target (0.71715 vs 0.88424), so we should improve performance with minimal, safe changes that keep the same core approach (TF‑IDF → LogisticRegression → predict_proba). The biggest issue is that you’re effectively optimizing for plain toxicity AUC only, while the competition metric heavily rewards reducing subgroup bias; we can do this without changing the model by switching to the competition’s standard sample-weighting scheme during `fit()`. I also fix a correctness leak in your notebook reporting (you split but never use it) and increase `max_iter` so SAG actually converges on this very large sparse matrix (improves score stability without changing the approach). Finally, I keep the same submission format and paths but ensure we always write a valid `submission.csv`.'
- What this solution (achieved 0.75466) has done: 'We make the smallest changes that are most likely to lift AUC toward your target while keeping the same TF‑IDF → LogisticRegression → predict_proba pipeline. Right now you create a train/validation split but still fit on the full dataset, and your `sample_weight` is computed for the full dataset while the model is trained on full `X,y`; we instead train on `X_train,y_train` with correctly aligned weights and evaluate on the held-out set (same core logic, but fixes a correctness issue and improves generalization). We also switch the solver to `liblinear` (still LogisticRegression) which is typically stronger than `sag` on sparse TF‑IDF for this problem at this scale, and raise `max_iter` to ensure convergence stability. Finally, we keep the submission schema/paths the same and still write `submission.csv` with `id,prediction`.'
- What this solution (achieved 0.5) has done: 'I fix the runtime failure by ensuring the sparse TF‑IDF matrix is `float64`, because `SGDClassifier`’s sparse dataset backend expects doubles when `sample_weight` is used (your current error is a float/double buffer mismatch). I also make the sample weights align exactly with the train/validation split indices to avoid any silent misalignment, and keep the existing TF‑IDF + `SGDClassifier(log_loss)` core approach unchanged. Finally, I make the I/O paths robust to the provided Kaggle directory layout and ensure we always write a valid `submission.csv` with `id,prediction` and the correct row count.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 score is far below the 0.88424 target, so we should push performance up with minimal changes while keeping the same TF‑IDF → linear classifier → predict_proba pipeline. The most likely cause of the 0.5 plateau is that `SGDClassifier` is under-trained here (`max_iter=20` with `tol=1e-3` often stops too early on this scale), so we increase iterations and require convergence more strictly without changing the model type. To better match the competition’s bias-focused metric while preserving your approach, we also slightly strengthen the existing identity/toxicity sample-weighting (still the same weighting formula/logic, just tuned) to reduce bias penalties. Submission writing and paths remain unchanged and we still emit a valid `submission.csv` with `id,prediction`.'
- What this solution (achieved 0.7666) has done: 'Your 0.5 score strongly suggests the submission is effectively uninformative (often caused by misaligned `id`/prediction rows or near-constant probabilities), so the smallest high-impact fix is to guarantee prediction-to-`id` alignment by building the submission from `test.csv` (not `sample_submission.csv`) and merging by `id`. Next, without changing the core TF‑IDF → `SGDClassifier(log_loss)` pipeline, we improve learning stability by standardizing the regularization strength via `alpha=1e-6` (your current `1/32` is far too strong for sparse TF‑IDF and commonly collapses probabilities), while keeping the same optimizer/training approach. Finally, we keep your bias-aware `sample_weight` logic but ensure it is aligned to the split indices (as you intended) and add a quick sanity check that predictions have non-trivial variance before writing `submission.csv`. These changes are minimal, preserve the approach, and are the most likely to move you up toward the 0.88424 target band.'

# 9. Code solution

## === cell 0
import os
import random
import gc
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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

if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = "/kaggle/input/train.csv"
if not os.path.exists(TEST_PATH):
    TEST_PATH = "/kaggle/input/test.csv"
if not os.path.exists(SUB_PATH):
    SUB_PATH = "/kaggle/input/sample_submission.csv"

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

dtype_train = {c: "float32" for c in identity_columns_all}
dtype_train.update({"target": "float32", "comment_text": "string"})
dtype_test = {"id": "int64", "comment_text": "string"}

read_csv_kwargs = dict()
try:
    import pyarrow  # noqa: F401

    read_csv_kwargs["engine"] = "pyarrow"
except Exception:
    pass

df_train = pd.read_csv(
    TRAIN_PATH, usecols=train_usecols, dtype=dtype_train, **read_csv_kwargs
)
df_test = pd.read_csv(
    TEST_PATH, usecols=["id", "comment_text"], dtype=dtype_test, **read_csv_kwargs
)
_ = pd.read_csv(
    SUB_PATH, usecols=["id", "prediction"], **read_csv_kwargs
)  # schema check only

df_train["comment_text"] = df_train["comment_text"].fillna("")
df_test["comment_text"] = df_test["comment_text"].fillna("")



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ArrowInvalid                              Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/arrow_parser_wrapper.py in read(self)
    265         try:
--> 266             table = pyarrow_csv.read_csv(
    267                 self.src,

/usr/local/lib/python3.11/dist-packages/pyarrow/_csv.pyx in pyarrow._csv.read_csv()

/usr/local/lib/python3.11/dist-packages/pyarrow/_csv.pyx in pyarrow._csv.read_csv()

/usr/local/lib/python3.11/dist-packages/pyarrow/error.pxi in pyarrow.lib.pyarrow_internal_check_status()

/usr/local/lib/python3.11/dist-packages/pyarrow/error.pxi in pyarrow.lib.check_status()

ArrowInvalid: CSV parse error: Expected 45 columns, got 44: Robin Hood is always highly popular - as long as he's stealing from someone else, and sharing th ...

The above exception was the direct cause of the following exception:

ParserError                               Traceback (most recent call last)
/tmp/ipykernel_11/3067503043.py in <cell line: 0>()
     40     pass
     41 
---> 42 df_train = pd.read_csv(
     43     TRAIN_PATH, usecols=train_usecols, dtype=dtype_train, **read_csv_kwargs
     44 )

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    624 
    625     with parser:
--> 626         return parser.read(nrows)
    627 
    628 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read(self, nrows)
   1909             try:
   1910                 # error: "ParserBase" has no attribute "read"
-> 1911                 df = self._engine.read()  # type: ignore[attr-defined]
   1912             except Exception:
   1913                 self.close()

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/arrow_parser_wrapper.py in read(self)
    271             )
    272         except pa.ArrowInvalid as e:
--> 273             raise ParserError(e) from e
    274 
    275         dtype_backend = self.kwds["dtype_backend"]

ParserError: CSV parse error: Expected 45 columns, got 44: Robin Hood is always highly popular - as long as he's stealing from someone else, and sharing th ...

## === cell 2
import hashlib
import joblib
from scipy import sparse

CACHE_DIR = "/kaggle/working/tfidf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def _file_sig(path: str) -> str:
    st = os.stat(path)
    return f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}"


sig = hashlib.md5(
    (
        "|".join([_file_sig(TRAIN_PATH), _file_sig(TEST_PATH)]) + f"|seed={SEED}|v=1"
    ).encode("utf-8")
).hexdigest()

VEC_PATH = os.path.join(CACHE_DIR, f"tfidf_vectorizer_{sig}.joblib")
X_PATH = os.path.join(CACHE_DIR, f"X_train_{sig}.npz")
TX_PATH = os.path.join(CACHE_DIR, f"X_test_{sig}.npz")



## === cell 3
Vectorize = TfidfVectorizer(
    stop_words=None,
    analyzer="char_wb",
    ngram_range=(3, 5),
    max_features=300000,
    min_df=2,
    dtype=np.float32,
)

train_text = df_train["comment_text"].to_numpy(copy=False)
test_text = df_test["comment_text"].to_numpy(copy=False)

if os.path.exists(VEC_PATH) and os.path.exists(X_PATH) and os.path.exists(TX_PATH):
    Vectorize = joblib.load(VEC_PATH)
    X = sparse.load_npz(X_PATH).tocsr()
    test_X = sparse.load_npz(TX_PATH).tocsr()
else:
    X = Vectorize.fit_transform(train_text)  # CSR
    test_X = Vectorize.transform(test_text)  # CSR
    joblib.dump(Vectorize, VEC_PATH, compress=3)
    sparse.save_npz(X_PATH, X)
    sparse.save_npz(TX_PATH, test_X)

y = (df_train["target"].to_numpy(dtype=np.float32, copy=False) >= 0.5).astype(
    np.int8, copy=False
)

del train_text, test_text
gc.collect()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4152904971.py in <cell line: 0>()
      8 )
      9 
---> 10 train_text = df_train["comment_text"].to_numpy(copy=False)
     11 test_text = df_test["comment_text"].to_numpy(copy=False)
     12 

NameError: name 'df_train' is not defined

## === cell 4
n = X.shape[0]
all_idx = np.arange(n, dtype=np.int32)

idx_train, idx_val, y_train, y_val = train_test_split(
    all_idx, y, test_size=0.2, random_state=SEED, stratify=y
)

X_train = X[idx_train]
X_val = X[idx_val]



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1773218177.py in <cell line: 0>()
      1 # Speed-only change: split indices first and then slice CSR once.
      2 # This avoids passing a huge sparse matrix through train_test_split internals and prevents extra copies.
----> 3 n = X.shape[0]
      4 all_idx = np.arange(n, dtype=np.int32)
      5 

NameError: name 'X' is not defined

## === cell 5
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
sub_and_tox = subgroup & is_toxic
sub_and_notox = subgroup & (~is_toxic)
sample_weight[sub_and_tox] = w_subgroup * w_subgroup_toxic
sample_weight[sub_and_notox] = w_subgroup * w_subgroup_nontoxic

sample_weight_train = sample_weight[idx_train]
sample_weight_val = sample_weight[idx_val]

del ident_mat, subgroup, target, is_toxic, sub_and_tox, sub_and_notox
gc.collect()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1142722691.py in <cell line: 0>()
----> 1 identity_columns = [c for c in identity_columns_all if c in df_train.columns]
      2 
      3 target = df_train["target"].to_numpy(dtype=np.float32, copy=False)
      4 is_toxic = target >= 0.5
      5 

/tmp/ipykernel_11/1142722691.py in <listcomp>(.0)
----> 1 identity_columns = [c for c in identity_columns_all if c in df_train.columns]
      2 
      3 target = df_train["target"].to_numpy(dtype=np.float32, copy=False)
      4 is_toxic = target >= 0.5
      5 

NameError: name 'df_train' is not defined

## === cell 6
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



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1646049389.py in <cell line: 0>()
     12 )
     13 
---> 14 clf.fit(X_train, y_train, sample_weight=sample_weight_train)
     15 

NameError: name 'X_train' is not defined

## === cell 7
y_val_pred = clf.predict(X_val)
print("Validation Accuracy is {0:.2f}%".format(accuracy_score(y_val, y_val_pred) * 100))



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/126519827.py in <cell line: 0>()
----> 1 y_val_pred = clf.predict(X_val)
      2 print("Validation Accuracy is {0:.2f}%".format(accuracy_score(y_val, y_val_pred) * 100))
      3 

NameError: name 'X_val' is not defined

## === cell 8
z = clf.decision_function(X_val).astype(np.float64, copy=False)
val_proba = np.empty_like(z, dtype=np.float64)
np.negative(z, out=val_proba)
np.exp(val_proba, out=val_proba)  # exp(-z)
val_proba += 1.0
np.reciprocal(val_proba, out=val_proba)  # 1/(1+exp(-z))

fpr, tpr, thr = roc_curve(y_val, val_proba)
auc_val = auc(fpr, tpr) * 100

print("Validation AUC:", float(auc_val))
print("Val proba mean/std:", float(np.mean(val_proba)), float(np.std(val_proba)))



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4001782950.py in <cell line: 0>()
----> 1 z = clf.decision_function(X_val).astype(np.float64, copy=False)
      2 val_proba = np.empty_like(z, dtype=np.float64)
      3 np.negative(z, out=val_proba)
      4 np.exp(val_proba, out=val_proba)  # exp(-z)
      5 val_proba += 1.0

NameError: name 'X_val' is not defined

## === cell 9
zt = clf.decision_function(test_X).astype(np.float64, copy=False)
predictions = np.empty_like(zt, dtype=np.float64)
np.negative(zt, out=predictions)
np.exp(predictions, out=predictions)  # exp(-zt)
predictions += 1.0
np.reciprocal(predictions, out=predictions)

np.clip(predictions, 0.0, 1.0, out=predictions)

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

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1977836479.py in <cell line: 0>()
----> 1 zt = clf.decision_function(test_X).astype(np.float64, copy=False)
      2 predictions = np.empty_like(zt, dtype=np.float64)
      3 np.negative(zt, out=predictions)
      4 np.exp(predictions, out=predictions)  # exp(-zt)
      5 predictions += 1.0

NameError: name 'test_X' is not defined
