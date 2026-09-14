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
import os
import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONHASHSEED", "42")
np.random.seed(42)

n_threads = str(os.cpu_count() or 1)
os.environ.setdefault("OMP_NUM_THREADS", n_threads)
os.environ.setdefault("OPENBLAS_NUM_THREADS", n_threads)
os.environ.setdefault("MKL_NUM_THREADS", n_threads)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", n_threads)
os.environ.setdefault("NUMEXPR_NUM_THREADS", n_threads)

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
        and any(fn.endswith(".csv") or fn.endswith(".csv.zip") for fn in os.listdir(p))
    ):
        INPUT_DIR = p
        break

if INPUT_DIR is None:
    print("Could not find a valid input directory. Listing ../input if it exists:")
    if os.path.exists("../input"):
        print(os.listdir("../input"))
    raise FileNotFoundError("No valid Kaggle input directory found among candidates.")

print("Using INPUT_DIR:", INPUT_DIR)
print(
    "Files:",
    [
        f
        for f in os.listdir(INPUT_DIR)
        if (f.endswith(".csv") or f.endswith(".csv.zip"))
    ][:10],
)




## === cell 1
def _pick_csv_or_zip(path_no_ext: str):
    csv_path = path_no_ext + ".csv"
    zip_path = path_no_ext + ".csv.zip"
    if os.path.exists(zip_path):
        return zip_path
    return csv_path


train_path = _pick_csv_or_zip(os.path.join(INPUT_DIR, "train"))
test_path = _pick_csv_or_zip(os.path.join(INPUT_DIR, "test"))
sub_path = _pick_csv_or_zip(os.path.join(INPUT_DIR, "sample_submission"))

train_usecols = ["id", "target", "comment_text"]
test_usecols = ["id", "comment_text"]

TRAIN_NROWS = int(os.environ.get("TRAIN_NROWS", "800000"))

train_df = pd.read_csv(
    train_path,
    usecols=train_usecols,
    dtype={"id": np.int64, "target": np.float32},
    nrows=TRAIN_NROWS,
)
test_df = pd.read_csv(
    test_path,
    usecols=test_usecols,
    dtype={"id": np.int64},
)
sub = pd.read_csv(sub_path, dtype={"id": np.int64, "prediction": np.float32})

print("train_df:", train_df.shape, "test_df:", test_df.shape, "sub:", sub.shape)
print("TRAIN_NROWS used:", TRAIN_NROWS)



## === cell 2
df = train_df



## === cell 3
pass



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
    dtype=np.float32,
)

Vectorize_char = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=25000,
    strip_accents="unicode",
    sublinear_tf=True,
    dtype=np.float32,
)



## === cell 5
try:
    from sklearnex import patch_sklearn  # scikit-learn-intelex

    patch_sklearn()
    print("Applied sklearnex patch_sklearn() for performance.")
except Exception as _e:
    print("sklearnex patch not applied (continuing with vanilla sklearn):", repr(_e))

from scipy.sparse import hstack
import pickle
from scipy import sparse as sp
from concurrent.futures import ThreadPoolExecutor

cache_dir = "/kaggle/working"
os.makedirs(cache_dir, exist_ok=True)

cache_all_npz = os.path.join(cache_dir, "tfidf_cache_v4_all_stacked.npz")
cache_meta_pkl = os.path.join(cache_dir, "tfidf_cache_v4_meta.pkl")
cache_vec_word_pkl = os.path.join(cache_dir, "tfidf_cache_v4_vec_word.pkl")
cache_vec_char_pkl = os.path.join(cache_dir, "tfidf_cache_v4_vec_char.pkl")

df_comment = df["comment_text"].fillna("")
test_comment = test_df["comment_text"].fillna("")

y_continuous = train_df["target"].astype(np.float32).values
y = (y_continuous >= 0.5).astype(np.int32)

train_text = df_comment.to_numpy(dtype=object, copy=False)
test_text = test_comment.to_numpy(dtype=object, copy=False)
all_text = np.concatenate([train_text, test_text]).astype(str, copy=False)

use_cache = (
    os.path.exists(cache_all_npz)
    and os.path.exists(cache_meta_pkl)
    and os.path.exists(cache_vec_word_pkl)
    and os.path.exists(cache_vec_char_pkl)
)

meta = None
X_all = None
if use_cache:
    try:
        with open(cache_meta_pkl, "rb") as f:
            meta = pickle.load(f)

        params_ok = (
            meta.get("n_train") == len(df_comment)
            and meta.get("n_test") == len(test_comment)
            and meta.get("vec_word_params") == Vectorize.get_params()
            and meta.get("vec_char_params") == Vectorize_char.get_params()
        )
        if not params_ok:
            use_cache = False
        else:
            X_all = sp.load_npz(cache_all_npz)
            if meta.get("all_shape") != tuple(X_all.shape):
                use_cache = False
                X_all = None
            else:
                with open(cache_vec_word_pkl, "rb") as f:
                    Vectorize = pickle.load(f)
                with open(cache_vec_char_pkl, "rb") as f:
                    Vectorize_char = pickle.load(f)
    except Exception:
        use_cache = False
        X_all = None

if not use_cache:
    def _fit_vec(vec, texts):
        vec.fit(texts)
        return vec

    with ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(_fit_vec, Vectorize, all_text)
        f2 = ex.submit(_fit_vec, Vectorize_char, all_text)
        Vectorize = f1.result()
        Vectorize_char = f2.result()

    train_text_str = train_text.astype(str, copy=False)
    test_text_str = test_text.astype(str, copy=False)

    def _transform_pair(vec, tr, te):
        return vec.transform(tr), vec.transform(te)

    with ThreadPoolExecutor(max_workers=2) as ex:
        fw = ex.submit(_transform_pair, Vectorize, train_text_str, test_text_str)
        fc = ex.submit(_transform_pair, Vectorize_char, train_text_str, test_text_str)
        X_word_train, X_word_test = fw.result()
        X_char_train, X_char_test = fc.result()

    X_train = hstack([X_word_train, X_char_train], format="csr")
    X_test = hstack([X_word_test, X_char_test], format="csr")

    X_train.eliminate_zeros()
    X_test.eliminate_zeros()
    X_train.sort_indices()
    X_test.sort_indices()

    if X_train.dtype != np.float32:
        X_train = X_train.astype(np.float32)
    if X_test.dtype != np.float32:
        X_test = X_test.astype(np.float32)

    X_all = sp.vstack([X_train, X_test], format="csr")
    del X_word_train, X_word_test, X_char_train, X_char_test, X_train, X_test

    sp.save_npz(cache_all_npz, X_all, compressed=True)
    with open(cache_vec_word_pkl, "wb") as f:
        pickle.dump(Vectorize, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(cache_vec_char_pkl, "wb") as f:
        pickle.dump(Vectorize_char, f, protocol=pickle.HIGHEST_PROTOCOL)
    meta = {
        "n_train": len(df_comment),
        "n_test": len(test_comment),
        "vec_word_params": Vectorize.get_params(),
        "vec_char_params": Vectorize_char.get_params(),
        "all_shape": tuple(X_all.shape),
    }
    with open(cache_meta_pkl, "wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)

n_train = len(df_comment)
X = X_all[:n_train]
test_X = X_all[n_train:]

if not sp.isspmatrix_csr(X):
    X = X.tocsr()
if not sp.isspmatrix_csr(test_X):
    test_X = test_X.tocsr()
if X.dtype != np.float32:
    X = X.astype(np.float32)
if test_X.dtype != np.float32:
    test_X = test_X.astype(np.float32)
X.sort_indices()
test_X.sort_indices()

print("X:", X.shape, "y:", y.shape, "positive rate:", float(y.mean()))
print("test_X:", test_X.shape)



## === cell 6
from sklearn.linear_model import LogisticRegression



## === cell 7
pass



## === cell 8
import hashlib

lr_cache_pkl = os.path.join(cache_dir, "lr_full_cache_v4.pkl")
lr_meta_pkl = os.path.join(cache_dir, "lr_full_cache_v4_meta.pkl")

lr_full = LogisticRegression(
    C=4,
    dual=False,
    n_jobs=-1,
    solver="saga",
    max_iter=1000,
    class_weight="balanced",
)


def _y_signature(y_arr: np.ndarray):
    y_arr = np.asarray(y_arr)
    return (int(y_arr.size), int(y_arr.sum()), int((y_arr == 0).sum()))


use_lr_cache = os.path.exists(lr_cache_pkl) and os.path.exists(lr_meta_pkl)
if use_lr_cache:
    try:
        with open(lr_meta_pkl, "rb") as f:
            lr_meta = pickle.load(f)
        ok = (
            lr_meta.get("lr_params") == lr_full.get_params()
            and lr_meta.get("X_shape") == tuple(X.shape)
            and lr_meta.get("y_sig") == _y_signature(y)
            and lr_meta.get("tfidf_meta") == (meta if meta is not None else None)
        )
        if ok:
            with open(lr_cache_pkl, "rb") as f:
                lr_full = pickle.load(f)
        else:
            use_lr_cache = False
    except Exception:
        use_lr_cache = False

if not use_lr_cache:
    lr_full.fit(X, y)
    with open(lr_cache_pkl, "wb") as f:
        pickle.dump(lr_full, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(lr_meta_pkl, "wb") as f:
        pickle.dump(
            {
                "lr_params": lr_full.get_params(),
                "X_shape": tuple(X.shape),
                "y_sig": _y_signature(y),
                "tfidf_meta": (meta if meta is not None else None),
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )



## === cell 9
predictions = lr_full.predict_proba(test_X)[:, 1]



## === cell 10
pass



## === cell 11
if "id" not in sub.columns or "id" not in test_df.columns:
    raise ValueError("Expected 'id' column in both sample submission and test data.")

pred_df = pd.DataFrame({"id": test_df["id"].values, "prediction": predictions})
sub_out = sub[["id"]].merge(pred_df, on="id", how="left")

if sub_out["prediction"].isna().any():
    missing = int(sub_out["prediction"].isna().sum())
    raise ValueError(f"Missing predictions after id-merge: {missing} rows.")

sub_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_out.shape)
print(sub_out.head(15))
