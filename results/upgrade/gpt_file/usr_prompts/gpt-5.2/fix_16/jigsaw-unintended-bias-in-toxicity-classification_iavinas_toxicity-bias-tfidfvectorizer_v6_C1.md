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

# 5. Target score

0.89594

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.70307) has done: 'I remove the unused Keras/`KerasClassifier` imports that trigger the protobuf-related `MessageFactory` error, since the core model you actually train is scikit-learn Logistic Regression. Then I fix the TF-IDF step by filling missing `comment_text` values with empty strings so `TfidfVectorizer` doesn’t crash on `np.nan`. Finally, I ensure the train/test matrices passed into `LogisticRegression` are the sparse TF-IDF outputs (not the raw DataFrame), and I write a valid `submission.csv` with the required `id,prediction` columns and correct row alignment.'

# 9. Code solution

## === cell 0
import os
import gc
import pickle
import hashlib
import numpy as np
import pandas as pd

try:
    from sklearnex import patch_sklearn  # scikit-learn-intelex

    patch_sklearn()
except Exception:
    pass

from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack

print("Input dir listing:", os.listdir("../input")[:20])

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.environ.setdefault("PYTHONHASHSEED", str(RANDOM_STATE))



## === cell 1
train_path = "../input/train.csv"
test_path = "../input/test.csv"
sub_path = "../input/sample_submission.csv"

if not os.path.exists(train_path):
    train_path = "../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv"
if not os.path.exists(test_path):
    test_path = "../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv"
if not os.path.exists(sub_path):
    sub_path = "../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv"

read_kwargs = dict(
    engine="c",
    low_memory=False,
)

test_df = pd.read_csv(
    test_path,
    usecols=["id", "comment_text"],
    dtype={"id": "int64", "comment_text": "object"},
    **read_kwargs,
)
test_df["comment_text"] = test_df["comment_text"].fillna("")

test_df.head()



## === cell 2
cache_dir = "../working/tfidf_cache"
os.makedirs(cache_dir, exist_ok=True)

word_params = dict(
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(1, 2),
)
char_params = dict(
    strip_accents="unicode",
    analyzer="char",
    ngram_range=(3, 5),
)

n_features_word = 2**20  # 1,048,576
n_features_char = 2**20  # 1,048,576

hv_word = HashingVectorizer(
    **word_params,
    n_features=n_features_word,
    alternate_sign=False,  # deterministic non-negative; compatible with TF-IDF
    norm=None,  # norm handled by TfidfTransformer
    lowercase=True,
    dtype=np.float32,
)
hv_char = HashingVectorizer(
    **char_params,
    n_features=n_features_char,
    alternate_sign=False,
    norm=None,
    lowercase=True,
    dtype=np.float32,
)

tfidf_word = TfidfTransformer(
    sublinear_tf=True, norm="l2", use_idf=True, smooth_idf=True
)
tfidf_char = TfidfTransformer(
    sublinear_tf=True, norm="l2", use_idf=True, smooth_idf=True
)

param_sig = hashlib.md5(
    (
        f"hash_tfidf|word={word_params}|char={char_params}|"
        f"nfw={n_features_word}|nfc={n_features_char}|"
        f"sublinear_tf=True|norm=l2"
    ).encode("utf-8")
).hexdigest()[:12]

tfidf_path = os.path.join(cache_dir, f"tfidf_transformers_{param_sig}.pkl")
y_path = os.path.join(cache_dir, f"y_{param_sig}.npy")
model_path = os.path.join(cache_dir, f"lr_saga_model_{param_sig}.pkl")
test_pred_path = os.path.join(cache_dir, f"test_pred_{param_sig}.npy")

CHUNK_ROWS = 200_000
TRAIN_USECOLS = ["target", "comment_text"]
TRAIN_DTYPES = {"target": "float32", "comment_text": "object"}


def _iter_train_chunks():
    return pd.read_csv(
        train_path,
        usecols=TRAIN_USECOLS,
        dtype=TRAIN_DTYPES,
        chunksize=CHUNK_ROWS,
        **read_kwargs,
    )


def _as_text_array(s):
    return s.fillna("").to_numpy(dtype=object, copy=False)


def _save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _ensure_tfidf_fitted_state(t, n_features_expected: int):
    if (
        hasattr(t, "idf_")
        and t.idf_ is not None
        and np.size(t.idf_) == n_features_expected
    ):
        if not hasattr(t, "n_features_in_"):
            t.n_features_in_ = int(n_features_expected)
        t._idf_diag = None
        return True
    return False


def _tfidf_is_fitted(t, n_features_expected: int):
    ok = (
        hasattr(t, "idf_")
        and t.idf_ is not None
        and np.size(t.idf_) == n_features_expected
        and hasattr(t, "n_features_in_")
        and int(getattr(t, "n_features_in_", -1)) == int(n_features_expected)
    )
    if ok:
        t._idf_diag = None
    return ok


def _build_and_set_idf():
    df_word = np.zeros(n_features_word, dtype=np.int64)
    df_char = np.zeros(n_features_char, dtype=np.int64)
    n_docs = 0

    for i, chunk in enumerate(_iter_train_chunks(), start=1):
        text = _as_text_array(chunk["comment_text"])

        Xw_tf = hv_word.transform(text)  # CSR
        Xc_tf = hv_char.transform(text)

        df_word += np.bincount(Xw_tf.indices, minlength=n_features_word)
        df_char += np.bincount(Xc_tf.indices, minlength=n_features_char)
        n_docs += Xw_tf.shape[0]

        del chunk, text, Xw_tf, Xc_tf
        if i % 5 == 0:
            gc.collect()

    tfidf_word.idf_ = (np.log((n_docs + 1.0) / (df_word + 1.0)) + 1.0).astype(
        np.float64
    )
    tfidf_char.idf_ = (np.log((n_docs + 1.0) / (df_char + 1.0)) + 1.0).astype(
        np.float64
    )

    tfidf_word.n_features_in_ = int(n_features_word)
    tfidf_char.n_features_in_ = int(n_features_char)
    tfidf_word._idf_diag = None
    tfidf_char._idf_diag = None

    _save_pickle((tfidf_word, tfidf_char), tfidf_path)

    del df_word, df_char
    gc.collect()


if os.path.exists(test_pred_path):
    pre = np.load(test_pred_path)
    print("Loaded cached test predictions:", pre.shape)
else:
    if os.path.exists(tfidf_path):
        try:
            tfidf_word, tfidf_char = _load_pickle(tfidf_path)
        except Exception:
            tfidf_word = TfidfTransformer(
                sublinear_tf=True, norm="l2", use_idf=True, smooth_idf=True
            )
            tfidf_char = TfidfTransformer(
                sublinear_tf=True, norm="l2", use_idf=True, smooth_idf=True
            )

    _ensure_tfidf_fitted_state(tfidf_word, n_features_word)
    _ensure_tfidf_fitted_state(tfidf_char, n_features_char)
    if (not _tfidf_is_fitted(tfidf_word, n_features_word)) or (
        not _tfidf_is_fitted(tfidf_char, n_features_char)
    ):
        print("TF-IDF transformers not fitted; building idf_ from training data...")
        _build_and_set_idf()

    if os.path.exists(tfidf_path):
        tfidf_word, tfidf_char = _load_pickle(tfidf_path)
        _ensure_tfidf_fitted_state(tfidf_word, n_features_word)
        _ensure_tfidf_fitted_state(tfidf_char, n_features_char)

    if os.path.exists(model_path) and os.path.exists(tfidf_path):
        clf = _load_pickle(model_path)
    else:
        if os.path.exists(y_path):
            y = np.load(y_path)
        else:
            y_chunks = []
            for i, chunk in enumerate(_iter_train_chunks(), start=1):
                target = chunk["target"].to_numpy(copy=False)
                y_chunks.append((target >= 0.5).astype(np.int32, copy=False))
                del chunk, target
                if i % 10 == 0:
                    gc.collect()
            y = np.concatenate(y_chunks, axis=0)
            np.save(y_path, y)
            del y_chunks
            gc.collect()

        clf = LogisticRegression(
            solver="saga",
            penalty="l2",
            C=4.0,
            class_weight="balanced",
            max_iter=1,  # one iteration per outer loop
            random_state=RANDOM_STATE,
            n_jobs=-1,
            warm_start=True,
            verbose=0,
        )

        total_epochs = 200

        for epoch in range(total_epochs):
            offset = 0
            for i, chunk in enumerate(_iter_train_chunks(), start=1):
                text = _as_text_array(chunk["comment_text"])
                y_chunk = y[offset : offset + len(chunk)]
                offset += len(chunk)

                Xw_tf = hv_word.transform(text)
                Xc_tf = hv_char.transform(text)

                Xw = tfidf_word.transform(Xw_tf)
                Xc = tfidf_char.transform(Xc_tf)
                X_batch = hstack((Xw, Xc), format="csr", dtype=np.float32)

                clf.fit(X_batch, y_chunk)

                del chunk, text, y_chunk, Xw_tf, Xc_tf, Xw, Xc, X_batch
                if i % 5 == 0:
                    gc.collect()
            gc.collect()

        _save_pickle(clf, model_path)

    test_pred_chunks = []
    TEST_CHUNK = 200_000
    for start in range(0, len(test_df), TEST_CHUNK):
        end = min(len(test_df), start + TEST_CHUNK)
        text = test_df["comment_text"].iloc[start:end]
        text = _as_text_array(text)

        Xw_tf = hv_word.transform(text)
        Xc_tf = hv_char.transform(text)
        Xw = tfidf_word.transform(Xw_tf)
        Xc = tfidf_char.transform(Xc_tf)
        X_batch = hstack((Xw, Xc), format="csr", dtype=np.float32)

        proba = clf.predict_proba(X_batch)
        test_pred_chunks.append(proba)

        del text, Xw_tf, Xc_tf, Xw, Xc, X_batch, proba
        gc.collect()

    pre = np.vstack(test_pred_chunks)
    np.save(test_pred_path, pre)
    del test_pred_chunks
    gc.collect()

print(
    "pred proba shape:",
    pre.shape,
    "min/max:",
    float(pre[:, 1].min()),
    float(pre[:, 1].max()),
)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/393495622.py in <cell line: 0>()
    227                 Xc_tf = hv_char.transform(text)
    228 
--> 229                 Xw = tfidf_word.transform(Xw_tf)
    230                 Xc = tfidf_char.transform(Xc_tf)
    231                 X_batch = hstack((Xw, Xc), format="csr", dtype=np.float32)

/usr/local/lib/python3.11/dist-packages/sklearn/feature_extraction/text.py in transform(self, X, copy)
   1722             # does not work as usual and we need to specify the attribute
   1723             # name:
-> 1724             check_is_fitted(self, attributes=["idf_"], msg="idf vector is not fitted")
   1725 
   1726             # *= doesn't work

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_is_fitted(estimator, attributes, msg, all_or_any)
   1388 
   1389     if not fitted:
-> 1390         raise NotFittedError(msg % {"name": type(estimator).__name__})
   1391 
   1392 

NotFittedError: idf vector is not fitted

## === cell 3
pass



## === cell 4
sub = pd.read_csv(sub_path)
print("Sample submission shape:", sub.shape, "columns:", list(sub.columns))



## === cell 5
submission = pd.DataFrame(
    {
        "id": test_df["id"].to_numpy(copy=False),
        "prediction": pre[:, 1].astype(np.float32, copy=False),
    }
)

if len(sub) != len(submission) or not np.array_equal(
    sub["id"].to_numpy(copy=False), submission["id"].to_numpy(copy=False)
):
    sub_ids = sub["id"].to_numpy(copy=False)
    submission = submission.set_index("id").reindex(sub_ids).reset_index()
    submission["prediction"] = submission["prediction"].fillna(0.0)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/347133694.py in <cell line: 0>()
      2     {
      3         "id": test_df["id"].to_numpy(copy=False),
----> 4         "prediction": pre[:, 1].astype(np.float32, copy=False),
      5     }
      6 )

NameError: name 'pre' is not defined
