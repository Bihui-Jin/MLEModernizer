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
Given questions and answers from various StackExchange properties, predict target values of 30 labels for each question-answer pair.

## Metric
Mean column-wise Spearman's correlation coefficient. The Spearman's rank correlation is computed for each target column, and the mean of these values is calculated for the submission score.

## Submission Format
For each qa_id in the test set, you must predict a probability for each target variable. The predictions should be in the range [0,1]. The file should contain a header and have the following format:

```
qa_id,question_asker_intent_understanding,...,answer_well_written
6,0.0,...,0.5
8,0.5,...,0.1
18,1.0,...,0.0
etc.
```

## Dataset
The list of 30 target labels are the same as the column names in the `sample_submission.csv` file. Target labels with the prefix `question_` relate to the `question_title` and/or `question_body` features in the data. Target labels with the prefix `answer_` relate to the `answer` feature.

Target labels are aggregated from multiple raters, and can have continuous values in the range `[0,1]`. Therefore, predictions must also be in that range.

- **train.csv** - the training data (target labels are the last 30 columns)
- **test.csv** - the test set (you must predict 30 labels for each test set row)
- **sample_submission.csv** - a sample submission file in the correct format; column names are the 30 target labels

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
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
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (83 lines)
            sample_submission.csv (609 lines)
            sample_submission.csv.zip (8.9 kB)
            test.csv (19551 lines)
            test.csv.zip (471.7 kB)
            train.csv (159837 lines)
            train.csv.zip (4.2 MB)
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
        input/
            description.md (83 lines)
            sample_submission.csv (609 lines)
            sample_submission.csv.zip (8.9 kB)
            test.csv (19551 lines)
            test.csv.zip (471.7 kB)
            train.csv (159837 lines)
            train.csv.zip (4.2 MB)
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
        working/
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
```

-> data/google-quest-challenge/sample_submission.csv has 608 rows and 31 columns.
The columns are: qa_id, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer, question_fact_seeking, question_has_commonly_accepted_answer, question_interestingness_others, question_interestingness_self, question_multi_intent, question_not_really_a_question, question_opinion_seeking, question_type_choice, question_type_compare, question_type_consequence... and 16 more columns

-> data/google-quest-challenge/test.csv has 19550 rows and 11 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host

-> data/google-quest-challenge/train.csv has 159836 rows and 41 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer... and 26 more columns

-> data/sample_submission.csv has 608 rows and 31 columns.
The columns are: qa_id, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer, question_fact_seeking, question_has_commonly_accepted_answer, question_interestingness_others, question_interestingness_self, question_multi_intent, question_not_really_a_question, question_opinion_seeking, question_type_choice, question_type_compare, question_type_consequence... and 16 more columns

-> data/test.csv has 19550 rows and 11 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host

-> data/train.csv has 159836 rows and 41 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer... and 26 more columns

-> (stopped after 10 files for performance)

# 5. Target score

-0.00164

# 6. Current score

0.03428

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.02955) has done: 'I fix the immediate import/runtime errors by removing the incorrect `cross_val_score` import, restoring missing imports, and preventing name shadowing where `model` was both a function and a variable. I also fix the preprocessing so it can run on both train and test consistently: label encoding be done with a stable mapping learned on train and applied to test, and TF‑IDF be fit on train then transformed on test. To avoid shape mismatches and hardcoded `input_dim`, the network’s input dimension be set from the actual TF‑IDF feature matrix, while keeping the same core 2‑Dense-layer architecture, loss, optimizer, epochs, and batch size. Finally, I generate `submission.csv` with the exact columns and row order from `sample_submission.csv`, ensuring predictions are clipped to `[0,1]`.'
- What this solution (achieved 0.03296) has done: 'The crash happens before any model code runs due to a known protobuf/TensorFlow import incompatibility in some Kaggle images; importing TensorFlow triggers `MessageFactory.GetPrototype` errors. The minimal fix is to force the pure-Python protobuf implementation via an environment variable before importing TensorFlow/Keras. I also update the cell numbering to start at 1 (Kaggle “cells” format) while keeping the model/training/prediction logic identical so your score behavior stays essentially unchanged. Finally, the script still write `submission.csv` with the correct columns/order and clipped predictions in `[0,1]`.'
- What this solution (achieved 0.01674) has done: 'I fix the immediate crash by setting the protobuf environment variables before TensorFlow/Keras import and forcing protobuf to the pure-Python implementation, which avoids the `MessageFactory.GetPrototype` incompatibility seen in some Kaggle images. I also add a small, safe fallback: if TensorFlow still fails to import, the code switch to the scikit-learn MLPRegressor with the same basic 2-layer dense structure and sigmoid-like output clipping—this guarantees an end-to-end run and a valid `submission.csv`. The submission creation be simplified to directly follow `test.csv` row order (by `qa_id`) to avoid any accidental misalignment. These changes are primarily stability/correctness; they should keep score behavior similar or slightly improved by ensuring consistent alignment and successful training/prediction.'
- What this solution (achieved 0.01441) has done: 'I fix the TensorFlow/protobuf crash that prevents the notebook from running by ensuring the protobuf environment variables are set before any TensorFlow/Keras import and by explicitly disabling the C++ protobuf implementation at runtime. This should make `TF_AVAILABLE` correctly fall back to the sklearn model only when TensorFlow truly can’t be imported, restoring end-to-end execution and consistent submission generation. I also adjust the cell numbering to start at 1 (your current script starts at cell 0), which avoids format issues. These changes are stability-focused and should keep the modeling/training logic and score behavior essentially the same while guaranteeing a valid `submission.csv`.'
- What this solution (achieved 0.03428) has done: 'We fix the immediate runtime crash in the first cell caused by the protobuf `MessageFactory.GetPrototype` incompatibility by ensuring the pure-Python protobuf implementation is forced *before* any TensorFlow-related import and by additionally setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` unconditionally (not via `setdefault`). To keep core logic intact and avoid score-changing edits, we won’t alter the model/training setup; instead we make the TensorFlow import robust: if TF still fails, we reliably fall back to the existing sklearn MLP path so the notebook always runs end-to-end. We also remove the `/kaggle/input` directory walk (it can be slow/noisy and isn’t needed) to reduce startup overhead and potential timeouts, without affecting training/predictions. Finally, we renumber cells to start at 1 (your required format) and keep submission creation unchanged so it always writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from scipy.stats import spearmanr

SEED = 1
np.random.seed(SEED)

TF_AVAILABLE = True
TF_IMPORT_ERROR = None
try:
    import tensorflow as tf

    tf.random.set_seed(SEED)

    from keras import Sequential
    from keras.layers import Dense
except Exception as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = repr(e)

print("TF_AVAILABLE:", TF_AVAILABLE)
if not TF_AVAILABLE:
    print("TensorFlow import error:", TF_IMPORT_ERROR)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_path = "/kaggle/input/google-quest-challenge/train.csv"
test_path = "/kaggle/input/google-quest-challenge/test.csv"
sample_path = "/kaggle/input/google-quest-challenge/sample_submission.csv"

data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

data.head()



## === cell 2
TARGET_COLS = [c for c in sample_sub.columns if c != "qa_id"]

FEATURE_COLS = [
    "qa_id",
    "question_title",
    "question_body",
    "question_user_name",
    "question_user_page",
    "answer",
    "answer_user_name",
    "answer_user_page",
    "url",
    "category",
    "host",
]

assert set(FEATURE_COLS).issubset(set(data.columns))
assert set(TARGET_COLS).issubset(set(data.columns))
assert set(FEATURE_COLS).issubset(set(test_data.columns))



## === cell 3
CATEGORICAL_COLS = [
    "question_title",
    "question_user_name",
    "question_user_page",
    "answer_user_name",
    "answer_user_page",
    "url",
    "category",
    "host",
]
TEXT_COLS = ["question_body", "answer"]


def fit_category_maps(df, cat_cols):
    maps = {}
    for col in cat_cols:
        s = df[col].astype(str).fillna("__nan__")
        uniques = pd.Index(s.unique())
        maps[col] = {v: i for i, v in enumerate(uniques)}
    return maps


def transform_with_maps(df, cat_cols, maps):
    out = df.copy()
    for col in cat_cols:
        s = out[col].astype(str).fillna("__nan__")
        mp = maps[col]
        unk_id = len(mp)  # unseen category
        out[col] = s.map(mp).fillna(unk_id).astype(np.int64)
    return out


def scale_numeric_like(df, cols, scaler=None):
    X = df[cols].astype(np.float32)
    if scaler is None:
        scaler = MinMaxScaler()
        Xs = scaler.fit_transform(X)
    else:
        Xs = scaler.transform(X)
    Xs = pd.DataFrame(Xs, columns=cols, index=df.index)
    return Xs, scaler




## === cell 4
def fit_vectorizer(train_df):
    corpus = (
        train_df["question_body"].fillna("").astype(str)
        + " "
        + train_df["answer"].fillna("").astype(str)
    )
    vec = TfidfVectorizer(
        max_features=50000,  # keep runtime reasonable
        ngram_range=(1, 2),
        stop_words="english",
    )
    vec.fit(corpus.values)
    return vec


def transform_vectorizer(vec, df):
    corpus = (
        df["question_body"].fillna("").astype(str)
        + " "
        + df["answer"].fillna("").astype(str)
    )
    X = vec.transform(corpus.values)
    return X




## === cell 5
train_feat = data[FEATURE_COLS].copy()

cat_maps = fit_category_maps(train_feat, CATEGORICAL_COLS)
train_enc = transform_with_maps(train_feat, CATEGORICAL_COLS, cat_maps)

train_scaled_cat, scaler = scale_numeric_like(train_enc, CATEGORICAL_COLS, scaler=None)

vec = fit_vectorizer(train_feat)
X_text = transform_vectorizer(vec, train_feat)

X_cat = train_scaled_cat.to_numpy(dtype=np.float32)
X = np.hstack([X_text.toarray().astype(np.float32), X_cat]).astype(np.float32)

y = data[TARGET_COLS].astype(np.float32).to_numpy()

X.shape, y.shape



## === cell 6
x_train, x_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)



## === cell 7
if TF_AVAILABLE:

    def build_model(input_dim: int):
        m = Sequential()
        m.add(Dense(30, input_dim=input_dim))
        m.add(Dense(30, activation="sigmoid"))
        m.compile(loss="mse", optimizer="sgd", metrics=["mse"])
        return m

    nn = build_model(input_dim=x_train.shape[1])

    history = nn.fit(
        x_train,
        y_train,
        epochs=40,
        batch_size=50,
        validation_data=(x_valid, y_valid),
        verbose=1,
    )
else:
    from sklearn.neural_network import MLPRegressor

    nn = MLPRegressor(
        hidden_layer_sizes=(30,),
        activation="relu",
        solver="sgd",
        learning_rate_init=0.01,
        momentum=0.0,
        max_iter=40,
        batch_size=50,
        random_state=SEED,
        verbose=True,
    )
    nn.fit(x_train, y_train)



## === cell 8
if TF_AVAILABLE:
    loss, mse = nn.evaluate(x_train, y_train, verbose=0)
    print("Training MSE: {:.6f}".format(mse))

    loss, mse = nn.evaluate(x_valid, y_valid, verbose=0)
    print("Validation MSE: {:.6f}".format(mse))

    y_pred_valid = nn.predict(x_valid, verbose=0)
else:
    from sklearn.metrics import mean_squared_error

    y_pred_train = nn.predict(x_train)
    print("Training MSE: {:.6f}".format(mean_squared_error(y_train, y_pred_train)))

    y_pred_valid = nn.predict(x_valid)
    print("Validation MSE: {:.6f}".format(mean_squared_error(y_valid, y_pred_valid)))

y_pred_valid = np.clip(np.asarray(y_pred_valid), 0.0, 1.0)
print(
    "Sanity Spearman (flattened):",
    spearmanr(y_valid.ravel(), y_pred_valid.ravel()).correlation,
)



## === cell 9
test_feat = test_data[FEATURE_COLS].copy()

test_enc = transform_with_maps(test_feat, CATEGORICAL_COLS, cat_maps)
test_scaled_cat, _ = scale_numeric_like(test_enc, CATEGORICAL_COLS, scaler=scaler)

X_text_test = transform_vectorizer(vec, test_feat)
X_cat_test = test_scaled_cat.to_numpy(dtype=np.float32)
X_test_final = np.hstack([X_text_test.toarray().astype(np.float32), X_cat_test]).astype(
    np.float32
)

X_test_final.shape



## === cell 10
if TF_AVAILABLE:
    pred = nn.predict(X_test_final, verbose=0)
else:
    pred = nn.predict(X_test_final)

pred = np.clip(np.asarray(pred, dtype=np.float32), 0.0, 1.0)

sub = pd.DataFrame(pred, columns=TARGET_COLS)
sub.insert(0, "qa_id", test_data["qa_id"].values)

sub = sub[sample_sub.columns.tolist()]
assert sub.shape[0] == test_data.shape[0]
assert sub.columns.tolist() == sample_sub.columns.tolist()

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(
    "First columns:",
    sub.columns.tolist()[:5],
    "... Last columns:",
    sub.columns.tolist()[-3:],
)
print(sub.head())
