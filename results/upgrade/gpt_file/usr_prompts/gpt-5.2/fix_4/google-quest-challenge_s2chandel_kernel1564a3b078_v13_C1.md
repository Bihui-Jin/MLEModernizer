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

0.00575

# 6. Current score

0.02209

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -0.00307) has done: 'I fix the import/runtime errors by removing the incorrect `cross_val_score` import, restoring missing imports that caused `NameError`s, and avoiding the function/name collision that turned your trained model into a Python function. I also make the preprocessing consistent between train and test by fitting encoders/scalers/TF-IDF on train and reusing them on test, which is required for correct shapes and a valid submission. To keep core logic intact, the model remains a simple two-layer dense network trained with MSE, but I set `input_dim` dynamically from the vectorized feature matrix instead of a hardcoded value so it runs reliably. Finally, I write `submission.csv` with the exact columns from `sample_submission.csv` and predictions clipped to `[0,1]`.'
- What this solution (achieved 0.06404) has done: 'I fix the immediate runtime crash caused by an upstream protobuf/TensorFlow incompatibility that is triggered when importing Keras; the safest minimal workaround in this environment is to force the pure-Python protobuf implementation before any TensorFlow/Keras import occurs. I also make the preprocessing consistent and stable by excluding the free-text columns (`question_title`) from label encoding/scaling (since you already model text via TF‑IDF on `question_body` and `answer`), which avoids injecting arbitrary integer IDs for huge-vocabulary text and should nudge Spearman correlation upward without changing the core model/training loop. Finally, I keep the exact same model architecture/training setup and ensure a valid `submission.csv` is written with the sample’s column order and predictions clipped to `[0,1]`.'
- What this solution (achieved 0.02209) has done: 'We fix the immediate crash by moving the protobuf/TensorFlow/Keras compatibility workaround to the very top and forcing the pure-Python protobuf implementation before anything that might import TensorFlow/Keras transitively. We also add a safe fallback to `tf_keras` if importing `keras` still fails in this environment, without changing your model architecture or training loop. Additionally, we correct the notebook cell numbering to start at 1 (your current script starts at cell 0) to match the required execution format and avoid tooling issues. No score-changing modeling changes are introduced; the goal is to restore end-to-end execution and a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

try:
    from keras import Sequential
    from keras.layers import Dense
except Exception:
    from tf_keras import Sequential
    from tf_keras.layers import Dense

from scipy.stats import spearmanr
from scipy.sparse import hstack, csr_matrix

np.random.seed(1)

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))



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
sample = pd.read_csv(sample_path)

target_cols = [c for c in sample.columns if c != "qa_id"]
assert len(target_cols) == 30, "Expected 30 target columns from sample_submission.csv"

data.head()



## === cell 2
feature_cols = [
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

X_raw = data[feature_cols].copy()
y = data[target_cols].copy()

X_raw.head()




## === cell 3
class MultiColumnLabelEncoder:
    def __init__(self, cols):
        self.cols = cols
        self.encoders = {c: LabelEncoder() for c in cols}
        self.known_classes_ = {}

    def fit(self, df):
        for c in self.cols:
            s = df[c].astype(str).fillna("")
            self.encoders[c].fit(s)
            self.known_classes_[c] = set(self.encoders[c].classes_.tolist())
        return self

    def transform(self, df):
        out = df.copy()
        for c in self.cols:
            s = out[c].astype(str).fillna("")
            unk = "__UNK__"
            if unk not in self.known_classes_[c]:
                classes = self.encoders[c].classes_.tolist()
                classes.append(unk)
                self.encoders[c].classes_ = np.array(classes, dtype=object)
                self.known_classes_[c].add(unk)

            known = self.known_classes_[c]
            s = s.map(lambda v: v if v in known else unk)
            out[c] = self.encoders[c].transform(s)
        return out


cat_cols = [
    "question_user_name",
    "question_user_page",
    "answer_user_name",
    "answer_user_page",
    "url",
    "category",
    "host",
]

le = MultiColumnLabelEncoder(cat_cols).fit(X_raw)




## === cell 4
def encoder_transform(df, le_obj):
    out = df[feature_cols].copy()
    out = le_obj.transform(out)
    return out


X_enc = encoder_transform(X_raw, le)
X_enc.head()



## === cell 5
scaler_cols = [
    "question_user_name",
    "question_user_page",
    "answer_user_name",
    "answer_user_page",
    "url",
    "category",
    "host",
]

scaler = MinMaxScaler()
scaler.fit(X_enc[scaler_cols])


def scale_transform(df_enc, scaler_obj):
    out = df_enc.copy()
    scaled = pd.DataFrame(
        scaler_obj.transform(out[scaler_cols]), columns=scaler_cols, index=out.index
    )
    out = out.drop(columns=scaler_cols)
    out = pd.concat([out, scaled], axis=1)
    out = out.drop(columns="qa_id")
    return out


X_scaled = scale_transform(X_enc, scaler)
X_scaled.head()



## === cell 6
tfidf_q = TfidfVectorizer(max_features=50000)
tfidf_a = TfidfVectorizer(max_features=50000)

tfidf_q.fit(X_raw["question_body"].astype(str).fillna(""))
tfidf_a.fit(X_raw["answer"].astype(str).fillna(""))


def vectorize(df_raw, df_scaled, tfidf_q_obj, tfidf_a_obj):
    qb = df_raw["question_body"].astype(str).fillna("")
    ans = df_raw["answer"].astype(str).fillna("")
    Xq = tfidf_q_obj.transform(qb)
    Xa = tfidf_a_obj.transform(ans)

    numeric = (
        df_scaled.drop(columns=["question_body", "answer", "question_title"])
        .astype(np.float32)
        .to_numpy()
    )
    return numeric, Xq, Xa


X_num, Xq, Xa = vectorize(X_raw, X_scaled, tfidf_q, tfidf_a)

X_all_sparse = hstack([csr_matrix(X_num), Xq, Xa], format="csr")
X_all = X_all_sparse.toarray().astype(np.float32)

X_all.shape



## === cell 7
x_train, x_valid, y_train, y_valid = train_test_split(
    X_all, y.values.astype(np.float32), random_state=1
)

input_dim = x_train.shape[1]
input_dim




## === cell 8
def build_model(input_dim_):
    m = Sequential()
    m.add(Dense(30, input_dim=input_dim_))
    m.add(Dense(30, activation="sigmoid"))
    m.compile(loss="mse", optimizer="sgd", metrics=["mse"])
    return m


model = build_model(input_dim)

history = model.fit(
    x_train,
    y_train,
    epochs=20,
    batch_size=200,
    validation_data=(x_valid, y_valid),
    verbose=1,
)



## === cell 9
loss_tr, mse_tr = model.evaluate(x_train, y_train, verbose=0)
print(f"Training MSE: {mse_tr:.6f}")

loss_va, mse_va = model.evaluate(x_valid, y_valid, verbose=0)
print(f"Validation MSE: {mse_va:.6f}")

y_pred_valid = model.predict(x_valid, verbose=0)

col_spearmans = []
for j in range(len(target_cols)):
    r = spearmanr(y_valid[:, j], y_pred_valid[:, j]).correlation
    col_spearmans.append(0.0 if np.isnan(r) else r)
print(f"Mean column-wise Spearman (valid): {np.mean(col_spearmans):.6f}")



## === cell 10
X_test_raw = test_data[feature_cols].copy()

X_test_enc = encoder_transform(X_test_raw, le)
X_test_scaled = scale_transform(X_test_enc, scaler)

X_test_num, X_test_q, X_test_a = vectorize(X_test_raw, X_test_scaled, tfidf_q, tfidf_a)
X_test_all_sparse = hstack([csr_matrix(X_test_num), X_test_q, X_test_a], format="csr")
X_test_all = X_test_all_sparse.toarray().astype(np.float32)

X_test_all.shape



## === cell 11
pred = model.predict(X_test_all, verbose=0)
pred = np.clip(pred, 0.0, 1.0)

submission = pd.DataFrame(pred, columns=target_cols)
submission.insert(0, "qa_id", test_data["qa_id"].values)

submission = submission[sample.columns.tolist()]
submission.head()



## === cell 12
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.columns.tolist()[:5], "...", submission.columns.tolist()[-5:])
