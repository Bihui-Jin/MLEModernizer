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

0.0095

# 6. Current score

0.00109

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.00109) has done: 'I fix two execution blockers while keeping your model and training loop intact: the TensorFlow/Protobuf import crash and the unseen-label failure from `LabelEncoder` at test time. The import crash is addressed by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before importing TensorFlow/Keras, which avoids the `MessageFactory.GetPrototype` issue in this environment. The unseen-label issue is fixed by replacing `LabelEncoder` with a tiny “safe label encoder” that maps unknown categories to `-1`, preventing runtime errors and keeping preprocessing semantics as close as possible. Finally, I ensure the pipeline always reaches the CSV write step and produces `submission.csv` with the exact sample-submission column order.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd

from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras import Sequential
from keras.layers import Dense

from scipy.stats import spearmanr

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))

np.random.seed(1)
tf.random.set_seed(1)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
data = pd.read_csv("/kaggle/input/google-quest-challenge/train.csv")
data.head()




## === cell 2
class SafeLabelEncoder:
    def fit(self, y):
        s = pd.Series(y).astype(str).fillna("")
        self.class_to_int_ = {k: i for i, k in enumerate(pd.unique(s))}
        return self

    def transform(self, y):
        s = pd.Series(y).astype(str).fillna("")
        return s.map(self.class_to_int_).fillna(-1).astype(int).to_numpy()

    def fit_transform(self, y):
        return self.fit(y).transform(y)


def fit_encoder(train_df):
    cat_cols = [
        "question_title",
        "question_user_name",
        "question_user_page",
        "answer_user_name",
        "answer_user_page",
        "url",
        "category",
        "host",
    ]
    encoders = {c: SafeLabelEncoder() for c in cat_cols}

    mapper = DataFrameMapper(
        [
            ("qa_id", None),
            ("question_title", encoders["question_title"]),
            ("question_body", None),
            ("question_user_name", encoders["question_user_name"]),
            ("question_user_page", encoders["question_user_page"]),
            ("answer", None),
            ("answer_user_name", encoders["answer_user_name"]),
            ("answer_user_page", encoders["answer_user_page"]),
            ("url", encoders["url"]),
            ("category", encoders["category"]),
            ("host", encoders["host"]),
        ],
        df_out=True,
    )

    X = mapper.fit_transform(train_df)
    return mapper, X


def transform_encoder(mapper, df):
    X = mapper.transform(df)
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    return X




## === cell 3
def fit_scale(X_df):
    scaler = MinMaxScaler()
    scale_cols = [
        "question_title",
        "question_user_name",
        "question_user_page",
        "answer_user_name",
        "answer_user_page",
        "url",
        "category",
        "host",
    ]
    X_scaled_part = pd.DataFrame(
        scaler.fit_transform(X_df[scale_cols].astype(float)),
        columns=scale_cols,
        index=X_df.index,
    )
    X_rest = X_df.drop(columns=scale_cols)
    X_out = pd.concat([X_rest, X_scaled_part], axis=1).drop(columns=["qa_id"])
    return scaler, X_out


def transform_scale(scaler, X_df):
    scale_cols = [
        "question_title",
        "question_user_name",
        "question_user_page",
        "answer_user_name",
        "answer_user_page",
        "url",
        "category",
        "host",
    ]
    X_scaled_part = pd.DataFrame(
        scaler.transform(X_df[scale_cols].astype(float)),
        columns=scale_cols,
        index=X_df.index,
    )
    X_rest = X_df.drop(columns=scale_cols)
    X_out = pd.concat([X_rest, X_scaled_part], axis=1).drop(columns=["qa_id"])
    return X_out




## === cell 4
def fit_text_vectorizers(X_df, max_features=20000):
    tfidf_q = TfidfVectorizer(max_features=max_features)
    tfidf_a = TfidfVectorizer(max_features=max_features)

    q_mat = tfidf_q.fit_transform(X_df["question_body"].fillna(""))
    a_mat = tfidf_a.fit_transform(X_df["answer"].fillna(""))
    return tfidf_q, tfidf_a, q_mat, a_mat


def transform_text_vectorizers(tfidf_q, tfidf_a, X_df):
    q_mat = tfidf_q.transform(X_df["question_body"].fillna(""))
    a_mat = tfidf_a.transform(X_df["answer"].fillna(""))
    return q_mat, a_mat


def build_final_matrix(X_df_scaled, q_mat, a_mat):
    num_df = X_df_scaled.drop(columns=["question_body", "answer"])
    num_mat = num_df.to_numpy(dtype=np.float32)

    q_dense = q_mat.toarray().astype(np.float32)
    a_dense = a_mat.toarray().astype(np.float32)

    X_final = np.concatenate([num_mat, q_dense, a_dense], axis=1)
    return X_final




## === cell 5
enc_mapper, X_enc = fit_encoder(data)
scaler, X_scaled = fit_scale(X_enc)

tfidf_q, tfidf_a, q_train_mat, a_train_mat = fit_text_vectorizers(
    X_scaled, max_features=20000
)
X = build_final_matrix(X_scaled, q_train_mat, a_train_mat)

X.shape



## === cell 6
target_cols = list(
    pd.read_csv("/kaggle/input/google-quest-challenge/sample_submission.csv").columns
)
target_cols.remove("qa_id")

y = data[target_cols].copy()
y.shape



## === cell 7
x_train, x_val, y_train, y_val = train_test_split(X, y, random_state=1)
x_train.shape, x_val.shape




## === cell 8
def build_model(input_dim):
    m = Sequential()
    m.add(Dense(30, input_dim=input_dim))
    m.add(Dense(30, activation="sigmoid"))
    m.compile(loss="mse", optimizer="sgd", metrics=["mse"])
    return m


nn_model = build_model(input_dim=x_train.shape[1])

history = nn_model.fit(
    x_train,
    y_train.to_numpy(dtype=np.float32),
    epochs=10,
    batch_size=200,
    validation_data=(x_val, y_val.to_numpy(dtype=np.float32)),
    verbose=1,
)



## === cell 9
loss, mse = nn_model.evaluate(x_train, y_train.to_numpy(dtype=np.float32), verbose=0)
print("Training MSE: {:.6f}".format(mse))

loss, mse = nn_model.evaluate(x_val, y_val.to_numpy(dtype=np.float32), verbose=0)
print("Validation MSE: {:.6f}".format(mse))

y_pred_val = nn_model.predict(x_val, verbose=0)
val_spears = []
for i, c in enumerate(target_cols):
    s = spearmanr(y_val[c].values, y_pred_val[:, i]).correlation
    if np.isnan(s):
        s = 0.0
    val_spears.append(s)
print("Mean Spearman (val): {:.6f}".format(float(np.mean(val_spears))))



## === cell 10
test_data = pd.read_csv("/kaggle/input/google-quest-challenge/test.csv")
test_ids = test_data["qa_id"].copy()

X_test_enc = transform_encoder(enc_mapper, test_data)
X_test_scaled = transform_scale(scaler, X_test_enc)

q_test_mat, a_test_mat = transform_text_vectorizers(tfidf_q, tfidf_a, X_test_scaled)
X_test = build_final_matrix(X_test_scaled, q_test_mat, a_test_mat)

X_test.shape



## === cell 11
pred = nn_model.predict(X_test, verbose=0).astype(np.float32)
pred = np.clip(pred, 0.0, 1.0)

submission = pd.DataFrame(pred, columns=target_cols)
submission.insert(0, "qa_id", test_ids.values)

sample_sub = pd.read_csv("/kaggle/input/google-quest-challenge/sample_submission.csv")
submission = submission[sample_sub.columns]

submission.head()



## === cell 12
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.describe().T[["min", "max", "mean"]].head())
