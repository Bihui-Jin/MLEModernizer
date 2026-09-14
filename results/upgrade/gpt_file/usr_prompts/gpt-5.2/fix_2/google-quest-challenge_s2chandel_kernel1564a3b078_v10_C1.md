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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
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
    encoders = {c: LabelEncoder() for c in cat_cols}

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



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/sklearn/utils/_encode.py in _encode(values, uniques, check_unknown)
    223         try:
--> 224             return _map_to_integer(values, uniques)
    225         except KeyError as e:

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_encode.py in _map_to_integer(values, uniques)
    163     table = _nandict({val: i for i, val in enumerate(uniques)})
--> 164     return np.array([table[v] for v in values])
    165 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_encode.py in <listcomp>(.0)
    163     table = _nandict({val: i for i, val in enumerate(uniques)})
--> 164     return np.array([table[v] for v in values])
    165 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_encode.py in __missing__(self, key)
    157             return self.nan_value
--> 158         raise KeyError(key)
    159 

KeyError: 'What is a word for somebody who lies to themselves'

During handling of the above exception, another exception occurred:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3617603885.py in <cell line: 0>()
      3 
      4 # Transform test using train-fitted preprocessors (critical for correctness and score)
----> 5 X_test_enc = transform_encoder(enc_mapper, test_data)
      6 X_test_scaled = transform_scale(scaler, X_test_enc)
      7 

/tmp/ipykernel_11/3901187837.py in transform_encoder(mapper, df)
     36 
     37 def transform_encoder(mapper, df):
---> 38     X = mapper.transform(df)
     39     # Ensure columns order is same and names exist
     40     if not isinstance(X, pd.DataFrame):

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn_pandas/dataframe_mapper.py in transform(self, X)
    430         X       the data to transform
    431         """
--> 432         return self._transform(X)
    433 
    434     def fit_transform(self, X, y=None):

/usr/local/lib/python3.11/dist-packages/sklearn_pandas/dataframe_mapper.py in _transform(self, X, y, do_fit)
    350 
    351                         t1 = datetime.now()
--> 352                         Xt = transformers.transform(Xt)
    353                         logger.info(f"[TRANSFORM] {columns}: {_elapsed_secs(t1)} secs")  # NOQA
    354 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_label.py in transform(self, y)
    137             return np.array([])
    138 
--> 139         return _encode(y, uniques=self.classes_)
    140 
    141     def inverse_transform(self, y):

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_encode.py in _encode(values, uniques, check_unknown)
    224             return _map_to_integer(values, uniques)
    225         except KeyError as e:
--> 226             raise ValueError(f"y contains previously unseen labels: {str(e)}")
    227     else:
    228         if check_unknown:

ValueError: question_title: y contains previously unseen labels: 'What is a word for somebody who lies to themselves'

## === cell 11
pred = nn_model.predict(X_test, verbose=0).astype(np.float32)
pred = np.clip(pred, 0.0, 1.0)

submission = pd.DataFrame(pred, columns=target_cols)
submission.insert(0, "qa_id", test_ids.values)

sample_sub = pd.read_csv("/kaggle/input/google-quest-challenge/sample_submission.csv")
submission = submission[sample_sub.columns]

submission.head()



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/226428662.py in <cell line: 0>()
----> 1 pred = nn_model.predict(X_test, verbose=0).astype(np.float32)
      2 pred = np.clip(pred, 0.0, 1.0)
      3 
      4 submission = pd.DataFrame(pred, columns=target_cols)
      5 submission.insert(0, "qa_id", test_ids.values)

NameError: name 'X_test' is not defined

## === cell 12
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.describe().T[["min", "max", "mean"]].head())

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1504556993.py in <cell line: 0>()
----> 1 submission.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with shape:", submission.shape)
      3 print(submission.describe().T[["min", "max", "mean"]].head())

NameError: name 'submission' is not defined
