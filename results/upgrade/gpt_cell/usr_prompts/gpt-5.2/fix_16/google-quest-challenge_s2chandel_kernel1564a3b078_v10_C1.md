# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.01688

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.02333) has done: 'Diagnosis: Cell 9 hard-codes `input_dim=82044`, but the TF-IDF feature matrix created earlier has `x_train.shape[1] == 76392`, so Keras raises an input-shape mismatch when calling the first `Dense` layer. The root cause is that TF-IDF vocabulary size (and thus feature count) is data-dependent and not guaranteed to equal 82044.  
Patch summary: In cell 9 only, replace the hard-coded `input_dim` with the actual number of columns in `x_train` (falling back to `x.shape[1]` if needed) so the model input matches the produced features. Keep the same model architecture, loss, optimizer, metrics, and training call.  
Updated cells: Only cell 9 is modified.  
Compatibility notes for cell k+1: The variables `model` and `history` are still created with the same types/semantics, so cell 10 run unchanged.  
Assumptions: `x_train` exists in scope by the time cell 9 runs (as created in cell 8), and it has a 2D shape `(n_samples, n_features)`.'
- What this solution (achieved -0.00905) has done: 'Your current score (0.02333) is above the target (0.0095), so we should make the smallest legitimate change that moves performance downward toward the target band without breaking the pipeline. The cleanest way (while keeping the same model, TF‑IDF, training loop, and submission semantics) is to reduce training epochs slightly so the model underfits more and yields a lower Spearman score. I only change the `epochs` value and also add a safe clip to keep predictions in `[0,1]` (shouldn’t materially improve score; it just ensures validity). The script still run end-to-end and write `submission.csv` with the correct columns.'
- What this solution (achieved 0.01688) has done: 'Diagnosis: The crash happens because `encoder(..., encoders=encoders)` creates a new `DataFrameMapper` and immediately calls `mapper.transform(...)` without fitting it first; `sklearn_pandas.DataFrameMapper` requires `fit()` to create internal attributes like `built_features`, otherwise `transform()` raises `AttributeError: 'DataFrameMapper' object has no attribute 'built_features'`. In earlier training code (cell 5) you explicitly do `_ = mapper.fit(...)` before `transform`, but `encoder()` doesn’t mirror that behavior when `encoders` are provided. This is an interface/usage bug, not a modeling bug. The minimal fix is to fit the mapper on the provided dataframe before transforming when `encoders` are passed.

Patch summary: Update cell 14 to fit a fresh `DataFrameMapper` on `test_data` before transforming, matching the same mapping/encoders used earlier, and then keep the rest of the pipeline (`scale` + `word_vectors.transform`) unchanged.

Updated cells: only cell 14 is changed.

Compatibility notes for cell k+1: `xtest` remains a pandas `DataFrame` (TF-IDF/transformed feature matrix) exactly as before, so `xtest.head()` in cell 15 still work.

Assumptions: Re-fitting `DataFrameMapper` on the test dataframe is safe because the categorical encoders are already fit on train+test (cell 5) and we are not refitting `word_vectors` (TF-IDF) here—only using `word_vectors.transform` as originally intended.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import pandas as pd
import numpy as np

from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras import Sequential
from keras.layers import Dense

from scipy.stats import spearmanr

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
data = pd.read_csv("/kaggle/input/google-quest-challenge/train.csv")
data.head()




## === cell 2
def _fit_label_encoders(train_df, test_df, cat_cols):
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train_df[c], test_df[c]], axis=0).astype(str).fillna("")
        le.fit(combined.values)
        encoders[c] = le
    return encoders


def encoder(data, encoders=None):
    local_fit = encoders is None
    if local_fit:
        encoders = {}
        for c in [
            "question_title",
            "question_user_name",
            "question_user_page",
            "answer_user_name",
            "answer_user_page",
            "url",
            "category",
            "host",
        ]:
            le = LabelEncoder()
            le.fit(data[c].astype(str).fillna("").values)
            encoders[c] = le

    cols = [
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
        ]
    )

    data_for_transform = data.copy()
    for c, le in encoders.items():
        data_for_transform[c] = data_for_transform[c].astype(str).fillna("")

    x = pd.DataFrame(
        (
            mapper.fit_transform(data_for_transform)
            if local_fit
            else mapper.transform(data_for_transform)
        ),
        columns=cols,
    )
    return x




## === cell 3
def scale(x):
    scaler = MinMaxScaler()

    df = x[
        [
            "question_title",
            "question_user_name",
            "question_user_page",
            "answer_user_name",
            "answer_user_page",
            "url",
            "category",
            "host",
        ]
    ]

    df = pd.DataFrame(
        scaler.fit_transform(df),
        columns=[
            "question_title",
            "question_user_name",
            "question_user_page",
            "answer_user_name",
            "answer_user_page",
            "url",
            "category",
            "host",
        ],
    )

    x = x.drop(
        columns=[
            "question_title",
            "question_user_name",
            "question_user_page",
            "answer_user_name",
            "answer_user_page",
            "url",
            "category",
            "host",
        ]
    )

    x = pd.concat([x, df], axis=1)
    x = x.drop(columns="qa_id")

    return x




## === cell 4
def word2vec(x):
    tfidf = TfidfVectorizer()

    mapper = DataFrameMapper(
        [
            ("question_title", None),
            ("question_body", tfidf),
            ("question_user_name", None),
            ("question_user_page", None),
            ("answer", tfidf),
            ("answer_user_name", None),
            ("answer_user_page", None),
            ("url", None),
            ("category", None),
            ("host", None),
        ]
    )

    vectors = mapper.fit(x)
    return vectors




## === cell 5
test_data = pd.read_csv("/kaggle/input/google-quest-challenge/test.csv")

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
encoders = _fit_label_encoders(data, test_data, cat_cols)

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
    ]
)

data_for_transform = data.copy()
for c, le in encoders.items():
    data_for_transform[c] = data_for_transform[c].astype(str).fillna("")

cols = [
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

_ = mapper.fit(data_for_transform)
x = pd.DataFrame(mapper.transform(data_for_transform), columns=cols)

x = scale(x)
word_vectors = word2vec(x)
x = pd.DataFrame(word_vectors.transform(x))

x.head()
len(x.columns)


## === cell 6
y = data[
    [
        "question_asker_intent_understanding",
        "question_body_critical",
        "question_conversational",
        "question_expect_short_answer",
        "question_fact_seeking",
        "question_has_commonly_accepted_answer",
        "question_interestingness_others",
        "question_interestingness_self",
        "question_multi_intent",
        "question_not_really_a_question",
        "question_opinion_seeking",
        "question_type_choice",
        "question_type_compare",
        "question_type_consequence",
        "question_type_definition",
        "question_type_entity",
        "question_type_instructions",
        "question_type_procedure",
        "question_type_reason_explanation",
        "question_type_spelling",
        "question_well_written",
        "answer_helpful",
        "answer_level_of_information",
        "answer_plausible",
        "answer_relevance",
        "answer_satisfaction",
        "answer_type_instructions",
        "answer_type_procedure",
        "answer_type_reason_explanation",
        "answer_well_written",
    ]
]



## === cell 7
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)




## === cell 8
def model():
    n_features = int(x_train.shape[1]) if "x_train" in globals() else int(x.shape[1])

    m = Sequential()
    m.add(Dense(30, input_dim=n_features))  # batch size 30
    m.add(Dense(30, activation="sigmoid"))  # shape 1
    m.compile(loss="mse", optimizer="sgd", metrics=["mse"])
    return m


model = model()

history = model.fit(
    x_train,
    y_train,
    epochs=3,
    batch_size=200,
    validation_data=(x_test, y_test),
    verbose=1,
)



## === cell 9
loss, mse = model.evaluate(x_train, y_train, verbose=0)
print("Training MSE: {:.4f}".format(mse))

loss, mse = model.evaluate(x_test, y_test, verbose=0)
print("Testing MSE:  {:.4f}".format(mse))

import matplotlib.pyplot as plt

plt.style.use("ggplot")


def plot_history(history):
    acc = history.history["mse"]
    val_acc = history.history["val_mse"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, "b", label="Training mse")
    plt.xlabel("epochs")
    plt.ylabel("mse")
    plt.plot(x, val_acc, "r", label="Validation mse")
    plt.title("Training and validation mse")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, "b", label="Training loss")
    plt.plot(x, val_loss, "r", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()


plot_history(history)



## === cell 10
y_pred = model.predict(x_test)



## === cell 11
spearmanr(y_test, y_pred, axis=None)



## === cell 12
test_ids = test_data["qa_id"]



## === cell 13
test_data.head()



## === cell 14
mapper_test = DataFrameMapper(
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
    ]
)

test_for_transform = test_data.copy()
for c, le in encoders.items():
    test_for_transform[c] = test_for_transform[c].astype(str).fillna("")

cols = [
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

_ = mapper_test.fit(test_for_transform)
xtest = pd.DataFrame(mapper_test.transform(test_for_transform), columns=cols)

xtest = scale(xtest)
xtest = pd.DataFrame(word_vectors.transform(xtest))


## === cell 15
xtest.head()



## === cell 16
pred = model.predict(xtest)
pred = np.clip(pred, 0.0, 1.0)



## === cell 17
predictions = pd.DataFrame(
    pred,
    columns=[
        "question_asker_intent_understanding",
        "question_body_critical",
        "question_conversational",
        "question_expect_short_answer",
        "question_fact_seeking",
        "question_has_commonly_accepted_answer",
        "question_interestingness_others",
        "question_interestingness_self",
        "question_multi_intent",
        "question_not_really_a_question",
        "question_opinion_seeking",
        "question_type_choice",
        "question_type_compare",
        "question_type_consequence",
        "question_type_definition",
        "question_type_entity",
        "question_type_instructions",
        "question_type_procedure",
        "question_type_reason_explanation",
        "question_type_spelling",
        "question_well_written",
        "answer_helpful",
        "answer_level_of_information",
        "answer_plausible",
        "answer_relevance",
        "answer_satisfaction",
        "answer_type_instructions",
        "answer_type_procedure",
        "answer_type_reason_explanation",
        "answer_well_written",
    ],
    index=test_ids,
)
predictions = predictions.reset_index()
predictions.head()



## === cell 18
predictions.describe()



## === cell 19
predictions.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", predictions.shape)
