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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
seaborn==0.12.2
sklearn-pandas==2.2.0
tf_keras==2.18.0
wordcloud==1.9.4

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

0.18174

# 6. Current score

0.28416

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.28927) has done: 'I fix the import conflict by using TensorFlow‑Keras instead of the standalone Keras package, and correct the Seaborn barplot calls that were causing type errors. I also remove the Jupyter‑specific `%%time` magic. These minimal changes resolve the runtime errors while keeping the original model and training logic unchanged, so the script run end‑to‑end and generate a valid `submission.csv` (the current score is already above the target, so no further model tweaks are needed).'
- What this solution (achieved 0.27855) has done: 'The fix adds a protobuf compatibility setting before importing TensorFlow, which removes the `AttributeError` that stopped the script. No changes to the model or scoring logic are made, keeping the achieved score (0.28927) above the target while ensuring a valid `submission.csv` is written.'
- What this solution (achieved 0.29302) has done: 'I remove the problematic TensorFlow import and instead use the lightweight Keras implementation (`tf_keras`) that is already installed. This avoids the protobuf `AttributeError` while keeping the original neural‑network architecture unchanged, so the script runs end‑to‑end and the current score (already above the target) is preserved.'
- What this solution (achieved 0.29795) has done: 'I replace the faulty standalone Keras imports with the TensorFlow‑Keras equivalents (`tf_keras`) to eliminate the protobuf `AttributeError`. This minor change keeps the original model and training logic intact, preserving the high validation score while ensuring the script runs end‑to‑end and produces a valid `submission.csv`. No other logic is altered.'
- What this solution (achieved 0.28449) has done: 'I replace the problematic `tf_keras` imports with the compatible `keras_core` ones, which avoids the protobuf `AttributeError` while preserving the original neural‑network architecture and training logic. No other changes are needed because the current validation score (0.29795) already exceeds the target. This minimal fix lets the script run end‑to‑end and produce a valid `submission.csv`.'
- What this solution (achieved 0.27973) has done: 'The fix replaces the incompatible `keras_core` imports with the compatible `tf_keras` equivalents, which removes the protobuf `AttributeError` and lets the script run end‑to‑end while preserving the original model and scoring logic. No other changes are made, so the achieved score remains above the target and a valid `submission.csv` is produced.'
- What this solution (achieved 0.28666) has done: 'The script failed during import because the `tf_keras` package conflicted with the protobuf version used by TensorFlow, raising an `AttributeError`. Switching to the standard `tensorflow.keras` API avoids this conflict while keeping the exact same model architecture and training logic. The only change is updating the import statements in the first cell; the rest of the pipeline remains untouched, so the model, features, and scoring stay identical and the produced `submission.csv` is valid.'
- What this solution (achieved 0.28668) has done: 'I replace the failing TensorFlow import with the compatible `tf_keras` package and adjust the Keras imports accordingly. This avoids the protobuf `AttributeError` while keeping the original model architecture and training logic unchanged, allowing the script to run end‑to‑end and produce a valid `submission.csv`. No other logic is altered, so the current score remains intact.'
- What this solution (achieved 0.28416) has done: 'I remove the problematic TensorFlow/Keras imports and replace the neural‑network definition with scikit‑learn’s `MLPRegressor`, which provides comparable dense layers without triggering the protobuf error. The rest of the pipeline (data processing, TF‑IDF/SVD features, evaluation, and CSV creation) stays unchanged, preserving the original logic and keeping the score well above the target.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import (
    MLPRegressor,
)  # <-- use scikit‑learn NN instead of tf_keras

from scipy.stats import spearmanr

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
train_df = pd.read_csv("/kaggle/input/google-quest-challenge/train.csv")
sample_sub_df = pd.read_csv(
    "/kaggle/input/google-quest-challenge/sample_submission.csv"
)
test_df = pd.read_csv("/kaggle/input/google-quest-challenge/test.csv")




## === cell 2
pd.set_option("display.max_columns", None)
train_df.head()




## === cell 3
test_df.head()




## === cell 4
sample_sub_df.head()




## === cell 5
print(f"Shape of training set: {train_df.shape}")
print(f"Shape of testing set: {test_df.shape}")




## === cell 6
train_df.columns




## === cell 7
sns.set(rc={"figure.figsize": (11, 8)})
sns.set(style="whitegrid")




## === cell 8
total = len(train_df)




## === cell 9
cat_counts = train_df["category"].value_counts()
ax = sns.barplot(x=cat_counts.index, y=cat_counts.values)
ax.set(xlabel="Category", ylabel="# of records", title="Category vs. # of records")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
for p in ax.patches:
    height = p.get_height()
    ax.text(
        p.get_x() + p.get_width() / 2.0,
        height + 5,
        f"{height/total*100:1.2f}%",
        ha="center",
        fontsize=15,
    )
plt.show()




## === cell 10
v = np.vectorize(lambda x: x.split(".")[0])
host_counts = train_df["host"].value_counts()
ax = sns.barplot(x=v(host_counts.index.values), y=host_counts.values)
ax.set(
    xlabel="Host platforms",
    ylabel="# of records",
    title="Host platforms vs. # of records",
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=50, ha="right")
plt.show()




## === cell 11
wc = WordCloud(background_color="white", max_font_size=85, width=700, height=350)
wc.generate(",".join(train_df["question_title"].tolist()))
plt.figure(figsize=(15, 10))
plt.axis("off")
plt.imshow(wc, interpolation="bilinear")
plt.show()




## === cell 12
wc.generate(
    ",".join(train_df["question_body"].tolist()).replace("gt", "").replace("lt", "")
)
plt.figure(figsize=(15, 10))
plt.axis("off")
plt.imshow(wc, interpolation="bilinear")
plt.show()




## === cell 13
wc.generate(",".join(train_df["answer"].tolist()).replace("gt", "").replace("lt", ""))
plt.figure(figsize=(15, 10))
plt.axis("off")
plt.imshow(wc, interpolation="bilinear")
plt.show()




## === cell 14
target_cols = sample_sub_df.drop(["qa_id"], axis=1).columns.values
target_cols




## === cell 15
X_train = train_df.drop(np.concatenate([target_cols, np.array(["qa_id"])]), axis=1)
Y_train = train_df[target_cols]




## === cell 16
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of Y_train: {Y_train.shape}")




## === cell 17
X_train.head()




## === cell 18
X_train = X_train.drop(
    [
        "question_user_name",
        "question_user_page",
        "answer_user_name",
        "answer_user_page",
        "url",
    ],
    axis=1,
)




## === cell 19
X_train.head()




## === cell 20
tfv = TfidfVectorizer(
    min_df=3,
    max_features=None,
    strip_accents="unicode",
    analyzer="word",
    token_pattern=r"\w{1,}",
    ngram_range=(1, 3),
    use_idf=1,
    smooth_idf=1,
    sublinear_tf=1,
    stop_words="english",
)
tsvd = TruncatedSVD(n_components=40)

question_title = tfv.fit_transform(X_train["question_title"].values)
question_title_test = tfv.transform(test_df["question_title"].values)
question_title = tsvd.fit_transform(question_title)
question_title_test = tsvd.transform(question_title_test)

question_body = tfv.fit_transform(X_train["question_body"].values)
question_body_test = tfv.transform(test_df["question_body"].values)
question_body = tsvd.fit_transform(question_body)
question_body_test = tsvd.transform(question_body_test)

answer = tfv.fit_transform(X_train["answer"].values)
answer_test = tfv.transform(test_df["answer"].values)
answer = tsvd.fit_transform(answer)
answer_test = tsvd.transform(answer_test)




## === cell 21
X_train = np.concatenate([question_title, question_body, answer], axis=1)
X_test = np.concatenate([question_title_test, question_body_test, answer_test], axis=1)




## === cell 22
print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)




## === cell 23
model = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64),
    activation="relu",
    solver="adam",
    max_iter=60,
    random_state=42,
    verbose=False,
)




## === cell 24
model.fit(X_train, Y_train)




## === cell 25
preds_train = model.predict(X_train)




## === cell 26
overall_score = 0
for col_index, col in enumerate(target_cols):
    overall_score += spearmanr(
        preds_train[:, col_index], Y_train[col].values
    ).correlation / len(target_cols)

overall_score




## === cell 27
preds_test = model.predict(X_test)




## === cell 28
print("Test predictions shape:", preds_test.shape)




## === cell 29
for col_index, col in enumerate(target_cols):
    sample_sub_df[col] = preds_test[:, col_index]




## === cell 30
sample_sub_df.to_csv("submission.csv", index=False)
