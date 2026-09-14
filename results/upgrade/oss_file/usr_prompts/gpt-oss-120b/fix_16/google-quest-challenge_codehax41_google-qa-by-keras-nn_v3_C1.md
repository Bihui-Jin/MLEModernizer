# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

gensim==4.4.0
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
nltk==3.9.2
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
tqdm==4.67.1

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

# 5. Code solution

## === cell 0
import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from scipy.stats import spearmanr

KERAS_AVAILABLE = False


def get_model(input_dim, output_dim):
    """
    Return a simple MultiOutputRegressor wrapping an MLPRegressor.
    This mimics the interface used later in the notebook.
    """
    base = MLPRegressor(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        max_iter=200,
        random_state=42,
        early_stopping=False,
        verbose=False,
    )
    return MultiOutputRegressor(base)


gc.collect()



## === cell 1
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 2
base_path = "/kaggle/input/google-quest-challenge"
train = pd.read_csv(os.path.join(base_path, "train.csv"))
test = pd.read_csv(os.path.join(base_path, "test.csv"))
df = pd.concat([train, test])
df.head()



## === cell 3
df.shape



## === cell 4
df.dtypes



## === cell 5
df.isnull().sum()



## === cell 6
target_cols = [
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



## === cell 7
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
width = 0.4
df.host.value_counts().plot(kind="bar", color="blue", ax=ax, width=width, position=1)
ax.set_xlabel("Sites")
ax.set_ylabel("Question Counts")
plt.close(fig)



## === cell 8
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111)
width = 0.2
df.category.value_counts().plot(
    kind="bar", color="green", ax=ax, width=width, position=1, legend=True
)
ax.set_xlabel("Sites")
ax.set_ylabel("Question Counts")
plt.close(fig)




## === cell 9
def char_count(s):
    return len(s)


def word_count(s):
    return s.count(" ")


train["question_title_n_chars"] = train["question_title"].apply(char_count)
train["question_title_n_words"] = train["question_title"].apply(word_count)
train["question_body_n_chars"] = train["question_body"].apply(char_count)
train["question_body_n_words"] = train["question_body"].apply(word_count)
train["answer_n_chars"] = train["answer"].apply(char_count)
train["answer_n_words"] = train["answer"].apply(word_count)

test["question_title_n_chars"] = test["question_title"].apply(char_count)
test["question_title_n_words"] = test["question_title"].apply(word_count)
test["question_body_n_chars"] = test["question_body"].apply(char_count)
test["question_body_n_words"] = test["question_body"].apply(word_count)
test["answer_n_chars"] = test["answer"].apply(char_count)
test["answer_n_words"] = test["answer"].apply(word_count)



## === cell 10
train["question_body_n_chars"].clip(0, 5000, inplace=True)
test["question_body_n_chars"].clip(0, 5000, inplace=True)
train["question_body_n_words"].clip(0, 1000, inplace=True)
test["question_body_n_words"].clip(0, 1000, inplace=True)

train["answer_n_chars"].clip(0, 5000, inplace=True)
test["answer_n_chars"].clip(0, 5000, inplace=True)
train["answer_n_words"].clip(0, 1000, inplace=True)
test["answer_n_words"].clip(0, 1000, inplace=True)



## === cell 11
num_question = train["question_user_name"].value_counts()
num_answer = train["answer_user_name"].value_counts()



## === cell 12
train["num_answer_user"] = train["answer_user_name"].map(num_answer)
train["num_question_user"] = train["question_user_name"].map(num_question)
test["num_answer_user"] = test["answer_user_name"].map(num_answer)
test["num_question_user"] = test["question_user_name"].map(num_question)



## === cell 13
test["num_answer_user"].fillna(1, inplace=True)
test["num_question_user"].fillna(1, inplace=True)

simple_feature_cols = [
    "question_title_n_chars",
    "question_title_n_words",
    "question_body_n_chars",
    "question_body_n_words",
    "answer_n_chars",
    "answer_n_words",
    "num_answer_user",
    "num_question_user",
]
simple_engineered_feature = train[simple_feature_cols].values
simple_engineered_feature_test = test[simple_feature_cols].values



## === cell 14
scaler = MaxAbsScaler()
simple_engineered_feature = scaler.fit_transform(simple_engineered_feature)
simple_engineered_feature_test = scaler.transform(simple_engineered_feature_test)




## === cell 15
def simple_prepro(s):
    return [
        w
        for w in s.replace("\n", " ")
        .replace(",", " , ")
        .replace("(", " ( ")
        .replace(")", " ) ")
        .replace(".", " . ")
        .replace("?", " ? ")
        .replace(":", " : ")
        .replace("n't", " not")
        .replace("'ve", " have")
        .replace("'re", " are")
        .replace("'s", " is")
        .split(" ")
        if w != ""
    ]


def simple_prepro_tfidf(s):
    return " ".join(
        [
            w
            for w in s.lower()
            .replace("\n", " ")
            .replace(",", " , ")
            .replace("(", " ( ")
            .replace(")", " ) ")
            .replace(".", " . ")
            .replace("?", " ? ")
            .replace(":", " : ")
            .replace("n't", " not")
            .replace("'ve", " have")
            .replace("'re", " are")
            .replace("'s", " is")
            .split(" ")
            if w != ""
        ]
    )




## === cell 16
qt_max = max([len(simple_prepro(l)) for l in list(train["question_title"].values)])
qb_max = max([len(simple_prepro(l)) for l in list(train["question_body"].values)])
an_max = max([len(simple_prepro(l)) for l in list(train["answer"].values)])
print("max length of question_title is", qt_max)
print("max length of question_body is", qb_max)
print("max length of answer is", an_max)




## === cell 17
def get_word_embeddings(text):
    return np.zeros(100)




## === cell 18
question_title = [get_word_embeddings(l) for l in tqdm(train["question_title"].values)]
question_title_test = [
    get_word_embeddings(l) for l in tqdm(test["question_title"].values)
]

question_body = [get_word_embeddings(l) for l in tqdm(train["question_body"].values)]
question_body_test = [
    get_word_embeddings(l) for l in tqdm(test["question_body"].values)
]

answer = [get_word_embeddings(l) for l in tqdm(train["answer"].values)]
answer_test = [get_word_embeddings(l) for l in tqdm(test["answer"].values)]

question_title = np.array(question_title)
question_title_test = np.array(question_title_test)
question_body = np.array(question_body)
question_body_test = np.array(question_body_test)
answer = np.array(answer)
answer_test = np.array(answer_test)



## === cell 19
tfidf = TfidfVectorizer(ngram_range=(1, 3))
tsvd = TruncatedSVD(n_components=50)

tfidf_question_title = tfidf.fit_transform(
    [simple_prepro_tfidf(l) for l in tqdm(train["question_title"].values)]
)
tfidf_question_title_test = tfidf.transform(
    [simple_prepro_tfidf(l) for l in tqdm(test["question_title"].values)]
)
tfidf_question_title = tsvd.fit_transform(tfidf_question_title)
tfidf_question_title_test = tsvd.transform(tfidf_question_title_test)

tfidf_question_body = tfidf.fit_transform(
    [simple_prepro_tfidf(l) for l in tqdm(train["question_body"].values)]
)
tfidf_question_body_test = tfidf.transform(
    [simple_prepro_tfidf(l) for l in tqdm(test["question_body"].values)]
)
tfidf_question_body = tsvd.fit_transform(tfidf_question_body)
tfidf_question_body_test = tsvd.transform(tfidf_question_body_test)

tfidf_answer = tfidf.fit_transform(
    [simple_prepro_tfidf(l) for l in tqdm(train["answer"].values)]
)
tfidf_answer_test = tfidf.transform(
    [simple_prepro_tfidf(l) for l in tqdm(test["answer"].values)]
)
tfidf_answer = tsvd.fit_transform(tfidf_answer)
tfidf_answer_test = tsvd.transform(tfidf_answer_test)



## === cell 20
type2int = {typ: i for i, typ in enumerate(sorted(set(train["category"])))}
cate = np.identity(len(type2int))[
    np.array(train["category"].apply(lambda x: type2int[x]))
].astype(np.float64)
cate_test = np.identity(len(type2int))[
    np.array(test["category"].apply(lambda x: type2int[x]))
].astype(np.float64)

train_features = np.concatenate(
    [
        question_title,
        question_body,
        answer,
        tfidf_question_title,
        tfidf_question_body,
        tfidf_answer,
        cate,
        simple_engineered_feature,
    ],
    axis=1,
)
test_features = np.concatenate(
    [
        question_title_test,
        question_body_test,
        answer_test,
        tfidf_question_title_test,
        tfidf_question_body_test,
        tfidf_answer_test,
        cate_test,
        simple_engineered_feature_test,
    ],
    axis=1,
)



## === cell 21
num_folds = 7
fold_scores = []
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
test_preds = np.zeros((len(test_features), len(target_cols)))

for train_index, val_index in kf.split(train_features):
    gc.collect()
    train_X = train_features[train_index, :]
    train_y = train.reindex(columns=target_cols).fillna(0).iloc[train_index]

    val_X = train_features[val_index, :]
    val_y = train.reindex(columns=target_cols).fillna(0).iloc[val_index]

    model = get_model(train_features.shape[1], len(target_cols))
    model.fit(train_X, train_y)

    preds = model.predict(val_X)
    preds = np.clip(preds * 0.4 + 0.3, 0, 1)

    overall_score = 0
    for col_index, col in enumerate(target_cols):
        col_score = spearmanr(preds[:, col_index], val_y[col].values).correlation
        if np.isnan(col_score):
            col_score = 0
        overall_score += col_score / len(target_cols)
    fold_scores.append(overall_score)

    test_fold_pred = model.predict(test_features)
    test_fold_pred = np.clip(test_fold_pred * 0.4 + 0.3, 0, 1)

    test_preds += test_fold_pred / num_folds

print("Fold scores:", fold_scores)



## === cell 22
submission = pd.DataFrame(
    np.concatenate([test["qa_id"].values.reshape(-1, 1), test_preds], axis=1),
    columns=["qa_id"] + target_cols,
)
submission.to_csv("submission.csv", index=False)
