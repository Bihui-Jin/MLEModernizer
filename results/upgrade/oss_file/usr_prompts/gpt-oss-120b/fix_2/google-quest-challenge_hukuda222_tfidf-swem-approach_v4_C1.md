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

gensim==4.4.0
geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
lightgbm==4.6.0
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

# 5. Target score

0.28711

# 6. Current score

0.35503

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.35503) has done: 'I fixed the import errors, added missing `tqdm` and corrected the `EarlyStopping` import. I also ensured that missing text values are filled, made the category encoding robust to unseen categories, and reorganized the cells so they run sequentially from 1 onward. These changes resolve the runtime failures and allow the script to produce a valid `submission.csv` file while keeping the original model logic intact.'

# 9. Code solution

## === cell 0
import os
import gc
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gensim
from nltk.corpus import brown
from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train = pd.read_csv("../input/google-quest-challenge/train.csv").fillna("")
test = pd.read_csv("../input/google-quest-challenge/test.csv").fillna("")



## === cell 2
sample_sub = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")



## === cell 3
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




## === cell 4
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




## === cell 5
qt_max = max([len(simple_prepro(l)) for l in train["question_title"].values])
qb_max = max([len(simple_prepro(l)) for l in train["question_body"].values])
an_max = max([len(simple_prepro(l)) for l in train["answer"].values])
print("max length of question_title:", qt_max)
print("max length of question_body:", qb_max)
print("max length of answer:", an_max)



## === cell 6
w2v_model = gensim.models.Word2Vec(
    brown.sents(), vector_size=100, window=5, min_count=1, workers=4
)




## === cell 7
def get_word_embeddings(text):
    np.random.seed(abs(hash(text)) % (10**8))
    words = simple_prepro(text)
    if len(words) == 0:
        return np.zeros(100)
    vectors = np.zeros((len(words), 100))
    for i, word in enumerate(words):
        if word in w2v_model.wv:
            vectors[i] = w2v_model.wv[word]
        else:
            vectors[i] = np.random.uniform(-0.01, 0.01, 100)
    return np.max(vectors, axis=0)




## === cell 8
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



## === cell 9
tfidf = TfidfVectorizer(ngram_range=(1, 3))
tsvd = TruncatedSVD(n_components=50, random_state=42)

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



## === cell 10
type2int = {cat: i for i, cat in enumerate(train["category"].unique())}
num_categories = len(type2int)
cate = np.identity(num_categories)[
    train["category"].apply(lambda x: type2int[x]).values
]
cate_test = np.identity(num_categories)[
    test["category"].apply(lambda x: type2int.get(x, 0)).values
]



## === cell 11
train_features = np.concatenate(
    [
        question_title,
        question_body,
        answer,
        tfidf_question_title,
        tfidf_question_body,
        tfidf_answer,
        cate,
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
    ],
    axis=1,
)



## === cell 12
num_folds = 10
fold_scores = []
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
test_preds = np.zeros((len(test_features), len(target_cols)))

for train_index, val_index in kf.split(train_features):
    gc.collect()
    train_X = train_features[train_index]
    train_y = train[target_cols].iloc[train_index].values

    val_X = train_features[val_index]
    val_y = train[target_cols].iloc[val_index].values

    model = Sequential(
        [
            Dense(512, input_shape=(train_features.shape[1],)),
            Activation("relu"),
            Dense(128),
            Activation("relu"),
            Dense(len(target_cols)),
            Activation("sigmoid"),
        ]
    )

    es = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=0
    )

    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit(
        train_X,
        train_y,
        epochs=100,
        validation_data=(val_X, val_y),
        callbacks=[es],
        verbose=0,
    )

    val_pred = model.predict(val_X)
    overall_score = 0.0
    for col_idx, col in enumerate(target_cols):
        corr = spearmanr(val_pred[:, col_idx], val_y[:, col_idx]).correlation
        overall_score += corr / len(target_cols)
        print(f"{col}: {corr:.4f}")
    fold_scores.append(overall_score)
    print(f"Fold score: {overall_score:.4f}")

    test_preds += model.predict(test_features) / num_folds

print("All fold scores:", fold_scores)



## === cell 13
sub = sample_sub.copy()
for idx, col in enumerate(target_cols):
    sub[col] = test_preds[:, idx]
sub.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
