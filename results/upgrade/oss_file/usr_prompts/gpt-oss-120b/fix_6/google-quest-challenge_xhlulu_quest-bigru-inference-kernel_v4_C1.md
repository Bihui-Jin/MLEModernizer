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

No external packages required in the script and installed.

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

0.1873609852665528

# 6. Current score

0.00307

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.04252) has done: 'I replace the missing tokenizer/model loading with a lightweight end‑to‑end pipeline: fit a Keras Tokenizer on the text fields, convert them to padded sequences, build a small multi‑input neural network (embeddings + GRU + dense) and train it briefly on a train/validation split. After training the model predicts the 30 target columns for the test set and writes a correctly‑named `submission.csv`. This fixes the import/file errors, defines all needed variables, and yields a valid submission whose score should move toward the target range.'
- What this solution (achieved 0.16402) has done: 'I replace the failing Ridge model with an SGDRegressor that works with sparse TF‑IDF matrices, keeping the same MultiOutputRegressor wrapper. This fixes the TypeError during fitting and allows the script to train and produce a valid `submission.csv` without altering the overall pipeline or feature engineering.'
- What this solution (achieved 0.00307) has done: 'I train on the full TF‑IDF matrix (instead of a 90 %/10 % split that isn’t used) and loosen the SGD regularisation while allowing more iterations. These small hyper‑parameter tweaks keep the same TF‑IDF + SGD + MultiOutputRegressor pipeline but give the model more data and capacity, which should raise the Spearman‑correlation toward the target score.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import SGDRegressor  # new import




## === cell 1
train_path = "/kaggle/input/google-quest-challenge/train.csv"
test_path = "/kaggle/input/google-quest-challenge/test.csv"
sample_sub_path = "/kaggle/input/google-quest-challenge/sample_submission.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
submission = pd.read_csv(sample_sub_path)

print("train shape:", train.shape, "test shape:", test.shape)




## === cell 2
target_cols = train.columns[-30:].tolist()

text_fields = ["question_title", "question_body", "answer"]
train_text = train[text_fields].astype(str).apply(lambda x: " ".join(x), axis=1)
test_text = test[text_fields].astype(str).apply(lambda x: " ".join(x), axis=1)

vectorizer = TfidfVectorizer(
    max_features=30000,  # increase features for richer representation
    ngram_range=(1, 2),
    stop_words="english",
    dtype=np.float32,
)

vectorizer.fit(pd.concat([train_text, test_text], ignore_index=True))

X_train_full = vectorizer.transform(train_text)
X_test = vectorizer.transform(test_text)

y = train[target_cols].values.astype(np.float32)

X_train = X_train_full
y_train = y




## === cell 3
base_model = SGDRegressor(
    loss="squared_error",
    penalty="l2",
    alpha=1e-5,  # weaker regularisation
    random_state=42,
    max_iter=2000,  # more iterations for convergence
    tol=1e-3,
    learning_rate="optimal",
)
model = MultiOutputRegressor(base_model)

model.fit(X_train, y_train)




## === cell 4
test_pred = model.predict(X_test)

test_pred = np.clip(test_pred, 0.0, 1.0)

submission.loc[:, target_cols] = test_pred
output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {os.path.abspath(output_path)}")
