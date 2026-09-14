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

gensim==4.4.0
geopandas==0.14.4
nltk==3.9.2
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

0.1726910959077161

# 6. Current score

0.31529

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.26304) has done: 'I fix the import order, set the protobuf environment variable before TensorFlow is imported, and combine the initial data‑loading steps into a single cell so that `train_df`, `test_df`, and `TARGET_COLS` are defined before they are used. This resolves the NameError issues and the TensorFlow import error while keeping the core modeling logic unchanged.'
- What this solution (achieved 0.25986) has done: 'I import TensorFlow right after setting the protobuf environment variable (before Gensim is loaded) to avoid the protobuf‑related AttributeError, and remove the separate TensorFlow import in the later cell. This fixes the runtime error while keeping the original model architecture and training unchanged, so the score stays near the current (already above the target).'
- What this solution (achieved 0.2593) has done: 'The fix moves the protobuf environment variable setting to the very top, adds a safe fallback import for TensorFlow (using `tensorflow.compat.v1` if the standard import fails), and silences unnecessary TensorFlow logs. This resolves the `AttributeError` on import while keeping the original model, preprocessing, and training logic unchanged, ensuring a valid `submission.csv` is produced and the score remains above the target.'
- What this solution (achieved 0.31594) has done: 'I fix the runtime error by switching the Ridge regression solver to one that works with the SciPy version used (e.g., `"lsqr"`), which avoids the deprecated `cg(..., tol=…)` call that caused the failure. This minimal change lets the model train and produce predictions, generating a valid `submission.csv` without altering the core modeling logic.'
- What this solution (achieved 0.31529) has done: 'I keep the existing model and preprocessing unchanged but add a simple calibration step after prediction: blend the model’s outputs with the overall mean of each target column (using α = 0.3). This monotonic linear combination reduces rank variation, which should lower the Spearman correlation and move the score from 0.31594 toward the target 0.17269 without altering the core logic.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress TF warnings (kept for safety)

import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

train_df = pd.read_csv("/kaggle/input/google-quest-challenge/train.csv")
test_df = pd.read_csv("/kaggle/input/google-quest-challenge/test.csv")
TARGET_COLS = train_df.columns[-30:].tolist()




## === cell 1
contractions = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "might've": "might have",
    "must've": "must have",
    "needn't": "need not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "we'd": "we would",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what's": "what is",
    "where's": "where is",
    "who's": "who is",
    "won't": "will not",
    "would've": "would have",
    "you'd": "you would",
    "you're": "you are",
    "you've": "you have",
}
rules = {"'t": " not", "'s": " is", "'d": " had", "'ve": " have", "'re": " are"}


def preprocess_string(s):
    if pd.isna(s) or s == "":
        return ""
    s = re.sub(r"https?://\S+", " ", s)  # remove URLs
    tokens = s.split()
    tokens = [contractions.get(tok.lower(), tok.lower()) for tok in tokens]
    for i, tok in enumerate(tokens):
        for key, repl in rules.items():
            tok = tok.replace(key, repl)
        tokens[i] = tok
    tokens = [re.sub("[^a-zA-Z\\s]+", " ", t) for t in tokens]
    cleaned = " ".join(tokens).strip()
    cleaned = re.sub("\\s+", " ", cleaned)
    return cleaned




## === cell 2
for col in ["question_title", "question_body", "answer"]:
    train_df[f"clean_{col}"] = train_df[col].apply(preprocess_string)
    test_df[f"clean_{col}"] = test_df[col].apply(preprocess_string)

train_df["full_text"] = (
    train_df["clean_question_title"]
    + " "
    + train_df["clean_question_body"]
    + " "
    + train_df["clean_answer"]
)
test_df["full_text"] = (
    test_df["clean_question_title"]
    + " "
    + test_df["clean_question_body"]
    + " "
    + test_df["clean_answer"]
)




## === cell 3
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    stop_words=stopwords.words("english"),
)
X_train = vectorizer.fit_transform(train_df["full_text"])
X_test = vectorizer.transform(test_df["full_text"])

y_train = train_df[TARGET_COLS].astype(np.float32).values

ridge = Ridge(alpha=1.0, solver="lsqr", random_state=42)
model = MultiOutputRegressor(ridge)
model.fit(X_train, y_train)




## === cell 4
test_preds = model.predict(X_test)

alpha = 0.3
col_means = train_df[TARGET_COLS].mean().values  # shape (30,)
test_preds = alpha * test_preds + (1 - alpha) * col_means

test_preds = np.clip(test_preds, 0.0, 1.0)

submission = pd.DataFrame(
    np.concatenate([test_df["qa_id"].values.reshape(-1, 1), test_preds], axis=1),
    columns=["qa_id"] + TARGET_COLS,
)

submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
