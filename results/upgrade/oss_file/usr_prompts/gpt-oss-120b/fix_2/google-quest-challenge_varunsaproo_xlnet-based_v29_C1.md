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

0.2585599180325146

# 6. Current score

-0.00639

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved -0.00639) has done: 'I fixed the import errors, replaced the unavailable BERT/Universal‑Sentence‑Encoder components with a lightweight TF‑IDF text vectorizer, built a simple dense neural network for the 30 target columns, and rewrote the pipeline so it runs from data loading to saving a proper `submission.csv` file.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import tensorflow as tf
import re
from scipy.stats import spearmanr
from tensorflow.keras.layers import TextVectorization, Dense, Dropout, Input
from tensorflow.keras.models import Model



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DIR = "/kaggle/input/google-quest-challenge"
BATCH_SIZE = 64




## === cell 2
def func(s):
    s = re.sub("\n+", " ", s)
    s = re.sub("[?]", " . ", s)
    s = re.sub("[!{}]", " . ", s)
    s = re.sub("\.{2,}", "", s)
    s = re.sub("\s+", " ", s)
    return s


def clean_data(df):
    df["question_body"] = df["question_body"].apply(func)
    df["question_title"] = df["question_title"].apply(func)
    df["answer"] = df["answer"].apply(func)
    return df




## === cell 3
train_df = pd.read_csv(DIR + "/train.csv")
test_df = pd.read_csv(DIR + "/test.csv")

train_df = clean_data(train_df)
test_df = clean_data(test_df)

train_df["combined"] = (
    train_df["question_title"]
    + " "
    + train_df["question_body"]
    + " "
    + train_df["answer"]
)
test_df["combined"] = (
    test_df["question_title"] + " " + test_df["question_body"] + " " + test_df["answer"]
)

vectorizer = TextVectorization(max_tokens=20000, output_mode="tf-idf", dtype=tf.float32)
vectorizer.adapt(train_df["combined"].values)

train_vec = vectorizer(train_df["combined"]).numpy()
test_vec = vectorizer(test_df["combined"]).numpy()

labels = train_df.iloc[:, -30:].values.astype(np.float32)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/843085640.py in <cell line: 0>()
     26 
     27 # Labels (last 30 columns)
---> 28 labels = train_df.iloc[:, -30:].values.astype(np.float32)
     29 
     30 

ValueError: could not convert string to float: "Which parts of fresh Fenugreek am I supposed to throw off before attempting to dry them out completely .  The fresh Fenugreek which I bought contains: - long stems - green leaves - yellow leaves I wish to know what parts of fresh Fenugreek am I supposed to throw off before attempting to place them drying out completely .  I would just pull off all the little stems with leaves on them off the big main stem and leave it at that. You can remove the yellow leaves if you wish, but there's no harm in leaving them on. I presume once dry you will crush all the leaves and small stems for storage, so leaving the yellow leaves on will not make any difference. Removing the big main stem will simply help speed up the drying time. For convenience of moving and turning you can leave them complete, but you should remove the smaller stems with leaves once they're dry, the large stem doesn't generally get used. "

## === cell 4
def SpearmanCorrCoeff_2(A, B):
    overall_score = 0.0
    x1 = np.random.normal(loc=1e-8, scale=1e-12, size=A.shape[0])
    x2 = np.random.normal(loc=1e-8, scale=1e-12, size=B.shape[0])
    for i in range(A.shape[1]):
        overall_score += spearmanr(A[:, i] + x1, B[:, i] + x2).correlation / 30.0
    return overall_score


def tf_SpearmanCorrCoeff(A, B):
    result = tf.numpy_function(SpearmanCorrCoeff_2, [A, B], tf.double)
    return result




## === cell 5
input_dim = train_vec.shape[1]
inputs = Input(shape=(input_dim,))
x = Dense(256, activation="relu")(inputs)
x = Dropout(0.2)(x)
outputs = Dense(30, activation="sigmoid")(x)
model = Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[tf_SpearmanCorrCoeff],
)

model.fit(train_vec, labels, batch_size=32, epochs=5, validation_split=0.1, verbose=2)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2669699368.py in <cell line: 0>()
     12 )
     13 
---> 14 model.fit(train_vec, labels, batch_size=32, epochs=5, validation_split=0.1, verbose=2)
     15 

NameError: name 'labels' is not defined

## === cell 6
test_pred = model.predict(test_vec, batch_size=64)



## === cell 7
sample_submission = pd.read_csv(DIR + "/sample_submission.csv")
submission = pd.DataFrame(test_pred, columns=sample_submission.columns[1:])
submission["qa_id"] = test_df["qa_id"].values
submission = submission[sample_submission.columns]
submission.to_csv("submission.csv", index=False)
