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

-0.01082

# 6. Current score

0.04386

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.04386) has done: 'Diagnosis: The crash happens because the first Dense layer is hard-coded with `input_dim=82044`, but the TF-IDF/mapper pipeline actually produced `x_train` with 76392 features in this run. Keras enforces that the last axis of the input matches `input_dim`, so `model.fit()` fails with a shape mismatch. The correct fix is to set `input_dim` dynamically from the already-created `x_train` feature matrix shape, keeping the same model structure and training loop.

Patch summary: In cell 12 only, replace the hard-coded `input_dim=82044` with `input_dim=x_train.shape[1]` (cast to `int`), so the model input size matches the TF-IDF feature dimensionality produced earlier.

Updated cells: cell 12 only.

Compatibility notes for cell k+1: The variable names `model` and `history` are preserved and produced exactly as before, so cell 13 run unchanged.

Assumptions: `x_train` is a 2D tabular matrix-like object (pandas DataFrame) with `.shape[1]` equal to the number of features, and it is defined in cell 11 as shown.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import sys
import subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.3"]
)

import pandas as pd
import numpy as np

from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras import Sequential, Model
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Input, Embedding
from scipy.stats import spearmanr

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


## === cell 1
data = pd.read_csv('/kaggle/input/google-quest-challenge/train.csv')
data.head()


## === cell 2
features = data[['question_title', 'question_body','answer','question_user_name','answer_user_name']]


## === cell 3
def encoder(data):

    encoder = LabelEncoder()

    df = data[[
        'qa_id', 'question_title', 'question_body', 'question_user_name',
        'question_user_page', 'answer', 
        'answer_user_name', 'answer_user_page',
        'url', 'category', 'host'
    ]]
    mapper = DataFrameMapper([
        ('qa_id', None),
        ('question_title', encoder),
        ('question_body', None),
        ('question_user_name', encoder),
        ('question_user_page', encoder),
        ('answer', None),
        ('answer_user_name', encoder),
        ('answer_user_page', encoder),
        ('url', encoder),
        ('category', encoder),
        ('host', encoder),
    ])
    x = pd.DataFrame(mapper.fit_transform(data),columns=[
        'qa_id','question_title',
        'question_body','question_user_name',
        'question_user_page','answer','answer_user_name',
        'answer_user_page','url','category','host'
    ])


    return x


## === cell 4
def scale(x):

    scaler = MinMaxScaler()

    df = x[[
        'question_title','question_user_name',
        'question_user_page','answer_user_name',
        'answer_user_page','url','category','host'
    ]]


    df = pd.DataFrame(scaler.fit_transform(df),columns=[
        'question_title','question_user_name',
        'question_user_page','answer_user_name',
        'answer_user_page','url','category','host'
    ])

    x = x.drop(columns=[
        'question_title','question_user_name',
        'question_user_page','answer_user_name',
        'answer_user_page','url','category','host'
    ])

    x = pd.concat([x,df],axis=1)
    x = x.drop(columns='qa_id')


    return x


## === cell 5
def word2vec(x):
    tfidf = TfidfVectorizer()

    mapper = DataFrameMapper([
            ('question_title', None),
            ('question_body', tfidf),
            ('question_user_name', None),
            ('question_user_page', None),
            ('answer', tfidf),
            ('answer_user_name', None),
            ('answer_user_page', None),
            ('url', None),
            ('category', None),
            ('host', None),
        ])

    vectors = mapper.fit(x)
    return vectors


## === cell 7
x = encoder(data)
x = scale(x)
word_vectors = word2vec(x)
x = pd.DataFrame(word_vectors.transform(x))


## === cell 8
x.head()


## === cell 9
x.head()
len(x.columns)


## === cell 10
y = data[[
    'question_asker_intent_understanding',
    'question_body_critical', 'question_conversational',
    'question_expect_short_answer', 'question_fact_seeking',
    'question_has_commonly_accepted_answer',
    'question_interestingness_others', 'question_interestingness_self',
    'question_multi_intent', 'question_not_really_a_question',
    'question_opinion_seeking', 'question_type_choice',
    'question_type_compare', 'question_type_consequence',
    'question_type_definition', 'question_type_entity',
    'question_type_instructions', 'question_type_procedure',
    'question_type_reason_explanation', 'question_type_spelling',
    'question_well_written', 'answer_helpful',
    'answer_level_of_information', 'answer_plausible', 'answer_relevance',
    'answer_satisfaction', 'answer_type_instructions',
    'answer_type_procedure', 'answer_type_reason_explanation',
    'answer_well_written'
]]


## === cell 11
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=1)


## === cell 12
def model():
    n_features = int(x_train.shape[1])

    model = Sequential()
    model.add(Dense(30, input_dim=n_features))  # batch size 30
    model.add(Dense(30, activation="sigmoid"))  # shape 1
    model.compile(loss="mse", optimizer="sgd", metrics=["mse"])
    return model


model = model()

history = model.fit(
    x_train,
    y_train,
    epochs=20,
    batch_size=200,
    validation_data=(x_test, y_test),
    verbose=1,
)


## === cell 13
loss, mse = model.evaluate(x_train,y_train, verbose=0)
print("Training MSE: {:.4f}".format(mse))

loss, mse = model.evaluate(x_test,y_test, verbose=0)
print("Testing MSE:  {:.4f}".format(mse))

acc = history.history


import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['mse']
    val_acc = history.history['val_mse']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training mse')
    plt.xlabel('epochs')
    plt.ylabel('mse')
    plt.plot(x, val_acc, 'r', label='Validation mse')
    plt.title('Training and validation mse')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')

    plt.legend()

plot_history(history)


## === cell 14
y_pred = model.predict(x_test)


## === cell 15
spearmanr(y_test, y_pred, axis=None)


## === cell 17
test_data = pd.read_csv('/kaggle/input/google-quest-challenge/test.csv')
test_ids = test_data['qa_id']
test_ids.columns = 'qa_id'


## === cell 18
test_data.head()


## === cell 20
xtest= pd.get_dummies(test_data['question_user_name'])
xtest= encoder(data=test_data)
xtest= scale(xtest)
xtest = pd.DataFrame(word_vectors.transform(xtest))


## === cell 21
xtest.head()


## === cell 22
pred = model.predict(xtest)


## === cell 23
predictions = pd.DataFrame(pred,columns=['question_asker_intent_understanding',
       'question_body_critical', 'question_conversational',
       'question_expect_short_answer', 'question_fact_seeking',
       'question_has_commonly_accepted_answer',
       'question_interestingness_others', 'question_interestingness_self',
       'question_multi_intent', 'question_not_really_a_question',
       'question_opinion_seeking', 'question_type_choice',
       'question_type_compare', 'question_type_consequence',
       'question_type_definition', 'question_type_entity',
       'question_type_instructions', 'question_type_procedure',
       'question_type_reason_explanation', 'question_type_spelling',
       'question_well_written', 'answer_helpful',
       'answer_level_of_information', 'answer_plausible', 'answer_relevance',
       'answer_satisfaction', 'answer_type_instructions',
       'answer_type_procedure', 'answer_type_reason_explanation',
       'answer_well_written'],index=test_ids)
predictions = predictions.reset_index()
predictions.head()


## === cell 24
predictions.describe()


## === cell 25
predictions.to_csv('submission.csv',index=False)
