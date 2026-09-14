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

0.0548585491919292

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Embedding
from keras.layers import Flatten, SimpleRNN, Reshape, Input, concatenate
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
from keras.models import Model
from keras.models import load_model
from keras import regularizers
from pathlib import Path
from tqdm import tqdm


def cat_to_numeric(category):
    if category=='LIFE_ARTS':
        return 1
    if category=='CULTURE':
        return 2
    if category=='SCIENCE':
        return 3
    if category=='STACKOVERFLOW':
      return 4
    if category=='TECHNOLOGY':
      return 5

def prepare_data(frame):
    to_drop = ['question_user_page', 'host', 'url', 'answer_user_page', 
               'answer_user_name', 'question_user_name', 'question_user_page']
    data = frame.drop(to_drop, axis=1)
    data['categoty_normalized'] = data['category'].apply(cat_to_numeric)
    data.drop('category', axis=1, inplace=True)
    return data

def get_vars_and_targets(train_data, test_data):
    
    train_cols = test_data.columns
    target_cols = set(train_data.columns).difference(set(test_data.columns))
    train_cols = set(train_data.columns) - target_cols
    target_cols = list(target_cols)
    train_cols = list(train_cols)
    return train_cols, target_cols

def transform_texts(frame):
    
    text_cols = ['question_title', 'question_body', 'answer']
    tokenizer = Tokenizer()
    for col in text_cols:
        tokenizer.fit_on_texts(frame[col])
    for col in text_cols:
        frame[col+'_tokenized'] = tokenizer.texts_to_sequences(frame[col])
        frame.drop([col], inplace=True, axis=1)
    return [frame, tokenizer]

def pad_texts(frame, cols, maxlen=1000):
    
    frame_padded = pd.DataFrame()
    for index in frame.index:
        padded_val = {'index': index}
        for col in cols:
            if len(frame.loc[index, col]) < maxlen:
                elem = frame.loc[index, col]
                while len(elem) < maxlen:
                    elem.append(0)
            else:
                elem = frame.loc[index, col][0:maxlen]
            padded_val[col+'_padded'] = elem
        frame_padded = frame_padded.append(padded_val, ignore_index=True)
    return frame_padded

def convert_to_arrays(frame):
    
    arrays = []
    for index in frame.index:
        merged = []
        for each in ['question_title_tokenized_padded', 'question_body_tokenized_padded', 'answer_tokenized_padded']:
            arr = np.array(frame.loc[index, each])
            merged.append(arr)
        arrays.append(np.array(merged))
    return np.array(arrays)
"""

def build_model(tokenizer, maxlen=750):
    input_qt = Input(shape=(maxlen, ), name='question_title_input')
    emb_qt = Embedding(len(tokenizer.word_index)+1, output_dim=512)(input_qt)
    lstm_qt = LSTM(512, return_sequences=True, recurrent_dropout=0.1)(emb_qt)
    lstm_qt2 = LSTM(512)(lstm_qt)
    
    input_qb = Input(shape=(maxlen, ), name='question_body_input')
    emb_qb = Embedding(len(tokenizer.word_index)+1, output_dim=512)(input_qb)
    lstm_qb = LSTM(512, return_sequences=True, recurrent_dropout=0.1)(emb_qb)
    lstm_qb2 = LSTM(512)(lstm_qb)
    
    input_ans = Input(shape=(maxlen, ), name='answer_input')
    emb_ans = Embedding(len(tokenizer.word_index)+1, output_dim=512)(input_ans)
    lstm_ans = LSTM(512, return_sequences=True, recurrent_dropout=0.1)(emb_ans)
    lstm_ans2 = LSTM(512)(lstm_ans)
    
    merged = concatenate([lstm_qt2, lstm_qb2, lstm_ans2])
    dense = Dense(128, activation='relu')(merged)
    dropout = Dropout(0.2)(dense)
    dense2 = Dense(64, activation='relu')(dropout)
    output = Dense(30, activation='relu', name='outputs')(dense2)
    
    model = Model(inputs=[input_qt, input_qb, input_ans], outputs=[output])
    model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])
    
    return model
"""

def build_model_2(tokenizer, maxlen=1000):
    input_qt = Input(shape=(3, maxlen))
    lstm_qt1 = LSTM(1024, return_sequences=True, recurrent_dropout=0.25, activation='sigmoid')(input_qt)
    lstm_qt2 = LSTM(512, return_sequences=True, recurrent_dropout=0.25, activation='sigmoid')(lstm_qt1)
    lstm_qt3 = LSTM(512)(lstm_qt2)
    
    dense = Dense(512, activation='relu')(lstm_qt3)
    dropout = Dropout(0.2)(dense)
    dense2 = Dense(256, activation='relu')(dropout)
    output = Dense(30, name='outputs')(dense2)
    
    model = Model(inputs=[input_qt], outputs=[output])
    model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])
    
    return model


def main():
    train_data = pd.read_csv('../input/google-quest-challenge/train.csv', encoding='utf-8')
    test_data = pd.read_csv('../input/google-quest-challenge/test.csv', encoding='utf-8')
    train_data = prepare_data(train_data)
    test_data = prepare_data(test_data)
    
    train_cols, target_cols = get_vars_and_targets(train_data, test_data)
    X_train, X_test, y_train, y_test = train_test_split(train_data.loc[:, train_cols], train_data.loc[:, target_cols], test_size=0.25)
    X_train, tokenizer = transform_texts(X_train)
    X_test, _ = transform_texts(X_test)
    
    X_train = pad_texts(X_train, cols=['question_title_tokenized', 'question_body_tokenized', 'answer_tokenized'])
    X_test = pad_texts(X_test, cols=['question_title_tokenized', 'question_body_tokenized', 'answer_tokenized'])

    X_train = convert_to_arrays(X_train)
    X_test = convert_to_arrays(X_test)
    model = build_model_2(tokenizer)
    model.fit(X_train, y_train, batch_size=16, epochs=1, validation_split=0.2)

    print(model.evaluate(X_test, y_test))
    
    test_data, _ = transform_texts(test_data.drop(['qa_id', 'categoty_normalized'], axis=1))
    test_data = pad_texts(test_data, cols=['question_title_tokenized', 'question_body_tokenized', 'answer_tokenized'])
    test_data = convert_to_arrays(test_data)
    predictions = model.predict(test_data)
        
    submission_data = pd.read_csv('../input/google-quest-challenge/sample_submission.csv', encoding='utf-8')
    print(submission_data)
    submissions = pd.DataFrame()
    print(len(submission_data.columns))
    print(submission_data.columns)
    row = 0
    for each in submission_data['qa_id']:
        col = 0
        submissions.loc[each, 'qa_id'] = str(each)
        for column in target_cols:
            try:
                if round(abs(predictions[row, col]), 5) < 0.0001:
                    submissions.loc[each, column] = 0
                else:
                    submissions.loc[each, column] = round(abs(predictions[row, col]), 5)
            except Exception:
                submissions.loc[each, column] = 0.5
            col+=1
        row+=1

    submissions['qa_id'] = submissions['qa_id'].astype(object)
    
    sample_csv = "../input/google-quest-challenge/sample_submission.csv"
    sample_df = pd.read_csv(sample_csv)
    n=0
    for line in tqdm(submissions.values):
        for i in range(len(sample_df)):
            if(sample_df.loc[i]['qa_id']==int(line[0])):
                for j in range(1,31):
                    sample_df.iloc[i,j] = line[j]
                break
    sample_df.head()
    sample_df.to_csv("submission.csv", index=False)
    print("done!")
    print(sample_df.head())

    
main()

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
