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

0.2724101302979739

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train_df = pd.read_csv('/kaggle/input/google-quest-challenge/train.csv')
test_df = pd.read_csv('/kaggle/input/google-quest-challenge/test.csv')

## === cell 2
import tensorflow as tf
import string
import re
from nltk.tokenize import word_tokenize 
import nltk
import tensorflow_hub as hub
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr
from nltk.stem import LancasterStemmer, PorterStemmer, WordNetLemmatizer
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as api
from nltk.corpus import stopwords 
import nltk
import gc


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
stop_words = stopwords.words('english')
w2v_model = KeyedVectors.load_word2vec_format("/kaggle/input/glove100/glove-100")
vector_dim = 100
encoder = hub.load("/kaggle/input/universalsentenceencoder")

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1862023555.py in <cell line: 0>()
      1 stop_words = stopwords.words('english')
----> 2 w2v_model = KeyedVectors.load_word2vec_format("/kaggle/input/glove100/glove-100")
      3 vector_dim = 100
      4 encoder = hub.load("/kaggle/input/universalsentenceencoder")

/usr/local/lib/python3.11/dist-packages/gensim/models/keyedvectors.py in load_word2vec_format(cls, fname, fvocab, binary, encoding, unicode_errors, limit, datatype, no_header)
   1719 
   1720         """
-> 1721         return _load_word2vec_format(
   1722             cls, fname, fvocab=fvocab, binary=binary, encoding=encoding, unicode_errors=unicode_errors,
   1723             limit=limit, datatype=datatype, no_header=no_header,

/usr/local/lib/python3.11/dist-packages/gensim/models/keyedvectors.py in _load_word2vec_format(cls, fname, fvocab, binary, encoding, unicode_errors, limit, datatype, no_header, binary_chunk_size)
   2048 
   2049     logger.info("loading projection weights from %s", fname)
-> 2050     with utils.open(fname, 'rb') as fin:
   2051         if no_header:
   2052             # deduce both vocab_size & vector_size from 1st pass over file

/usr/local/lib/python3.11/dist-packages/smart_open/smart_open_lib.py in open(uri, mode, buffering, encoding, errors, newline, closefd, opener, compression, transport_params)
    168         transport_params = {}
    169 
--> 170     fobj = _shortcut_open(
    171         uri,
    172         mode,

/usr/local/lib/python3.11/dist-packages/smart_open/smart_open_lib.py in _shortcut_open(uri, mode, compression, buffering, encoding, errors, newline)
    366         open_kwargs['errors'] = errors
    367 
--> 368     return _builtin_open(local_path, mode, buffering=buffering, **open_kwargs)
    369 
    370 

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/glove100/glove-100'

## === cell 4
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
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

rules = {
    "'t": " not",
    "'cause": " because",
    "'ve": " have",
    "'t": " not",
    "'s": " is",
    "'d": " had"
}

## === cell 5
def combine_text(s):
  return (s['question_title'] + ' ' + s['question_body'] + ' ' +s['answer'])

def preprocess_string(s):
  if s == '':
    return ''
  s = re.sub(r'https{0,1}:\/\/([^\s*\n]*)?', ' ', s)
  tokens = s.split()
  
  tokens = [contractions.get(tokens[i].lower(), tokens[i].lower()) for i, _ in enumerate(tokens)]
  for i, _ in enumerate(tokens):
    for key, item in rules.items():
      tokens[i] = tokens[i].replace(key, item)
  tokens = [re.sub('[^a-zA-Z\s]+',' ', x) for x in tokens]
  s = ' '.join(tokens).strip()
  s = re.sub('\s+', ' ', s)

  return s

## === cell 6
train_df['clean_title'] = train_df.question_title.apply(preprocess_string)
train_df['clean_question_body'] = train_df.question_body.apply(preprocess_string)
train_df['clean_answer'] = train_df.answer.apply(preprocess_string)
train_df['combine_text'] = train_df[['question_title', 'question_body', 'answer']].apply(combine_text, axis = 1)

test_df['clean_title'] = test_df.question_title.apply(preprocess_string)
test_df['clean_question_body'] = test_df.question_body.apply(preprocess_string)
test_df['clean_answer'] = test_df.answer.apply(preprocess_string)
test_df['combine_text'] = test_df[['question_title', 'question_body', 'answer']].apply(combine_text, axis = 1)

## === cell 7
def get_embed(input, time_steps, vector_dim):
  final_embed = []
  for index, _ in enumerate(input):
    lst = input[index].split()
    empty_array = np.zeros(shape = (vector_dim))
    tmp_embed = []
    for i in range(time_steps):
      try:
        tmp_embed.append(w2v_model[lst[i]])
      except:
        tmp_embed.append(empty_array)
    tmp_embed = np.vstack(tmp_embed)
    final_embed += [tmp_embed]
  return np.stack(final_embed, axis = 0)

train_title_embed = get_embed(train_df.clean_title, 30, vector_dim).astype(np.float16)
train_question_body_embed = get_embed(train_df.clean_question_body, 256, vector_dim).astype(np.float16)
train_answer_embed = get_embed(train_df.clean_answer, 256, vector_dim).astype(np.float16)
train_combine_embed = encoder(train_df.combine_text).numpy().astype(np.float16).astype(np.float16)


test_title_embed = get_embed(test_df.clean_title, 30, vector_dim).astype(np.float16)
test_question_body_embed = get_embed(test_df.clean_question_body, 256, vector_dim).astype(np.float16)
test_answer_embed = get_embed(test_df.clean_answer, 256, vector_dim).astype(np.float16)
test_combine_embed = encoder(test_df.combine_text).numpy().astype(np.float16)

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2299850246.py in <cell line: 0>()
     14   return np.stack(final_embed, axis = 0)
     15 
---> 16 train_title_embed = get_embed(train_df.clean_title, 30, vector_dim).astype(np.float16)
     17 train_question_body_embed = get_embed(train_df.clean_question_body, 256, vector_dim).astype(np.float16)
     18 train_answer_embed = get_embed(train_df.clean_answer, 256, vector_dim).astype(np.float16)

NameError: name 'vector_dim' is not defined

## === cell 8
del w2v_model, encoder
tf.keras.backend.clear_session()
gc.collect()

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/909061671.py in <cell line: 0>()
----> 1 del w2v_model, encoder
      2 tf.keras.backend.clear_session()
      3 gc.collect()

NameError: name 'w2v_model' is not defined

## === cell 9
def SpearmanCorrCoeff(A, B):
  overall_score = 0
  x1 = np.random.normal(loc = 1e-8, scale = 1e-12, size = A.shape[0])
  x2 = np.random.normal(loc = 1e-8, scale = 1e-12, size = B.shape[0])
  for index in range(30):
      overall_score += spearmanr(A[:, index]+x1, B[:, index] + x2).correlation
  return overall_score/30

def tf_SpearmanCorrCoeff(A, B):
  result = tf.numpy_function(SpearmanCorrCoeff, [A, B], Tout = tf.double)
  return result

## === cell 10
final_outputs = train_df.iloc[:, 11:-4].values

## === cell 11
def create_model():
  i1 = tf.keras.Input(shape = (None, vector_dim), dtype = tf.float16)
  i2 = tf.keras.Input(shape = (None, vector_dim), dtype = tf.float16)
  i3 = tf.keras.Input(shape = (None, vector_dim), dtype = tf.float16)
  i4 = tf.keras.Input(shape = (512), dtype = tf.float16)
  input_concat_1 = tf.keras.layers.Concatenate(axis = 1)([i1, i2])
  lstm_1 = tf.keras.layers.LSTM(64)(input_concat_1)
  lstm_2 = tf.keras.layers.LSTM(64)(i3)

  dense_1 = tf.keras.layers.Dense(64)(i4)
  add = tf.keras.layers.Add()([lstm_1, lstm_2])
  concat = tf.keras.layers.Concatenate(axis = -1)([add, dense_1])
  dense = tf.keras.layers.Dense(30, activation = 'sigmoid')(concat)
  model = tf.keras.Model(inputs = [i1, i2, i3, i4], outputs = [dense])

  return model

## === cell 12
gkf = GroupKFold(n_splits=5).split(X=train_df.url, groups = train_df.url)
valid_preds = []
for fold, (train_idx, valid_idx) in enumerate(gkf):
  if fold in [0,2,4]:
    tf.keras.backend.clear_session()
    model = create_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    train_inputs = [
                train_title_embed[train_idx], train_question_body_embed[train_idx], train_answer_embed[train_idx], train_combine_embed[train_idx]
               ]
    val_inputs = [
                train_title_embed[valid_idx], train_question_body_embed[valid_idx], train_answer_embed[valid_idx], train_combine_embed[valid_idx]
               ]
    model.fit(train_inputs, final_outputs[train_idx], epochs=20, batch_size=16,)

    valid_preds.append(model.predict(val_inputs))
    print('validation score = ', SpearmanCorrCoeff(final_outputs[valid_idx], valid_preds[-1]))

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/102418551.py in <cell line: 0>()
      4   if fold in [0,2,4]:
      5     tf.keras.backend.clear_session()
----> 6     model = create_model()
      7     optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
      8     model.compile(loss='binary_crossentropy', optimizer=optimizer)

/tmp/ipykernel_11/972359239.py in create_model()
      1 def create_model():
----> 2   i1 = tf.keras.Input(shape = (None, vector_dim), dtype = tf.float16)
      3   i2 = tf.keras.Input(shape = (None, vector_dim), dtype = tf.float16)
      4   i3 = tf.keras.Input(shape = (None, vector_dim), dtype = tf.float16)
      5   i4 = tf.keras.Input(shape = (512), dtype = tf.float16)

NameError: name 'vector_dim' is not defined

## === cell 13
test_preds = model.predict([test_title_embed, test_question_body_embed, test_answer_embed, test_combine_embed])
submission = pd.read_csv('/kaggle/input/google-quest-challenge/sample_submission.csv')
submission.iloc[:,1:] = test_preds
submission.to_csv("submission.csv", index = False)
submission.head()

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/531116263.py in <cell line: 0>()
----> 1 test_preds = model.predict([test_title_embed, test_question_body_embed, test_answer_embed, test_combine_embed])
      2 submission = pd.read_csv('/kaggle/input/google-quest-challenge/sample_submission.csv')
      3 submission.iloc[:,1:] = test_preds
      4 submission.to_csv("submission.csv", index = False)
      5 submission.head()

NameError: name 'model' is not defined
