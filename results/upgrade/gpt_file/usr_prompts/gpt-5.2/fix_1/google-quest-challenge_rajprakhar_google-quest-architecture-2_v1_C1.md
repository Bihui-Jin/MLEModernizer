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

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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

0.1205653379877138

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import pandas as pd
import numpy as np

import tensorflow as tf
print(tf.__version__)

import re
from tqdm import tqdm

from scipy.stats import spearmanr

import warnings
warnings.simplefilter('ignore')

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
PATH = '../input/google-quest-challenge/'
PATH_w2vec_300d = '../input/glove-300d/'

df_train = pd.read_csv(PATH+'train.csv')
df_test = pd.read_csv(PATH+'test.csv')
df_sub = pd.read_csv(PATH+'sample_submission.csv')
print('Train Shape =', df_train.shape)
print('Test Shape =', df_test.shape)

output_categories = list(df_train.columns[11:])
input_categories = list(df_train.columns[[1,2,5]])
print('\nOutput Categories:\n\t', output_categories)
print('\nInput Categories:\n\t', input_categories)

## === cell 4

stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"]

def decontracted(phrase): # https://stackoverflow.com/a/47091490/4084039
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def preprocess_text(text_data):
    preprocessed_text = []
    for sentance in tqdm(text_data):
        sent = decontracted(sentance)
        sent = sent.replace('\\r', ' ')
        sent = sent.replace('\\n', ' ')
        sent = sent.replace('\\"', ' ')
        sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
        sent = ' '.join(e for e in sent.split() if e.lower() not in stopwords)
        preprocessed_text.append(sent.lower().strip())
    return preprocessed_text

def perform_preprocessing(text_array):
    lower_text_array = pd.Series(text_array).str.lower()
    preprocessed_text_array = preprocess_text(lower_text_array)

    return pd.Series(preprocessed_text_array)


df_train['Preproc_Question_Title'] = perform_preprocessing(df_train['question_title'].values)
df_train['Preproc_Question_Body'] = perform_preprocessing(df_train['question_body'].values)
df_train['Preproc_Answer'] = perform_preprocessing(df_train['answer'].values)
  
df_test['Preproc_Question_Title'] = perform_preprocessing(df_test['question_title'].values)
df_test['Preproc_Question_Body'] = perform_preprocessing(df_test['question_body'].values)
df_test['Preproc_Answer'] = perform_preprocessing(df_test['answer'].values)

print("\n")
print("="*70 + "Question Title" + "="*70)
print("Before Preprocessing:\n", df_train['question_title'][0])
print("\nAfter Preprocessing:\n", df_train['Preproc_Question_Title'][0])

print("="*70 + "Question Body" + "="*70)
print("Before Preprocessing:\n", df_train['question_body'][0])
print("\nAfter Preprocessing:\n", df_train['Preproc_Question_Body'][0])

print("="*70 + "Answer" + "="*70)
print("Before Preprocessing:\n", df_train['answer'][0])
print("\nAfter Preprocessing:\n", df_train['Preproc_Answer'][0])

## === cell 6

def prepare_embedding(input_series_train, input_series_test, column_name):

    print("="*70 + column_name + "="*70)

    tokenizer_obj = tf.keras.preprocessing.text.Tokenizer()
    tokenizer_obj.fit_on_texts(input_series_train.values)

    word_index = tokenizer_obj.word_index
    print('Found %s unique tokens.' % len(word_index))

    train_sequences = tokenizer_obj.texts_to_sequences(input_series_train.values)
    test_sequences = tokenizer_obj.texts_to_sequences(input_series_test.values)
    print("Train Sequences Length", len(train_sequences))
    print("Test Sequences Length", len(test_sequences))

    MAX_SEQUENCE_LENGTH = int(np.percentile(pd.Series(train_sequences).apply(lambda x: len(x)), 96))
    print("Around 96 percentile of " + column_name + " have length of words less than ", MAX_SEQUENCE_LENGTH)

    vocab_size = len(word_index)+1
    train_sequences_pad = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    test_sequences_pad = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print("Shape of padded train sequences: ", train_sequences_pad.shape)
    print("Shape of padded test sequences: ", test_sequences_pad.shape)


    embeddings_index = {}
    f = open(PATH_w2vec_300d+'glove-840B-300d-char_embed.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((vocab_size, 300))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return vocab_size, embedding_matrix, MAX_SEQUENCE_LENGTH, train_sequences_pad, test_sequences_pad


## === cell 7
vocab_size_question_title, embedding_matrix_question_title, MAX_SEQUENCE_LENGTH_question_title, train_sequences_pad_qt, test_sequences_pad_qt = \
prepare_embedding(df_train['question_title'], df_test['question_title'], 'Question Title')

vocab_size_question_body, embedding_matrix_question_body, MAX_SEQUENCE_LENGTH_question_body, train_sequences_pad_qb, test_sequences_pad_qb = \
prepare_embedding(df_train['question_body'], df_test['question_body'], 'Question Body')

vocab_size_answer, embedding_matrix_answer, MAX_SEQUENCE_LENGTH_answer, train_sequences_pad_ans, test_sequences_pad_ans = \
prepare_embedding(df_train['answer'], df_test['answer'], 'Answer')

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3215742443.py in <cell line: 0>()
      1 #Calling prepare_embedding method to generate embedding for 3 inputs for both train and test
      2 vocab_size_question_title, embedding_matrix_question_title, MAX_SEQUENCE_LENGTH_question_title, train_sequences_pad_qt, test_sequences_pad_qt = \
----> 3 prepare_embedding(df_train['question_title'], df_test['question_title'], 'Question Title')
      4 
      5 vocab_size_question_body, embedding_matrix_question_body, MAX_SEQUENCE_LENGTH_question_body, train_sequences_pad_qb, test_sequences_pad_qb = \

/tmp/ipykernel_11/2282510748.py in prepare_embedding(input_series_train, input_series_test, column_name)
     33     # Loading Glove embedding layer
     34     embeddings_index = {}
---> 35     f = open(PATH_w2vec_300d+'glove-840B-300d-char_embed.txt')
     36     for line in f:
     37         values = line.split()

FileNotFoundError: [Errno 2] No such file or directory: '../input/glove-300d/glove-840B-300d-char_embed.txt'

## === cell 8
validation_sequences_pad_qt = train_sequences_pad_qt[5000:]
train_sequences_pad_qt = train_sequences_pad_qt[:5000]

validation_sequences_pad_qb = train_sequences_pad_qb[5000:]
train_sequences_pad_qb = train_sequences_pad_qb[:5000]

validation_sequences_pad_ans = train_sequences_pad_ans[5000:]
train_sequences_pad_ans = train_sequences_pad_ans[:5000]

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1452466285.py in <cell line: 0>()
      1 #Splitting into train and validation
----> 2 validation_sequences_pad_qt = train_sequences_pad_qt[5000:]
      3 train_sequences_pad_qt = train_sequences_pad_qt[:5000]
      4 
      5 validation_sequences_pad_qb = train_sequences_pad_qb[5000:]

NameError: name 'train_sequences_pad_qt' is not defined

## === cell 9

embedding_layer_question_title = tf.keras.layers.Embedding(vocab_size_question_title,
                                            300,
                                            weights=[embedding_matrix_question_title],
                                            input_length=MAX_SEQUENCE_LENGTH_question_title,
                                            name = 'Question_Title',
                                            trainable=False)

embedding_layer_question_body = tf.keras.layers.Embedding(vocab_size_question_body,
                                            300,
                                            weights=[embedding_matrix_question_body],
                                            input_length=MAX_SEQUENCE_LENGTH_question_body,
                                            name = 'Question_Body',
                                            trainable=False)

embedding_layer_answer = tf.keras.layers.Embedding(vocab_size_answer,
                                            300,
                                            weights=[embedding_matrix_answer],
                                            input_length=MAX_SEQUENCE_LENGTH_answer,
                                            name = 'Answer',
                                            trainable=False)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3551920909.py in <cell line: 0>()
      1 #Creating Embedding Layers for all 3 inputs:
      2 
----> 3 embedding_layer_question_title = tf.keras.layers.Embedding(vocab_size_question_title,
      4                                             300,
      5                                             weights=[embedding_matrix_question_title],

NameError: name 'vocab_size_question_title' is not defined

## === cell 10
def create_model_2(): 

    input_question_title = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH_question_title,), name = 'IP_Question_Title')
    embedded_question_title = embedding_layer_question_title(input_question_title)

    input_question_body = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH_question_body,), name = 'IP_Question_Body')
    embedded_question_body = embedding_layer_question_body(input_question_body)

    input_answer = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH_answer,), name = 'IP_Answer')
    embedded_answer = embedding_layer_answer(input_answer)

    tower_1 = tf.keras.layers.Conv1D(64, 5, activation='relu')(embedded_question_title) #Kernel Size(M) = 3
    tower_2 = tf.keras.layers.Conv1D(64, 5, activation='relu')(embedded_question_body) #Kernel Size(N) = 5
    tower_3 = tf.keras.layers.Conv1D(64, 5, activation='relu')(embedded_answer) #Kernel Size(O) = 7

    concat = tf.keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)
    max_pool = tf.keras.layers.MaxPooling1D(9)(concat)#9

    tower_1a = tf.keras.layers.Conv1D(64, 5, activation='relu')(max_pool) #Kernel Size(i) = 5
    tower_2b = tf.keras.layers.Conv1D(64, 7, activation='relu')(max_pool) #Kernel Size(j) = 7
    tower_3c = tf.keras.layers.Conv1D(64, 9, activation='relu')(max_pool) #Kernel Size(k) = 9 

    concat2 = tf.keras.layers.concatenate([tower_1a, tower_2b, tower_3c], axis=1)
    max_pool2 = tf.keras.layers.MaxPooling1D(9)(concat2)#9

    convP = tf.keras.layers.Conv1D(64, 9, activation='relu')(max_pool2) #Kernel Size(P) = 9
    flatten = tf.keras.layers.Flatten()(convP)
    dropout = tf.keras.layers.Dropout(0.7)(flatten) #Taking Dropout Rate = 0.2

    dense = tf.keras.layers.Dense(128, activation='relu')(dropout) #128
    preds = tf.keras.layers.Dense(30, activation='sigmoid', name='Output')(dense)

    model_created = tf.keras.models.Model([input_question_title, input_question_body, input_answer], preds, name='Model_Google_QUEST')

    return model_created

model_Google_QUEST = create_model_2()
print(model_Google_QUEST.summary())

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3617742976.py in <cell line: 0>()
     39 
     40 #Calling create_model method and printing summary of model
---> 41 model_Google_QUEST = create_model_2()
     42 print(model_Google_QUEST.summary())

/tmp/ipykernel_11/3617742976.py in create_model_2()
      2 
      3     #Path 1
----> 4     input_question_title = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH_question_title,), name = 'IP_Question_Title')
      5     embedded_question_title = embedding_layer_question_title(input_question_title)
      6 

NameError: name 'MAX_SEQUENCE_LENGTH_question_title' is not defined

## === cell 11
tf.keras.utils.plot_model(model_Google_QUEST)

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1802298219.py in <cell line: 0>()
      1 #Plotting Architecture_1 of Google QUEST:
----> 2 tf.keras.utils.plot_model(model_Google_QUEST)

NameError: name 'model_Google_QUEST' is not defined

## === cell 12

def compute_spearmanr(trues, preds):
    rhos = []
    for col_trues, col_pred in zip(trues.T, preds.T):
        rhos.append(
            spearmanr(col_trues, col_pred + np.random.normal(0, 1e-7, col_pred.shape[0])).correlation)
    return np.nanmean(rhos)


class CustomCallback(tf.keras.callbacks.Callback):
    
    def on_train_begin(self, logs={}):
        self.train_data = {'IP_Question_Title': train_sequences_pad_qt, 'IP_Question_Body': train_sequences_pad_qb, 'IP_Answer': train_sequences_pad_ans}
        self.train_target = df_train[output_categories].values[:5000]

        self.validation_data = {'IP_Question_Title': validation_sequences_pad_qt, 'IP_Question_Body': validation_sequences_pad_qb, 'IP_Answer': validation_sequences_pad_ans}
        self.validation_target = df_train[output_categories].values[5000:]

        self.valid_predictions = []
        self.test_predictions = []
        
    def on_epoch_end(self, epoch, logs={}):
        self.valid_predictions.append(
            self.model.predict(self.validation_data))
        
        rho_val = compute_spearmanr(
            self.validation_target, np.average(self.valid_predictions, axis=0))
        
        print("\nvalidation rho: %.4f" % rho_val)
        
        

custom_callback = CustomCallback()

## === cell 13

train_data = {'IP_Question_Title': train_sequences_pad_qt, 'IP_Question_Body': train_sequences_pad_qb, 'IP_Answer': train_sequences_pad_ans}
train_target = df_train[output_categories].values[:5000]

test_data = {'IP_Question_Title': validation_sequences_pad_qt, 'IP_Question_Body': validation_sequences_pad_qb, 'IP_Answer': validation_sequences_pad_ans}
test_target = df_train[output_categories].values[5000:]

optimizer_adam = tf.keras.optimizers.Adam(learning_rate=0.01)
model_Google_QUEST.compile(loss='mean_squared_error', optimizer=optimizer_adam)
model_Google_QUEST.fit(train_data, train_target, validation_data = (test_data, test_target),
           epochs=100, batch_size=64, verbose=1, callbacks=[custom_callback])

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2017854881.py in <cell line: 0>()
      1 #Compile and fit Model:
      2 
----> 3 train_data = {'IP_Question_Title': train_sequences_pad_qt, 'IP_Question_Body': train_sequences_pad_qb, 'IP_Answer': train_sequences_pad_ans}
      4 train_target = df_train[output_categories].values[:5000]
      5 

NameError: name 'train_sequences_pad_qt' is not defined

## === cell 14
train_prediction = model_Google_QUEST.predict({'IP_Question_Title': train_sequences_pad_qt, 'IP_Question_Body': train_sequences_pad_qb, 'IP_Answer': train_sequences_pad_ans})
validation_prediction = model_Google_QUEST.predict({'IP_Question_Title': validation_sequences_pad_qt, 'IP_Question_Body': validation_sequences_pad_qb, 'IP_Answer': validation_sequences_pad_ans})
test_prediction = model_Google_QUEST.predict({'IP_Question_Title': test_sequences_pad_qt, 'IP_Question_Body': test_sequences_pad_qb, 'IP_Answer': test_sequences_pad_ans})

print("Train Spearman Rank Correlation: ",compute_spearmanr(df_train[output_categories].values[:5000], train_prediction))

print("Validation Spearman Rank Correlation: ",compute_spearmanr(df_train[output_categories].values[5000:], validation_prediction))

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2060167204.py in <cell line: 0>()
----> 1 train_prediction = model_Google_QUEST.predict({'IP_Question_Title': train_sequences_pad_qt, 'IP_Question_Body': train_sequences_pad_qb, 'IP_Answer': train_sequences_pad_ans})
      2 validation_prediction = model_Google_QUEST.predict({'IP_Question_Title': validation_sequences_pad_qt, 'IP_Question_Body': validation_sequences_pad_qb, 'IP_Answer': validation_sequences_pad_ans})
      3 test_prediction = model_Google_QUEST.predict({'IP_Question_Title': test_sequences_pad_qt, 'IP_Question_Body': test_sequences_pad_qb, 'IP_Answer': test_sequences_pad_ans})
      4 
      5 #Train Spearman Rank Correlation

NameError: name 'model_Google_QUEST' is not defined

## === cell 15
test_prediction = model_Google_QUEST.predict({'IP_Question_Title': test_sequences_pad_qt, 'IP_Question_Body': test_sequences_pad_qb, 'IP_Answer': test_sequences_pad_ans})
submission_df = pd.concat([pd.DataFrame(df_test['qa_id']), pd.DataFrame(test_prediction, columns=output_categories)], axis=1)
submission_df.to_csv('submission.csv', index=False)

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2699303170.py in <cell line: 0>()
      1 #Prepare submission file:
----> 2 test_prediction = model_Google_QUEST.predict({'IP_Question_Title': test_sequences_pad_qt, 'IP_Question_Body': test_sequences_pad_qb, 'IP_Answer': test_sequences_pad_ans})
      3 submission_df = pd.concat([pd.DataFrame(df_test['qa_id']), pd.DataFrame(test_prediction, columns=output_categories)], axis=1)
      4 submission_df.to_csv('submission.csv', index=False)

NameError: name 'model_Google_QUEST' is not defined
