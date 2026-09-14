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

3.9

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

0.1664010301247311

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
import pandas as pd
import re
import numpy as np
from pandas.core.frame import DataFrame
from tensorflow.python.framework.indexed_slices import _LARGE_SPARSE_NUM_ELEMENTS
from tensorflow.python.keras.backend import sigmoid
import tensorflow as tf
from numpy.core.defchararray import title
from pandas.core.arrays import categorical
import sklearn.preprocessing
DEFAULT_FILE_PATH = "/kaggle/input/filepython/glove.6B.50d.txt"

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2

def df_process(df=None,train=False,model='rnn_glove'): # train arguement can be used somewhere
    if df is None:
        df = pd.read_csv("google-quest-challenge/train.csv",index_col='qa_id')
    if(model=='rnn_glove'):
        df['question_user_page']=df['question_user_page'].apply(lambda x: x.split("/")[-1])
        df['answer_user_page']=df['answer_user_page'].apply(lambda x: x.split("/")[-1])
        df['category']=df['category'].apply(lambda x:re.sub("[^a-zA-Z]"," ",x).lower())
        df['answer_user_page']=df['answer_user_page'].apply(lambda x: x.split("/")[-1])
        df['host']=df['host'].apply(lambda x: x.split(".")[0])
        df.host=pd.Categorical(df.host)
        df['host'] = df.host.cat.codes
        df.category=pd.Categorical(df.category)
        df['category'] = df.category.cat.codes
        df=df.drop(['question_user_name','answer_user_name','url'],axis=1)
        df['q_len']=df['question_body'].str.len()
        df['t_len']=df['question_title'].str.len()
        df['ans_len']=df['answer'].str.len()
       
        return df




## === cell 3
def loadWordVectors(tokens, filepath=DEFAULT_FILE_PATH, dimensions=50): #tokens is the set of words in training data
    """Read pretrained GloVe vectors"""
    wordVectors = np.random.randn(len(tokens), dimensions) #random normal distribution
    with open(filepath) as ifs:
        for line in ifs:
            line = line.strip()
            if not line:
                continue
            row = line.split()
            token = row[0]
            if token not in tokens:
                continue
            data = [float(x) for x in row[1:]]
            if len(data) != dimensions:
                raise RuntimeError("wrong number of dimensions")
            wordVectors[tokens[token]] = np.asarray(data)
    return wordVectors

## === cell 4
class wordbank:
    def __init__(self,df=None,train=True):
        if df is None:
            df = pd.read_csv("google-quest-challenge/train.csv",index_col='qa_id')
        self.df=df
        self.train=train

    def tokens(self):
        if hasattr(self, "_tokens") and self._tokens: # if already attribute present 
            print("Token attribute present already")
            return self._tokens

        tokens = dict()
        tokenfreq = dict()
        wordcount = 0
        idx = 0
        tokens["<pad>"] = idx
        tokenfreq["<pad>"] = 0 
        wordcount += 1
        idx += 1
        self._sentences=self.questions()+self.answers()+self.titles()
        for sentence in self._sentences:
            for w in sentence:
                wordcount += 1
                if not w in tokens:
                    tokens[w] = idx
                    tokenfreq[w] = 1
                    idx += 1
                else:
                    tokenfreq[w] += 1

        

        self._tokens = tokens
        self._tokenfreq = tokenfreq
        self._wordcount = wordcount
        self._titles=[[tokens[word] for word in sent] for sent in self.titles()]
        self._questions=[[tokens[word] for word in sent] for sent in self.questions()]
        self._answers=[[tokens[word] for word in sent] for sent in self.answers()]
        return self._tokens

    '''Different functions if diffferent type of preprocessing required'''

    def answers(self):
        if hasattr(self, "_answers") and self._answers:
            return self._answers

        ans_df=self.df['answer']
        ans_df=ans_df.apply(lambda x: (re.sub("[^a-zA-Z]"," ",x)).strip().split())
        ans_df=ans_df.apply(lambda x: [w.lower() for w in x])
        self._answers = ans_df.tolist()
        return self._answers
    def titles(self):
        if hasattr(self, "_titles") and self._titles:
            return self._titles

        titles_df=self.df['question_title']
        titles_df=titles_df.apply(lambda x: (re.sub("[^a-zA-Z]"," ",x)).strip().split())
        titles_df=titles_df.apply(lambda x: [w.lower() for w in x])
        self._titles = titles_df.tolist()
        return self._titles
    def questions(self):
        if hasattr(self, "_questions") and self._questions:
            return self._questions

        q_df=self.df['question_body']
        q_df=q_df.apply(lambda x: (re.sub("[^a-zA-Z]"," ",x)).strip().split())
        q_df=q_df.apply(lambda x: [w.lower() for w in x])
        self._questions = q_df.tolist()
        return self._questions

    def numSentences(self):
        if hasattr(self, "_numSentences") and self._numSentences:
            return self._numSentences
        else:
            self._numSentences = len(self._answers())
            return self._numSentences
    def numQ(self): # numbe of unique questions or titles
        if hasattr(self, "_numQ") and self._numQ:
            return self._numQ
        else:
            self._numQ = self.df['question_title'].nunique()
            return self._numQ
    def test(self):
        ind_unk=self._tokens['unknown']
        self._titles=[[self._tokens.get(word,ind_unk) for word in sent] for sent in self.titles()]
        self._questions=[[self._tokens.get(word,ind_unk) for word in sent] for sent in self.questions()]
        self._answers=[[self._tokens.get(word,ind_unk) for word in sent] for sent in self.answers()]
        

## === cell 5
'''First Model'''
class rnnGoogleQuest:
    def __init__(self,data,vocab,embedding_matrix=None,max_length=0):
        self._data=data
        self._vocab=vocab
        self._embedding_matrix=embedding_matrix
        self._embedding_dim=self._embedding_matrix.shape[1]
        self._num_vocab=len(self._vocab)
        self._max_length=max_length
    def embeddings(self,input):
        embedding_layer = tf.keras.layers.Embedding(
            self._num_vocab,
            self._embedding_dim,
            embeddings_initializer=tf.keras.initializers.Constant(self._embedding_matrix),
            trainable=True,mask_zero=True)(input)
        lstm_layer=tf.keras.layers.LSTM(64,return_state=True)(embedding_layer)
        return lstm_layer
    def rnn_model(self):
        
        title_input = tf.keras.Input(shape=(None,), name="title")  # Variable-length sequence of ints
        question_input = tf.keras.Input(shape=(None,), name="question")  # Variable-length sequence of ints
        answer_input=tf.keras.Input(shape=(None,), name="answer") # Variable-length sequence of ints
        category_input = tf.keras.Input(shape=(5,), name="category")  # category binary input vector (5 categories)
        host_input=tf.keras.Input(shape=(59,), name="host") #host category(59 total)
        stats_input=tf.keras.Input(shape=(3,),name='stats') #new created features
        encoder_outputs, title_state_h, state_c  = self.embeddings(title_input)
        
        encoder_outputs, question_state_h, state_c = self.embeddings(question_input)
        encoder_outputs, answer_state_h, state_c = self.embeddings(answer_input)
    
        features= tf.keras.layers.concatenate([title_state_h, question_state_h, answer_state_h,category_input,host_input,stats_input])
        hidden=tf.keras.layers.Dense(512,activation='relu',name='hidden')(features)
        drop1=tf.keras.layers.Dropout(0.4)(hidden)
        hidden2=tf.keras.layers.Dense(128,activation='relu',name='hidden2')(drop1)
        drop2=tf.keras.layers.Dropout(0.3)(hidden2)
        pred=tf.keras.layers.Dense(30,activation='sigmoid',name='prediction')(drop2)
        model = tf.keras.Model(inputs=[title_input, question_input, answer_input,category_input,host_input,stats_input],outputs=pred)
        self._model=model
        tf.keras.utils.plot_model(self._model, "rnn_model.png", show_shapes=True)
    
    '''def answerEmbeddings(self):  
    def questionEmbeddings(self):
    def titleEmbeddings(self):
    def categoryEmbeddings(self):'''
    
    def train(self):
        self.rnn_model()
        self._model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
        name='Adam'
        ),
        loss='mean_absolute_error',metrics=['mean_squared_error'])
        self._model.fit(
        {"title": self._data['title'], "question": self._data['question'], "answer": self._data['answer'],"category":self._data['category'],"host":self._data['host'],"stats":self._data['stats']},
        {"prediction": self._data['output']},
        epochs=25,
        batch_size=128,
        )
        self._model.save('rnn_glove_model')


## === cell 6
def multi_hot_enc(array,label=None):
    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    label_binarizer.fit(range(max(train[label].to_numpy())+1)) # for test
    array=label_binarizer.transform(array)
    return array
def normalize(df):
    x = df.values #returns a numpy array
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    return x_scaled

train=pd.read_csv("/kaggle/input/google-quest-challenge/train.csv",index_col='qa_id') #6079 rows
train=df_process(train,True)
Gq=wordbank(train)
tokens=Gq.tokens()

word_vectors=loadWordVectors(tokens)
titles_vec=Gq._titles
questions_vec=Gq._questions
answers_vec=Gq._answers
'''if not any(Gq._titles+Gq._answers+Gq._questions):
    print("Empty list!")
data={}
data['question']=tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences(
    questions_vec, padding="post"))
data['answer']=tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences(
    answers_vec, padding="post"))
data['title']=tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences(
    titles_vec, padding="post"))
data['host']=tf.convert_to_tensor(multi_hot_enc(train['host'].to_numpy(),'host'))
data['category']=tf.convert_to_tensor(multi_hot_enc(train['category'].to_numpy(),'category'))
data['stats']=tf.convert_to_tensor(normalize(train[['q_len','t_len','ans_len']]))
data['output']=tf.convert_to_tensor(train.iloc[:,7:37].values)
model=rnnGoogleQuest(data,tokens,word_vectors)
model.train()

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1532024.py in <cell line: 0>()
     15 tokens=Gq.tokens()
     16 
---> 17 word_vectors=loadWordVectors(tokens)
     18 titles_vec=Gq._titles
     19 #title_vec = tf.keras.preprocessing.sequence.pad_sequences(titles_vec, padding="post")

/tmp/ipykernel_11/1048423058.py in loadWordVectors(tokens, filepath, dimensions)
      2     """Read pretrained GloVe vectors"""
      3     wordVectors = np.random.randn(len(tokens), dimensions) #random normal distribution
----> 4     with open(filepath) as ifs:
      5         for line in ifs:
      6             line = line.strip()

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/filepython/glove.6B.50d.txt'

## === cell 7
pred_model=tf.keras.models.load_model('rnn_glove_model')

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/360043706.py in <cell line: 0>()
----> 1 pred_model=tf.keras.models.load_model('rnn_glove_model')

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    204         )
    205     else:
--> 206         raise ValueError(
    207             f"File format not supported: filepath={filepath}. "
    208             "Keras 3 only supports V3 `.keras` files and "

ValueError: File format not supported: filepath=rnn_glove_model. Keras 3 only supports V3 `.keras` files and legacy H5 format files (`.h5` extension). Note that the legacy SavedModel format is not supported by `load_model()` in Keras 3. In order to reload a TensorFlow SavedModel as an inference-only layer in Keras 3, use `keras.layers.TFSMLayer(rnn_glove_model, call_endpoint='serving_default')` (note that your `call_endpoint` might have a different name).

## === cell 8
test=pd.read_csv("/kaggle/input/google-quest-challenge/test.csv",index_col='qa_id')
test2=test.copy()
test2=df_process(test2,False)
GqTest=wordbank(test2,train=False)
GqTest._tokens=tokens # train tokens

GqTest.test()
titles_vecT=GqTest._titles
questions_vecT=GqTest._questions
answers_vecT=GqTest._answers
dataT={}
dataT['question']=tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences(
    questions_vecT, padding="post"))
dataT['answer']=tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences(
    answers_vecT, padding="post"))
dataT['title']=tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences(
    titles_vecT, padding="post"))
dataT['host']=tf.convert_to_tensor(multi_hot_enc(test2['host'].to_numpy(),'host'))
dataT['category']=tf.convert_to_tensor(multi_hot_enc(test2['category'].to_numpy(),'category'))
dataT['stats']=tf.convert_to_tensor(normalize(test2[['q_len','t_len','ans_len']]))
df = pd.DataFrame(np.array(pred_model.predict(
    {"title": dataT['title'], "question": dataT['question'], "answer": dataT['answer'],"category":dataT['category'],"host":dataT['host'],"stats":dataT['stats']})))

df.index=test.index
test[list(train.iloc[:,7:37].columns)]=df.iloc[:,0:30]
test.to_csv('submission.csv') 

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2106293778.py in <cell line: 0>()
     20 dataT['category']=tf.convert_to_tensor(multi_hot_enc(test2['category'].to_numpy(),'category'))
     21 dataT['stats']=tf.convert_to_tensor(normalize(test2[['q_len','t_len','ans_len']]))
---> 22 df = pd.DataFrame(np.array(pred_model.predict(
     23     {"title": dataT['title'], "question": dataT['question'], "answer": dataT['answer'],"category":dataT['category'],"host":dataT['host'],"stats":dataT['stats']})))
     24 

NameError: name 'pred_model' is not defined

## === cell 9
test=test.drop(['question_title', 'question_body','question_user_name','question_user_page','answer','answer_user_name','answer_user_page','url','category','host'], axis = 1)

## === cell 10
test.to_csv('submission.csv') 

## --- ERROR in outputing the csv:
Invalid submission: Submission is missing the following columns: {'answer_type_procedure', 'question_has_commonly_accepted_answer', 'question_type_definition', 'question_multi_intent', 'question_type_choice', 'answer_type_reason_explanation', 'question_type_compare', 'question_type_instructions', 'question_type_procedure', 'answer_helpful', 'question_type_spelling', 'question_conversational', 'question_opinion_seeking', 'question_not_really_a_question', 'question_asker_intent_understanding', 'answer_level_of_information', 'answer_plausible', 'answer_type_instructions', 'question_type_reason_explanation', 'question_interestingness_self', 'question_type_entity', 'question_fact_seeking', 'question_interestingness_others', 'answer_well_written', 'question_type_consequence', 'question_expect_short_answer', 'question_well_written', 'answer_satisfaction', 'answer_relevance', 'question_body_critical'}
