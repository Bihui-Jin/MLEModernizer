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
Given a dataset of comments from Wikipedia's talk page edits, predict the probability of each comment being toxic.

## Metric
Mean column-wise ROC AUC; the average of the individual AUCs of each predicted column.

## Submission Format
For each `id` in the test set, you must predict a probability for each of the six possible types of comment toxicity (toxic, severe_toxic, obscene, threat, insult, identity_hate). The columns must be in the same order as shown below. The file should contain a header and have the following format:

```
id,toxic,severe_toxic,obscene,threat,insult,identity_hate
00001cee341fdb12,0.5,0.5,0.5,0.5,0.5,0.5
0000247867823ef7,0.5,0.5,0.5,0.5,0.5,0.5
etc.
```

## Dataset 
- **train.csv** - the training set, contains comments with their binary labels
- **test.csv** - the test set, you must predict the toxicity probabilities for these comments.
- **sample_submission.csv** - a sample submission file in the correct format

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
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        input/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        working/
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
```

-> data/jigsaw-toxic-comment-classification-challenge/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/jigsaw-toxic-comment-classification-challenge/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/jigsaw-toxic-comment-classification-challenge/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> (stopped after 10 files for performance)

# 5. Target score

0.96744

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
import matplotlib.pyplot as plt
import re
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer 
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Dropout, Conv1D, GlobalMaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.vis_utils import plot_model
from sklearn.metrics import roc_auc_score

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
!unzip /kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip
!unzip /kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip


## === cell 3
df_train=pd.read_csv('./train.csv')
print('Shape=>',df_train.shape)
df_train.head()


## === cell 4
df_test=pd.read_csv('./test.csv')
print('Shape=>',df_test.shape)
df_test.head()


## === cell 5
for i in ['toxic','severe_toxic','obscene','threat','insult','identity_hate']:
    print(df_train[i].value_counts(normalize=True)*100)


## === cell 6
fig,axes=plt.subplots(3,2,figsize=(15,15))

for ax,class_name in zip(axes.flatten(),['toxic','severe_toxic','obscene','threat','insult','identity_hate']):
    pd.value_counts(df_train[class_name],sort=True).plot(kind='bar',rot=0,ax=ax)
    ax.set_title('{} Distribution'.format(class_name))
    ax.set_xticks([0,1])
    ax.set_xlabel('Labels')
    ax.set_ylabel('Frequency')

plt.show()


## === cell 7
def cleaner(text):
    text=text.lower()
    text=re.sub("[^a-z]+"," ",text)
    text=re.sub("[ ]+"," ",text)
    
    return text


## === cell 8
df_train['cleaned']=df_train['comment_text'].apply(cleaner)


## === cell 9
df_train['comment_text'][:2].values


## === cell 10
df_train['cleaned'][:2].values


## === cell 11
df_test['cleaned']=df_test['comment_text'].apply(cleaner)


## === cell 12
df_train.head()


## === cell 13
df_test.head()


## === cell 14
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_train['cleaned'])


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3087657149.py in <cell line: 0>()
----> 1 tokenizer = Tokenizer()
      2 #creating index for words
      3 tokenizer.fit_on_texts(df_train['cleaned'])

NameError: name 'Tokenizer' is not defined

## === cell 15
print('Vocabulary Size=>',len(tokenizer.word_index))


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/858116680.py in <cell line: 0>()
----> 1 print('Vocabulary Size=>',len(tokenizer.word_index))

NameError: name 'tokenizer' is not defined

## === cell 16
train_seq = tokenizer.texts_to_sequences(df_train['cleaned']) 
test_seq = tokenizer.texts_to_sequences(df_test['cleaned'])


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2728590566.py in <cell line: 0>()
      1 # Converting word sequence to integer sequence
----> 2 train_seq = tokenizer.texts_to_sequences(df_train['cleaned'])
      3 test_seq = tokenizer.texts_to_sequences(df_test['cleaned'])

NameError: name 'tokenizer' is not defined

## === cell 17
train_seq=pad_sequences(train_seq,maxlen=100,padding='post')
test_seq=pad_sequences(test_seq,maxlen=100,padding='post')


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3026023900.py in <cell line: 0>()
      1 # Padding with zero
----> 2 train_seq=pad_sequences(train_seq,maxlen=100,padding='post')
      3 test_seq=pad_sequences(test_seq,maxlen=100,padding='post')

NameError: name 'train_seq' is not defined

## === cell 18
vocabulary=len(tokenizer.word_index)+1
print('Vocabulary Size=>',vocabulary)


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1324246316.py in <cell line: 0>()
----> 1 vocabulary=len(tokenizer.word_index)+1
      2 print('Vocabulary Size=>',vocabulary)

NameError: name 'tokenizer' is not defined

## === cell 19
print('Shape of train_sequence=>',train_seq.shape)
print('Shape of test_sequence=>',test_seq.shape)


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4265567356.py in <cell line: 0>()
----> 1 print('Shape of train_sequence=>',train_seq.shape)
      2 print('Shape of test_sequence=>',test_seq.shape)

NameError: name 'train_seq' is not defined

## === cell 20
y_train=df_train[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].values
print('Shape of Training Labels=>',y_train.shape)


## === cell 21
!pip install iterative-stratification


## === cell 22
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


## === cell 23
msss=MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)


## === cell 24
for train_index, val_index in msss.split(train_seq, y_train):
    x_train_split,y_train_split=train_seq[train_index],y_train[train_index]
    x_valid_split,y_valid_split=train_seq[val_index],y_train[val_index]


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1463602180.py in <cell line: 0>()
----> 1 for train_index, val_index in msss.split(train_seq, y_train):
      2     # Creating Train Set
      3     x_train_split,y_train_split=train_seq[train_index],y_train[train_index]
      4     # Creating Test Set
      5     x_valid_split,y_valid_split=train_seq[val_index],y_train[val_index]

NameError: name 'train_seq' is not defined

## === cell 25
print('Shape of Train Split=>',x_train_split.shape,y_train_split.shape)
print('Shape of Validation Split=>',x_valid_split.shape,y_valid_split.shape)


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2315397513.py in <cell line: 0>()
----> 1 print('Shape of Train Split=>',x_train_split.shape,y_train_split.shape)
      2 print('Shape of Validation Split=>',x_valid_split.shape,y_valid_split.shape)

NameError: name 'x_train_split' is not defined

## === cell 26
print('Class Distribution of Train Split in Percentage')
for i,v in enumerate(['toxic','severe_toxic','obscene','threat','insult','identity_hate']):
    print(v)
    print(pd.Series(y_train_split[:,i]).value_counts(normalize=True)*100)


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2585741069.py in <cell line: 0>()
      2 for i,v in enumerate(['toxic','severe_toxic','obscene','threat','insult','identity_hate']):
      3     print(v)
----> 4     print(pd.Series(y_train_split[:,i]).value_counts(normalize=True)*100)

NameError: name 'y_train_split' is not defined

## === cell 27
print('Class Distribution of Validation Split in Percentage')
for i,v in enumerate(['toxic','severe_toxic','obscene','threat','insult','identity_hate']):
    print(v)
    print(pd.Series(y_valid_split[:,i]).value_counts(normalize=True)*100)


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1618156633.py in <cell line: 0>()
      2 for i,v in enumerate(['toxic','severe_toxic','obscene','threat','insult','identity_hate']):
      3     print(v)
----> 4     print(pd.Series(y_valid_split[:,i]).value_counts(normalize=True)*100)

NameError: name 'y_valid_split' is not defined

## === cell 28
input_1=Input(shape=(100,))
embedding_1=Embedding(vocabulary,100)(input_1)
conv_1=Conv1D(filters=352,kernel_size=7,padding="same")(embedding_1)
dropout_1=Dropout(0.06675)(conv_1)
pool_1=GlobalMaxPooling1D()(dropout_1)

dense=Dense(128,activation='relu')(pool_1)
output=Dense(6,activation='sigmoid')(dense)

model=Model(inputs=[input_1],outputs=output)

model.summary()


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1216011674.py in <cell line: 0>()
----> 1 input_1=Input(shape=(100,))
      2 embedding_1=Embedding(vocabulary,100)(input_1)
      3 conv_1=Conv1D(filters=352,kernel_size=7,padding="same")(embedding_1)
      4 dropout_1=Dropout(0.06675)(conv_1)
      5 pool_1=GlobalMaxPooling1D()(dropout_1)

NameError: name 'Input' is not defined

## === cell 29
plot_model(model, to_file= 'model.png', show_shapes=True)


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1491628891.py in <cell line: 0>()
----> 1 plot_model(model, to_file= 'model.png', show_shapes=True)

NameError: name 'plot_model' is not defined

## === cell 30
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=["accuracy"])


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1669191790.py in <cell line: 0>()
      1 # Compile Model
----> 2 model.compile(optimizer='adam',loss='binary_crossentropy',metrics=["accuracy"])

NameError: name 'model' is not defined

## === cell 31
es=EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=5,min_delta=1e-5)
mc = ModelCheckpoint("/kaggle/working/model.hdf5", monitor='val_loss', verbose=0, save_best_only=True, mode='min')


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3445514338.py in <cell line: 0>()
      1 # Callbacks
----> 2 es=EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=5,min_delta=1e-5)
      3 mc = ModelCheckpoint("/kaggle/working/model.hdf5", monitor='val_loss', verbose=0, save_best_only=True, mode='min')

NameError: name 'EarlyStopping' is not defined

## === cell 32
model.fit(x_train_split,y_train_split, batch_size=512, epochs=100, verbose=1, validation_data=(x_valid_split,y_valid_split), callbacks=[es,mc])


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1098553906.py in <cell line: 0>()
----> 1 model.fit(x_train_split,y_train_split, batch_size=512, epochs=100, verbose=1, validation_data=(x_valid_split,y_valid_split), callbacks=[es,mc])

NameError: name 'model' is not defined

## === cell 33
train_pred=model.predict(x_train_split)
print('In-sample Evaluation ROC-AUC Score:\n',roc_auc_score(y_train_split,train_pred))


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/62049048.py in <cell line: 0>()
      1 # In-sample Evaluation
----> 2 train_pred=model.predict(x_train_split)
      3 print('In-sample Evaluation ROC-AUC Score:\n',roc_auc_score(y_train_split,train_pred))

NameError: name 'model' is not defined

## === cell 34
valid_pred=model.predict(x_valid_split)
print('In-sample Evaluation ROC-AUC Score:\n',roc_auc_score(y_valid_split,valid_pred))


## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2405947013.py in <cell line: 0>()
      1 # Out-of-sample Evaluation
----> 2 valid_pred=model.predict(x_valid_split)
      3 print('In-sample Evaluation ROC-AUC Score:\n',roc_auc_score(y_valid_split,valid_pred))

NameError: name 'model' is not defined

## === cell 35
final_pred=model.predict(test_seq)


## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3760480301.py in <cell line: 0>()
----> 1 final_pred=model.predict(test_seq)

NameError: name 'model' is not defined

## === cell 36
prob=pd.DataFrame(columns=['id','toxic','severe_toxic','obscene','threat','insult','identity_hate'])
prob['id']=df_test['id']
for index,value in enumerate(['toxic','severe_toxic','obscene','threat','insult','identity_hate']):
    prob[value]=final_pred[:,index]


## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3959838350.py in <cell line: 0>()
      3 prob['id']=df_test['id']
      4 for index,value in enumerate(['toxic','severe_toxic','obscene','threat','insult','identity_hate']):
----> 5     prob[value]=final_pred[:,index]

NameError: name 'final_pred' is not defined

## === cell 37
prob


## === cell 38
prob.to_csv('submission-CNN-final.csv',index=False)


## --- ERROR in outputing the csv:
Invalid submission: Submission is missing the following columns: {'insult', 'obscene', 'threat', 'toxic', 'identity_hate', 'severe_toxic'}
