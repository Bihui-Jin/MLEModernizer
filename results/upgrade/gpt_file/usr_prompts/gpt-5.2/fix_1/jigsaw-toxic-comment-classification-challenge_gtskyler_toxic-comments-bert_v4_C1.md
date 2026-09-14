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

3.9

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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sentence-transformers==4.1.0
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
transformers==4.53.3

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

0.98559

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
from transformers import TFBertModel,  BertConfig, BertTokenizerFast, TFAutoModel

from tensorflow.keras.layers import Input, Dropout, Dense, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical

import pandas as pd
import tensorflow as tf
import re
import numpy as np
from sklearn.model_selection import train_test_split


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
!unzip ../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip
!unzip ../input/jigsaw-toxic-comment-classification-challenge/test.csv.zip


## === cell 2
df=pd.read_csv('train.csv')

df = df.sample(frac=1)


## === cell 3
df.head()


## === cell 4
def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text


## === cell 5
df['comment_text'] = df['comment_text'].map(lambda x : clean_text(x))


## === cell 6
train_sentences = df["comment_text"].fillna("CVxTz").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
train_y = df[list_classes].values


## === cell 7
df.head()


## === cell 8
model_name = 'bert-base-uncased'

max_length = 128

config = BertConfig.from_pretrained(model_name)

tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = config)
bert = TFAutoModel.from_pretrained(model_name)


## === cell 9
input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
attention_mask = Input(shape=(max_length,), name='attention_mask', dtype='int32') 
inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
x = bert.bert(inputs)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1445753935.py in <cell line: 0>()
      2 attention_mask = Input(shape=(max_length,), name='attention_mask', dtype='int32')
      3 inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
----> 4 x = bert.bert(inputs)

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
     68             # To get the full stack trace, call:
     69             # `tf.debugging.disable_traceback_filtering()`
---> 70             raise e.with_traceback(filtered_tb) from None
     71         finally:
     72             del filtered_tb

/usr/local/lib/python3.11/dist-packages/transformers/modeling_tf_utils.py in run_call_with_unpacked_inputs(self, *args, **kwargs)
    434             config = self.config
    435 
--> 436         unpacked_inputs = input_processing(func, config, **fn_args_and_kwargs)
    437         return func(self, **unpacked_inputs)
    438 

/usr/local/lib/python3.11/dist-packages/transformers/modeling_tf_utils.py in input_processing(func, config, **kwargs)
    559                 continue
    560             else:
--> 561                 raise ValueError(f"Data of type {type(v)} is not allowed only {allowed_types} is accepted for {k}.")
    562     else:
    563         if tf.is_tensor(main_input) or main_input is None:

ValueError: Exception encountered when calling layer 'bert' (type TFBertMainLayer).

Data of type <class 'keras.src.backend.common.keras_tensor.KerasTensor'> is not allowed only (<class 'tensorflow.python.framework.tensor.Tensor'>, <class 'bool'>, <class 'int'>, <class 'transformers.utils.generic.ModelOutput'>, <class 'tuple'>, <class 'list'>, <class 'dict'>, <class 'numpy.ndarray'>) is accepted for input_ids.

Call arguments received by layer 'bert' (type TFBertMainLayer):
  • input_ids={'input_ids': '<KerasTensor shape=(None, 128), dtype=int32, sparse=False, name=input_ids>', 'attention_mask': '<KerasTensor shape=(None, 128), dtype=int32, sparse=False, name=attention_mask>'}
  • attention_mask=None
  • token_type_ids=None
  • position_ids=None
  • head_mask=None
  • inputs_embeds=None
  • encoder_hidden_states=None
  • encoder_attention_mask=None
  • past_key_values=None
  • use_cache=None
  • output_attentions=None
  • output_hidden_states=None
  • return_dict=None
  • training=False

## === cell 10
x


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3618968493.py in <cell line: 0>()
----> 1 x

NameError: name 'x' is not defined

## === cell 11
x2 = GlobalAveragePooling1D()(x[0])
y =Dense(len(list_classes), activation='sigmoid', name='outputs')(x2)

model = Model(inputs=inputs, outputs=y)

model.summary()


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/542367789.py in <cell line: 0>()
      1 #x2 =Dense(512, activation='relu')(x[1])
----> 2 x2 = GlobalAveragePooling1D()(x[0])
      3 #x3 = Dropout(0.5)(x2)
      4 y =Dense(len(list_classes), activation='sigmoid', name='outputs')(x2)
      5 

NameError: name 'x' is not defined

## === cell 12
optimizer = Adam(lr=1e-5, decay=1e-6)
model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3387398201.py in <cell line: 0>()
----> 1 optimizer = Adam(lr=1e-5, decay=1e-6)
      2 model.compile(loss='binary_crossentropy',
      3                   optimizer=optimizer,
      4                   metrics=['accuracy'])

/usr/local/lib/python3.11/dist-packages/keras/src/optimizers/adam.py in __init__(self, learning_rate, beta_1, beta_2, epsilon, amsgrad, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, **kwargs)
     60         **kwargs,
     61     ):
---> 62         super().__init__(
     63             learning_rate=learning_rate,
     64             name=name,

/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/optimizer.py in __init__(self, *args, **kwargs)
     19 class TFOptimizer(KerasAutoTrackable, base_optimizer.BaseOptimizer):
     20     def __init__(self, *args, **kwargs):
---> 21         super().__init__(*args, **kwargs)
     22         self._distribution_strategy = tf.distribute.get_strategy()
     23 

/usr/local/lib/python3.11/dist-packages/keras/src/optimizers/base_optimizer.py in __init__(self, learning_rate, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, **kwargs)
     88             )
     89         if kwargs:
---> 90             raise ValueError(f"Argument(s) not recognized: {kwargs}")
     91 
     92         if name is None:

ValueError: Argument(s) not recognized: {'lr': 1e-05}

## === cell 13
x = tokenizer(
    text=list(train_sentences),
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)


## === cell 14
history = model.fit(
    x={'input_ids': x['input_ids'], 'attention_mask': x['attention_mask']},
    y={'outputs': train_y},
    validation_split=0.1,
    batch_size=32,
    epochs=1)


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/36320832.py in <cell line: 0>()
----> 1 history = model.fit(
      2     x={'input_ids': x['input_ids'], 'attention_mask': x['attention_mask']},
      3     #x={'input_ids': x['input_ids']},
      4     y={'outputs': train_y},
      5     validation_split=0.1,

NameError: name 'model' is not defined

## === cell 15
test_df=pd.read_csv("test.csv")
test_df['comment_text']=test_df['comment_text'].map(lambda x : clean_text(x))
test_sentences = test_df["comment_text"].fillna("CVxTz").values


## === cell 16
test_x = tokenizer(
    text=list(test_sentences),
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)


## === cell 17
del test_sentences
del x
del df


## === cell 18
import gc
gc.collect()


## === cell 19
test_x


## === cell 20
predictions=model.predict(x={'input_ids': test_x['input_ids'], 'attention_mask': test_x['attention_mask']},batch_size=32)


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1135755896.py in <cell line: 0>()
----> 1 predictions=model.predict(x={'input_ids': test_x['input_ids'], 'attention_mask': test_x['attention_mask']},batch_size=32)

NameError: name 'model' is not defined

## === cell 21
predictions


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4129765282.py in <cell line: 0>()
----> 1 predictions

NameError: name 'predictions' is not defined

## === cell 22
submission=pd.DataFrame(predictions,columns=list_classes)
submission['id'] = test_df['id']
submission=submission[['id']+(list_classes)]
submission.to_csv("submission.csv", index=False)


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1961808539.py in <cell line: 0>()
----> 1 submission=pd.DataFrame(predictions,columns=list_classes)
      2 submission['id'] = test_df['id']
      3 submission=submission[['id']+(list_classes)]
      4 submission.to_csv("submission.csv", index=False)

NameError: name 'predictions' is not defined

## === cell 23
submission


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/351652756.py in <cell line: 0>()
----> 1 submission

NameError: name 'submission' is not defined

## === cell 24
!ls


## --- ERROR in outputing the csv:
Invalid submission: Submission is missing the following columns: {'obscene', 'insult', 'toxic', 'severe_toxic', 'threat', 'identity_hate'}
