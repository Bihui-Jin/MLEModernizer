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
Given pairs of phrases (an `anchor` and a `target` phrase), build a model to rate how similar they are.  

## Metric
Pearson correlation coefficient.

## Submission Format
For each `id` (representing a pair of phrases) in the test set, you must predict the similarity `score`. The file should contain a header and have the following format:

```
id,score
4112d61851461f60,0
09e418c93a776564,0.25
36baf228038e314b,1
etc.

```

## Dataset
The scores are in the 0-1 range with increments of 0.25 with the following meanings:

- **1.0** - Very close match. This is typically an exact match except possibly for differences in conjugation, quantity (e.g. singular vs. plural), and addition or removal of stopwords (e.g. "the", "and", "or").
- **0.75** - Close synonym, e.g. "mobile phone" vs. "cellphone". This also includes abbreviations, e.g. "TCP" -> "transmission control protocol".
- **0.5** - Synonyms which don't have the same meaning (same function, same properties). This includes broad-narrow (hyponym) and narrow-broad (hypernym) matches.
- **0.25** - Somewhat related, e.g. the two phrases are in the same high level domain but are not synonyms. This also includes antonyms.
- **0.0** - Unrelated.

Files
-----

- **train.csv** - the training set, containing phrases, contexts, and their similarity scores
- **test.csv** - the test set set, identical in structure to the training set but without the score
- **sample_submission.csv** - a sample submission file in the correct format

Columns
-------

- `id` - a unique identifier for a pair of phrases
- `anchor` - the first phrase
- `target` - the second phrase
- `context` - the [CPC classification (version 2021.05)](https://en.wikipedia.org/wiki/Cooperative_Patent_Classification), which indicates the subject within which the similarity is to be scored
- `score` - the similarity. This is sourced from a combination of one or more manual expert ratings.

# 2. Python version

3.10

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
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        input/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        working/
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
```

-> data/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> data/us-patent-phrase-to-phrase-matching/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/us-patent-phrase-to-phrase-matching/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/us-patent-phrase-to-phrase-matching/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> input/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> (stopped after 10 files for performance)

# 5. Target score

-0.3844

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
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
train= pd.read_csv('../input/us-patent-phrase-to-phrase-matching/train.csv')
test= pd.read_csv('../input/us-patent-phrase-to-phrase-matching/test.csv')


## === cell 3
train.head()


## === cell 4
X_train, X_test, y_train, y_test = train_test_split(train[['anchor', 'target']], train['score'], test_size=0.25, random_state=42)


## === cell 5
print(X_train.head())
print(X_test.head())
print(y_train.head())
print(y_test.head())


## === cell 6
X_train['text'] = X_train[['anchor', 'target']].apply(lambda x:str(x[0])+" "+str(x[1]), axis=1)


## === cell 7
X_train.head()


## === cell 8
X_train['anchor'].str.len().plot(kind='hist')


## === cell 9
X_train['target'].str.len().plot(kind='hist')


## === cell 10
print(max(X_train['anchor'].str.len()))

print(max(X_train['target'].str.len()))


## === cell 11
X_train.head()


## === cell 12
vocab_size = 7905
embedding_dim = 16
max_length = 4
trunc_type='post'
oov_tok = ""


## === cell 13
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train['text'].values)


## === cell 14
word_index = tokenizer.word_index


## === cell 15
anchor_sequences = tokenizer.texts_to_sequences(X_train['anchor'].values)
target_sequences = tokenizer.texts_to_sequences(X_train['target'].values)

padded_anchor_sequences = pad_sequences(anchor_sequences, maxlen=max_length, truncating=trunc_type)
padded_target_sequences = pad_sequences(target_sequences, maxlen=max_length, truncating=trunc_type)


## === cell 16
padded_anchor_sequences.shape[1]


## === cell 17
val_anchor_sequences = tokenizer.texts_to_sequences(X_test['anchor'].values)
val_target_sequences = tokenizer.texts_to_sequences(X_test['target'].values)

val_padded_anchor_sequences = pad_sequences(val_anchor_sequences, maxlen=max_length, truncating=trunc_type)
val_padded_target_sequences = pad_sequences(val_target_sequences, maxlen=max_length, truncating=trunc_type)


## === cell 18
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.keras.utils.vis_utils import plot_model
from keras import backend as K


## === cell 19
'''
def base_network():
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
    ])
    
'''


## === cell 20
def initialize_base_network():
    input= Input(shape=(padded_anchor_sequences.shape[1],))
    common_embedding= tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length)(input)
    common_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True))(common_embedding)
    flatten_layer= Flatten()(common_lstm)
    return Model(inputs=input, outputs= flatten_layer)


## === cell 21
def euclidean_distance(vects):
    x,y = vects
    sum_square= K.sum(K.square(x-y), axis=1, keepdims= True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2= shapes
    return (shape1[0],1)


## === cell 22
base_network= initialize_base_network()
plot_model(base_network, show_shapes= True)


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/1307627165.py in <cell line: 0>()
      1 base_network= initialize_base_network()
----> 2 plot_model(base_network, show_shapes= True)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/keras/utils/vis_utils.py in plot_model(model, to_file, show_shapes, show_dtype, show_layer_names, rankdir, expand_nested, dpi)
    317     This enables in-line display of the model plots in notebooks.
    318   """
--> 319   dot = model_to_dot(
    320       model,
    321       show_shapes=show_shapes,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/keras/utils/vis_utils.py in model_to_dot(model, show_shapes, show_dtype, show_layer_names, rankdir, expand_nested, dpi, subgraph)
     94     ImportError: if graphviz or pydot are not available.
     95   """
---> 96   from tensorflow.python.keras.layers import wrappers
     97   from tensorflow.python.keras.engine import sequential
     98   from tensorflow.python.keras.engine import functional

ImportError: cannot import name 'wrappers' from 'tensorflow.python.keras.layers' (/usr/local/lib/python3.11/dist-packages/tensorflow/python/keras/layers/__init__.py)

## === cell 23
input_1 = Input(shape=(padded_anchor_sequences.shape[1],), name= 'input_1')
input_2 = Input(shape=(padded_target_sequences.shape[1],), name= 'input_2')

vec_1 = base_network(input_1)
vec_2 = base_network(input_2)

output = Lambda(euclidean_distance, name="output_layer")([vec_1, vec_2])

siamese_model = Model([input_1, input_2], output)

plot_model(siamese_model, show_shapes=True, show_layer_names=True, to_file='outer-model.png')


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NotImplementedError                       Traceback (most recent call last)
/tmp/ipykernel_11/2023786234.py in <cell line: 0>()
      6 
      7 #output = Lambda(euclidean_distance, name="output_layer", output_shape=eucl_dist_output_shape)([vec_1, vec_2])
----> 8 output = Lambda(euclidean_distance, name="output_layer")([vec_1, vec_2])
      9 
     10 siamese_model = Model([input_1, input_2], output)

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/layers/core/lambda_layer.py in compute_output_shape(self, input_shape)
     93                 return tree.map_structure(lambda x: x.shape, output_spec)
     94             except:
---> 95                 raise NotImplementedError(
     96                     "We could not automatically infer the shape of "
     97                     "the Lambda's output. Please specify the `output_shape` "

NotImplementedError: Exception encountered when calling Lambda.call().

We could not automatically infer the shape of the Lambda's output. Please specify the `output_shape` argument for this Lambda layer.

Arguments received by Lambda.call():
  • args=(['<KerasTensor shape=(None, 256), dtype=float32, sparse=False, name=keras_tensor_4>', '<KerasTensor shape=(None, 256), dtype=float32, sparse=False, name=keras_tensor_5>'],)
  • kwargs={'mask': ['None', 'None']}

## === cell 24
def contrastive_loss_with_margin(margin):
    def contrastive_loss(y_true, y_pred):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
    return contrastive_loss


## === cell 25
rms = RMSprop()
siamese_model.compile(loss=contrastive_loss_with_margin(margin=1), optimizer=rms)
siamese_model.summary()


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/953389205.py in <cell line: 0>()
      1 rms = RMSprop()
----> 2 siamese_model.compile(loss=contrastive_loss_with_margin(margin=1), optimizer=rms)
      3 siamese_model.summary()

NameError: name 'siamese_model' is not defined

## === cell 26
history = siamese_model.fit([padded_anchor_sequences,padded_target_sequences], y_train,  epochs=10,
                   batch_size=64, validation_data=([val_padded_anchor_sequences, val_padded_target_sequences],y_test))


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1590639514.py in <cell line: 0>()
----> 1 history = siamese_model.fit([padded_anchor_sequences,padded_target_sequences], y_train,  epochs=10,
      2                    batch_size=64, validation_data=([val_padded_anchor_sequences, val_padded_target_sequences],y_test))

NameError: name 'siamese_model' is not defined

## === cell 27
similarity= siamese_model.predict([val_padded_anchor_sequences, val_padded_target_sequences])


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/48678453.py in <cell line: 0>()
----> 1 similarity= siamese_model.predict([val_padded_anchor_sequences, val_padded_target_sequences])

NameError: name 'siamese_model' is not defined

## === cell 28
print(similarity.shape)
similarity= similarity.reshape((len(similarity),))
print(similarity.shape)


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/161360274.py in <cell line: 0>()
----> 1 print(similarity.shape)
      2 similarity= similarity.reshape((len(similarity),))
      3 print(similarity.shape)

NameError: name 'similarity' is not defined

## === cell 31
test_anchor_sequences = tokenizer.texts_to_sequences(test['anchor'].values)
test_target_sequences = tokenizer.texts_to_sequences(test['target'].values)

test_padded_anchor_sequences = pad_sequences(test_anchor_sequences, maxlen=max_length, truncating=trunc_type)
test_padded_target_sequences = pad_sequences(test_target_sequences, maxlen=max_length, truncating=trunc_type)


## === cell 32
similarity= siamese_model.predict([test_padded_anchor_sequences, test_padded_target_sequences])


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3671986459.py in <cell line: 0>()
----> 1 similarity= siamese_model.predict([test_padded_anchor_sequences, test_padded_target_sequences])

NameError: name 'siamese_model' is not defined

## === cell 33
similarity.shape


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/587611881.py in <cell line: 0>()
----> 1 similarity.shape

NameError: name 'similarity' is not defined

## === cell 34
similarity= similarity.reshape((len(similarity),))
similarity.shape


## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1142527017.py in <cell line: 0>()
----> 1 similarity= similarity.reshape((len(similarity),))
      2 similarity.shape

NameError: name 'similarity' is not defined

## === cell 35
test['score'] = similarity


## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/428849571.py in <cell line: 0>()
----> 1 test['score'] = similarity

NameError: name 'similarity' is not defined

## === cell 36
test.head(20)


## === cell 37
test= test[['id', 'score']]


## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3567956529.py in <cell line: 0>()
----> 1 test= test[['id', 'score']]

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['score'] not in index"

## === cell 38
test.head(20)


## === cell 39
test.to_csv("submission.csv", index=False)


## --- ERROR in outputing the csv:
Invalid submission: Submission must have columns ['id', 'score'], got Index(['id', 'anchor', 'target', 'context'], dtype='object')
