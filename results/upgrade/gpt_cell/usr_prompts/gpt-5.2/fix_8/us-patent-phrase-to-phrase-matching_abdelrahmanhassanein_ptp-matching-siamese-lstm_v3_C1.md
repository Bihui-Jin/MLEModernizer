# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.10

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import os

import sys, subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


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
base_network = initialize_base_network()

try:
    tf.keras.utils.plot_model(base_network, show_shapes=True)
except Exception:
    try:
        plot_model(base_network, show_shapes=True)
    except Exception:
        pass


## === cell 23
input_1 = Input(shape=(padded_anchor_sequences.shape[1],), name= 'input_1')
input_2 = Input(shape=(padded_target_sequences.shape[1],), name= 'input_2')

vec_1 = base_network(input_1)
vec_2 = base_network(input_2)

output = Lambda(euclidean_distance, name="output_layer")([vec_1, vec_2])

siamese_model = Model([input_1, input_2], output)

plot_model(siamese_model, show_shapes=True, show_layer_names=True, to_file='outer-model.png')


## --- ERROR in cell 23, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mNotImplementedError[0m                       Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2023786234.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      6[0m [0;34m[0m[0m
[1;32m      7[0m [0;31m#output = Lambda(euclidean_distance, name="output_layer", output_shape=eucl_dist_output_shape)([vec_1, vec_2])[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 8[0;31m [0moutput[0m [0;34m=[0m [0mLambda[0m[0;34m([0m[0meuclidean_distance[0m[0;34m,[0m [0mname[0m[0;34m=[0m[0;34m"output_layer"[0m[0;34m)[0m[0;34m([0m[0;34m[[0m[0mvec_1[0m[0;34m,[0m [0mvec_2[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      9[0m [0;34m[0m[0m
[1;32m     10[0m [0msiamese_model[0m [0;34m=[0m [0mModel[0m[0;34m([0m[0;34m[[0m[0minput_1[0m[0;34m,[0m [0minput_2[0m[0;34m][0m[0;34m,[0m [0moutput[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py[0m in [0;36merror_handler[0;34m(*args, **kwargs)[0m
[1;32m    120[0m             [0;31m# To get the full stack trace, call:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    121[0m             [0;31m# `keras.config.disable_traceback_filtering()`[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 122[0;31m             [0;32mraise[0m [0me[0m[0;34m.[0m[0mwith_traceback[0m[0;34m([0m[0mfiltered_tb[0m[0;34m)[0m [0;32mfrom[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    123[0m         [0;32mfinally[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    124[0m             [0;32mdel[0m [0mfiltered_tb[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/layers/core/lambda_layer.py[0m in [0;36mcompute_output_shape[0;34m(self, input_shape)[0m
[1;32m     93[0m                 [0;32mreturn[0m [0mtree[0m[0;34m.[0m[0mmap_structure[0m[0;34m([0m[0;32mlambda[0m [0mx[0m[0;34m:[0m [0mx[0m[0;34m.[0m[0mshape[0m[0;34m,[0m [0moutput_spec[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     94[0m             [0;32mexcept[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 95[0;31m                 raise NotImplementedError(
[0m[1;32m     96[0m                     [0;34m"We could not automatically infer the shape of "[0m[0;34m[0m[0;34m[0m[0m
[1;32m     97[0m                     [0;34m"the Lambda's output. Please specify the `output_shape` "[0m[0;34m[0m[0;34m[0m[0m

[0;31mNotImplementedError[0m: Exception encountered when calling Lambda.call().

[1mWe could not automatically infer the shape of the Lambda's output. Please specify the `output_shape` argument for this Lambda layer.[0m

Arguments received by Lambda.call():
  • args=(['<KerasTensor shape=(None, 256), dtype=float32, sparse=False, name=keras_tensor_4>', '<KerasTensor shape=(None, 256), dtype=float32, sparse=False, name=keras_tensor_5>'],)
  • kwargs={'mask': ['None', 'None']}

## === cell 24
"""
def contrastive_loss_with_margin(margin):
    def contrastive_loss(y_true, y_pred):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
    return contrastive_loss
"""
