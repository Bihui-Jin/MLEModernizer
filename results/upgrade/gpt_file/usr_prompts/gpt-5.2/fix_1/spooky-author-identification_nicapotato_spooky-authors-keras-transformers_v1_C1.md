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
Given some text, predict the author.

## Metric
Multi-class logarithmic loss. 

The submitted probabilities for a given sentences are not required to sum to one because they are rescaled prior to being scored (each row is divided by the row sum).

In order to avoid the extremes of the log function, predicted probabilities are replaced with \\(max(min(p,1-10^{-15}),10^{-15})\\).

## Submission Format
You must submit a csv file with the id, and a probability for each of the three classes. The order of the rows does not matter. The file must have a header and should look like the following:

```
id,EAP,HPL,MWS
id07943,0.33,0.33,0.33
...
```

## Dataset 
### File descriptions
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Data fields
- **id** - a unique identifier for each sentence
- **text** - some text written by one of the authors
- **author** - the author of the sentence (EAP: Edgar Allan Poe, HPL: HP Lovecraft; MWS: Mary Wollstonecraft Shelley)

# 2. Python version

3.9

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
            description.md (98 lines)
            sample_submission.csv (1959 lines)
            sample_submission.csv.zip (7.4 kB)
            test.csv (1959 lines)
            test.csv.zip (133.3 kB)
            train.csv (17622 lines)
            train.csv.zip (1.2 MB)
            train.zip (1.2 MB)
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
        input/
            description.md (98 lines)
            sample_submission.csv (1959 lines)
            sample_submission.csv.zip (7.4 kB)
            test.csv (1959 lines)
            test.csv.zip (133.3 kB)
            train.csv (17622 lines)
            train.csv.zip (1.2 MB)
            train.zip (1.2 MB)
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
        working/
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
```

-> data/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> data/spooky-author-identification/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> data/spooky-author-identification/test.csv has 1958 rows and 2 columns.
The columns are: id, text

-> data/spooky-author-identification/train.csv has 17621 rows and 3 columns.
The columns are: id, text, author

-> data/test.csv has 1958 rows and 2 columns.
The columns are: id, text

-> data/train.csv has 17621 rows and 3 columns.
The columns are: id, text, author

-> input/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> (stopped after 10 files for performance)

# 5. Target score

0.45013

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
"""
Title: Text classification with Transformer
Author: [Apoorv Nandan](https://twitter.com/NandanApoorv)
Date created: 2020/05/10
Last modified: 2020/05/10
Description: Implement a Transformer block as a Keras layer and use it for text classification.
"""
"""
## Setup
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from keras.utils import to_categorical
from keras.preprocessing import text, sequence
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

"""
## Implement multi head self attention as a Keras layer
"""

print("Tensorflow Version: ", tf.__version__)
print("Eager mode enabled: ", tf.executing_eagerly())
print("GPU available: ", tf.test.is_gpu_available())

notebookstart = time.time()


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output


"""
## Implement a Transformer block as a layer
"""


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


"""
## Implement embedding layer
Two seperate embedding layers, one for tokens, one for token index (positions).
"""


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


## === cell 2
MAXLEN = 250
VOCABLEN = 10000
NCHANNEL = 3
BATCHSIZE = 32
EPOCHS = 60

TEXTCOL = "text"
TARGETCOL = "author"
CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'


## === cell 3
train = pd.read_csv("../input/spooky-author-identification/train.zip")
test = pd.read_csv("../input/spooky-author-identification/test.zip")
testdex = test.id
submission = pd.read_csv("../input/spooky-author-identification/sample_submission.zip")
sub_cols = submission.columns

xtrain = train[TEXTCOL].astype(str)
xtest = test[TEXTCOL].astype(str)

label_mapper = {name: i for i,name in enumerate(set(train[TARGETCOL].values))}
num_label = np.vectorize(label_mapper.get)(train[TARGETCOL].values)
y_train = to_categorical(num_label)

tokenizer = text.Tokenizer(
    filters=CHARS_TO_REMOVE,
    lower=False,
    num_words=VOCABLEN)
tokenizer.fit_on_texts(list(xtrain) + list(xtest))

xtrain = tokenizer.texts_to_sequences(xtrain)
xtest = tokenizer.texts_to_sequences(xtest)

xtrain = sequence.pad_sequences(xtrain, maxlen=MAXLEN)
xtest = sequence.pad_sequences(xtest, maxlen=MAXLEN)


print("X Shape: {}".format(xtrain.shape))
print("y Shape: {}".format(xtrain.shape))

X_train, X_val, y_train, y_val = train_test_split(
    xtrain, y_train, test_size=0.33, random_state=42)

print("X_train Shape: {}".format(X_train.shape))
print("y_train Shape: {}".format(y_train.shape))


del train, test, xtrain


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1559714578.py in <cell line: 0>()
----> 1 train = pd.read_csv("../input/spooky-author-identification/train.zip")
      2 test = pd.read_csv("../input/spooky-author-identification/test.zip")
      3 testdex = test.id
      4 submission = pd.read_csv("../input/spooky-author-identification/sample_submission.zip")
      5 sub_cols = submission.columns

NameError: name 'pd' is not defined

## === cell 4
embed_dim = 32  # Embedding size for each token
num_heads = 4  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(MAXLEN,))
embedding_layer = TokenAndPositionEmbedding(MAXLEN, VOCABLEN, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(32, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NCHANNEL, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1757986846.py in <cell line: 0>()
      7 x = embedding_layer(inputs)
      8 transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
----> 9 x = transformer_block(x)
     10 x = layers.GlobalAveragePooling1D()(x)
     11 x = layers.Dropout(0.3)(x)

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/lib/python3.11/inspect.py in bind(self, *args, **kwargs)
   3193         if the passed arguments can not be bound.
   3194         """
-> 3195         return self._bind(args, kwargs)
   3196 
   3197     def bind_partial(self, /, *args, **kwargs):

/usr/lib/python3.11/inspect.py in _bind(self, args, kwargs, partial)
   3108                             msg = 'missing a required argument: {arg!r}'
   3109                             msg = msg.format(arg=param.name)
-> 3110                             raise TypeError(msg) from None
   3111             else:
   3112                 # We have a positional argument to process

TypeError: missing a required argument: 'training'

## === cell 5
"""
## Train and Evaluate
"""
es = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001,
                             patience=4, verbose=1, mode='min', baseline=None,
                             restore_best_weights=False)


model.compile(Adam(lr=5e-5), "categorical_crossentropy", metrics=["accuracy"])
history = model.fit(
    X_train, y_train, batch_size=BATCHSIZE, epochs=EPOCHS, validation_data=(X_val, y_val),
    callbacks=[es]
)


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/358305096.py in <cell line: 0>()
      8 
      9 
---> 10 model.compile(Adam(lr=5e-5), "categorical_crossentropy", metrics=["accuracy"])
     11 history = model.fit(
     12     X_train, y_train, batch_size=BATCHSIZE, epochs=EPOCHS, validation_data=(X_val, y_val),

NameError: name 'model' is not defined

## === cell 6
plot_metrics = ['loss']

f, ax = plt.subplots(1,figsize = [7,4])
for p_i,metric in enumerate(plot_metrics):
    ax.plot(history.history[metric], label='Train ' + metric)
    ax.plot(history.history['val_' + metric], label='Val ' + metric)
    ax.set_title("Loss Curve - {}".format(metric))
    ax.legend()
plt.show()


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1298875726.py in <cell line: 0>()
      1 plot_metrics = ['loss']
      2 
----> 3 f, ax = plt.subplots(1,figsize = [7,4])
      4 for p_i,metric in enumerate(plot_metrics):
      5     ax.plot(history.history[metric], label='Train ' + metric)

NameError: name 'plt' is not defined

## === cell 7
import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


val_pred = model.predict(X_val)
cnf_matrix = confusion_matrix(np.argmax(y_val,axis=1), np.argmax(val_pred,axis=1))
np.set_printoptions(precision=2)

plt.figure(figsize=(4,4))
plot_confusion_matrix(cnf_matrix, classes=['EAP', 'HPL', 'MWS'],
                      title='Confusion matrix, without normalization')
plt.show()


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1522465764.py in <cell line: 0>()
      7                           normalize=False,
      8                           title='Confusion matrix',
----> 9                           cmap=plt.cm.Blues):
     10     """
     11     This function prints and plots the confusion matrix.

NameError: name 'plt' is not defined

## === cell 8
test_pred = model.predict(xtest)
test_pred.shape


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3359117740.py in <cell line: 0>()
      1 # model.load_weights('model.h5')
----> 2 test_pred = model.predict(xtest)
      3 test_pred.shape

NameError: name 'model' is not defined

## === cell 9
submission = pd.DataFrame(test_pred, columns=label_mapper.keys())
submission['id'] = testdex

submission = submission[sub_cols]
submission.to_csv('submission_transfomer_keras.csv', index=False)
print(submission.shape)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2894700646.py in <cell line: 0>()
----> 1 submission = pd.DataFrame(test_pred, columns=label_mapper.keys())
      2 submission['id'] = testdex
      3 
      4 submission = submission[sub_cols]
      5 submission.to_csv('submission_transfomer_keras.csv', index=False)

NameError: name 'pd' is not defined

## === cell 10
!head submission_transfomer_keras.csv


## === cell 11
print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1768269731.py in <cell line: 0>()
----> 1 print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))

NameError: name 'time' is not defined
