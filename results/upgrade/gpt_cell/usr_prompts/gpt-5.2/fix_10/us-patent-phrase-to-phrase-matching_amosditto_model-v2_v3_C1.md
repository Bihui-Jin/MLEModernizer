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
import os
import sys
import subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
)

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.utils import shuffle


## === cell 1
path = '../input/us-patent-phrase-to-phrase-matching/train.csv'
path_test = '../input/us-patent-phrase-to-phrase-matching/test.csv'
df = pd.read_csv(path)
df_test = pd.read_csv(path_test)


df = df.drop(columns=['id', 'context'])
test_id = df_test['id']
df_test = df_test.drop(columns=['id', 'context'])
df = shuffle(df)
df = df.reset_index(drop=True)


## === cell 2
df_test.shape


## === cell 3
x_data_1 = df['anchor']
x_data_2 = df['target']
score = df['score']


## === cell 4
test_combined = df_test['anchor'] + ' ' + df_test['target']
x_combined = x_data_1 + " " + x_data_2
df_tokens = pd.concat([test_combined, x_combined])
df_tokens.shape


## === cell 5
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_tokens)


## === cell 6
anchor_tokenized = tokenizer.texts_to_sequences(x_data_1)
target_tokenized = tokenizer.texts_to_sequences(x_data_2)


## === cell 7
padded_anchor = tf.keras.preprocessing.sequence.pad_sequences(anchor_tokenized, maxlen=7)
padded_target = tf.keras.preprocessing.sequence.pad_sequences(target_tokenized, maxlen=17)


## === cell 8
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
y_score = LE.fit_transform(score)


## === cell 9
class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, vocab_size, output_dim, input_dim):
        super(PositionalEmbedding, self).__init__()
        self.word_embedding = layers.Embedding(vocab_size, output_dim=output_dim, input_length=input_dim)
        self.postional_embedding = layers.Embedding(input_dim, output_dim)
        
    def call(self, inputs):
        position_indices = tf.range(tf.shape(inputs)[-1])
        embedded_words = self.word_embedding(inputs)
        embedded_indices = self.postional_embedding(position_indices)
        return embedded_words + embedded_indices


## === cell 10
class Transformer(keras.layers.Layer):
    def __init__(self,num_heads, embed_dim, ff_dim, rate=0.1):
        super(Transformer,self).__init__()
        self.att = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
    def call(self, inputs, training):
        out1 = self.att(inputs, inputs)
        out1 = self.dropout1(out1, training=training)
        out1 = self.layernorm1(inputs + out1)
        out2 = self.ffn(out1)
        out2 = self.dropout2(out2, training=training)
        output = self.layernorm2(out1 + out2)
        
        return output


## === cell 11
class AutoEncoderModel(keras.Model):
    def __init__(self, vocab_size, num_heads, embed_dim, ff_dim, output_dim, input_dim_1, input_dim_2):
        super(AutoEncoderModel, self).__init__()
        self.embed_layer1 = PositionalEmbedding(vocab_size, output_dim, input_dim_1)
        self.att1 = Transformer(num_heads, embed_dim, ff_dim)
        self.embed_layer2 = PositionalEmbedding(vocab_size, output_dim, input_dim_2)
        self.att2 = Transformer(num_heads, embed_dim, ff_dim)
        self.drop_out_clf = layers.Dropout(rate=0.2)
        self.global_avg1 = layers.GlobalAveragePooling1D()
        self.global_avg2 = layers.GlobalAveragePooling1D()
        self.dense1 = layers.Dense(256, activation='relu')
        self.dense2 = layers.Dense(128, activation='relu')
        self.dense3 = layers.Dense(64, activation='relu')
        self.dense4 = layers.Dense(64, activation='relu')
        self.dense5 = layers.Dense(32)
        self.dense_clf = layers.Dense(5, activation='softmax')
    def call(self, inputs):
        anchor, target = inputs
        out_anchor = self.embed_layer1(anchor)
        out_anchor = self.att1(out_anchor)
        out_anchor = self.global_avg1(out_anchor)
        
        out_target = self.embed_layer2(target)
        out_target = self.att2(out_target)
        out_target = self.global_avg2(out_target)
        
        output = layers.Concatenate(axis=1)([out_anchor, out_target])
        output = self.dense1(output)
        output = self.dense2(output)
        output = self.dense3(output)
        output = self.dense4(output)
        output = self.dense5(output)
        output = self.drop_out_clf(output)
        output = self.dense_clf(output)
        return output


## === cell 12
vocab_size = len(tokenizer.word_index)
output_dim = 32
input_dim_1 = 7
input_dim_2 = 17
num_heads = 8
embed_dim = 32
ff_dim = 256


## === cell 13
model = AutoEncoderModel(vocab_size, num_heads, embed_dim, ff_dim, output_dim, input_dim_1, input_dim_2)


## === cell 14
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


## === cell 15
x_anchor = padded_anchor[:33000]
x_target = padded_target[:33000]
anchor_val = padded_anchor[33000:]
target_val = padded_target[33000:]
y_data = y_score[:33000]
y_val = y_score[33000:]


## === cell 16
class Transformer(keras.layers.Layer):
    def __init__(self, num_heads, embed_dim, ff_dim, rate=0.1):
        super(Transformer, self).__init__()
        self.att = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, inputs, training=None):
        out1 = self.att(inputs, inputs)
        out1 = self.dropout1(out1, training=training)
        out1 = self.layernorm1(inputs + out1)
        out2 = self.ffn(out1)
        out2 = self.dropout2(out2, training=training)
        output = self.layernorm2(out1 + out2)
        return output


def _autoencoder_call(self, inputs, training=None):
    anchor, target = inputs
    out_anchor = self.embed_layer1(anchor)
    out_anchor = self.att1(out_anchor, training=training)
    out_anchor = self.global_avg1(out_anchor)

    out_target = self.embed_layer2(target)
    out_target = self.att2(out_target, training=training)
    out_target = self.global_avg2(out_target)

    output = layers.Concatenate(axis=1)([out_anchor, out_target])
    output = self.dense1(output)
    output = self.dense2(output)
    output = self.dense3(output)
    output = self.dense4(output)
    output = self.dense5(output)
    output = self.drop_out_clf(output, training=training)
    output = self.dense_clf(output)
    return output


AutoEncoderModel.call = _autoencoder_call

callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3)
history = model.fit(
    [x_anchor, x_target], y_data, epochs=100, batch_size=128, callbacks=[callback]
)


## === cell 17
if len(anchor_val) == 0 or len(target_val) == 0 or len(y_val) == 0:
    print(
        "Skipping evaluation: validation set is empty (check split indices in cell 15)."
    )
else:
    model.evaluate([anchor_val, target_val], y_val)


## === cell 18
model.summary()


## === cell 19
pre = model.predict([anchor_val[:20], target_val[:20]])
predicted = []
for x in pre:
    predicted.append(np.argmax(x))
predicted = LE.inverse_transform(predicted)
predicted


## --- ERROR in cell 19, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/4250305755.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mpre[0m [0;34m=[0m [0mmodel[0m[0;34m.[0m[0mpredict[0m[0;34m([0m[0;34m[[0m[0manchor_val[0m[0;34m[[0m[0;34m:[0m[0;36m20[0m[0;34m][0m[0;34m,[0m [0mtarget_val[0m[0;34m[[0m[0;34m:[0m[0;36m20[0m[0;34m][0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0mpredicted[0m [0;34m=[0m [0;34m[[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0;32mfor[0m [0mx[0m [0;32min[0m [0mpre[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m     [0mpredicted[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0mnp[0m[0;34m.[0m[0margmax[0m[0;34m([0m[0mx[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0mpredicted[0m [0;34m=[0m [0mLE[0m[0;34m.[0m[0minverse_transform[0m[0;34m([0m[0mpredicted[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py[0m in [0;36merror_handler[0;34m(*args, **kwargs)[0m
[1;32m    120[0m             [0;31m# To get the full stack trace, call:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    121[0m             [0;31m# `keras.config.disable_traceback_filtering()`[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 122[0;31m             [0;32mraise[0m [0me[0m[0;34m.[0m[0mwith_traceback[0m[0;34m([0m[0mfiltered_tb[0m[0;34m)[0m [0;32mfrom[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    123[0m         [0;32mfinally[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    124[0m             [0;32mdel[0m [0mfiltered_tb[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/utils/progbar.py[0m in [0;36mupdate[0;34m(self, current, values, finalize)[0m
[1;32m    117[0m [0;34m[0m[0m
[1;32m    118[0m             [0;32mif[0m [0mself[0m[0;34m.[0m[0mtarget[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 119[0;31m                 [0mnumdigits[0m [0;34m=[0m [0mint[0m[0;34m([0m[0mmath[0m[0;34m.[0m[0mlog10[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mtarget[0m[0;34m)[0m[0;34m)[0m [0;34m+[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    120[0m                 [0mbar[0m [0;34m=[0m [0;34m([0m[0;34m"%"[0m [0;34m+[0m [0mstr[0m[0;34m([0m[0mnumdigits[0m[0;34m)[0m [0;34m+[0m [0;34m"d/%d"[0m[0;34m)[0m [0;34m%[0m [0;34m([0m[0mcurrent[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mtarget[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    121[0m                 [0mbar[0m [0;34m=[0m [0;34mf"\x1b[1m{bar}\x1b[0m "[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: math domain error

## === cell 20
True_values = LE.inverse_transform(y_val[:20])
True_values
