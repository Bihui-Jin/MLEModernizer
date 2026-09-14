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

3.9

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

# 4. Code solution

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

import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

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
[0;31m---------------------------------------------------------------------------[0m
[0;31mImportError[0m                               Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/759710921.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     19[0m [0mos[0m[0;34m.[0m[0menviron[0m[0;34m.[0m[0mpop[0m[0;34m([0m[0;34m"PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"[0m[0;34m,[0m [0;32mNone[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     20[0m [0;34m[0m[0m
[0;32m---> 21[0;31m [0;32mimport[0m [0mtensorflow[0m [0;32mas[0m [0mtf[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     22[0m [0;32mfrom[0m [0mtensorflow[0m [0;32mimport[0m [0mkeras[0m[0;34m[0m[0;34m[0m[0m
[1;32m     23[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mkeras[0m [0;32mimport[0m [0mlayers[0m[0;34m,[0m [0mcallbacks[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m     47[0m [0m_tf2[0m[0;34m.[0m[0menable[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     48[0m [0;34m[0m[0m
[0;32m---> 49[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0m_api[0m[0;34m.[0m[0mv2[0m [0;32mimport[0m [0m__internal__[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     50[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0m_api[0m[0;34m.[0m[0mv2[0m [0;32mimport[0m [0m__operators__[0m[0;34m[0m[0;34m[0m[0m
[1;32m     51[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0m_api[0m[0;34m.[0m[0mv2[0m [0;32mimport[0m [0maudio[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      6[0m [0;32mimport[0m [0msys[0m [0;32mas[0m [0m_sys[0m[0;34m[0m[0;34m[0m[0m
[1;32m      7[0m [0;34m[0m[0m
[0;32m----> 8[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0m_api[0m[0;34m.[0m[0mv2[0m[0;34m.[0m[0m__internal__[0m [0;32mimport[0m [0mautograph[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      9[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0m_api[0m[0;34m.[0m[0mv2[0m[0;34m.[0m[0m__internal__[0m [0;32mimport[0m [0mdecorator[0m[0;34m[0m[0;34m[0m[0m
[1;32m     10[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0m_api[0m[0;34m.[0m[0mv2[0m[0;34m.[0m[0m__internal__[0m [0;32mimport[0m [0mdispatch[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/autograph/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      6[0m [0;32mimport[0m [0msys[0m [0;32mas[0m [0m_sys[0m[0;34m[0m[0;34m[0m[0m
[1;32m      7[0m [0;34m[0m[0m
[0;32m----> 8[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mautograph[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mag_ctx[0m [0;32mimport[0m [0mcontrol_status_ctx[0m [0;31m# line: 34[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      9[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mautograph[0m[0;34m.[0m[0mimpl[0m[0;34m.[0m[0mapi[0m [0;32mimport[0m [0mtf_convert[0m [0;31m# line: 493[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/ag_ctx.py[0m in [0;36m<module>[0;34m[0m
[1;32m     19[0m [0;32mimport[0m [0mthreading[0m[0;34m[0m[0;34m[0m[0m
[1;32m     20[0m [0;34m[0m[0m
[0;32m---> 21[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mautograph[0m[0;34m.[0m[0mutils[0m [0;32mimport[0m [0mag_logging[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     22[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mutil[0m[0;34m.[0m[0mtf_export[0m [0;32mimport[0m [0mtf_export[0m[0;34m[0m[0;34m[0m[0m
[1;32m     23[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m     15[0m [0;34m"""Utility module that contains APIs usable in the generated code."""[0m[0;34m[0m[0;34m[0m[0m
[1;32m     16[0m [0;34m[0m[0m
[0;32m---> 17[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mautograph[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mcontext_managers[0m [0;32mimport[0m [0mcontrol_dependency_on_returns[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     18[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mautograph[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mmisc[0m [0;32mimport[0m [0malias_tensors[0m[0;34m[0m[0;34m[0m[0m
[1;32m     19[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mautograph[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mtensor_list[0m [0;32mimport[0m [0mdynamic_list_append[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/context_managers.py[0m in [0;36m<module>[0;34m[0m
[1;32m     17[0m [0;32mimport[0m [0mcontextlib[0m[0;34m[0m[0;34m[0m[0m
[1;32m     18[0m [0;34m[0m[0m
[0;32m---> 19[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mframework[0m [0;32mimport[0m [0mops[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     20[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mops[0m [0;32mimport[0m [0mtensor_array_ops[0m[0;34m[0m[0;34m[0m[0m
[1;32m     21[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py[0m in [0;36m<module>[0;34m[0m
[1;32m     31[0m [0;34m[0m[0m
[1;32m     32[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mmessage[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 33[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mframework[0m [0;32mimport[0m [0mattr_value_pb2[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     34[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mframework[0m [0;32mimport[0m [0mfull_type_pb2[0m[0;34m[0m[0;34m[0m[0m
[1;32m     35[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mframework[0m [0;32mimport[0m [0mfunction_pb2[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/core/framework/attr_value_pb2.py[0m in [0;36m<module>[0;34m[0m
[1;32m      3[0m [0;31m# source: tensorflow/core/framework/attr_value.proto[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;34m"""Generated protocol buffer code."""[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 5[0;31m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m[0;34m.[0m[0minternal[0m [0;32mimport[0m [0mbuilder[0m [0;32mas[0m [0m_builder[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      6[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mdescriptor[0m [0;32mas[0m [0m_descriptor[0m[0;34m[0m[0;34m[0m[0m
[1;32m      7[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mdescriptor_pool[0m [0;32mas[0m [0m_descriptor_pool[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/builder.py[0m in [0;36m<module>[0;34m[0m
[1;32m     16[0m [0;34m[0m[0m
[1;32m     17[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m[0;34m.[0m[0minternal[0m [0;32mimport[0m [0menum_type_wrapper[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 18[0;31m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m[0;34m.[0m[0minternal[0m [0;32mimport[0m [0mpython_message[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     19[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mmessage[0m [0;32mas[0m [0m_message[0m[0;34m[0m[0;34m[0m[0m
[1;32m     20[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mreflection[0m [0;32mas[0m [0m_reflection[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/python_message.py[0m in [0;36m<module>[0;34m[0m
[1;32m     36[0m [0;32mimport[0m [0mweakref[0m[0;34m[0m[0;34m[0m[0m
[1;32m     37[0m [0;34m[0m[0m
[0;32m---> 38[0;31m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mdescriptor[0m [0;32mas[0m [0mdescriptor_mod[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     39[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mmessage[0m [0;32mas[0m [0mmessage_mod[0m[0;34m[0m[0;34m[0m[0m
[1;32m     40[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mtext_format[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/google/protobuf/descriptor.py[0m in [0;36m<module>[0;34m[0m
[1;32m     27[0m   [0;31m# TODO: Remove this import after fix api_implementation[0m[0;34m[0m[0;34m[0m[0m
[1;32m     28[0m   [0;32mif[0m [0m_message[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 29[0;31m     [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m[0;34m.[0m[0mpyext[0m [0;32mimport[0m [0m_message[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     30[0m   [0m_USE_C_DESCRIPTORS[0m [0;34m=[0m [0;32mTrue[0m[0;34m[0m[0;34m[0m[0m
[1;32m     31[0m [0;34m[0m[0m

[0;31mImportError[0m: cannot import name '_message' from 'google.protobuf.pyext' (/usr/local/lib/python3.11/dist-packages/google/protobuf/pyext/__init__.py)

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
