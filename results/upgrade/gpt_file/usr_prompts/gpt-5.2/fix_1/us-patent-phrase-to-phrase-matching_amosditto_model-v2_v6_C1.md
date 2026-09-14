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

0.2177

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.utils import shuffle


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(64, activation='relu')
        self.dense3 = layers.Dense(64, activation='relu')
        self.dense4 = layers.Dense(32)
        self.dense5 = layers.Dense(16)
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
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
history = model.fit([x_anchor, x_target], y_data, epochs=100, batch_size=128, callbacks=[callback])


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_10/307066228.py in <cell line: 0>()
      1 callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
----> 2 history = model.fit([x_anchor, x_target], y_data, epochs=100, batch_size=128, callbacks=[callback])

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_10/4248822512.py in call(self, inputs)
     20         anchor, target = inputs
     21         out_anchor = self.embed_layer1(anchor)
---> 22         out_anchor = self.att1(out_anchor)
     23         out_anchor = self.global_avg1(out_anchor)
     24         # out_anchor = self.drop_out1(out_anchor)

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

TypeError: Exception encountered when calling AutoEncoderModel.call().

missing a required argument: 'training'

Arguments received by AutoEncoderModel.call():
  • inputs=('tf.Tensor(shape=(None, 7), dtype=int32)', 'tf.Tensor(shape=(None, 17), dtype=int32)')

## === cell 17
model.evaluate([anchor_val, target_val], y_val)


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_10/2484127513.py in <cell line: 0>()
----> 1 model.evaluate([anchor_val, target_val], y_val)

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_10/4248822512.py in call(self, inputs)
     20         anchor, target = inputs
     21         out_anchor = self.embed_layer1(anchor)
---> 22         out_anchor = self.att1(out_anchor)
     23         out_anchor = self.global_avg1(out_anchor)
     24         # out_anchor = self.drop_out1(out_anchor)

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

TypeError: Exception encountered when calling AutoEncoderModel.call().

missing a required argument: 'training'

Arguments received by AutoEncoderModel.call():
  • inputs=('tf.Tensor(shape=(32, 7), dtype=int32)', 'tf.Tensor(shape=(32, 17), dtype=int32)')

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
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_10/4250305755.py in <cell line: 0>()
----> 1 pre = model.predict([anchor_val[:20], target_val[:20]])
      2 predicted = []
      3 for x in pre:
      4     predicted.append(np.argmax(x))
      5 predicted = LE.inverse_transform(predicted)

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/progbar.py in update(self, current, values, finalize)
    117 
    118             if self.target is not None:
--> 119                 numdigits = int(math.log10(self.target)) + 1
    120                 bar = ("%" + str(numdigits) + "d/%d") % (current, self.target)
    121                 bar = f"\x1b[1m{bar}\x1b[0m "

ValueError: math domain error

## === cell 20
True_values = LE.inverse_transform(y_val[:20])
True_values


## === cell 22
anchor_test= tokenizer.texts_to_sequences(df_test['anchor'])
target_test = tokenizer.texts_to_sequences(df_test['target'])


## === cell 23
padded_anchor_test = tf.keras.preprocessing.sequence.pad_sequences(anchor_test, maxlen=7)
padded_target_test = tf.keras.preprocessing.sequence.pad_sequences(target_test, maxlen=17)


## === cell 24
test_predicted = model.predict([padded_anchor_test[:], padded_target_test[:]])


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_10/1504911264.py in <cell line: 0>()
----> 1 test_predicted = model.predict([padded_anchor_test[:], padded_target_test[:]])

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_10/4248822512.py in call(self, inputs)
     20         anchor, target = inputs
     21         out_anchor = self.embed_layer1(anchor)
---> 22         out_anchor = self.att1(out_anchor)
     23         out_anchor = self.global_avg1(out_anchor)
     24         # out_anchor = self.drop_out1(out_anchor)

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

TypeError: Exception encountered when calling AutoEncoderModel.call().

missing a required argument: 'training'

Arguments received by AutoEncoderModel.call():
  • inputs=('tf.Tensor(shape=(32, 7), dtype=int32)', 'tf.Tensor(shape=(32, 17), dtype=int32)')

## === cell 25
predicted_arr = []
for x in test_predicted:
    predicted_arr.append(np.argmax(x))


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/2453809234.py in <cell line: 0>()
      1 predicted_arr = []
----> 2 for x in test_predicted:
      3     predicted_arr.append(np.argmax(x))

NameError: name 'test_predicted' is not defined

## === cell 26
predicted_arr = LE.inverse_transform(predicted_arr)


## === cell 27
test_id_1 = np.array(test_id)
predicted_arr_1 = np.array(predicted_arr)
print(test_id_1.shape, predicted_arr_1.shape)


## === cell 28
Submission = pd.DataFrame({'id': test_id_1, 'score': predicted_arr_1})


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_10/1149998660.py in <cell line: 0>()
----> 1 Submission = pd.DataFrame({'id': test_id_1, 'score': predicted_arr_1})

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    776         elif isinstance(data, dict):
    777             # GH#38939 de facto copy defaults to False only in non-dict cases
--> 778             mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy, typ=manager)
    779         elif isinstance(data, ma.MaskedArray):
    780             from numpy.ma import mrecords

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in dict_to_mgr(data, index, columns, dtype, typ, copy)
    501             arrays = [x.copy() if hasattr(x, "dtype") else x for x in arrays]
    502 
--> 503     return arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ, consolidate=copy)
    504 
    505 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in arrays_to_mgr(arrays, columns, index, dtype, verify_integrity, typ, consolidate)
    112         # figure out the index, if necessary
    113         if index is None:
--> 114             index = _extract_index(arrays)
    115         else:
    116             index = ensure_index(index)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in _extract_index(data)
    675         lengths = list(set(raw_lengths))
    676         if len(lengths) > 1:
--> 677             raise ValueError("All arrays must be of the same length")
    678 
    679         if have_dicts:

ValueError: All arrays must be of the same length

## === cell 29
filename = 'submission.csv'
Submission.to_csv(filename, index=False)


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/2600084013.py in <cell line: 0>()
      2 # os.makedirs('Submissions')
      3 filename = 'submission.csv'
----> 4 Submission.to_csv(filename, index=False)

NameError: name 'Submission' is not defined
