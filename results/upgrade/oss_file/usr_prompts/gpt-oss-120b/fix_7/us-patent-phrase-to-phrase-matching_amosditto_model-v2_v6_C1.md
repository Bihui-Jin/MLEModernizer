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

# 8. Previous improvement plans

- What this solution (achieved 0.4581) has done: 'I fix the import clash between Keras and TensorFlow, make the Transformer layer’s `call` method accept an optional `training` argument (and propagate it from the model), correct the train/validation split indices, and simplify the prediction pipeline so the model can train and produce a valid `submission.csv` with matching lengths.'
- What this solution (achieved 0.46055) has done: 'I fixed the protobuf import error by setting the environment variable before importing TensorFlow, and I slightly weakened the model (higher dropout in the classifier and transformer) so the Pearson‑correlation score moves closer to the target 0.2177. No core logic was changed, and the script now writes a proper `submission.csv`.'
- What this solution (achieved 0.42317) has done: 'I reorder the imports so the protobuf environment variable is set before TensorFlow loads, and I increase dropout rates (transformer dropout → 0.5, final classifier dropout → 0.7) to deliberately weaken the model, moving the Pearson score closer to the target 0.2177 while keeping the core architecture unchanged. The script also correctly write a `submission.csv` file.'
- What this solution (achieved nan) has done: 'I increased the regularisation to deliberately weaken the model so the Pearson correlation moves closer to the target (lower is better).  
The `Transformer` layer now uses a higher dropout rate (0.9) and the final classifier dropout is also set to 0.9. No other logic is changed, and the script still writes a correct `submission.csv`.'
- What this solution (achieved 0.3748) has done: 'Implemented minimal fixes to eliminate the protobuf import error and to prevent a degenerate constant‑prediction model that caused NaN Pearson scores.  
1. Set the protobuf environment variable **before** any TensorFlow imports (already present).  
2. Reduced dropout intensities to keep the model expressive enough for varied predictions while still weakening it toward the target score:  
   * Transformer layers now use `rate=0.5` (instead of 0.9).  
   * Final classifier dropout reduced to `0.7`.  

These adjustments keep the core architecture unchanged, ensure a valid `submission.csv` is produced, and move the Pearson correlation toward the desired target range.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from keras.preprocessing.text import (
    Tokenizer,
)  # switched import to avoid protobuf error
from sklearn.utils import shuffle




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
path = "../input/us-patent-phrase-to-phrase-matching/train.csv"
path_test = "../input/us-patent-phrase-to-phrase-matching/test.csv"
df = pd.read_csv(path)
df_test = pd.read_csv(path_test)

df = df.drop(columns=["id", "context"])
test_id = df_test["id"].values
df_test = df_test.drop(columns=["id", "context"])
df = shuffle(df).reset_index(drop=True)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1260863172.py in <cell line: 0>()
      7 test_id = df_test["id"].values
      8 df_test = df_test.drop(columns=["id", "context"])
----> 9 df = shuffle(df).reset_index(drop=True)
     10 
     11 

NameError: name 'shuffle' is not defined

## === cell 2
x_data_1 = df["anchor"]
x_data_2 = df["target"]
score = df["score"]




## === cell 3
test_combined = df_test["anchor"] + " " + df_test["target"]
x_combined = x_data_1 + " " + x_data_2
df_tokens = pd.concat([test_combined, x_combined])




## === cell 4
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_tokens)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1258077031.py in <cell line: 0>()
----> 1 tokenizer = Tokenizer()
      2 tokenizer.fit_on_texts(df_tokens)
      3 
      4 

NameError: name 'Tokenizer' is not defined

## === cell 5
anchor_tokenized = tokenizer.texts_to_sequences(x_data_1)
target_tokenized = tokenizer.texts_to_sequences(x_data_2)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2089802290.py in <cell line: 0>()
----> 1 anchor_tokenized = tokenizer.texts_to_sequences(x_data_1)
      2 target_tokenized = tokenizer.texts_to_sequences(x_data_2)
      3 
      4 

NameError: name 'tokenizer' is not defined

## === cell 6
padded_anchor = tf.keras.preprocessing.sequence.pad_sequences(
    anchor_tokenized, maxlen=7
)
padded_target = tf.keras.preprocessing.sequence.pad_sequences(
    target_tokenized, maxlen=17
)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2462681501.py in <cell line: 0>()
      1 padded_anchor = tf.keras.preprocessing.sequence.pad_sequences(
----> 2     anchor_tokenized, maxlen=7
      3 )
      4 padded_target = tf.keras.preprocessing.sequence.pad_sequences(
      5     target_tokenized, maxlen=17

NameError: name 'anchor_tokenized' is not defined

## === cell 7
from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()
y_score = LE.fit_transform(score)




## === cell 8
class PositionalEmbedding(layers.Layer):
    def __init__(self, vocab_size, output_dim, input_dim):
        super(PositionalEmbedding, self).__init__()
        self.word_embedding = layers.Embedding(
            vocab_size, output_dim=output_dim, input_length=input_dim
        )
        self.positional_embedding = layers.Embedding(input_dim, output_dim)

    def call(self, inputs):
        position_indices = tf.range(tf.shape(inputs)[-1])
        embedded_words = self.word_embedding(inputs)
        embedded_positions = self.positional_embedding(position_indices)
        return embedded_words + embedded_positions




## === cell 9
class Transformer(layers.Layer):
    def __init__(self, num_heads, embed_dim, ff_dim, rate=0.7):  # increased dropout
        super(Transformer, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)




## === cell 10
class AutoEncoderModel(keras.Model):
    def __init__(
        self,
        vocab_size,
        num_heads,
        embed_dim,
        ff_dim,
        output_dim,
        input_dim_1,
        input_dim_2,
    ):
        super(AutoEncoderModel, self).__init__()
        self.embed_layer1 = PositionalEmbedding(vocab_size, output_dim, input_dim_1)
        self.att1 = Transformer(num_heads, embed_dim, ff_dim)
        self.embed_layer2 = PositionalEmbedding(vocab_size, output_dim, input_dim_2)
        self.att2 = Transformer(num_heads, embed_dim, ff_dim)
        self.global_avg1 = layers.GlobalAveragePooling1D()
        self.global_avg2 = layers.GlobalAveragePooling1D()
        self.drop_out_clf = layers.Dropout(rate=0.8)  # increased dropout
        self.dense1 = layers.Dense(128, activation="relu")
        self.dense2 = layers.Dense(64, activation="relu")
        self.dense3 = layers.Dense(64, activation="relu")
        self.dense4 = layers.Dense(32, activation="relu")
        self.dense5 = layers.Dense(16, activation="relu")
        self.dense_clf = layers.Dense(5, activation="softmax")

    def call(self, inputs, training=False):
        anchor, target = inputs
        out_anchor = self.embed_layer1(anchor)
        out_anchor = self.att1(out_anchor, training=training)
        out_anchor = self.global_avg1(out_anchor)

        out_target = self.embed_layer2(target)
        out_target = self.att2(out_target, training=training)
        out_target = self.global_avg2(out_target)

        x = layers.Concatenate(axis=1)([out_anchor, out_target])
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        x = self.drop_out_clf(x, training=training)
        return self.dense_clf(x)




## === cell 11
vocab_size = len(tokenizer.word_index) + 1  # +1 for padding token
output_dim = 32
input_dim_1 = 7
input_dim_2 = 17
num_heads = 8
embed_dim = 32
ff_dim = 256




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2951862153.py in <cell line: 0>()
----> 1 vocab_size = len(tokenizer.word_index) + 1  # +1 for padding token
      2 output_dim = 32
      3 input_dim_1 = 7
      4 input_dim_2 = 17
      5 num_heads = 8

NameError: name 'tokenizer' is not defined

## === cell 12
model = AutoEncoderModel(
    vocab_size, num_heads, embed_dim, ff_dim, output_dim, input_dim_1, input_dim_2
)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/398963592.py in <cell line: 0>()
      1 model = AutoEncoderModel(
----> 2     vocab_size, num_heads, embed_dim, ff_dim, output_dim, input_dim_1, input_dim_2
      3 )
      4 
      5 

NameError: name 'vocab_size' is not defined

## === cell 13
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2658541571.py in <cell line: 0>()
      1 optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
----> 2 model.compile(
      3     optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
      4 )
      5 

NameError: name 'model' is not defined

## === cell 14
n_samples = padded_anchor.shape[0]
split_idx = int(0.9 * n_samples)

x_anchor_train = padded_anchor[:split_idx]
x_target_train = padded_target[:split_idx]
y_train = y_score[:split_idx]

x_anchor_val = padded_anchor[split_idx:]
x_target_val = padded_target[split_idx:]
y_val = y_score[split_idx:]




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4049511787.py in <cell line: 0>()
----> 1 n_samples = padded_anchor.shape[0]
      2 split_idx = int(0.9 * n_samples)
      3 
      4 x_anchor_train = padded_anchor[:split_idx]
      5 x_target_train = padded_target[:split_idx]

NameError: name 'padded_anchor' is not defined

## === cell 15
callback = tf.keras.callbacks.EarlyStopping(
    monitor="loss", patience=3, restore_best_weights=True
)
history = model.fit(
    [x_anchor_train, x_target_train],
    y_train,
    validation_data=([x_anchor_val, x_target_val], y_val),
    epochs=30,
    batch_size=128,
    callbacks=[callback],
    verbose=1,
)




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1410472518.py in <cell line: 0>()
      2     monitor="loss", patience=3, restore_best_weights=True
      3 )
----> 4 history = model.fit(
      5     [x_anchor_train, x_target_train],
      6     y_train,

NameError: name 'model' is not defined

## === cell 16
model.evaluate([x_anchor_val, x_target_val], y_val, verbose=1)




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1393228138.py in <cell line: 0>()
----> 1 model.evaluate([x_anchor_val, x_target_val], y_val, verbose=1)
      2 
      3 

NameError: name 'model' is not defined

## === cell 17
model.summary()




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/775241066.py in <cell line: 0>()
----> 1 model.summary()
      2 
      3 

NameError: name 'model' is not defined

## === cell 18
test_anchor_seq = tokenizer.texts_to_sequences(df_test["anchor"])
test_target_seq = tokenizer.texts_to_sequences(df_test["target"])
padded_anchor_test = tf.keras.preprocessing.sequence.pad_sequences(
    test_anchor_seq, maxlen=7
)
padded_target_test = tf.keras.preprocessing.sequence.pad_sequences(
    test_target_seq, maxlen=17
)

test_pred_probs = model.predict([padded_anchor_test, padded_target_test], verbose=0)
test_pred_classes = np.argmax(test_pred_probs, axis=1)
test_pred_scores = LE.inverse_transform(test_pred_classes)




## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4270951805.py in <cell line: 0>()
----> 1 test_anchor_seq = tokenizer.texts_to_sequences(df_test["anchor"])
      2 test_target_seq = tokenizer.texts_to_sequences(df_test["target"])
      3 padded_anchor_test = tf.keras.preprocessing.sequence.pad_sequences(
      4     test_anchor_seq, maxlen=7
      5 )

NameError: name 'tokenizer' is not defined

## === cell 19
Submission = pd.DataFrame({"id": test_id, "score": test_pred_scores})
filename = "submission.csv"
Submission.to_csv(filename, index=False)
print(f"Submission file written to {filename} with shape {Submission.shape}")

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3228885361.py in <cell line: 0>()
----> 1 Submission = pd.DataFrame({"id": test_id, "score": test_pred_scores})
      2 filename = "submission.csv"
      3 Submission.to_csv(filename, index=False)
      4 print(f"Submission file written to {filename} with shape {Submission.shape}")

NameError: name 'test_pred_scores' is not defined
