# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.2165

# 6. Current score

0.46369

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.47753) has done: 'I fix the initial TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` issue in this environment. Then I fix the Keras layer signature bug by threading the `training` argument through your `Transformer` and `AutoEncoderModel` `call()` methods so `fit/evaluate/predict` work. Finally, I ensure prediction runs end-to-end and always produces a valid `submission.csv` with the correct `id,score` columns and matching lengths by predicting on the full test set and mapping the 5-way softmax back to the original score labels.'
- What this solution (achieved 0.46369) has done: 'I fix the protobuf/TensorFlow crash by forcing a compatible protobuf version at runtime and ensuring the pure-Python implementation is used *before* importing TensorFlow, which resolves the `MessageFactory.GetPrototype` error in this Kaggle environment. I keep your model and training logic intact; the only “score-related” change be to **avoid EarlyStopping** (your constraints disallow it) so training matches the intended 100 epochs and is deterministic under the fixed seeds. I also make the data split robust (so it never produces an empty train set) and keep the submission writing exactly as required (`submission.csv` with `id,score`). These changes are minimal and aimed at correctness/stability while nudging score upward only as a side effect of removing early stopping.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import subprocess


def _ensure_protobuf_compat():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        major = int(pb_ver.split(".", 1)[0])
        if major >= 4:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf<4"]
            )
            for k in list(sys.modules.keys()):
                if k.startswith("google.protobuf"):
                    sys.modules.pop(k, None)
    except Exception:
        pass


_ensure_protobuf_compat()

import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.utils import shuffle

tf.random.set_seed(42)
np.random.seed(42)



## === cell 1
path = "../input/us-patent-phrase-to-phrase-matching/train.csv"
path_test = "../input/us-patent-phrase-to-phrase-matching/test.csv"
df = pd.read_csv(path)
df_test = pd.read_csv(path_test)

df = df.drop(columns=["id", "context"])
test_id = df_test["id"].copy()
df_test = df_test.drop(columns=["id", "context"])
df = shuffle(df, random_state=42).reset_index(drop=True)



## === cell 2
df_test.shape



## === cell 3
x_data_1 = df["anchor"]
x_data_2 = df["target"]
score = df["score"]



## === cell 4
test_combined = df_test["anchor"] + " " + df_test["target"]
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
padded_anchor = tf.keras.preprocessing.sequence.pad_sequences(
    anchor_tokenized, maxlen=7
)
padded_target = tf.keras.preprocessing.sequence.pad_sequences(
    target_tokenized, maxlen=17
)



## === cell 8
from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()
y_score = LE.fit_transform(score)




## === cell 9
class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, vocab_size, output_dim, input_dim):
        super(PositionalEmbedding, self).__init__()
        self.word_embedding = layers.Embedding(
            vocab_size, output_dim=output_dim, input_length=input_dim
        )
        self.postional_embedding = layers.Embedding(input_dim, output_dim)

    def call(self, inputs):
        position_indices = tf.range(tf.shape(inputs)[-1])
        embedded_words = self.word_embedding(inputs)
        embedded_indices = self.postional_embedding(position_indices)
        return embedded_words + embedded_indices




## === cell 10
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




## === cell 11
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
        self.drop_out_clf = layers.Dropout(rate=0.2)
        self.global_avg1 = layers.GlobalAveragePooling1D()
        self.global_avg2 = layers.GlobalAveragePooling1D()
        self.dense1 = layers.Dense(256, activation="relu")
        self.dense2 = layers.Dense(128, activation="relu")
        self.dense3 = layers.Dense(64, activation="relu")
        self.dense4 = layers.Dense(64, activation="relu")
        self.dense5 = layers.Dense(32)
        self.dense_clf = layers.Dense(5, activation="softmax")

    def call(self, inputs, training=None):
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




## === cell 12
vocab_size = len(tokenizer.word_index) + 1  # +1 to include 0-padding token safely
output_dim = 32
input_dim_1 = 7
input_dim_2 = 17
num_heads = 8
embed_dim = 32
ff_dim = 256



## === cell 13
model = AutoEncoderModel(
    vocab_size, num_heads, embed_dim, ff_dim, output_dim, input_dim_1, input_dim_2
)



## === cell 14
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)



## === cell 15
split_idx = min(33000, len(padded_anchor))
if split_idx <= 0:
    raise ValueError("Empty training data after preprocessing; check input files.")

x_anchor = padded_anchor[:split_idx]
x_target = padded_target[:split_idx]
anchor_val = padded_anchor[split_idx:]
target_val = padded_target[split_idx:]
y_data = y_score[:split_idx]
y_val = y_score[split_idx:]



## === cell 16
history = model.fit(
    [x_anchor, x_target],
    y_data,
    epochs=100,
    batch_size=128,
    verbose=1,
)



## === cell 17
if len(y_val) > 0:
    model.evaluate([anchor_val, target_val], y_val, verbose=1)



## === cell 18
model.summary()



## === cell 19
if len(y_val) > 0:
    pre = model.predict([anchor_val[:20], target_val[:20]], verbose=0)
    predicted = np.argmax(pre, axis=1)
    predicted_scores = LE.inverse_transform(predicted)
    True_values = LE.inverse_transform(y_val[:20])
    print("Pred:", predicted_scores)
    print("True:", True_values)



## === cell 20
anchor_test = tokenizer.texts_to_sequences(df_test["anchor"])
target_test = tokenizer.texts_to_sequences(df_test["target"])

padded_anchor_test = tf.keras.preprocessing.sequence.pad_sequences(
    anchor_test, maxlen=7
)
padded_target_test = tf.keras.preprocessing.sequence.pad_sequences(
    target_test, maxlen=17
)



## === cell 21
test_predicted = model.predict(
    [padded_anchor_test, padded_target_test], batch_size=128, verbose=1
)



## === cell 22
predicted_arr = np.argmax(test_predicted, axis=1)
predicted_arr = LE.inverse_transform(predicted_arr).astype(float)

test_id_1 = np.array(test_id)
predicted_arr_1 = np.array(predicted_arr)

assert test_id_1.shape[0] == predicted_arr_1.shape[0], (
    test_id_1.shape,
    predicted_arr_1.shape,
)

Submission = pd.DataFrame({"id": test_id_1, "score": predicted_arr_1})
Submission = Submission[["id", "score"]]
Submission["score"] = Submission["score"].astype(float)

filename = "submission.csv"
Submission.to_csv(filename, index=False)
print("Wrote:", filename, "shape:", Submission.shape)
print(Submission.head())
