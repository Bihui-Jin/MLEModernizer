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

-0.0835

# 6. Current score

0.12463

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.12463) has done: 'I fix the environment-breaking TensorFlow/Protobuf import error by using the Kaggle-provided `tf_keras` package (compatible with TF 2.18 here) instead of `tensorflow.keras`. Then I remove the `plot_model` calls (they fail due to missing/deprecated internal TF modules) without changing any training/inference logic. Finally, I fix the Siamese `Lambda` layer shape inference by specifying `output_shape`, allowing the model to build/compile/train and produce predictions. With those fixes in place, the script generate a valid `submission.csv` containing exactly `id,score`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import tensorflow as tf
import tf_keras as keras
from tf_keras.preprocessing.text import Tokenizer
from tf_keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

np.random.seed(42)
tf.random.set_seed(42)

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:10]:
        print(os.path.join(dirname, filename))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/train.csv")
test = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv")

print(train.shape, test.shape)
train.head()



## === cell 2
X_train, X_test, y_train, y_test = train_test_split(
    train[["anchor", "target"]], train["score"], test_size=0.25, random_state=42
)

print(X_train.head())
print(X_test.head())
print(y_train.head())
print(y_test.head())



## === cell 3
X_train = X_train.copy()
X_test = X_test.copy()
X_train["text"] = X_train[["anchor", "target"]].apply(
    lambda x: str(x.iloc[0]) + " " + str(x.iloc[1]), axis=1
)

X_train.head()



## === cell 4
vocab_size = 7905
embedding_dim = 16
max_length = 4
trunc_type = "post"
oov_tok = ""

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train["text"].values)

word_index = tokenizer.word_index
print("Vocab size (word_index):", len(word_index))



## === cell 5
anchor_sequences = tokenizer.texts_to_sequences(X_train["anchor"].values)
target_sequences = tokenizer.texts_to_sequences(X_train["target"].values)

padded_anchor_sequences = pad_sequences(
    anchor_sequences, maxlen=max_length, truncating=trunc_type
)
padded_target_sequences = pad_sequences(
    target_sequences, maxlen=max_length, truncating=trunc_type
)

print("Train anchor padded shape:", padded_anchor_sequences.shape)
print("Train target padded shape:", padded_target_sequences.shape)



## === cell 6
val_anchor_sequences = tokenizer.texts_to_sequences(X_test["anchor"].values)
val_target_sequences = tokenizer.texts_to_sequences(X_test["target"].values)

val_padded_anchor_sequences = pad_sequences(
    val_anchor_sequences, maxlen=max_length, truncating=trunc_type
)
val_padded_target_sequences = pad_sequences(
    val_target_sequences, maxlen=max_length, truncating=trunc_type
)

print("Val anchor padded shape:", val_padded_anchor_sequences.shape)
print("Val target padded shape:", val_padded_target_sequences.shape)



## === cell 7
from tf_keras.models import Model
from tf_keras.layers import Input, Flatten, Lambda
from tf_keras.optimizers import RMSprop
from tf_keras import backend as K



def initialize_base_network():
    inp = Input(shape=(padded_anchor_sequences.shape[1],))
    common_embedding = keras.layers.Embedding(
        vocab_size, embedding_dim, input_length=max_length
    )(inp)
    common_lstm = keras.layers.Bidirectional(
        keras.layers.LSTM(32, return_sequences=True)
    )(common_embedding)
    flatten_layer = Flatten()(common_lstm)
    return Model(inputs=inp, outputs=flatten_layer)


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, _ = shapes
    return (shape1[0], 1)


base_network = initialize_base_network()
base_network.summary()



## === cell 8
input_1 = Input(shape=(padded_anchor_sequences.shape[1],), name="input_1")
input_2 = Input(shape=(padded_target_sequences.shape[1],), name="input_2")

vec_1 = base_network(input_1)
vec_2 = base_network(input_2)

output = Lambda(
    euclidean_distance, name="output_layer", output_shape=eucl_dist_output_shape
)([vec_1, vec_2])

siamese_model = Model([input_1, input_2], output)
siamese_model.summary()



## === cell 9
rms = RMSprop()
siamese_model.compile(loss="mse", optimizer=rms)

history = siamese_model.fit(
    [padded_anchor_sequences, padded_target_sequences],
    y_train.values,
    epochs=10,
    batch_size=64,
    validation_data=(
        [val_padded_anchor_sequences, val_padded_target_sequences],
        y_test.values,
    ),
    verbose=2,
)



## === cell 10
val_pred = siamese_model.predict(
    [val_padded_anchor_sequences, val_padded_target_sequences], verbose=0
)
print("Raw val_pred shape:", val_pred.shape)

val_pred = val_pred.reshape((len(val_pred),))
print("Flattened val_pred shape:", val_pred.shape)
print(
    "val_pred stats:",
    float(np.min(val_pred)),
    float(np.max(val_pred)),
    float(np.mean(val_pred)),
)



## === cell 11
test_anchor_sequences = tokenizer.texts_to_sequences(test["anchor"].values)
test_target_sequences = tokenizer.texts_to_sequences(test["target"].values)

test_padded_anchor_sequences = pad_sequences(
    test_anchor_sequences, maxlen=max_length, truncating=trunc_type
)
test_padded_target_sequences = pad_sequences(
    test_target_sequences, maxlen=max_length, truncating=trunc_type
)

test_pred = siamese_model.predict(
    [test_padded_anchor_sequences, test_padded_target_sequences], verbose=0
)
test_pred = test_pred.reshape((len(test_pred),))

print("Test pred shape:", test_pred.shape)



## === cell 12
submission = pd.DataFrame({"id": test["id"].values, "score": test_pred})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Columns:", submission.columns.tolist())
