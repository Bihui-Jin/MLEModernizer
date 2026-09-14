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

3.10

# 3. Installed packages

geopandas==0.14.4
nltk==3.9.2
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
scipy==1.15.3
seaborn==0.12.2
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

0.95569

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
class Config:
    vocab_size = 15000  # Vocabulary Size
    tfidf_vocab_size = 40000
    sequence_length = 100  # Length of sequence
    batch_size = 1024
    validation_split = 0.15
    embed_dim = 256
    latent_dim = 256
    epochs = 10  # Number of Epochs to train
    best_auc_model_path = "model_best_auc.weights.h5"
    best_acc_model_path = "model_best_acc.weights.h5"
    latest_model_path = "model_latest.weights.h5"
    labels = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]


config = Config()




## === cell 1
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from scipy.stats import rankdata
import json



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
train_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv"
test_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv"
sample_sub_path = (
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv"
)

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_sub_path)




## === cell 3
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<.*?>", " ")
    text = tf.strings.regex_replace(
        stripped_html, f"[{re.escape(string.punctuation)}]", ""
    )
    text = tf.strings.regex_replace(text, f"[0-9]+", " ")
    text = tf.strings.regex_replace(text, f"[ ]+", " ")
    text = tf.strings.strip(text)
    return text




## === cell 4
tfidf_vectorizer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=config.tfidf_vocab_size,
    output_mode="tf-idf",
    ngrams=2,
)
tfidf_vectorizer.adapt(list(train["comment_text"]))




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3044032009.py in <cell line: 0>()
      6     ngrams=2,
      7 )
----> 8 tfidf_vectorizer.adapt(list(train["comment_text"]))
      9 
     10 

/usr/local/lib/python3.11/dist-packages/keras/src/layers/preprocessing/text_vectorization.py in adapt(self, data, batch_size, steps)
    426                 # is treated as as many documents
    427                 data = tf.expand_dims(data, -1)
--> 428             self.update_state(data)
    429         self.finalize_state()
    430 

/usr/local/lib/python3.11/dist-packages/keras/src/layers/preprocessing/text_vectorization.py in update_state(self, data)
    430 
    431     def update_state(self, data):
--> 432         self._lookup_layer.update_state(self._preprocess(data))
    433 
    434     def finalize_state(self):

/usr/local/lib/python3.11/dist-packages/keras/src/layers/preprocessing/text_vectorization.py in _preprocess(self, inputs)
    531             )
    532         if callable(self._standardize):
--> 533             inputs = self._standardize(inputs)
    534 
    535         if self._split is not None:

/tmp/ipykernel_55/1739815564.py in custom_standardization(input_data)
      4     stripped_html = tf.strings.regex_replace(lowercase, "<.*?>", " ")
      5     text = tf.strings.regex_replace(
----> 6         stripped_html, f"[{re.escape(string.punctuation)}]", ""
      7     )
      8     text = tf.strings.regex_replace(text, f"[0-9]+", " ")

NameError: name 're' is not defined

## === cell 5
word2vec_vectorizer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=config.vocab_size,
    output_sequence_length=config.sequence_length,
)
word2vec_vectorizer.adapt(list(train["comment_text"]))




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3215923123.py in <cell line: 0>()
      5     output_sequence_length=config.sequence_length,
      6 )
----> 7 word2vec_vectorizer.adapt(list(train["comment_text"]))
      8 
      9 

/usr/local/lib/python3.11/dist-packages/keras/src/layers/preprocessing/text_vectorization.py in adapt(self, data, batch_size, steps)
    426                 # is treated as as many documents
    427                 data = tf.expand_dims(data, -1)
--> 428             self.update_state(data)
    429         self.finalize_state()
    430 

/usr/local/lib/python3.11/dist-packages/keras/src/layers/preprocessing/text_vectorization.py in update_state(self, data)
    430 
    431     def update_state(self, data):
--> 432         self._lookup_layer.update_state(self._preprocess(data))
    433 
    434     def finalize_state(self):

/usr/local/lib/python3.11/dist-packages/keras/src/layers/preprocessing/text_vectorization.py in _preprocess(self, inputs)
    531             )
    532         if callable(self._standardize):
--> 533             inputs = self._standardize(inputs)
    534 
    535         if self._split is not None:

/tmp/ipykernel_55/1739815564.py in custom_standardization(input_data)
      4     stripped_html = tf.strings.regex_replace(lowercase, "<.*?>", " ")
      5     text = tf.strings.regex_replace(
----> 6         stripped_html, f"[{re.escape(string.punctuation)}]", ""
      7     )
      8     text = tf.strings.regex_replace(text, f"[0-9]+", " ")

NameError: name 're' is not defined

## === cell 6
X_train, X_val, y_train, y_val = train_test_split(
    train["comment_text"],
    train[config.labels],
    test_size=config.validation_split,
    random_state=42,
)




## === cell 7
X_train.shape, y_train.shape, X_val.shape, y_val.shape




## === cell 8
def make_dataset(X, y, batch_size, mode):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if mode == "train":
        dataset = dataset.shuffle(256, seed=42)
    dataset = dataset.batch(batch_size)
    dataset = dataset.cache().prefetch(tf.data.AUTOTUNE).repeat(1)
    return dataset




## === cell 9
train_ds = make_dataset(X_train, y_train, batch_size=config.batch_size, mode="train")
valid_ds = make_dataset(X_val, y_val, batch_size=config.batch_size, mode="valid")




## === cell 10
for batch in train_ds.take(1):
    print(batch)




## === cell 11
class FNetEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, dropout_rate=0.1, **kwargs):
        super(FNetEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs):
        inp_complex = tf.cast(inputs, tf.complex64)
        fft = tf.math.real(tf.signal.fft2d(inp_complex))
        proj_input = self.layernorm_1(inputs + fft)
        proj_output = self.dense_proj(proj_input)
        layer_norm = self.layernorm_2(proj_input + proj_output)
        return layer_norm




## === cell 12
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions




## === cell 13
def get_word2vec_model(config, inputs):
    x = word2vec_vectorizer(inputs)
    x = PositionalEmbedding(
        config.sequence_length, config.vocab_size, config.embed_dim
    )(x)
    x = FNetEncoder(config.embed_dim, config.latent_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    for _ in range(3):
        x = layers.Dense(100, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
    return x




## === cell 14
def get_tfidf_model(config, inputs):
    x = tfidf_vectorizer(inputs)
    x = layers.Dense(256, activation="relu", kernel_regularizer="l2")(x)
    x = layers.Dense(100, activation="relu", kernel_regularizer="l2")(x)
    return x




## === cell 15
def get_model(config):
    inputs = keras.Input(shape=(None,), dtype="string", name="inputs")
    word2vec_x = get_word2vec_model(config, inputs)
    tfidf_x = get_tfidf_model(config, inputs)
    x = layers.Concatenate()([word2vec_x, tfidf_x])
    output = layers.Dense(6, activation="sigmoid")(x)
    model = keras.Model(inputs, output, name="toxicity_model")
    return model




## === cell 16
model = get_model(config)
model.summary()




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1240837086.py in <cell line: 0>()
----> 1 model = get_model(config)
      2 model.summary()
      3 
      4 

/tmp/ipykernel_55/3135907718.py in get_model(config)
      2     inputs = keras.Input(shape=(None,), dtype="string", name="inputs")
      3     word2vec_x = get_word2vec_model(config, inputs)
----> 4     tfidf_x = get_tfidf_model(config, inputs)
      5     x = layers.Concatenate()([word2vec_x, tfidf_x])
      6     output = layers.Dense(6, activation="sigmoid")(x)

/tmp/ipykernel_55/757224984.py in get_tfidf_model(config, inputs)
      1 def get_tfidf_model(config, inputs):
      2     x = tfidf_vectorizer(inputs)
----> 3     x = layers.Dense(256, activation="relu", kernel_regularizer="l2")(x)
      4     x = layers.Dense(100, activation="relu", kernel_regularizer="l2")(x)
      5     return x

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/variables.py in _validate_shape(self, shape)
    207         shape = standardize_shape(shape)
    208         if None in shape:
--> 209             raise ValueError(
    210                 "Shapes used to initialize variables must be "
    211                 "fully-defined (no `None` dimensions). Received: "

ValueError: Shapes used to initialize variables must be fully-defined (no `None` dimensions). Received: shape=(None, 256) for variable path='dense_5/kernel'

## === cell 17
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["categorical_accuracy", keras.metrics.AUC(name="auc")],
)




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/153683437.py in <cell line: 0>()
----> 1 model.compile(
      2     optimizer="adam",
      3     loss="binary_crossentropy",
      4     metrics=["categorical_accuracy", keras.metrics.AUC(name="auc")],
      5 )

NameError: name 'model' is not defined

## === cell 18
acc_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=config.best_acc_model_path,
    monitor="val_categorical_accuracy",
    save_weights_only=True,
    save_best_only=True,
    mode="max",
    verbose=1,
)
auc_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=config.best_auc_model_path,
    monitor="val_auc",
    save_weights_only=True,
    save_best_only=True,
    mode="max",
    verbose=1,
)
early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    patience=5, min_delta=1e-4, min_lr=1e-6, verbose=1
)

model.fit(
    train_ds,
    epochs=config.epochs,
    validation_data=valid_ds,
    callbacks=[acc_checkpoint, auc_checkpoint, early_stopping, reduce_lr],
)
model.save_weights(config.latest_model_path)




## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/376641125.py in <cell line: 0>()
     21 )
     22 
---> 23 model.fit(
     24     train_ds,
     25     epochs=config.epochs,

NameError: name 'model' is not defined

## === cell 19
from sklearn.metrics import classification_report

y_val_pred_prob = model.predict(valid_ds)
y_val_pred = (y_val_pred_prob > 0.5).astype(int)
print(classification_report(y_val, y_val_pred, target_names=config.labels))




## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/894445492.py in <cell line: 0>()
      2 from sklearn.metrics import classification_report
      3 
----> 4 y_val_pred_prob = model.predict(valid_ds)
      5 y_val_pred = (y_val_pred_prob > 0.5).astype(int)
      6 print(classification_report(y_val, y_val_pred, target_names=config.labels))

NameError: name 'model' is not defined

## === cell 20
test_ds = (
    tf.data.Dataset.from_tensor_slices(test["comment_text"])
    .batch(config.batch_size)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)




## === cell 21
scores = []
for path in [config.best_acc_model_path, config.latest_model_path]:
    model.load_weights(path)
    pred = model.predict(test_ds)
    scores.append(pred)
score = np.mean(scores, axis=0)  # shape (num_test, 6)




## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1627506524.py in <cell line: 0>()
      2 scores = []
      3 for path in [config.best_acc_model_path, config.latest_model_path]:
----> 4     model.load_weights(path)
      5     pred = model.predict(test_ds)
      6     scores.append(pred)

NameError: name 'model' is not defined

## === cell 22
submission = sample_submission.copy()
submission[config.labels] = score
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4118794879.py in <cell line: 0>()
      1 # Create submission file
      2 submission = sample_submission.copy()
----> 3 submission[config.labels] = score
      4 submission_path = "/kaggle/working/submission.csv"
      5 submission.to_csv(submission_path, index=False)

NameError: name 'score' is not defined
