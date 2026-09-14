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

0.66487

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.88404) has done: 'I remove the protobuf environment setting that causes an import error, fix the model’s input shape to match Keras expectations (using a scalar string input), and ensure the training, validation, and test pipelines work with this corrected input. These minimal changes eliminate the runtime failures, allow the model to train, generate predictions, and finally write a proper `submission.csv` file, moving the solution toward the target score.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

import re, string
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


class Config:
    vocab_size = 15000
    tfidf_vocab_size = 40000
    sequence_length = 100
    batch_size = 1024
    validation_split = 0.15
    embed_dim = 256
    latent_dim = 256
    epochs = 10
    best_auc_model_path = "model_best_auc.weights.h5"
    best_acc_model_path = "model_best_acc.weights.h5"
    lastest_model_path = "model_latest.weights.h5"
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


config = Config()



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv"
test_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv"
sample_sub_path = (
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv"
)

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_sub_path)




## === cell 2
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "", " ")
    text = tf.strings.regex_replace(
        stripped_html, f"[{re.escape(string.punctuation)}]", ""
    )
    text = tf.strings.regex_replace(text, r"[0-9]+", " ")
    text = tf.strings.regex_replace(text, r"[ ]+", " ")
    text = tf.strings.strip(text)
    return text




## === cell 3
tfidf_vectorizer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=config.tfidf_vocab_size,
    output_mode="tf-idf",
    ngrams=2,
)
tfidf_adapt_ds = tf.data.Dataset.from_tensor_slices(
    train["comment_text"].astype(str)
).batch(1024)
tfidf_vectorizer.adapt(tfidf_adapt_ds)



## === cell 4
word2vec_vectorizer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=config.vocab_size,
    output_sequence_length=config.sequence_length,
)
word2vec_adapt_ds = tf.data.Dataset.from_tensor_slices(
    train["comment_text"].astype(str)
).batch(1024)
word2vec_vectorizer.adapt(word2vec_adapt_ds)



## === cell 5
X_train, X_val, y_train, y_val = train_test_split(
    train["comment_text"],
    train[config.labels],
    test_size=config.validation_split,
    random_state=42,
)




## === cell 6
def make_dataset(X, y, batch_size, mode):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if mode == "train":
        dataset = dataset.shuffle(256)
    dataset = dataset.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    return dataset




## === cell 7
train_ds = make_dataset(X_train, y_train, config.batch_size, mode="train")
valid_ds = make_dataset(X_val, y_val, config.batch_size, mode="valid")




## === cell 8
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
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




## === cell 9
class FNetEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, **kwargs):
        super().__init__(**kwargs)
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
        return self.layernorm_2(proj_input + proj_output)




## === cell 10
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




## === cell 11
def get_tfidf_model(config, inputs):
    x = tfidf_vectorizer(inputs)
    x = layers.Dense(256, activation="relu", kernel_regularizer="l2")(x)
    x = layers.Dense(100, activation="relu", kernel_regularizer="l2")(x)
    return x




## === cell 12
def get_model(config):
    inputs = keras.Input(shape=(), dtype="string", name="inputs")
    word2vec_x = get_word2vec_model(config, inputs)
    tfidf_x = get_tfidf_model(config, inputs)
    x = layers.Concatenate()([word2vec_x, tfidf_x])
    output = layers.Dense(6, activation="sigmoid")(x)
    return keras.Model(inputs, output, name="toxic_comment_model")




## === cell 13
model = get_model(config)
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["binary_accuracy", keras.metrics.AUC(name="auc")],
)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_54/4202532080.py in <cell line: 0>()
----> 1 model = get_model(config)
      2 model.compile(
      3     optimizer="adam",
      4     loss="binary_crossentropy",
      5     metrics=["binary_accuracy", keras.metrics.AUC(name="auc")],

/tmp/ipykernel_54/3609461418.py in get_model(config)
      1 def get_model(config):
      2     inputs = keras.Input(shape=(), dtype="string", name="inputs")
----> 3     word2vec_x = get_word2vec_model(config, inputs)
      4     tfidf_x = get_tfidf_model(config, inputs)
      5     x = layers.Concatenate()([word2vec_x, tfidf_x])

/tmp/ipykernel_54/2315512234.py in get_word2vec_model(config, inputs)
      4         config.sequence_length, config.vocab_size, config.embed_dim
      5     )(x)
----> 6     x = FNetEncoder(config.embed_dim, config.latent_dim)(x)
      7     x = layers.GlobalAveragePooling1D()(x)
      8     x = layers.Dropout(0.5)(x)

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_54/1124715928.py in call(self, inputs)
     14         inp_complex = tf.cast(inputs, tf.complex64)
     15         fft = tf.math.real(tf.signal.fft2d(inp_complex))
---> 16         proj_input = self.layernorm_1(inputs + fft)
     17         proj_output = self.dense_proj(proj_input)
     18         return self.layernorm_2(proj_input + proj_output)

TypeError: Exception encountered when calling FNetEncoder.call().

Could not automatically infer the output shape / dtype of 'f_net_encoder' (of type FNetEncoder). Either the `FNetEncoder.call()` method is incorrect, or you need to implement the `FNetEncoder.compute_output_spec() / compute_output_shape()` method. Error encountered:

Input 'y' of 'AddV2' Op has type float32 that does not match type float16 of argument 'x'.

Arguments received by FNetEncoder.call():
  • args=('<KerasTensor shape=(None, 100, 256), dtype=float16, sparse=False, name=keras_tensor_2>',)
  • kwargs=<class 'inspect._empty'>

## === cell 14
acc_checkpoint = keras.callbacks.ModelCheckpoint(
    config.best_acc_model_path,
    monitor="val_binary_accuracy",
    save_weights_only=True,
    save_best_only=True,
)
auc_checkpoint = keras.callbacks.ModelCheckpoint(
    config.best_auc_model_path,
    monitor="val_auc",
    save_weights_only=True,
    save_best_only=True,
)
reduce_lr = keras.callbacks.ReduceLROnPlateau(patience=5, min_delta=1e-4, min_lr=1e-6)

model.fit(
    train_ds,
    epochs=config.epochs,
    validation_data=valid_ds,
    callbacks=[acc_checkpoint, auc_checkpoint, reduce_lr],
)
model.save_weights(config.lastest_model_path)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/462920970.py in <cell line: 0>()
     13 reduce_lr = keras.callbacks.ReduceLROnPlateau(patience=5, min_delta=1e-4, min_lr=1e-6)
     14 
---> 15 model.fit(
     16     train_ds,
     17     epochs=config.epochs,

NameError: name 'model' is not defined

## === cell 15
y_pred_val = (model.predict(valid_ds) > 0.5).astype(int)
from sklearn.metrics import classification_report

print(classification_report(y_val, y_pred_val, target_names=config.labels))



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/3853129096.py in <cell line: 0>()
----> 1 y_pred_val = (model.predict(valid_ds) > 0.5).astype(int)
      2 from sklearn.metrics import classification_report
      3 
      4 print(classification_report(y_val, y_pred_val, target_names=config.labels))
      5 

NameError: name 'model' is not defined

## === cell 16
test_ds = tf.data.Dataset.from_tensor_slices(test["comment_text"]).batch(
    config.batch_size
)
predictions = []
for path in [
    config.best_acc_model_path,
    config.best_auc_model_path,
    config.lastest_model_path,
]:
    if os.path.isfile(path):
        model.load_weights(path)
        preds = model.predict(test_ds)
        predictions.append(preds)

if not predictions:
    predictions.append(model.predict(test_ds))

avg_preds = np.mean(predictions, axis=0)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/3531391422.py in <cell line: 0>()
     14 
     15 if not predictions:
---> 16     predictions.append(model.predict(test_ds))
     17 
     18 avg_preds = np.mean(predictions, axis=0)

NameError: name 'model' is not defined

## === cell 17
submission = sample_submission.copy()
submission[config.labels] = avg_preds
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("Submission saved to /kaggle/working/submission.csv")

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/75671367.py in <cell line: 0>()
      1 submission = sample_submission.copy()
----> 2 submission[config.labels] = avg_preds
      3 submission.to_csv("/kaggle/working/submission.csv", index=False)
      4 print("Submission saved to /kaggle/working/submission.csv")

NameError: name 'avg_preds' is not defined
