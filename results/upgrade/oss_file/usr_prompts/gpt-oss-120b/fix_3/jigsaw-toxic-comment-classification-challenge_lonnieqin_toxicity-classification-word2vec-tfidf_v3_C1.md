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
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 2 other files
        input/
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 2 other files
        working/
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 2 other files
```

-> data/jigsaw-toxic-comment-classification-challenge/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/jigsaw-toxic-comment-classification-challenge/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/jigsaw-toxic-comment-classification-challenge/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> input/jigsaw-toxic-comment-classification-challenge/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> input/jigsaw-toxic-comment-classification-challenge/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> working/jigsaw-toxic-comment-classification-challenge/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> (stopped after 10 files for performance)

# 5. Target score

0.96129

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pandas as pd
import tensorflow as tf
import numpy as np
import re, string
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split



## === cell 1
train_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv"
train = pd.read_csv(train_path)
train.head()




## === cell 2
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<.*?>", " ")
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
with tf.device("CPU"):
    tfidf_vectorizer.adapt(list(train["comment_text"]))



## === cell 4
word2vec_vectorizer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=config.vocab_size,
    output_sequence_length=config.sequence_length,
)
with tf.device("CPU"):
    word2vec_vectorizer.adapt(train["comment_text"])



## === cell 5
X_train, X_val, y_train, y_val = train_test_split(
    train["comment_text"],
    train[config.labels],
    test_size=config.validation_split,
    random_state=42,
)



## === cell 6
X_train.shape, y_train.shape, X_val.shape, y_val.shape




## === cell 7
def make_dataset(X, y, batch_size, mode):
    X_np = np.array(X)
    y_np = np.array(y)
    dataset = tf.data.Dataset.from_tensor_slices((X_np, y_np))
    if mode == "train":
        dataset = dataset.shuffle(256, seed=42)
    dataset = dataset.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    return dataset




## === cell 8
train_ds = make_dataset(X_train, y_train, batch_size=config.batch_size, mode="train")
valid_ds = make_dataset(X_val, y_val, batch_size=config.batch_size, mode="valid")



## === cell 9
for batch in train_ds.take(1):
    print(batch)




## === cell 10
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




## === cell 11
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




## === cell 12
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




## === cell 13
def get_tfidf_model(config, inputs):
    x = tfidf_vectorizer(inputs)
    x = layers.Dense(256, activation="relu", kernel_regularizer="l2")(x)
    x = layers.Dense(100, activation="relu", kernel_regularizer="l2")(x)
    return x




## === cell 14
def get_model(config):
    inputs = keras.Input(shape=(), dtype="string", name="inputs")
    word2vec_x = get_word2vec_model(config, inputs)
    tfidf_x = get_tfidf_model(config, inputs)
    x = layers.Concatenate()([word2vec_x, tfidf_x])
    output = layers.Dense(6, activation="sigmoid")(x)
    model = keras.Model(inputs, output, name="toxicity_model")
    return model




## === cell 15
model = get_model(config)
model.summary()



## === cell 16
try:
    keras.utils.plot_model(model, show_shapes=True, to_file="model.png")
except Exception:
    pass



## === cell 17
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["binary_accuracy", keras.metrics.AUC(name="auc")],
)



## === cell 18
acc_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=config.best_acc_model_path,
    monitor="val_binary_accuracy",
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
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    patience=5, min_delta=1e-4, min_lr=1e-6, verbose=1
)

model.fit(
    train_ds,
    epochs=config.epochs,
    validation_data=valid_ds,
    callbacks=[acc_checkpoint, auc_checkpoint, reduce_lr],
)
model.save_weights(config.latest_model_path)



## === cell 19
from sklearn.metrics import classification_report

y_pred = (model.predict(valid_ds) > 0.5).astype(int)
print(classification_report(y_val, y_pred, target_names=config.labels))



## === cell 20
test_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv"
test = pd.read_csv(test_path)
test.head()



## === cell 21
sample_submission_path = (
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv"
)
sample_submission = pd.read_csv(sample_submission_path)
sample_submission.head()



## === cell 22
test_ds = (
    tf.data.Dataset.from_tensor_slices(test["comment_text"])
    .batch(config.batch_size)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

scores = []
for path in [
    config.best_acc_model_path,
    config.best_auc_model_path,
    config.latest_model_path,
]:
    model.load_weights(path)
    scores.append(model.predict(test_ds))
score = np.mean(scores, axis=0)
print(score.shape)



## === cell 23
sample_submission[config.labels] = score
submission_path = "submission.csv"
sample_submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
