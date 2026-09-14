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

3.7

# 3. Installed packages

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tf_keras==2.18.0

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

0.96436

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

print(os.listdir("../input"))



## === cell 1
import pandas as pd
import numpy as np

np.random.seed(42)



## === cell 2
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
print("train shape {} rows, {} cols".format(*df_train.shape))
print("test shape {} rows, {} cols".format(*df_test.shape))



## === cell 3
target_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
for label in target_cols:
    print(label)
    print(df_train[label].value_counts())
    print("*" * 80)



## === cell 4
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_features = 30000
maxlen = 100

X_train_raw = df_train["comment_text"].fillna("fillna").values
X_test_raw = df_test["comment_text"].fillna("fillna").values
y_train = df_train[target_cols].values

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train_raw) + list(X_test_raw))

X_train_seq = tokenizer.texts_to_sequences(X_train_raw)
X_test_seq = tokenizer.texts_to_sequences(X_test_raw)

x_train = pad_sequences(X_train_seq, maxlen=maxlen)
x_test = pad_sequences(X_test_seq, maxlen=maxlen)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 5
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Layer,
    Input,
    Embedding,
    GlobalMaxPooling1D,
    Dropout,
    Dense,
)
from tensorflow.keras.models import Model


class Position_Embedding(Layer):
    def __init__(self, size=None, mode="sum", **kwargs):
        super(Position_Embedding, self).__init__(**kwargs)
        self.size = size
        self.mode = mode

    def call(self, x):
        if (self.size is None) or (self.mode == "sum"):
            self.size = int(x.shape[-1])
        batch_size, seq_len = K.shape(x)[0], K.shape(x)[1]
        position_j = 1.0 / K.pow(
            10000.0, 2 * K.arange(self.size / 2, dtype="float32") / self.size
        )
        position_j = K.expand_dims(position_j, 0)
        position_i = K.cumsum(K.ones_like(x[:, :, 0]), 1) - 1
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        if self.mode == "sum":
            return position_ij + x
        elif self.mode == "concat":
            return K.concatenate([position_ij, x], 2)

    def compute_output_shape(self, input_shape):
        if self.mode == "sum":
            return input_shape
        else:  # concat
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)


class Attention(Layer):
    def __init__(self, nb_head, size_per_head, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head

    def build(self, input_shape):
        self.WQ = self.add_weight(
            name="WQ",
            shape=(input_shape[0][-1], self.output_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.WK = self.add_weight(
            name="WK",
            shape=(input_shape[1][-1], self.output_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.WV = self.add_weight(
            name="WV",
            shape=(input_shape[2][-1], self.output_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        super(Attention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode="mul"):
        if seq_len is None:
            return inputs
        mask = K.one_hot(seq_len[:, 0], K.shape(inputs)[1])
        mask = 1 - K.cumsum(mask, 1)
        for _ in range(len(inputs.shape) - 2):
            mask = K.expand_dims(mask, 2)
        if mode == "mul":
            return inputs * mask
        else:  # add
            return inputs - (1 - mask) * 1e12

    def call(self, x):
        if len(x) == 3:
            Q_seq, K_seq, V_seq = x
            Q_len, V_len = None, None
        else:
            Q_seq, K_seq, V_seq, Q_len, V_len = x
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(
            Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head)
        )
        Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))

        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(
            K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head)
        )
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))

        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(
            V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head)
        )
        V_seq = K.permute_dimensions(V_seq, (0, 2, 1, 3))

        A = K.batch_dot(Q_seq, K_seq, axes=[3, 3]) / (self.size_per_head**0.5)
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = self.Mask(A, V_len, "add")
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = K.softmax(A)

        O_seq = K.batch_dot(A, V_seq, axes=[3, 2])
        O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, "mul")
        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)




## === cell 6
from tensorflow.keras.callbacks import Callback, EarlyStopping
from sklearn.metrics import roc_auc_score


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(RocAucEvaluation, self).__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch + 1, score))




## === cell 7
S_inputs = Input(shape=(None,), dtype="int32")
emb = Embedding(max_features, 128)(S_inputs)
emb = Position_Embedding()(emb)
att = Attention(8, 16)([emb, emb, emb])
pool = GlobalMaxPooling1D()(att)
drop = Dropout(0.5)(pool)
outputs = Dense(6, activation="sigmoid")(drop)
model = Model(inputs=S_inputs, outputs=outputs)



## === cell 8
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])



## === cell 9
from sklearn.model_selection import train_test_split

X_tra, X_val, y_tra, y_val = train_test_split(
    x_train, y_train, test_size=0.3, random_state=233
)

roc_auc_cb = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)



## === cell 10
hist = model.fit(
    X_tra,
    y_tra,
    validation_data=(X_val, y_val),
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True), roc_auc_cb],
    batch_size=32,
    epochs=5,  # limited epochs for reasonable runtime
    verbose=2,
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3010912299.py in <cell line: 0>()
----> 1 hist = model.fit(
      2     X_tra,
      3     y_tra,
      4     validation_data=(X_val, y_val),
      5     callbacks=[EarlyStopping(patience=5, restore_best_weights=True), roc_auc_cb],

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_55/4242926768.py in call(self, x)
    107 
    108         A = K.batch_dot(Q_seq, K_seq, axes=[3, 3]) / (self.size_per_head**0.5)
--> 109         A = K.permute_dimensions(A, (0, 3, 2, 1))
    110         A = self.Mask(A, V_len, "add")
    111         A = K.permute_dimensions(A, (0, 3, 2, 1))

ValueError: Exception encountered when calling Attention.call().

Dimension must be 5 but is 4 for '{{node functional_1/attention_1/transpose_7}} = Transpose[T=DT_FLOAT, Tperm=DT_INT32](functional_1/attention_1/truediv, functional_1/attention_1/transpose_7/perm)' with input shapes: [?,8,100,8,100], [4].

Arguments received by Attention.call():
  • x=['tf.Tensor(shape=(None, 100, 128), dtype=float32)', 'tf.Tensor(shape=(None, 100, 128), dtype=float32)', 'tf.Tensor(shape=(None, 100, 128), dtype=float32)']

## === cell 11
y_pred = model.predict(x_test, batch_size=1024)
submission = pd.read_csv("../input/sample_submission.csv")
submission[target_cols] = y_pred
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3022568964.py in <cell line: 0>()
----> 1 y_pred = model.predict(x_test, batch_size=1024)
      2 submission = pd.read_csv("../input/sample_submission.csv")
      3 submission[target_cols] = y_pred
      4 submission.to_csv("submission.csv", index=False)
      5 print("Submission file saved as submission.csv")

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_55/4242926768.py in call(self, x)
    107 
    108         A = K.batch_dot(Q_seq, K_seq, axes=[3, 3]) / (self.size_per_head**0.5)
--> 109         A = K.permute_dimensions(A, (0, 3, 2, 1))
    110         A = self.Mask(A, V_len, "add")
    111         A = K.permute_dimensions(A, (0, 3, 2, 1))

ValueError: Exception encountered when calling Attention.call().

Dimension must be 5 but is 4 for '{{node functional_1/attention_1/transpose_7}} = Transpose[T=DT_FLOAT, Tperm=DT_INT32](functional_1/attention_1/truediv, functional_1/attention_1/transpose_7/perm)' with input shapes: [1024,8,100,8,100], [4].

Arguments received by Attention.call():
  • x=['tf.Tensor(shape=(1024, 100, 128), dtype=float32)', 'tf.Tensor(shape=(1024, 100, 128), dtype=float32)', 'tf.Tensor(shape=(1024, 100, 128), dtype=float32)']
