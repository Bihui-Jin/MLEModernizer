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

BASE_INPUT = "../input"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "/kaggle/input"

os.listdir(BASE_INPUT)[:10]



## === cell 1
import pandas as pd
import numpy as np



## === cell 2
np.random.seed(42)



## === cell 3
train_path = os.path.join(BASE_INPUT, "train.csv")
test_path = os.path.join(BASE_INPUT, "test.csv")
sub_path = os.path.join(BASE_INPUT, "sample_submission.csv")

if not os.path.exists(train_path):
    train_path = os.path.join(
        BASE_INPUT, "jigsaw-toxic-comment-classification-challenge", "train.csv"
    )
    test_path = os.path.join(
        BASE_INPUT, "jigsaw-toxic-comment-classification-challenge", "test.csv"
    )
    sub_path = os.path.join(
        BASE_INPUT,
        "jigsaw-toxic-comment-classification-challenge",
        "sample_submission.csv",
    )

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

print("train shape {} rows, {} cols".format(*df_train.shape))
print("test shape {} rows, {} cols".format(*df_test.shape))



## === cell 4
df_train.head()



## === cell 5
df_test.head()



## === cell 6
target_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
for label in target_cols:
    print(label)
    print(df_train[label].value_counts())
    print("*" * 80)



## === cell 7
from tf_keras.preprocessing import text, sequence



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 8
X_train = df_train["comment_text"].fillna("fillna").values
y_train = df_train[target_cols].values
X_test = df_test["comment_text"].fillna("fillna").values



## === cell 9
max_features = 30000
maxlen = 100
embed_size = 300  # kept as in original (unused in this architecture)

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))

X_train_seq = tokenizer.texts_to_sequences(X_train)
x_train = sequence.pad_sequences(X_train_seq, maxlen=maxlen)

X_test_seq = tokenizer.texts_to_sequences(X_test)
x_test = sequence.pad_sequences(X_test_seq, maxlen=maxlen)

print(x_train.shape, y_train.shape, x_test.shape)



## === cell 10
import tf_keras as keras
from tf_keras import backend as K
from tf_keras.layers import Layer, Input, Embedding, GlobalMaxPooling1D, Dropout, Dense
from tf_keras.models import Model


class Position_Embedding(Layer):
    def __init__(self, size=None, mode="sum", **kwargs):
        self.size = size  # must be even when explicitly set
        self.mode = mode
        super(Position_Embedding, self).__init__(**kwargs)

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
        return x

    def compute_output_shape(self, input_shape):
        if self.mode == "sum":
            return input_shape
        elif self.mode == "concat":
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)
        return input_shape




## === cell 11
class Attention(Layer):
    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head
        super(Attention, self).__init__(**kwargs)

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
        if mode == "add":
            return inputs - (1 - mask) * 1e12
        return inputs

    def call(self, x):
        if len(x) == 3:
            Q_seq, K_seq, V_seq = x
            Q_len, V_len = None, None
        elif len(x) == 5:
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




## === cell 12
from tf_keras.callbacks import Callback, EarlyStopping
from sklearn.metrics import roc_auc_score


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch + 1, score))




## === cell 13
S_inputs = Input(shape=(None,), dtype="int32")
embeddings = Embedding(max_features, 128)(S_inputs)
embeddings = Position_Embedding()(embeddings)
O_seq = Attention(8, 16)([embeddings, embeddings, embeddings])
O_seq = GlobalMaxPooling1D()(O_seq)
O_seq = Dropout(0.5)(O_seq)
outputs = Dense(6, activation="sigmoid")(O_seq)
model = Model(inputs=S_inputs, outputs=outputs)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/248852669.py in <cell line: 0>()
      3 embeddings = Embedding(max_features, 128)(S_inputs)
      4 embeddings = Position_Embedding()(embeddings)
----> 5 O_seq = Attention(8, 16)([embeddings, embeddings, embeddings])
      6 O_seq = GlobalMaxPooling1D()(O_seq)
      7 O_seq = Dropout(0.5)(O_seq)

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
     68             # To get the full stack trace, call:
     69             # `tf.debugging.disable_traceback_filtering()`
---> 70             raise e.with_traceback(filtered_tb) from None
     71         finally:
     72             del filtered_tb

/tmp/__autograph_generated_filehn3apc8h.py in tf__call(self, x)
     60                 V_seq = ag__.converted_call(ag__.ld(K).permute_dimensions, (ag__.ld(V_seq), (0, 2, 1, 3)), None, fscope)
     61                 A = ag__.converted_call(ag__.ld(K).batch_dot, (ag__.ld(Q_seq), ag__.ld(K_seq)), dict(axes=[3, 3]), fscope) / ag__.ld(self).size_per_head ** 0.5
---> 62                 A = ag__.converted_call(ag__.ld(K).permute_dimensions, (ag__.ld(A), (0, 3, 2, 1)), None, fscope)
     63                 A = ag__.converted_call(ag__.ld(self).Mask, (ag__.ld(A), ag__.ld(V_len), 'add'), None, fscope)
     64                 A = ag__.converted_call(ag__.ld(K).permute_dimensions, (ag__.ld(A), (0, 3, 2, 1)), None, fscope)

ValueError: Exception encountered when calling layer "attention" (type Attention).

in user code:

    File "/tmp/ipykernel_11/652366099.py", line 68, in call  *
        A = K.permute_dimensions(A, (0, 3, 2, 1))
    File "/usr/local/lib/python3.11/dist-packages/tf_keras/src/backend.py", line 3646, in permute_dimensions
        return tf.compat.v1.transpose(x, perm=pattern)

    ValueError: Dimension must be 5 but is 4 for '{{node attention/transpose_7}} = Transpose[T=DT_FLOAT, Tperm=DT_INT32](attention/truediv, attention/transpose_7/perm)' with input shapes: [?,8,?,8,?], [4].


Call arguments received by layer "attention" (type Attention):
  • x=['tf.Tensor(shape=(None, None, 128), dtype=float32)', 'tf.Tensor(shape=(None, None, 128), dtype=float32)', 'tf.Tensor(shape=(None, None, 128), dtype=float32)']

## === cell 14
try:
    from IPython.display import SVG
    from tf_keras.utils import model_to_dot

    SVG(model_to_dot(model).create(prog="dot", format="svg"))
except Exception as e:
    print("Model visualization skipped:", repr(e))



## === cell 15
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/878093567.py in <cell line: 0>()
----> 1 model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
      2 

NameError: name 'model' is not defined

## === cell 16
model.summary()



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1903595429.py in <cell line: 0>()
----> 1 model.summary()
      2 

NameError: name 'model' is not defined

## === cell 17
from sklearn.model_selection import train_test_split

X_tra, X_val, y_tra, y_val = train_test_split(
    x_train, y_train, test_size=0.3, random_state=233
)
roc_auc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)



## === cell 18
hist = model.fit(
    X_tra,
    y_tra,
    callbacks=[EarlyStopping(patience=5), roc_auc],
    batch_size=32,
    epochs=100,
    validation_data=(X_val, y_val),
    verbose=2,
)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3636334966.py in <cell line: 0>()
----> 1 hist = model.fit(
      2     X_tra,
      3     y_tra,
      4     callbacks=[EarlyStopping(patience=5), roc_auc],
      5     batch_size=32,

NameError: name 'model' is not defined

## === cell 19
import matplotlib.pyplot as plt

acc_key = (
    "accuracy"
    if "accuracy" in hist.history
    else ("acc" if "acc" in hist.history else None)
)
val_acc_key = (
    "val_accuracy"
    if "val_accuracy" in hist.history
    else ("val_acc" if "val_acc" in hist.history else None)
)

if acc_key and val_acc_key:
    plt.plot(hist.history[acc_key])
    plt.plot(hist.history[val_acc_key])
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.show()

plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper left")
plt.show()



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4121476254.py in <cell line: 0>()
      4 acc_key = (
      5     "accuracy"
----> 6     if "accuracy" in hist.history
      7     else ("acc" if "acc" in hist.history else None)
      8 )

NameError: name 'hist' is not defined

## === cell 20
y_pred = model.predict(x_test, batch_size=1024, verbose=1)

submission = pd.read_csv(sub_path)
submission[target_cols] = y_pred
submission = submission[["id"] + target_cols]
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/271994220.py in <cell line: 0>()
----> 1 y_pred = model.predict(x_test, batch_size=1024, verbose=1)
      2 
      3 submission = pd.read_csv(sub_path)
      4 # Ensure correct column order and shape
      5 submission[target_cols] = y_pred

NameError: name 'model' is not defined
