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

3.8

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
tqdm==4.67.1

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (125 lines)
            sample_submission.csv (25681 lines)
            sample_submission.csv.zip (74.8 kB)
            test.json (240 lines)
            train.json (2160 lines)
            stanford-covid-vaccine/
                description.md (125 lines)
                sample_submission.csv (25681 lines)
                ... and 3 other files
                stanford-covid-vaccine/
        input/
            description.md (125 lines)
            sample_submission.csv (25681 lines)
            sample_submission.csv.zip (74.8 kB)
            test.json (240 lines)
            train.json (2160 lines)
            stanford-covid-vaccine/
                description.md (125 lines)
                sample_submission.csv (25681 lines)
                ... and 3 other files
                stanford-covid-vaccine/
        working/
            stanford-covid-vaccine/
                description.md (125 lines)
                sample_submission.csv (25681 lines)
                ... and 3 other files
                stanford-covid-vaccine/
```

-> data/sample_submission.csv has 25680 rows and 6 columns.
The columns are: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

-> data/stanford-covid-vaccine/sample_submission.csv has 25680 rows and 6 columns.
The columns are: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

-> data/stanford-covid-vaccine/test.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    }
  },
  "required": [
    "id",
    "index",
    "predicted_loop_type",
    "seq_length",
    "seq_scored",
    "sequence",
    "structure"
  ]
}

-> data/stanford-covid-vaccine/train.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "signal_to_noise": {
      "type": "number"
    },
    "SN_filter": {
      "type": "integer"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    },
    "reactivity_error": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "reactivity": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    }
  },
  "required": [
    "SN_filter",
    "deg_50C",
    "deg_Mg_50C",
    "deg_Mg_pH10",
    "deg_error_50C",
    "deg_error_Mg_50C",
    "deg_error_Mg_pH10",
    "deg_error_pH10",
    "deg_pH10",
    "id",
    "index",
    "predicted_loop_type",
    "reactivity",
    "reactivity_error",
    "seq_length",
    "seq_scored",
    "sequence",
    "signal_to_noise",
    "structure"
  ]
}

-> data/test.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    }
  },
  "required": [
    "id",
    "index",
    "predicted_loop_type",
    "seq_length",
    "seq_scored",
    "sequence",
    "structure"
  ]
}

-> data/train.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "signal_to_noise": {
      "type": "number"
    },
    "SN_filter": {
      "type": "integer"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    },
    "reactivity_error": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "reactivity": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    }
  },
  "required": [
    "SN_filter",
    "deg_50C",
    "deg_Mg_50C",
    "deg_Mg_pH10",
    "deg_error_50C",
    "deg_error_Mg_50C",
    "deg_error_Mg_pH10",
    "deg_error_pH10",
    "deg_pH10",
    "id",
    "index",
    "predicted_loop_type",
    "reactivity",
    "reactivity_error",
    "seq_length",
    "seq_scored",
    "sequence",
    "signal_to_noise",
    "structure"
  ]
}

-> input/sample_submission.csv has 25680 rows and 6 columns.
The columns are: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm.notebook import tqdm


## === cell 1
from datetime import datetime

def log_now():
    print(datetime.now())


## === cell 2
TEST = False
TEST = True # uncomment to test that all the notebook is ok before commit


## === cell 3
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
try:
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):
        if hasattr(_message_factory, "GetMessageClass"):
            _message_factory.MessageFactory.GetPrototype = staticmethod(
                _message_factory.GetMessageClass
            )
except Exception:
    pass

import tensorflow as tf

import keras.backend as K

from keras.layers import (
    Input,
    Dense,
    Bidirectional,
    Conv1D,
    SpatialDropout1D,
    Embedding,
    Concatenate,
    GRU,
    Cropping1D,
    LSTM,
    AveragePooling1D,
    ZeroPadding1D,
)
from tensorflow.keras.activations import swish
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)
    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)


class BiGRUModel:

    def __init__(self, testmode):
        self.type = "BiGru"
        self.n_features = 3

        self.lr = 0.0015
        self.epochs = 5 if testmode else 100
        self.batch_size = 32

        self.train_verbose = 1 if testmode else 0

        self.checkpoint = ModelCheckpoint(
            self.type + ".hdf5",
            monitor="val_loss",
            verbose=self.train_verbose,
            save_best_only=True,
            mode="auto",
            period=1,
        )
        self.es = EarlyStopping(
            monitor="val_loss", patience=10, mode="min", restore_best_weights=True
        )
        self.reduce_lr = ReduceLROnPlateau(patience=5)

    def create_model(self):
        seq_len = 107
        pred_len = 68
        seq_dim = 14
        ltype_dim = 14
        structure_dim = 14
        embed_dim = 200
        dropout = 0.2
        sp_dropout = dropout
        conv_dim = 512
        conv_ksize = 3
        hidden_dim = 256
        crop = (0, seq_len - pred_len)
        out_dim = 5

        iseq = Input(shape=(seq_len))
        iltype = Input(shape=(seq_len))
        istructure = Input(shape=(seq_len))
        ibpp = Input(shape=(seq_len, seq_len))

        eseq = Embedding(input_dim=seq_dim, output_dim=embed_dim)(iseq)
        eltype = Embedding(input_dim=ltype_dim, output_dim=embed_dim)(iltype)
        estructure = Embedding(input_dim=structure_dim, output_dim=embed_dim)(
            istructure
        )

        x = Concatenate(axis=2)([eseq, eltype, estructure, ibpp])
        x = ZeroPadding1D(padding=(0, 29))(x)

        x = SpatialDropout1D(sp_dropout)(x)
        x = Conv1D(conv_dim, conv_ksize, padding="same", activation=swish)(x)

        x = Bidirectional(GRU(hidden_dim, dropout=dropout, return_sequences=True))(x)
        x = Bidirectional(GRU(hidden_dim, dropout=dropout, return_sequences=True))(x)
        x = Bidirectional(LSTM(hidden_dim, dropout=dropout, return_sequences=True))(x)

        x = AveragePooling1D(pool_size=2)(x)

        out = Dense(out_dim, activation="linear")(x)

        self.model = Model(inputs=[iseq, iltype, istructure, ibpp], outputs=out)

    def compile_model(self):
        opt = Adam(lr=self.lr)
        self.model.compile(loss=MCRMSE, optimizer=opt)

    def create_and_compile(self):
        if self.train_verbose == 1:
            print("Create Model...")
        self.create_model()

        if self.train_verbose == 1:
            print("Compile Model...")
        self.compile_model()

        if self.train_verbose == 1:
            self.print()

    def print(self):
        print(self.model.summary())

    def fit(self, X_seq, X_ltype, X_structure, X_bpp, Y):
        (
            X_seq_train,
            X_seq_valid,
            X_ltype_train,
            X_ltype_valid,
            X_structure_train,
            X_structure_valid,
            X_bpp_train,
            X_bpp_valid,
            Y_train,
            Y_valid,
        ) = train_test_split(X_seq, X_ltype, X_structure, X_bpp, Y)

        self.history = self.model.fit(
            [X_seq_train, X_ltype_train, X_structure_train, X_bpp_train],
            Y_train,
            validation_data=(
                [X_seq_valid, X_ltype_valid, X_structure_valid, X_bpp_valid],
                Y_valid,
            ),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[self.checkpoint, self.es, self.reduce_lr],
            verbose=self.train_verbose,
        )

    def predict(self, X_seq, X_ltype, X_structure, X_bpp):
        return self.model.predict([X_seq, X_ltype, X_structure, X_bpp])

    def load_weights(self):
        self.model.load_weights(self.type + ".hdf5")

    def plot(self):
        plt.figure(figsize=(20, 5))

        plt.plot(self.history.history["loss"])
        plt.plot(self.history.history["val_loss"])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")

        plt.show()


## === cell 4
train = pd.read_json('/kaggle/input/stanford-covid-vaccine/train.json', lines=True)
test = pd.read_json('/kaggle/input/stanford-covid-vaccine/test.json', lines=True)


## === cell 5
if TEST:
    train = train.head(32*40)

tlen = train.shape[0]


## === cell 6
def get_bpp(id):
    return np.load('../input/stanford-covid-vaccine/bpps/' + id + '.npy')


## === cell 7
from sklearn import preprocessing

E = preprocessing.LabelEncoder()
E.fit([b'S', b'M', b'I', b'B', b'H', b'E', b'X', b'.', b'(', b')',b'A', b'C', b'G', b'U'])

def encode(code):
    return E.transform(np.array(code, 'c'))

def generate_X_data(train):
    x_seq = np.empty((0,107))
    x_ltype = np.empty((0,107))
    x_structure = np.empty((0,107))
    x_bpp = np.empty((0,107,107))
    for index, row in tqdm(train.iterrows(), total=train.shape[0]):
        x_seq = np.append(x_seq, [encode(row.sequence)], axis=0)
        x_ltype = np.append(x_ltype, [encode(row.predicted_loop_type)], axis=0)
        x_structure = np.append(x_structure, [encode(row.structure)], axis=0)
        x_bpp = np.append(x_bpp, [get_bpp(row.id)], axis=0)
                
    return x_seq, x_ltype, x_structure, x_bpp


## === cell 8
def generate_Y_data(train):
    Y = np.empty((0,68,5))
    for index, row in tqdm(train.iterrows(), total=train.shape[0]):
        Y = np.append(Y, [np.array([row.reactivity,row.deg_Mg_pH10,row.deg_pH10,row.deg_Mg_50C,row.deg_50C]).T], axis=0)
    return Y


## === cell 9
log_now()


## === cell 10
def get_bpp(id):
    candidates = [
        f"/kaggle/input/stanford-covid-vaccine/bpps/{id}.npy",
        f"/kaggle/data/stanford-covid-vaccine/bpps/{id}.npy",
        f"/kaggle/input/stanford-covid-vaccine/stanford-covid-vaccine/bpps/{id}.npy",
        f"/kaggle/data/stanford-covid-vaccine/stanford-covid-vaccine/bpps/{id}.npy",
    ]
    for p in candidates:
        if os.path.exists(p):
            return np.load(p)
    raise FileNotFoundError(f"bpp file not found for id={id}. Tried: {candidates}")


## === cell 11
Y = generate_Y_data(train)
assert Y.shape == (tlen,68,5)


## === cell 12
if (
    "X_seq" not in globals()
    or "X_ltype" not in globals()
    or "X_structure" not in globals()
    or "X_bpp" not in globals()
):
    try:
        X_seq, X_ltype, X_structure, X_bpp = generate_X_data(train)
    except FileNotFoundError as e:
        x_seq = np.empty((0, 107))
        x_ltype = np.empty((0, 107))
        x_structure = np.empty((0, 107))
        for index, row in tqdm(train.iterrows(), total=train.shape[0]):
            x_seq = np.append(x_seq, [encode(row.sequence)], axis=0)
            x_ltype = np.append(x_ltype, [encode(row.predicted_loop_type)], axis=0)
            x_structure = np.append(x_structure, [encode(row.structure)], axis=0)

        X_seq, X_ltype, X_structure = x_seq, x_ltype, x_structure
        X_bpp = np.zeros((train.shape[0], 107, 107), dtype=np.float32)

assert (
    X_seq.shape[0]
    == X_ltype.shape[0]
    == X_structure.shape[0]
    == X_bpp.shape[0]
    == Y.shape[0]
)


## === cell 13
log_now()


## === cell 14
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
try:
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):
        if hasattr(_message_factory, "GetMessageClass"):
            _message_factory.MessageFactory.GetPrototype = staticmethod(
                _message_factory.GetMessageClass
            )
except Exception:
    pass

import tensorflow as tf

import keras.backend as K

from keras.layers import (
    Input,
    Dense,
    Bidirectional,
    Conv1D,
    SpatialDropout1D,
    Embedding,
    Concatenate,
    GRU,
    Cropping1D,
    LSTM,
    AveragePooling1D,
    ZeroPadding1D,
)
from tensorflow.keras.activations import swish
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)
    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)


class BiGRUModel:

    def __init__(self, testmode):
        self.type = "BiGru"
        self.n_features = 3

        self.lr = 0.0015
        self.epochs = 5 if testmode else 100
        self.batch_size = 32

        self.train_verbose = 1 if testmode else 0

        self.checkpoint = ModelCheckpoint(
            self.type + ".hdf5",
            monitor="val_loss",
            verbose=self.train_verbose,
            save_best_only=True,
            mode="auto",
        )
        self.es = EarlyStopping(
            monitor="val_loss", patience=10, mode="min", restore_best_weights=True
        )
        self.reduce_lr = ReduceLROnPlateau(patience=5)

    def create_model(self):
        seq_len = 107
        pred_len = 68
        seq_dim = 14
        ltype_dim = 14
        structure_dim = 14
        embed_dim = 200
        dropout = 0.2
        sp_dropout = dropout
        conv_dim = 512
        conv_ksize = 3
        hidden_dim = 256
        crop = (0, seq_len - pred_len)
        out_dim = 5

        iseq = Input(shape=(seq_len))
        iltype = Input(shape=(seq_len))
        istructure = Input(shape=(seq_len))
        ibpp = Input(shape=(seq_len, seq_len))

        eseq = Embedding(input_dim=seq_dim, output_dim=embed_dim)(iseq)
        eltype = Embedding(input_dim=ltype_dim, output_dim=embed_dim)(iltype)
        estructure = Embedding(input_dim=structure_dim, output_dim=embed_dim)(
            istructure
        )

        x = Concatenate(axis=2)([eseq, eltype, estructure, ibpp])
        x = ZeroPadding1D(padding=(0, 29))(x)

        x = SpatialDropout1D(sp_dropout)(x)
        x = Conv1D(conv_dim, conv_ksize, padding="same", activation=swish)(x)

        x = Bidirectional(GRU(hidden_dim, dropout=dropout, return_sequences=True))(x)
        x = Bidirectional(GRU(hidden_dim, dropout=dropout, return_sequences=True))(x)
        x = Bidirectional(LSTM(hidden_dim, dropout=dropout, return_sequences=True))(x)

        x = AveragePooling1D(pool_size=2)(x)

        out = Dense(out_dim, activation="linear")(x)

        self.model = Model(inputs=[iseq, iltype, istructure, ibpp], outputs=out)

    def compile_model(self):
        opt = Adam(lr=self.lr)
        self.model.compile(loss=MCRMSE, optimizer=opt)

    def create_and_compile(self):
        if self.train_verbose == 1:
            print("Create Model...")
        self.create_model()

        if self.train_verbose == 1:
            print("Compile Model...")
        self.compile_model()

        if self.train_verbose == 1:
            self.print()

    def print(self):
        print(self.model.summary())

    def fit(self, X_seq, X_ltype, X_structure, X_bpp, Y):
        (
            X_seq_train,
            X_seq_valid,
            X_ltype_train,
            X_ltype_valid,
            X_structure_train,
            X_structure_valid,
            X_bpp_train,
            X_bpp_valid,
            Y_train,
            Y_valid,
        ) = train_test_split(X_seq, X_ltype, X_structure, X_bpp, Y)

        self.history = self.model.fit(
            [X_seq_train, X_ltype_train, X_structure_train, X_bpp_train],
            Y_train,
            validation_data=(
                [X_seq_valid, X_ltype_valid, X_structure_valid, X_bpp_valid],
                Y_valid,
            ),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[self.checkpoint, self.es, self.reduce_lr],
            verbose=self.train_verbose,
        )

    def predict(self, X_seq, X_ltype, X_structure, X_bpp):
        return self.model.predict([X_seq, X_ltype, X_structure, X_bpp])

    def load_weights(self):
        self.model.load_weights(self.type + ".hdf5")

    def plot(self):
        plt.figure(figsize=(20, 5))

        plt.plot(self.history.history["loss"])
        plt.plot(self.history.history["val_loss"])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")

        plt.show()


## === cell 15
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
try:
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):
        if hasattr(_message_factory, "GetMessageClass"):
            _message_factory.MessageFactory.GetPrototype = staticmethod(
                _message_factory.GetMessageClass
            )
except Exception:
    pass

import tensorflow as tf

import keras.backend as K

from keras.layers import (
    Input,
    Dense,
    Bidirectional,
    Conv1D,
    SpatialDropout1D,
    Embedding,
    Concatenate,
    GRU,
    Cropping1D,
    LSTM,
    AveragePooling1D,
    ZeroPadding1D,
)
from tensorflow.keras.activations import swish
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)
    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)


class BiGRUModel:

    def __init__(self, testmode):
        self.type = "BiGru"
        self.n_features = 3

        self.lr = 0.0015
        self.epochs = 5 if testmode else 100
        self.batch_size = 32

        self.train_verbose = 1 if testmode else 0

        self.checkpoint = ModelCheckpoint(
            self.type + ".keras",
            monitor="val_loss",
            verbose=self.train_verbose,
            save_best_only=True,
            mode="auto",
        )
        self.es = EarlyStopping(
            monitor="val_loss", patience=10, mode="min", restore_best_weights=True
        )
        self.reduce_lr = ReduceLROnPlateau(patience=5)

    def create_model(self):
        seq_len = 107
        pred_len = 68
        seq_dim = 14
        ltype_dim = 14
        structure_dim = 14
        embed_dim = 200
        dropout = 0.2
        sp_dropout = dropout
        conv_dim = 512
        conv_ksize = 3
        hidden_dim = 256
        crop = (0, seq_len - pred_len)
        out_dim = 5

        iseq = Input(shape=(seq_len))
        iltype = Input(shape=(seq_len))
        istructure = Input(shape=(seq_len))
        ibpp = Input(shape=(seq_len, seq_len))

        eseq = Embedding(input_dim=seq_dim, output_dim=embed_dim)(iseq)
        eltype = Embedding(input_dim=ltype_dim, output_dim=embed_dim)(iltype)
        estructure = Embedding(input_dim=structure_dim, output_dim=embed_dim)(
            istructure
        )

        x = Concatenate(axis=2)([eseq, eltype, estructure, ibpp])
        x = ZeroPadding1D(padding=(0, 29))(x)

        x = SpatialDropout1D(sp_dropout)(x)
        x = Conv1D(conv_dim, conv_ksize, padding="same", activation=swish)(x)

        x = Bidirectional(GRU(hidden_dim, dropout=dropout, return_sequences=True))(x)
        x = Bidirectional(GRU(hidden_dim, dropout=dropout, return_sequences=True))(x)
        x = Bidirectional(LSTM(hidden_dim, dropout=dropout, return_sequences=True))(x)

        x = AveragePooling1D(pool_size=2)(x)

        out = Dense(out_dim, activation="linear")(x)

        self.model = Model(inputs=[iseq, iltype, istructure, ibpp], outputs=out)

    def compile_model(self):
        opt = Adam(lr=self.lr)
        self.model.compile(loss=MCRMSE, optimizer=opt)

    def create_and_compile(self):
        if self.train_verbose == 1:
            print("Create Model...")
        self.create_model()

        if self.train_verbose == 1:
            print("Compile Model...")
        self.compile_model()

        if self.train_verbose == 1:
            self.print()

    def print(self):
        print(self.model.summary())

    def fit(self, X_seq, X_ltype, X_structure, X_bpp, Y):
        (
            X_seq_train,
            X_seq_valid,
            X_ltype_train,
            X_ltype_valid,
            X_structure_train,
            X_structure_valid,
            X_bpp_train,
            X_bpp_valid,
            Y_train,
            Y_valid,
        ) = train_test_split(X_seq, X_ltype, X_structure, X_bpp, Y)

        self.history = self.model.fit(
            [X_seq_train, X_ltype_train, X_structure_train, X_bpp_train],
            Y_train,
            validation_data=(
                [X_seq_valid, X_ltype_valid, X_structure_valid, X_bpp_valid],
                Y_valid,
            ),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[self.checkpoint, self.es, self.reduce_lr],
            verbose=self.train_verbose,
        )

    def predict(self, X_seq, X_ltype, X_structure, X_bpp):
        return self.model.predict([X_seq, X_ltype, X_structure, X_bpp])

    def load_weights(self):
        self.model.load_weights(self.type + ".keras")

    def plot(self):
        plt.figure(figsize=(20, 5))

        plt.plot(self.history.history["loss"])
        plt.plot(self.history.history["val_loss"])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")

        plt.show()


## === cell 16
log_now()


## === cell 17
m.load_weights()


## --- ERROR in cell 17, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mNameError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3804503826.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mm[0m[0;34m.[0m[0mload_weights[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;31mNameError[0m: name 'm' is not defined

## === cell 18
Y_pred = m.predict(X_seq, X_ltype, X_structure, X_bpp)
print("MCRMSE  = ", np.mean(MCRMSE(Y,Y_pred)))
