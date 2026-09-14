# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Overview
Predict likely degradation rates at each base of an RNA molecule.

## Metric
Mean columnwise root mean squared error:

$\textrm{MCRMSE} = \frac{1}{N_{t}}\sum_{j=1}^{N_{t}}\sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_{ij} - \hat{y}_{ij})^2}$

where $N_{t}$ is the number of scored ground truth target columns, and $y$ and $\hat{y}$ are the actual and predicted values, respectively.

There are multiple ground truth values provided in the training data. While the submission format requires all 5 to be predicted, only the following are scored: reactivity, deg_Mg_pH10, and deg_Mg_50C.

## Submission Formats
For each sample `id` in the test set, you must predict targets for *each* sequence position (`seqpos`), one per row. If the length of the `sequence` of an `id` is, e.g., 107, then you should make 107 predictions. Positions greater than the `seq_scored` value of a sample are not scored, but still need a value in the solution file.

```csv
id_seqpos,reactivity,deg_Mg_pH10,deg_pH10,deg_Mg_50C,deg_50C
id_d190610e8_0,0.1,0.3,0.2,0.5,0.4
id_d190610e8_1,0.3,0.2,0.5,0.4,0.2
id_d190610e8_2,0.5,0.4,0.2,0.1,0.2
etc.
```

## Dataset 
- **train.json** - the training data
- **test.json** - the test set, without any columns associated with the ground truth.
- **sample_submission.csv** - a sample submission file in the correct format

#### Columns
- `id` - An arbitrary identifier for each sample.
- `seq_scored` - (68 in Train and Public Test, 68 in Private Test) Integer value denoting the number of positions used in scoring with predicted values. This should match the length of `reactivity`, `deg_*` and `*_error_*` columns.
- `seq_length` - (107 in Train and Public Test, 107 in Private Test) Integer values, denotes the length of `sequence`.
- `sequence` - (1x107 string in Train and Public Test, 107 in Private Test) Describes the RNA sequence, a combination of `A`, `G`, `U`, and `C` for each sample. Should be 107 characters long, and the first 68 bases should correspond to the 68 positions specified in `seq_scored` (note: indexed starting at 0).
- `structure` - (1x107 string in Train and Public Test, 107 in Private Test) An array of `(`, `)`, and `.` characters that describe whether a base is estimated to be paired or unpaired. Paired bases are denoted by opening and closing parentheses e.g. (....) means that base 0 is paired to base 5, and bases 1-4 are unpaired.
- `reactivity` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likely secondary structure of the RNA sample.
- `deg_pH10` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating without magnesium at high pH (pH 10).
- `deg_Mg_pH10` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating with magnesium in high pH (pH 10).
- `deg_50C` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating without magnesium at high temperature (50 degrees Celsius).
- `deg_Mg_50C` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating with magnesium at high temperature (50 degrees Celsius).
- `*_error_*` - An array of floating point numbers, should have the same length as the corresponding `reactivity` or `deg_*` columns, calculated errors in experimental values obtained in `reactivity` and `deg_*` columns.
- `predicted_loop_type` - (1x107 string) Describes the structural context (also referred to as 'loop type')of each character in `sequence`. Loop types assigned by bpRNA from Vienna RNAfold 2 structure. From the bpRNA_documentation: S: paired "Stem" M: Multiloop I: Internal loop B: Bulge H: Hairpin loop E: dangling End X: eXternal loop
    - `S/N filter` Indicates if the sample passed filters described below in `Additional Notes`.

#### Additional Notes
At the beginning of the competition, Stanford scientists have data on 2400 RNA sequences of length 107. For technical reasons, measurements cannot be carried out on the final bases of these RNA sequences, so we have experimental data (ground truth) in 5 conditions for the first 68 bases.

We have split out 240 of these 2400 sequences for a public test set to allow for continuous evaluation through the competition, on the public leaderboard. These sequences, in `test.json`, have been additionally filtered based on three criteria detailed below to ensure that this subset is not dominated by any large cluster of RNA molecules with poor data, which might bias the public leaderboard. The remaining 2160 sequences for which we have data are in `train.json`.

For our final and most important scoring (the Private Leaderbooard), Stanford scientists are carrying out measurements on 240 new RNAs. For these data, we expect to have measurements for the first 68 bases, again missing the ends of the RNA. These sequences constitute the 240 sequences in `test.json`.

For those interested in how the sequences in `test.json` were filtered, here were the steps to ensure a diverse and high quality test set for public leaderboard scoring:

1. Minimum value across all 5 conditions must be greater than -0.5.
2. Mean signal/noise across all 5 conditions must be greater than 1.0. [Signal/noise is defined as mean( measurement value over 68 nts )/mean( statistical error in measurement value over 68 nts)]
3. To help ensure sequence diversity, the resulting sequences were clustered into clusters with less than 50% sequence similarity, and the 240 test set sequences were chosen from clusters with 3 or fewer members. That is, any sequence in the test set should be sequence similar to at most 2 other sequences.

Note that these filters have not been applied to the 2160 RNAs in the public training data `train.json` -- some of those measurements have negative values or poor signal-to-noise, or some RNA sequences have near-identical sequences in that set. But we are providing all those data in case competitors can squeeze out more signal.

# 2. Python version

3.8

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
tqdm==4.67.1

# 4. Data file paths

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

# 5. Target score

0.55506

# 6. Current score

0.326

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'I fix the environment/runtime blockers without changing the model’s core architecture or training semantics: (1) remove a Keras/TF + protobuf crash by avoiding the legacy `keras` import mix and using `tf.keras` consistently, (2) update `ModelCheckpoint(period=...)` to the supported API, and (3) fix the missing BPP files by computing BPP matrices on the fly from `structure` (so the 4-input model still receives an `(107,107)` matrix). I also replace deprecated `DataFrame.append` with `pd.concat` and build the submission by following `sample_submission.csv` order to guarantee the exact 25680 rows/columns expected. These changes make the notebook run end-to-end and produce a valid `submission.csv` file.'
- What this solution (achieved 0.4041) has done: 'I fix the two runtime blockers that prevent the model from training and generating predictions: the protobuf/TF incompatibility error by forcing the Python protobuf implementation before importing TensorFlow, and the Keras 3 `ModelCheckpoint` filepath extension requirement by switching to a `.keras` filename (and loading from it). These changes are minimal and do not alter the model architecture, training loop, loss, or data processing, so the score behavior should remain consistent while producing a valid submission. I also keep paths and submission construction unchanged, only ensuring the checkpoint filename is consistent everywhere. The script run end-to-end and write `submission.csv`.'
- What this solution (achieved 0.63824) has done: 'We fix the TensorFlow/protobuf crash causing `MessageFactory.GetPrototype` errors by forcing a compatible protobuf runtime before importing TensorFlow (and falling back to the pure-Python implementation if needed). We also make the current code actually train for a meaningful number of epochs by setting `TEST=False` (right now it is hard-overridden to `True`, which severely undertrains and explains the worse score), without changing the model architecture, loss, or data pipeline. Finally, we keep the Keras 3 checkpointing fix (`.keras` filepath) and ensure the submission is written as `submission.csv` with the correct row order and columns as per `sample_submission.csv`.'
- What this solution (achieved 0.326) has done: 'I fix the TensorFlow import crash by forcing protobuf to use the pure-Python implementation *before* TensorFlow is imported (the current `"cpp"` setting is what triggers the missing `_message` error in this environment). This allow cell 3 to execute so `BiGRUModel` is defined, which in turn resolves the downstream `NameError: m is not defined` failures. I also keep the existing model/training logic intact, but add a couple of small stability guards: explicitly set random seeds for reproducibility and make sure predictions are generated with the correct method signature during test inference. These changes are runtime/stability fixes and should let the script run end-to-end and produce a valid `submission.csv`, with score behavior consistent with the current approach (and likely improved vs a run that previously failed to train at all).'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import random
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)



## === cell 1
from datetime import datetime


def log_now():
    print(datetime.now())




## === cell 2
TEST = False



## === cell 3
import tensorflow as tf

tf.random.set_seed(SEED)

from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input,
    Dense,
    Bidirectional,
    Conv1D,
    SpatialDropout1D,
    Embedding,
    Concatenate,
    GRU,
    LSTM,
    AveragePooling1D,
    ZeroPadding1D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
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

        self.ckpt_path = self.type + ".keras"
        self.checkpoint = ModelCheckpoint(
            filepath=self.ckpt_path,
            monitor="val_loss",
            verbose=self.train_verbose,
            save_best_only=True,
            mode="min",
        )
        self.es = EarlyStopping(
            monitor="val_loss", patience=10, mode="min", restore_best_weights=True
        )
        self.reduce_lr = ReduceLROnPlateau(patience=5)

    def create_model(self):
        seq_len = 107
        seq_dim = 14
        ltype_dim = 14
        structure_dim = 14
        embed_dim = 200
        dropout = 0.2
        sp_dropout = dropout
        conv_dim = 512
        conv_ksize = 3
        hidden_dim = 256
        out_dim = 5

        iseq = Input(shape=(seq_len,))
        iltype = Input(shape=(seq_len,))
        istructure = Input(shape=(seq_len,))
        ibpp = Input(shape=(seq_len, seq_len))

        eseq = Embedding(input_dim=seq_dim, output_dim=embed_dim)(iseq)
        eltype = Embedding(input_dim=ltype_dim, output_dim=embed_dim)(iltype)
        estructure = Embedding(input_dim=structure_dim, output_dim=embed_dim)(
            istructure
        )

        x = Concatenate(axis=2)([eseq, eltype, estructure, ibpp])
        x = ZeroPadding1D(padding=(0, 29))(x)

        x = SpatialDropout1D(sp_dropout)(x)
        x = Conv1D(
            conv_dim, conv_ksize, padding="same", activation=tf.keras.activations.swish
        )(x)

        x = Bidirectional(GRU(hidden_dim, dropout=dropout, return_sequences=True))(x)
        x = Bidirectional(GRU(hidden_dim, dropout=dropout, return_sequences=True))(x)
        x = Bidirectional(LSTM(hidden_dim, dropout=dropout, return_sequences=True))(x)

        x = AveragePooling1D(pool_size=2)(x)

        out = Dense(out_dim, activation="linear")(x)

        self.model = Model(inputs=[iseq, iltype, istructure, ibpp], outputs=out)

    def compile_model(self):
        opt = Adam(learning_rate=self.lr)
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
        ) = train_test_split(
            X_seq, X_ltype, X_structure, X_bpp, Y, test_size=0.1, random_state=SEED
        )

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
        return self.model.predict([X_seq, X_ltype, X_structure, X_bpp], verbose=0)

    def load_weights(self):
        self.model.load_weights(self.ckpt_path)

    def plot(self):
        plt.figure(figsize=(20, 5))
        plt.plot(self.history.history["loss"])
        plt.plot(self.history.history["val_loss"])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "valid"], loc="upper left")
        plt.show()




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
train = pd.read_json("/kaggle/input/stanford-covid-vaccine/train.json", lines=True)
test = pd.read_json("/kaggle/input/stanford-covid-vaccine/test.json", lines=True)
sample_sub = pd.read_csv("/kaggle/input/stanford-covid-vaccine/sample_submission.csv")



## === cell 5
if TEST:
    train = train.head(32 * 40)

tlen = train.shape[0]



## === cell 6
from sklearn import preprocessing

E = preprocessing.LabelEncoder()
E.fit(
    [b"S", b"M", b"I", b"B", b"H", b"E", b"X", b".", b"(", b")", b"A", b"C", b"G", b"U"]
)


def encode(code):
    return E.transform(np.array(code, "c"))


def _paired_index_from_dotbracket(structure_str):
    stack = []
    pair = {}
    for i, ch in enumerate(structure_str):
        if ch == "(":
            stack.append(i)
        elif ch == ")":
            if stack:
                j = stack.pop()
                pair[i] = j
                pair[j] = i
    return pair


def get_bpp_from_structure(structure_str, seq_len=107):
    pair = _paired_index_from_dotbracket(structure_str[:seq_len])
    bpp = np.zeros((seq_len, seq_len), dtype=np.float32)
    for i, j in pair.items():
        bpp[i, j] = 1.0
    return bpp


def generate_X_data(df):
    x_seq = np.empty((0, 107), dtype=np.int32)
    x_ltype = np.empty((0, 107), dtype=np.int32)
    x_structure = np.empty((0, 107), dtype=np.int32)
    x_bpp = np.empty((0, 107, 107), dtype=np.float32)

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        x_seq = np.append(x_seq, [encode(row.sequence)], axis=0)
        x_ltype = np.append(x_ltype, [encode(row.predicted_loop_type)], axis=0)
        x_structure = np.append(x_structure, [encode(row.structure)], axis=0)
        x_bpp = np.append(x_bpp, [get_bpp_from_structure(row.structure)], axis=0)

    return x_seq, x_ltype, x_structure, x_bpp




## === cell 7
def generate_Y_data(df):
    Y = np.empty((0, 68, 5), dtype=np.float32)
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        Y = np.append(
            Y,
            [
                np.array(
                    [
                        row.reactivity,
                        row.deg_Mg_pH10,
                        row.deg_pH10,
                        row.deg_Mg_50C,
                        row.deg_50C,
                    ],
                    dtype=np.float32,
                ).T
            ],
            axis=0,
        )
    return Y




## === cell 8
log_now()



## === cell 9
X_seq, X_ltype, X_structure, X_bpp = generate_X_data(train)

assert X_seq.shape == (tlen, 107)
assert X_ltype.shape == (tlen, 107)
assert X_structure.shape == (tlen, 107)
assert X_bpp.shape == (tlen, 107, 107)



## === cell 10
Y = generate_Y_data(train)
assert Y.shape == (tlen, 68, 5)



## === cell 11
assert (
    X_seq.shape[0]
    == X_ltype.shape[0]
    == X_structure.shape[0]
    == X_bpp.shape[0]
    == Y.shape[0]
)



## === cell 12
log_now()



## === cell 13
m = BiGRUModel(TEST)
m.create_and_compile()



## === cell 14
m.fit(X_seq, X_ltype, X_structure, X_bpp, Y)
m.plot()



## === cell 15
log_now()



## === cell 16
m.load_weights()



## === cell 17
Y_pred = m.predict(X_seq, X_ltype, X_structure, X_bpp)
print("MCRMSE  = ", float(np.mean(MCRMSE(Y, Y_pred))))



## === cell 18
log_now()



## === cell 19
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
sub = sample_sub.copy()

pred_map = {}
for _, row in tqdm(test.iterrows(), total=test.shape[0]):
    Xs = np.array([encode(row.sequence)], dtype=np.int32)[:, :107]
    Xl = np.array([encode(row.predicted_loop_type)], dtype=np.int32)[:, :107]
    Xst = np.array([encode(row.structure)], dtype=np.int32)[:, :107]
    Xb = np.array([get_bpp_from_structure(row.structure)], dtype=np.float32)[
        :, :107, :107
    ]

    predicted = m.predict(Xs, Xl, Xst, Xb)[0]

    if predicted.shape[0] < row.seq_length:
        pad = np.zeros(
            (row.seq_length - predicted.shape[0], predicted.shape[1]), dtype=np.float32
        )
        predicted_full = np.vstack([predicted, pad])
    else:
        predicted_full = predicted[: row.seq_length]

    for pos in range(row.seq_length):
        pred_map[f"{row.id}_{pos}"] = predicted_full[pos]

pred_arr = np.vstack([pred_map[k] for k in sub["id_seqpos"].values]).astype(np.float32)
sub.loc[:, target_cols] = pred_arr

assert sub.shape[0] == 25680
assert list(sub.columns) == ["id_seqpos"] + target_cols



## === cell 20
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)



## === cell 21
log_now()
