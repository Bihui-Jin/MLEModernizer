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

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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
import tensorflow as tf

import keras.backend as K

from keras.layers import Input, Dense, Bidirectional, Conv1D, SpatialDropout1D, Embedding, Concatenate, GRU, Cropping1D, LSTM, AveragePooling1D, ZeroPadding1D
from tensorflow.keras.activations import swish
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)
    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)

class BiGRUModel():
    
    def __init__(self, testmode):
        self.type = 'BiGru'
        self.n_features = 3
        
        self.lr = 0.0015
        self.epochs = 5 if testmode else 100
        self.batch_size = 32
        
        self.train_verbose = 1 if testmode else 0

        self.checkpoint = ModelCheckpoint(self.type + ".hdf5", 
                                          monitor='val_loss',
                                          verbose=self.train_verbose,
                                          save_best_only=True,
                                          mode='auto',
                                          period=1)
        self.es = EarlyStopping(monitor='val_loss', patience = 10, mode = 'min', restore_best_weights=True)
        self.reduce_lr = ReduceLROnPlateau(patience=5)

    def create_model(self):
        seq_len = 107
        pred_len = 68
        seq_dim = 14
        ltype_dim = 14
        structure_dim = 14
        embed_dim = 200
        dropout = .2
        sp_dropout = dropout
        conv_dim = 512
        conv_ksize = 3
        hidden_dim = 256
        crop = (0,seq_len-pred_len)
        out_dim = 5
        
        iseq = Input(shape = (seq_len))
        iltype = Input(shape = (seq_len))
        istructure = Input(shape = (seq_len))
        ibpp = Input(shape= (seq_len,seq_len))
        
        eseq = Embedding(input_dim=seq_dim,output_dim=embed_dim)(iseq)
        eltype = Embedding(input_dim=ltype_dim,output_dim=embed_dim)(iltype)
        estructure = Embedding(input_dim=structure_dim,output_dim=embed_dim)(istructure)
        
        x = Concatenate(axis=2)([eseq,eltype,estructure,ibpp])
        x = ZeroPadding1D(padding=(0,29))(x)

        x = SpatialDropout1D(sp_dropout)(x)
        x = Conv1D(conv_dim, conv_ksize, padding='same', activation=swish)(x)
        
        x = Bidirectional(GRU(hidden_dim, dropout=dropout, return_sequences=True))(x)        
        x = Bidirectional(GRU(hidden_dim, dropout=dropout, return_sequences=True))(x)        
        x = Bidirectional(LSTM(hidden_dim, dropout=dropout, return_sequences=True))(x)        
        
        x = AveragePooling1D(pool_size=2)(x)

        
        out = Dense(out_dim, activation='linear')(x)

        self.model = Model(inputs = [iseq, iltype, istructure, ibpp], outputs = out)
    
    def compile_model(self):
        opt = Adam(lr = self.lr)
        self.model.compile(loss = MCRMSE, optimizer = opt)
    
    def create_and_compile(self):
        if self.train_verbose == 1:
            print('Create Model...')
        self.create_model()
        
        if self.train_verbose == 1:
            print('Compile Model...')
        self.compile_model()
        
        if self.train_verbose == 1:
            self.print()


    def print(self):
        print(self.model.summary())
        
    def fit(self, X_seq, X_ltype, X_structure, X_bpp, Y):
        X_seq_train, X_seq_valid, X_ltype_train, X_ltype_valid, X_structure_train, X_structure_valid, X_bpp_train, X_bpp_valid, Y_train, Y_valid = train_test_split(X_seq, X_ltype, X_structure, X_bpp, Y)

        self.history = self.model.fit([X_seq_train, X_ltype_train, X_structure_train, X_bpp_train],
                                      Y_train,
                                      validation_data = ([X_seq_valid, X_ltype_valid, X_structure_valid, X_bpp_valid], Y_valid),
                                      epochs = self.epochs,
                                      batch_size = self.batch_size,
                                      callbacks = [self.checkpoint, self.es, self.reduce_lr],
                                      verbose = self.train_verbose)

    def predict(self, X_seq, X_ltype, X_structure, X_bpp):
        return self.model.predict([X_seq, X_ltype, X_structure, X_bpp])
    
    def load_weights(self):
        self.model.load_weights(self.type + ".hdf5")
    
    def plot(self):
        plt.figure(figsize=(20,5))

        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        plt.show()


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
X_seq, X_ltype, X_structure, X_bpp = generate_X_data(train)

assert X_seq.shape == (tlen,107)
assert X_ltype.shape == (tlen,107)
assert X_structure.shape == (tlen,107)
assert X_bpp.shape == (tlen,107,107)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2290529082.py in <cell line: 0>()
----> 1 X_seq, X_ltype, X_structure, X_bpp = generate_X_data(train)
      2 
      3 assert X_seq.shape == (tlen,107)
      4 assert X_ltype.shape == (tlen,107)
      5 assert X_structure.shape == (tlen,107)

/tmp/ipykernel_11/2679581328.py in generate_X_data(train)
     16         x_ltype = np.append(x_ltype, [encode(row.predicted_loop_type)], axis=0)
     17         x_structure = np.append(x_structure, [encode(row.structure)], axis=0)
---> 18         x_bpp = np.append(x_bpp, [get_bpp(row.id)], axis=0)
     19 
     20     return x_seq, x_ltype, x_structure, x_bpp

/tmp/ipykernel_11/708644247.py in get_bpp(id)
      1 def get_bpp(id):
----> 2     return np.load('../input/stanford-covid-vaccine/bpps/' + id + '.npy')

/usr/local/lib/python3.11/dist-packages/numpy/lib/npyio.py in load(file, mmap_mode, allow_pickle, fix_imports, encoding, max_header_size)
    425             own_fid = False
    426         else:
--> 427             fid = stack.enter_context(open(os_fspath(file), "rb"))
    428             own_fid = True
    429 

FileNotFoundError: [Errno 2] No such file or directory: '../input/stanford-covid-vaccine/bpps/id_001f94081.npy'

## === cell 11
Y = generate_Y_data(train)
assert Y.shape == (tlen,68,5)


## === cell 12
assert X_seq.shape[0] == X_ltype.shape[0] == X_structure.shape[0] == X_bpp.shape[0] == Y.shape[0]


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/10387614.py in <cell line: 0>()
----> 1 assert X_seq.shape[0] == X_ltype.shape[0] == X_structure.shape[0] == X_bpp.shape[0] == Y.shape[0]

NameError: name 'X_seq' is not defined

## === cell 13
log_now()


## === cell 14
m = BiGRUModel(TEST)
m.create_and_compile()


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1893344134.py in <cell line: 0>()
----> 1 m = BiGRUModel(TEST)
      2 m.create_and_compile()

/tmp/ipykernel_11/1484548562.py in __init__(self, testmode)
     29         self.train_verbose = 1 if testmode else 0
     30 
---> 31         self.checkpoint = ModelCheckpoint(self.type + ".hdf5", 
     32                                           monitor='val_loss',
     33                                           verbose=self.train_verbose,

TypeError: ModelCheckpoint.__init__() got an unexpected keyword argument 'period'

## === cell 15
m.fit(X_seq, X_ltype, X_structure, X_bpp, Y)
m.plot()


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/113190871.py in <cell line: 0>()
----> 1 m.fit(X_seq, X_ltype, X_structure, X_bpp, Y)
      2 m.plot()

NameError: name 'm' is not defined

## === cell 16
log_now()


## === cell 17
m.load_weights()


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3804503826.py in <cell line: 0>()
----> 1 m.load_weights()

NameError: name 'm' is not defined

## === cell 18
Y_pred = m.predict(X_seq, X_ltype, X_structure, X_bpp)
print("MCRMSE  = ", np.mean(MCRMSE(Y,Y_pred)))


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3930237541.py in <cell line: 0>()
----> 1 Y_pred = m.predict(X_seq, X_ltype, X_structure, X_bpp)
      2 print("MCRMSE  = ", np.mean(MCRMSE(Y,Y_pred)))

NameError: name 'm' is not defined

## === cell 19
log_now()


## === cell 20
submission = pd.DataFrame({
                "id_seqpos" :  [],
                "reactivity":  [],
                "deg_Mg_pH10": [],
                "deg_pH10":    [],
                "deg_Mg_50C":  [],
                "deg_50C":     []
            })


## === cell 21
for index, row in tqdm(test.iterrows(), total=test.shape[0]):
    X_seq = np.array([encode(row.sequence)])[:,:107]
    X_ltype = np.array([encode(row.predicted_loop_type)])[:,:107]
    X_structure = np.array([encode(row.structure)])[:,:107]
    X_bpp = np.array([get_bpp(row.id)])[:,:107,:107]

    predicted = m.predict(X_seq, X_ltype, X_structure, X_bpp)

    df = pd.DataFrame(data=predicted[0], columns=["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"])
    dfz = pd.DataFrame(data=np.zeros((row.seq_length-df.shape[0],5)), columns=["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"])
    df = df.append(dfz) # complete with zeros
    df = df.reset_index(drop=True)
    df["id_seqpos"] = df.apply(lambda r : f"{row.id}_{r.name}", axis = 1) 

    submission = submission.append(df)


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/607122979.py in <cell line: 0>()
      3     X_ltype = np.array([encode(row.predicted_loop_type)])[:,:107]
      4     X_structure = np.array([encode(row.structure)])[:,:107]
----> 5     X_bpp = np.array([get_bpp(row.id)])[:,:107,:107]
      6 
      7     predicted = m.predict(X_seq, X_ltype, X_structure, X_bpp)

/tmp/ipykernel_11/708644247.py in get_bpp(id)
      1 def get_bpp(id):
----> 2     return np.load('../input/stanford-covid-vaccine/bpps/' + id + '.npy')

/usr/local/lib/python3.11/dist-packages/numpy/lib/npyio.py in load(file, mmap_mode, allow_pickle, fix_imports, encoding, max_header_size)
    425             own_fid = False
    426         else:
--> 427             fid = stack.enter_context(open(os_fspath(file), "rb"))
    428             own_fid = True
    429 

FileNotFoundError: [Errno 2] No such file or directory: '../input/stanford-covid-vaccine/bpps/id_00b436dec.npy'

## === cell 22
assert submission.shape[0] == 457953


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/675734886.py in <cell line: 0>()
----> 1 assert submission.shape[0] == 457953

AssertionError: 

## === cell 23
submission.to_csv("submission.csv", index = False)


## === cell 24
log_now()


## --- ERROR in outputing the csv:
Invalid submission: Expected submission to be the same length as answers, but got 0 instead of 25680.
