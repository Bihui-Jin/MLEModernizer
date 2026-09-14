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

0.69003

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


## === cell 1
train = pd.read_json("../input/stanford-covid-vaccine/train.json",lines=True)
test = pd.read_json("../input/stanford-covid-vaccine/test.json",lines=True)
ss = pd.read_csv("../input/stanford-covid-vaccine/sample_submission.csv")

train = train.set_index('index')
test = test.set_index('index')


## === cell 2
bpps_dir = '../input/stanford-covid-vaccine/bpps/'
bpps_list = os.listdir(bpps_dir)
bpps_npy = np.load(f'../input/stanford-covid-vaccine/bpps/{bpps_list[25]}')
print('Count of npy files: ', len(bpps_list))
print('Size of image: ', bpps_npy.shape)


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2667085122.py in <cell line: 0>()
      1 # read npy data file
      2 bpps_dir = '../input/stanford-covid-vaccine/bpps/'
----> 3 bpps_list = os.listdir(bpps_dir)
      4 bpps_npy = np.load(f'../input/stanford-covid-vaccine/bpps/{bpps_list[25]}')
      5 print('Count of npy files: ', len(bpps_list))

FileNotFoundError: [Errno 2] No such file or directory: '../input/stanford-covid-vaccine/bpps/'

## === cell 3
targets = ['reactivity','deg_Mg_pH10','deg_Mg_50C','deg_pH10','deg_50C']


## === cell 4
train = train[['id']+targets ]


## === cell 5
train['reactivity'] = train['reactivity'].apply(lambda x: np.mean(x))
train['deg_Mg_pH10'] = train['deg_Mg_pH10'].apply(lambda x: np.mean(x))
train['deg_Mg_50C'] = train['deg_Mg_50C'].apply(lambda x: np.mean(x))
train['deg_pH10'] = train['deg_pH10'].apply(lambda x: np.mean(x))
train['deg_50C'] = train['deg_50C'].apply(lambda x: np.mean(x))


## === cell 6
train


## === cell 7
train_data_ids = train['id'].values


## === cell 8
train_img = []
for ID in train_data_ids:
    img_path = os.path.join(bpps_dir,ID+'.npy')
    img = np.load(img_path)
    train_img.append(img)


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1744473713.py in <cell line: 0>()
      2 for ID in train_data_ids:
      3     img_path = os.path.join(bpps_dir,ID+'.npy')
----> 4     img = np.load(img_path)
      5     train_img.append(img)

/usr/local/lib/python3.11/dist-packages/numpy/lib/npyio.py in load(file, mmap_mode, allow_pickle, fix_imports, encoding, max_header_size)
    425             own_fid = False
    426         else:
--> 427             fid = stack.enter_context(open(os_fspath(file), "rb"))
    428             own_fid = True
    429 

FileNotFoundError: [Errno 2] No such file or directory: '../input/stanford-covid-vaccine/bpps/id_001f94081.npy'

## === cell 9
y = train[targets].values


## === cell 10
train_img = np.array(train_img).reshape(-1, 107, 107, 1)


## === cell 11
X_train, X_val, y_train, y_val = train_test_split(train_img, y, test_size=0.1, random_state=32)


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2214000399.py in <cell line: 0>()
----> 1 X_train, X_val, y_train, y_val = train_test_split(train_img, y, test_size=0.1, random_state=32)

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in train_test_split(test_size, train_size, random_state, shuffle, stratify, *arrays)
   2557         raise ValueError("At least one array required as input")
   2558 
-> 2559     arrays = indexable(*arrays)
   2560 
   2561     n_samples = _num_samples(arrays[0])

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in indexable(*iterables)
    441 
    442     result = [_make_indexable(X) for X in iterables]
--> 443     check_consistent_length(*result)
    444     return result
    445 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_consistent_length(*arrays)
    395     uniques = np.unique(lengths)
    396     if len(uniques) > 1:
--> 397         raise ValueError(
    398             "Found input variables with inconsistent numbers of samples: %r"
    399             % [int(l) for l in lengths]

ValueError: Found input variables with inconsistent numbers of samples: [0, 2160]

## === cell 12
from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
import tensorflow as tf


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 13
model = Sequential()

model.add(Conv2D(64,(3,3), padding='same', input_shape=(107, 107,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(5, activation='linear'))

opt = Adam(lr=0.005)
model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
model.summary()


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4250273619.py in <cell line: 0>()
     20 model.add(Dense(5, activation='linear'))
     21 
---> 22 opt = Adam(lr=0.005)
     23 model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
     24 model.summary()

/usr/local/lib/python3.11/dist-packages/keras/src/optimizers/adam.py in __init__(self, learning_rate, beta_1, beta_2, epsilon, amsgrad, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, **kwargs)
     60         **kwargs,
     61     ):
---> 62         super().__init__(
     63             learning_rate=learning_rate,
     64             name=name,

/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/optimizer.py in __init__(self, *args, **kwargs)
     19 class TFOptimizer(KerasAutoTrackable, base_optimizer.BaseOptimizer):
     20     def __init__(self, *args, **kwargs):
---> 21         super().__init__(*args, **kwargs)
     22         self._distribution_strategy = tf.distribute.get_strategy()
     23 

/usr/local/lib/python3.11/dist-packages/keras/src/optimizers/base_optimizer.py in __init__(self, learning_rate, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, **kwargs)
     88             )
     89         if kwargs:
---> 90             raise ValueError(f"Argument(s) not recognized: {kwargs}")
     91 
     92         if name is None:

ValueError: Argument(s) not recognized: {'lr': 0.005}

## === cell 14
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=2, min_lr=0.00001, mode='auto')

callbacks = [reduce_lr]

history = model.fit(
    x=X_train,
    y=y_train,
    epochs=50,
    validation_data = (X_val,y_val),
    callbacks=callbacks
)


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3254421775.py in <cell line: 0>()
      5 
      6 history = model.fit(
----> 7     x=X_train,
      8     y=y_train,
      9     epochs=50,

NameError: name 'X_train' is not defined

## === cell 15
plt.figure(figsize=(15,7))
ax1 = plt.subplot(1,2,1)
ax1.plot(history.history['loss'], color='b', label='Training Loss') 
ax1.plot(history.history['val_loss'], color='r', label = 'Validation Loss',axes=ax1)
legend = ax1.legend(loc='best', shadow=True)
ax2 = plt.subplot(1,2,2)
ax2.plot(history.history['accuracy'], color='b', label='Training Accuracy') 
ax2.plot(history.history['val_accuracy'], color='r', label = 'Validation Accuracy')
legend = ax2.legend(loc='best', shadow=True)


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3871689184.py in <cell line: 0>()
      1 plt.figure(figsize=(15,7))
      2 ax1 = plt.subplot(1,2,1)
----> 3 ax1.plot(history.history['loss'], color='b', label='Training Loss')
      4 ax1.plot(history.history['val_loss'], color='r', label = 'Validation Loss',axes=ax1)
      5 legend = ax1.legend(loc='best', shadow=True)

NameError: name 'history' is not defined

## === cell 16
test


## === cell 17
test_public = test[test.seq_length == 107]
test_private = test[test.seq_length == 130]


## === cell 18
test_public_ids = test_public['id'].values
test_private_ids = test_private['id'].values


## === cell 19
test_public_img = []
for ID in test_public_ids:
    img_path = os.path.join(bpps_dir,ID+'.npy')
    img = np.load(img_path)
    test_public_img.append(img)

test_private_img = []
for ID in test_private_ids:
    img_path = os.path.join(bpps_dir,ID+'.npy')
    img = np.load(img_path)
    test_private_img.append(img)
    


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3346118310.py in <cell line: 0>()
      2 for ID in test_public_ids:
      3     img_path = os.path.join(bpps_dir,ID+'.npy')
----> 4     img = np.load(img_path)
      5     test_public_img.append(img)
      6 

/usr/local/lib/python3.11/dist-packages/numpy/lib/npyio.py in load(file, mmap_mode, allow_pickle, fix_imports, encoding, max_header_size)
    425             own_fid = False
    426         else:
--> 427             fid = stack.enter_context(open(os_fspath(file), "rb"))
    428             own_fid = True
    429 

FileNotFoundError: [Errno 2] No such file or directory: '../input/stanford-covid-vaccine/bpps/id_00b436dec.npy'

## === cell 20
plt.imshow(test_private_img[0])


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2505979730.py in <cell line: 0>()
----> 1 plt.imshow(test_private_img[0])

NameError: name 'test_private_img' is not defined

## === cell 21
test_public_img = np.array(test_public_img).reshape(-1, 107, 107, 1)
test_private_img = np.array(test_private_img).reshape(-1, 130, 130, 1)


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4158386314.py in <cell line: 0>()
      1 test_public_img = np.array(test_public_img).reshape(-1, 107, 107, 1)
----> 2 test_private_img = np.array(test_private_img).reshape(-1, 130, 130, 1)

NameError: name 'test_private_img' is not defined

## === cell 22
pred_public = model.predict(test_public_img)


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3184079940.py in <cell line: 0>()
----> 1 pred_public = model.predict(test_public_img)
      2 #pred_private = model.predict(test_private_img)

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/progbar.py in update(self, current, values, finalize)
    117 
    118             if self.target is not None:
--> 119                 numdigits = int(math.log10(self.target)) + 1
    120                 bar = ("%" + str(numdigits) + "d/%d") % (current, self.target)
    121                 bar = f"\x1b[1m{bar}\x1b[0m "

ValueError: math domain error

## === cell 23
len(test_private_img)


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2532052815.py in <cell line: 0>()
----> 1 len(test_private_img)

NameError: name 'test_private_img' is not defined

## === cell 24
pred_private.shape


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/915736536.py in <cell line: 0>()
----> 1 pred_private.shape

NameError: name 'pred_private' is not defined

## === cell 25
pred_public = np.repeat(pred_public,repeats=107,axis=0)


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2642368635.py in <cell line: 0>()
----> 1 pred_public = np.repeat(pred_public,repeats=107,axis=0)

NameError: name 'pred_public' is not defined

## === cell 26
pred_private = np.repeat(np.array([0,0,0,0,0]),repeats=130*3005,axis=0).reshape(-1,5)


## === cell 27
prediction = np.concatenate((pred_public,pred_private),axis=0)


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4051332543.py in <cell line: 0>()
----> 1 prediction = np.concatenate((pred_public,pred_private),axis=0)

NameError: name 'pred_public' is not defined

## === cell 28
sub = pd.DataFrame(prediction)


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1767818850.py in <cell line: 0>()
----> 1 sub = pd.DataFrame(prediction)

NameError: name 'prediction' is not defined

## === cell 29
seqpos = ss.id_seqpos.values


## === cell 30
sub['id_seqpos'] = seqpos


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/496078287.py in <cell line: 0>()
----> 1 sub['id_seqpos'] = seqpos

NameError: name 'sub' is not defined

## === cell 31
sub


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1231690419.py in <cell line: 0>()
----> 1 sub

NameError: name 'sub' is not defined

## === cell 32
sub = sub.rename(columns={0: "reactivity", 1: "deg_Mg_pH10",2: "deg_Mg_50C", 3: "deg_pH10", 4: "deg_50C"})


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4033185068.py in <cell line: 0>()
----> 1 sub = sub.rename(columns={0: "reactivity", 1: "deg_Mg_pH10",2: "deg_Mg_50C", 3: "deg_pH10", 4: "deg_50C"})

NameError: name 'sub' is not defined

## === cell 33
sub.to_csv("submission.csv",index=False)


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/776095622.py in <cell line: 0>()
----> 1 sub.to_csv("submission.csv",index=False)

NameError: name 'sub' is not defined
