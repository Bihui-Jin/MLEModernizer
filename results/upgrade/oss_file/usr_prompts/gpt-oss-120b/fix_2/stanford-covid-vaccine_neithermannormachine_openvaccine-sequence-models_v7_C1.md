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

0.41112

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow.keras.layers as layers



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]




## === cell 2
def read_json(filename):
    """Read train/test json data as pandas DataFrame."""
    with open(filename, "r") as f:
        df = pd.read_json(path_or_buf=f, orient="records", lines=True)
    return df




## === cell 3
train_df = read_json("../input/stanford-covid-vaccine/train.json")
print("train samples:", train_df.shape[0])



## === cell 4
test_df = read_json("../input/stanford-covid-vaccine/test.json")
print("test samples:", test_df.shape[0])



## === cell 6
def unpack_df_lists(df, col_names):
    """Explode list‑like columns into long format."""
    if isinstance(col_names, str):
        col_names = [col_names]
    all_series = [df[c] for c in col_names]
    unpacked = [ser.explode() for ser in all_series]
    data = pd.concat(unpacked, axis=1)
    original = df.drop(col_names, axis=1)
    return original.join(data)




## === cell 8
char_set = set(
    "".join(train_df["sequence"].astype(str))
    + "".join(train_df["structure"].astype(str))
    + "".join(train_df["predicted_loop_type"].astype(str))
)
char2idx = {
    ch: i + 1 for i, ch in enumerate(sorted(char_set))
}  # 0 reserved for unknown


def tokenize_series(series):
    return series.apply(lambda s: [char2idx.get(ch, 0) for ch in s])




## === cell 9
train_df = train_df[train_df["SN_filter"] == 1].reset_index(drop=True)

train_df["sequence"] = tokenize_series(train_df["sequence"])
train_df["structure"] = tokenize_series(train_df["structure"])
train_df["predicted_loop_type"] = tokenize_series(train_df["predicted_loop_type"])



## === cell 10
corr_data = train_df.drop(
    [
        "index",
        "id",
        "sequence",
        "structure",
        "predicted_loop_type",
        "seq_length",
        "seq_scored",
    ],
    axis=1,
)
corr_data = unpack_df_lists(
    corr_data,
    [
        "reactivity_error",
        "deg_error_Mg_pH10",
        "deg_error_pH10",
        "deg_error_Mg_50C",
        "deg_error_50C",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ],
).convert_dtypes()
corr_matrix = corr_data.corr()




## === cell 11
def score(raw_values=False, use_tf=False, **kwargs):
    col_dict = {"reactivity": 0, "deg_Mg_pH10": 1, "deg_Mg_50C": 3}
    unscored = set(range(5)) - set(col_dict.values())
    multi = "raw_values" if raw_values else "uniform_average"

    def loss(y_true, y_pred):
        from sklearn.metrics import mean_squared_error

        y_true = np.array(y_true)[:, list(col_dict.values())]
        y_pred = np.array(y_pred)[:, list(col_dict.values())]
        return mean_squared_error(y_true, y_pred, squared=False, multioutput=multi)

    def loss_tf(y_true, y_pred):
        import tensorflow.keras.backend as K

        colwise_mse = K.mean(K.square(y_true - y_pred), axis=1)
        return K.mean(K.sqrt(colwise_mse), axis=1)

    return loss_tf if use_tf else loss




## === cell 12
def build_input_array(df):
    seq = np.stack(df["sequence"].apply(lambda x: np.array(x, dtype=np.int32)).values)
    struct = np.stack(
        df["structure"].apply(lambda x: np.array(x, dtype=np.int32)).values
    )
    loop = np.stack(
        df["predicted_loop_type"].apply(lambda x: np.array(x, dtype=np.int32)).values
    )
    return np.stack([seq, struct, loop], axis=1)  # shape (n, 3, seq_len)


X_train = build_input_array(train_df)

y_train = (
    train_df[target_cols]
    .apply(lambda row: [e for e in row], axis=1)
    .apply(lambda e: np.array(e))
)
y_train = np.stack(y_train.values, axis=0)  # shape (n, 5, seq_scored)



## === cell 13
sw = train_df["signal_to_noise"].values
sw = np.log1p(sw + 5) / 2




## === cell 14
def make_model(vocab_size):
    EMBEDDING_PARAMS = {"input_dim": vocab_size, "output_dim": 100}
    inputs = tf.keras.Input(shape=(3, None), dtype=tf.int32)
    embed = layers.Embedding(**EMBEDDING_PARAMS)(inputs)

    def rnn_layer():
        return layers.Bidirectional(layers.LSTM(30, return_sequences=True))

    rnn_outputs = []
    for i in range(embed.shape[1]):  # iterate over the 3 channels
        r = rnn_layer()(embed[:, i])
        r = rnn_layer()(r)
        rnn_outputs.append(r)

    x = layers.Concatenate()(rnn_outputs)
    x = layers.Dense(100, activation="relu")(x)
    x = layers.Dense(5, activation="linear")(x)  # (batch, seq_len, 5)
    x = tf.transpose(x, perm=(0, 2, 1))  # (batch, 5, seq_len)
    x = x[:, :, :-39]  # keep first 68 positions (107‑39) or 91 (130‑39)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer="adam", loss=score(use_tf=True), metrics=["mse"])
    return model




## === cell 15
vocab_size = len(char2idx) + 1
model = make_model(vocab_size)
model.summary()



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3299079583.py in <cell line: 0>()
      1 vocab_size = len(char2idx) + 1
----> 2 model = make_model(vocab_size)
      3 model.summary()
      4 

/tmp/ipykernel_55/495306943.py in make_model(vocab_size)
     16     x = layers.Dense(100, activation="relu")(x)
     17     x = layers.Dense(5, activation="linear")(x)  # (batch, seq_len, 5)
---> 18     x = tf.transpose(x, perm=(0, 2, 1))  # (batch, 5, seq_len)
     19     x = x[:, :, :-39]  # keep first 68 positions (107‑39) or 91 (130‑39)
     20     model = tf.keras.Model(inputs=inputs, outputs=x)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/weak_tensor_ops.py in wrapper(*args, **kwargs)
     86   def wrapper(*args, **kwargs):
     87     if not ops.is_auto_dtype_conversion_enabled():
---> 88       return op(*args, **kwargs)
     89     bound_arguments = signature.bind(*args, **kwargs)
     90     bound_arguments.apply_defaults()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/keras_tensor.py in __tf_tensor__(self, dtype, name)
    136 
    137     def __tf_tensor__(self, dtype=None, name=None):
--> 138         raise ValueError(
    139             "A KerasTensor cannot be used as input to a TensorFlow function. "
    140             "A KerasTensor is a symbolic placeholder for a shape and dtype, "

ValueError: A KerasTensor cannot be used as input to a TensorFlow function. A KerasTensor is a symbolic placeholder for a shape and dtype, used when constructing Keras Functional models or Keras Functions. You can only use it as input to a Keras layer or a Keras operation (from the namespaces `keras.layers` and `keras.operations`). You are likely doing something like:

```
x = Input(...)
...
tf_fn(x)  # Invalid.
```

What you should do instead is wrap `tf_fn` in a layer:

```
class MyLayer(Layer):
    def call(self, x):
        return tf_fn(x)

x = MyLayer()(x)
```


## === cell 16
history = model.fit(
    X_train, y_train, epochs=100, batch_size=100, sample_weight=sw, verbose=0
)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1185077224.py in <cell line: 0>()
----> 1 history = model.fit(
      2     X_train, y_train, epochs=100, batch_size=100, sample_weight=sw, verbose=0
      3 )
      4 

NameError: name 'model' is not defined

## === cell 17
for metric, values in history.history.items():
    plt.plot(values, label=metric)
plt.legend()
plt.show()



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1450235935.py in <cell line: 0>()
      1 # Plot training loss (optional)
----> 2 for metric, values in history.history.items():
      3     plt.plot(values, label=metric)
      4 plt.legend()
      5 plt.show()

NameError: name 'history' is not defined

## === cell 18
test_public = test_df[test_df["seq_length"] == 107].reset_index(drop=True)
test_private = test_df[test_df["seq_length"] == 130].reset_index(drop=True)

test_public["sequence"] = tokenize_series(test_public["sequence"])
test_public["structure"] = tokenize_series(test_public["structure"])
test_public["predicted_loop_type"] = tokenize_series(test_public["predicted_loop_type"])

if not test_private.empty:
    test_private["sequence"] = tokenize_series(test_private["sequence"])
    test_private["structure"] = tokenize_series(test_private["structure"])
    test_private["predicted_loop_type"] = tokenize_series(
        test_private["predicted_loop_type"]
    )

X_test_public = build_input_array(test_public)

if not test_private.empty:
    X_test_private = build_input_array(test_private)



## === cell 19
test_pred_public = model.predict(X_test_public, verbose=0)

if not test_private.empty:
    test_pred_private = model.predict(X_test_private, verbose=0)




## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/779589169.py in <cell line: 0>()
      1 # Predict
----> 2 test_pred_public = model.predict(X_test_public, verbose=0)
      3 
      4 if not test_private.empty:
      5     test_pred_private = model.predict(X_test_private, verbose=0)

NameError: name 'model' is not defined

## === cell 20
def create_sub_df(test_df, predictions, length):
    sub_df = test_df.drop(
        ["sequence", "structure", "predicted_loop_type", "index", "seq_scored"], axis=1
    )
    sub_df["seqpos"] = sub_df.apply(lambda row: list(range(row["seq_length"])), axis=1)
    sub_df = unpack_df_lists(sub_df, "seqpos")
    sub_df["id_seqpos"] = sub_df.apply(
        lambda row: f"{row['id']}_{row['seqpos']}", axis=1
    )

    def pad_pred(p, target_len):
        start = list(p)
        padding = [0] * (target_len - len(start))
        return start + padding

    pred_df = pd.DataFrame(
        [[pad_pred(l, length) for l in e] for e in predictions], columns=target_cols
    )
    pred_df = unpack_df_lists(pred_df, target_cols)
    pred_df.index = sub_df.index
    pred_df = pred_df.reset_index(drop=True)

    sub_df = sub_df.reset_index(drop=True)
    sub_df = sub_df[["id_seqpos"]].join(pred_df)
    return sub_df




## === cell 21
sub_public = create_sub_df(test_public, test_pred_public, 107)

if not test_private.empty:
    sub_private = create_sub_df(test_private, test_pred_private, 130)
    sub_df = pd.concat([sub_public, sub_private], ignore_index=True)
else:
    sub_df = sub_public.copy()

sub_df.to_csv("submission.csv", index=False)
print("submission.csv written with shape:", sub_df.shape)

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/812808308.py in <cell line: 0>()
----> 1 sub_public = create_sub_df(test_public, test_pred_public, 107)
      2 
      3 if not test_private.empty:
      4     sub_private = create_sub_df(test_private, test_pred_private, 130)
      5     sub_df = pd.concat([sub_public, sub_private], ignore_index=True)

NameError: name 'test_pred_public' is not defined
