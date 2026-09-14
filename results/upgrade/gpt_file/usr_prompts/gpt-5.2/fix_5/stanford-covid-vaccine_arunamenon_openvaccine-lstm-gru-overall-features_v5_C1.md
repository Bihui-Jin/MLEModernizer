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

0.39813

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.25159) has done: 'I fix the environment-breaking import issue by ensuring TensorFlow/Protobuf are imported in a compatible order and by avoiding the problematic Keras 3/TensorFlow mix. Then I make BPPS feature loading robust by checking for the `/kaggle/input/.../bpps` directory and falling back to zero-filled features when those files aren’t present, so preprocessing always works. Next I fix the model build error by replacing the raw `tf.reshape` on a KerasTensor with a Keras `Reshape` layer (same semantics), and I also make the embedding accept integer categorical inputs explicitly. Finally, I ensure predictions are generated for all test rows and merged back into the sample submission with correct ordering/shape, writing a valid `submission.csv`.'
- What this solution (achieved 0.2521) has done: 'I fix the environment-breaking TensorFlow import error caused by an incompatibility between TensorFlow 2.18 and protobuf 6 by forcing the pure-Python protobuf implementation before importing TensorFlow. I also ensure we consistently use `tf.keras` (not standalone `keras`) for losses to avoid Keras 3 / tf.keras mixing issues that can cause subtle runtime problems. Finally, I keep the existing model/training/prediction logic unchanged, only making these minimal compatibility fixes so the notebook runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.24864) has done: 'I fix the TensorFlow import crash by pinning protobuf to the pure-Python implementation *and* using the compatible internal C++ implementation version before TensorFlow is imported (this resolves the `MessageFactory.GetPrototype` error with TF 2.18 + protobuf 6). I also ensure we consistently use `tf.keras` only (no standalone `keras`) to avoid Keras 3 / tf.keras mismatches. Everything else (data loading, preprocessing, model, training loop, inference, and submission formatting) be kept the same so the score behavior stays essentially unchanged while the notebook runs end-to-end and writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import json

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

try:
    import tensorflow as tf
except Exception as e:
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
    import tensorflow as tf

print("TF version:", tf.__version__)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/3974650291.py in <cell line: 0>()
      9 try:
---> 10     import tensorflow as tf
     11 except Exception as e:

/usr/local/lib/python3.11/dist-packages/tensorflow/__init__.py in <module>
     48 
---> 49 from tensorflow._api.v2 import __internal__
     50 from tensorflow._api.v2 import __operators__

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/__init__.py in <module>
      7 
----> 8 from tensorflow._api.v2.__internal__ import autograph
      9 from tensorflow._api.v2.__internal__ import decorator

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/autograph/__init__.py in <module>
      7 
----> 8 from tensorflow.python.autograph.core.ag_ctx import control_status_ctx # line: 34
      9 from tensorflow.python.autograph.impl.api import tf_convert # line: 493

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/ag_ctx.py in <module>
     20 
---> 21 from tensorflow.python.autograph.utils import ag_logging
     22 from tensorflow.python.util.tf_export import tf_export

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/__init__.py in <module>
     16 
---> 17 from tensorflow.python.autograph.utils.context_managers import control_dependency_on_returns
     18 from tensorflow.python.autograph.utils.misc import alias_tensors

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/context_managers.py in <module>
     18 
---> 19 from tensorflow.python.framework import ops
     20 from tensorflow.python.ops import tensor_array_ops

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in <module>
     32 from google.protobuf import message
---> 33 from tensorflow.core.framework import attr_value_pb2
     34 from tensorflow.core.framework import full_type_pb2

/usr/local/lib/python3.11/dist-packages/tensorflow/core/framework/attr_value_pb2.py in <module>
      4 """Generated protocol buffer code."""
----> 5 from google.protobuf.internal import builder as _builder
      6 from google.protobuf import descriptor as _descriptor

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/builder.py in <module>
     17 from google.protobuf.internal import enum_type_wrapper
---> 18 from google.protobuf.internal import python_message
     19 from google.protobuf import message as _message

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/python_message.py in <module>
     37 
---> 38 from google.protobuf import descriptor as descriptor_mod
     39 from google.protobuf import message as message_mod

/usr/local/lib/python3.11/dist-packages/google/protobuf/descriptor.py in <module>
     28   if _message is None:
---> 29     from google.protobuf.pyext import _message
     30   _USE_C_DESCRIPTORS = True

ImportError: cannot import name '_message' from 'google.protobuf.pyext' (/usr/local/lib/python3.11/dist-packages/google/protobuf/pyext/__init__.py)

During handling of the above exception, another exception occurred:

ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/3974650291.py in <cell line: 0>()
     13     os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
     14     os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
---> 15     import tensorflow as tf
     16 
     17 print("TF version:", tf.__version__)

/usr/local/lib/python3.11/dist-packages/tensorflow/__init__.py in <module>
     47 _tf2.enable()
     48 
---> 49 from tensorflow._api.v2 import __internal__
     50 from tensorflow._api.v2 import __operators__
     51 from tensorflow._api.v2 import audio

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/__init__.py in <module>
      6 import sys as _sys
      7 
----> 8 from tensorflow._api.v2.__internal__ import autograph
      9 from tensorflow._api.v2.__internal__ import decorator
     10 from tensorflow._api.v2.__internal__ import dispatch

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/autograph/__init__.py in <module>
      6 import sys as _sys
      7 
----> 8 from tensorflow.python.autograph.core.ag_ctx import control_status_ctx # line: 34
      9 from tensorflow.python.autograph.impl.api import tf_convert # line: 493

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/ag_ctx.py in <module>
     19 import threading
     20 
---> 21 from tensorflow.python.autograph.utils import ag_logging
     22 from tensorflow.python.util.tf_export import tf_export
     23 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/__init__.py in <module>
     15 """Utility module that contains APIs usable in the generated code."""
     16 
---> 17 from tensorflow.python.autograph.utils.context_managers import control_dependency_on_returns
     18 from tensorflow.python.autograph.utils.misc import alias_tensors
     19 from tensorflow.python.autograph.utils.tensor_list import dynamic_list_append

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/context_managers.py in <module>
     17 import contextlib
     18 
---> 19 from tensorflow.python.framework import ops
     20 from tensorflow.python.ops import tensor_array_ops
     21 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in <module>
     31 
     32 from google.protobuf import message
---> 33 from tensorflow.core.framework import attr_value_pb2
     34 from tensorflow.core.framework import full_type_pb2
     35 from tensorflow.core.framework import function_pb2

/usr/local/lib/python3.11/dist-packages/tensorflow/core/framework/attr_value_pb2.py in <module>
      3 # source: tensorflow/core/framework/attr_value.proto
      4 """Generated protocol buffer code."""
----> 5 from google.protobuf.internal import builder as _builder
      6 from google.protobuf import descriptor as _descriptor
      7 from google.protobuf import descriptor_pool as _descriptor_pool

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/builder.py in <module>
     16 
     17 from google.protobuf.internal import enum_type_wrapper
---> 18 from google.protobuf.internal import python_message
     19 from google.protobuf import message as _message
     20 from google.protobuf import reflection as _reflection

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/python_message.py in <module>
     36 import weakref
     37 
---> 38 from google.protobuf import descriptor as descriptor_mod
     39 from google.protobuf import message as message_mod
     40 from google.protobuf import text_format

/usr/local/lib/python3.11/dist-packages/google/protobuf/descriptor.py in <module>
     27   # TODO: Remove this import after fix api_implementation
     28   if _message is None:
---> 29     from google.protobuf.pyext import _message
     30   _USE_C_DESCRIPTORS = True
     31 

ImportError: cannot import name '_message' from 'google.protobuf.pyext' (/usr/local/lib/python3.11/dist-packages/google/protobuf/pyext/__init__.py)

## === cell 2
os.chdir("/kaggle/")
os.getcwd()



## === cell 3
train_data = pd.read_json("/kaggle/input/stanford-covid-vaccine/train.json", lines=True)
test_data = pd.read_json("/kaggle/input/stanford-covid-vaccine/test.json", lines=True)
submission_format = pd.read_csv(
    "/kaggle/input/stanford-covid-vaccine/sample_submission.csv", encoding="utf-8-sig"
)



## === cell 4
train_data.head()



## === cell 5
train_data.shape



## === cell 6
train_data.groupby(["SN_filter"]).size()



## === cell 7
test_data.head()



## === cell 8
test_data.shape



## === cell 9
submission_format.head()



## === cell 10
print(train_data.shape)
print(test_data.shape)
print(submission_format.shape)



## === cell 11
print("Training data:\n", train_data["seq_scored"].value_counts())
print("Test data:\n", test_data["seq_scored"].value_counts())
len(train_data["reactivity"].iloc[0])



## === cell 12
len(train_data["sequence"].iloc[0])



## === cell 13
flag = False
for i in range(0, len(train_data)):
    if (
        ([x < 0 for x in train_data["reactivity_error"].iloc[0]].count(True) > 0)
        | ([x < 0 for x in train_data["deg_error_Mg_pH10"].iloc[0]].count(True) > 0)
        | ([x < 0 for x in train_data["deg_error_pH10"].iloc[0]].count(True) > 0)
        | ([x < 0 for x in train_data["deg_error_Mg_50C"].iloc[0]].count(True) > 0)
        | ([x < 0 for x in train_data["deg_error_50C"].iloc[0]].count(True) > 0)
    ):
        flag = True
print(flag)



## === cell 14
train_data.columns



## === cell 15
token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]



## === cell 16
token2int



## === cell 17
BPPS_DIR = "/kaggle/input/stanford-covid-vaccine/bpps"


def _zero_bpps_features(df, seq_col="sequence"):
    lens = df[seq_col].str.len().to_numpy()
    out = [np.zeros((L,), dtype=np.float32) for L in lens]
    return out


def read_bpps_sum(df):
    if not os.path.isdir(BPPS_DIR):
        return _zero_bpps_features(df)
    bpps_arr = []
    for mol_id in df.id.to_list():
        path = os.path.join(BPPS_DIR, f"{mol_id}.npy")
        if os.path.exists(path):
            bpps_arr.append(np.load(path).sum(axis=1).astype(np.float32))
        else:
            bpps_arr.append(
                np.zeros(
                    (len(df.loc[df["id"] == mol_id, "sequence"].iloc[0]),),
                    dtype=np.float32,
                )
            )
    return bpps_arr


def read_bpps_max(df):
    if not os.path.isdir(BPPS_DIR):
        return _zero_bpps_features(df)
    bpps_arr = []
    for mol_id in df.id.to_list():
        path = os.path.join(BPPS_DIR, f"{mol_id}.npy")
        if os.path.exists(path):
            bpps_arr.append(np.load(path).max(axis=1).astype(np.float32))
        else:
            bpps_arr.append(
                np.zeros(
                    (len(df.loc[df["id"] == mol_id, "sequence"].iloc[0]),),
                    dtype=np.float32,
                )
            )
    return bpps_arr


def read_bpps_nb(df):
    bpps_nb_mean = 0.077522
    bpps_nb_std = 0.08914
    if not os.path.isdir(BPPS_DIR):
        return _zero_bpps_features(df)
    bpps_arr = []
    for mol_id in df.id.to_list():
        path = os.path.join(BPPS_DIR, f"{mol_id}.npy")
        if os.path.exists(path):
            bpps = np.load(path)
            bpps_nb = (bpps > 0).sum(axis=0) / bpps.shape[0]
            bpps_nb = (bpps_nb - bpps_nb_mean) / bpps_nb_std
            bpps_arr.append(bpps_nb.astype(np.float32))
        else:
            bpps_arr.append(
                np.zeros(
                    (len(df.loc[df["id"] == mol_id, "sequence"].iloc[0]),),
                    dtype=np.float32,
                )
            )
    return bpps_arr


os.chdir("/kaggle/working/")

train_data["bpps_sum"] = read_bpps_sum(train_data)
test_data["bpps_sum"] = read_bpps_sum(test_data)
train_data["bpps_max"] = read_bpps_max(train_data)
test_data["bpps_max"] = read_bpps_max(test_data)
train_data["bpps_nb"] = read_bpps_nb(train_data)
test_data["bpps_nb"] = read_bpps_nb(test_data)

train_data.head()




## === cell 18
def preprocess_inputs(df, cols=["sequence", "structure", "predicted_loop_type"]):
    base_fea = np.transpose(
        np.array(
            df[cols].applymap(lambda seq: [token2int[x] for x in seq]).values.tolist()
        ),
        (0, 2, 1),
    ).astype(np.int32)

    bpps_sum_fea = np.array(df["bpps_sum"].to_list(), dtype=np.float32)[
        :, :, np.newaxis
    ]
    bpps_max_fea = np.array(df["bpps_max"].to_list(), dtype=np.float32)[
        :, :, np.newaxis
    ]
    bpps_nb_fea = np.array(df["bpps_nb"].to_list(), dtype=np.float32)[:, :, np.newaxis]

    out = np.concatenate(
        [base_fea.astype(np.float32), bpps_sum_fea, bpps_max_fea, bpps_nb_fea], 2
    ).astype(np.float32)
    return out




## === cell 19
train_inputs = preprocess_inputs(train_data.loc[train_data["signal_to_noise"] > 1])
train_labels = np.array(
    train_data.loc[train_data["signal_to_noise"] > 1][target_cols].values.tolist(),
    dtype=np.float32,
).transpose((0, 2, 1))

print("train_inputs:", train_inputs.shape, train_inputs.dtype)
print("train_labels:", train_labels.shape, train_labels.dtype)



## === cell 20
train_data.loc[[0]]



## === cell 21
preprocess_inputs(train_data.loc[[0]]).shape



## === cell 22
test_data.head()



## === cell 23
mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)


def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(mse(y_true, y_pred))


def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)
    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)


def lstm_layer(hidden_dim, dropout):
    return tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            hidden_dim,
            dropout=dropout,
            return_sequences=True,
            kernel_initializer="orthogonal",
        )
    )


def build_model(seq_len=107, embed_dim=100, hidden_dim=256, dropout=0.2, pred_len=68):
    inputs = tf.keras.layers.Input(shape=(seq_len, 6))
    categorical_feats = inputs[:, :, :3]
    numerical_feats = inputs[:, :, 3:]

    categorical_feats_int = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.int32))(
        categorical_feats
    )

    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(
        categorical_feats_int
    )
    reshaped = tf.keras.layers.Reshape((seq_len, 3 * embed_dim))(embed)

    reshaped = tf.keras.layers.Concatenate(axis=2)([reshaped, numerical_feats])

    LSTM_layer = lstm_layer(hidden_dim, dropout)(reshaped)
    truncated = LSTM_layer[:, :pred_len]
    out = tf.keras.layers.Dense(5, activation="linear")(truncated)

    model = tf.keras.Model(inputs=inputs, outputs=out)
    adam = tf.optimizers.Adam()
    model.compile(optimizer=adam, loss=MCRMSE)
    return model




## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1032003343.py in <cell line: 0>()
----> 1 mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
      2 
      3 
      4 def root_mean_squared_error(y_true, y_pred):
      5     return tf.sqrt(mse(y_true, y_pred))

NameError: name 'tf' is not defined

## === cell 24
EPOCHS = 60
BATCH_SIZE = 32

model_on_train_data = build_model()
model_on_train_data.summary()

ckpt_path = "LSTM_model.weights.h5"
model_callback = tf.keras.callbacks.ModelCheckpoint(
    ckpt_path, save_weights_only=True, save_best_only=False
)

history = model_on_train_data.fit(
    train_inputs,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=2,
    callbacks=[model_callback],
)



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1600909755.py in <cell line: 0>()
      2 BATCH_SIZE = 32
      3 
----> 4 model_on_train_data = build_model()
      5 model_on_train_data.summary()
      6 

NameError: name 'build_model' is not defined

## === cell 25
print(f" LSTM mean training loss: {min(history.history['loss'])}")



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/864973356.py in <cell line: 0>()
----> 1 print(f" LSTM mean training loss: {min(history.history['loss'])}")
      2 

NameError: name 'history' is not defined

## === cell 26
public_df = test_data.query("seq_length == 107").copy()
private_df = test_data.query("seq_length == 130").copy()

public_inputs = preprocess_inputs(public_df) if len(public_df) else None
private_inputs = preprocess_inputs(private_df) if len(private_df) else None

print("public_df:", public_df.shape, "private_df:", private_df.shape)



## === cell 27
pred_test_data_public = np.zeros((0, 0, 5), dtype=np.float32)
if len(public_df):
    model_on_test_data_public = build_model(seq_len=107, pred_len=107)
    model_on_test_data_public.load_weights(ckpt_path)
    pred_test_data_public = model_on_test_data_public.predict(
        public_inputs, batch_size=64, verbose=0
    )
    print("pred_test_data_public:", pred_test_data_public.shape)



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1791387887.py in <cell line: 0>()
      1 pred_test_data_public = np.zeros((0, 0, 5), dtype=np.float32)
      2 if len(public_df):
----> 3     model_on_test_data_public = build_model(seq_len=107, pred_len=107)
      4     model_on_test_data_public.load_weights(ckpt_path)
      5     pred_test_data_public = model_on_test_data_public.predict(

NameError: name 'build_model' is not defined

## === cell 28
pred_test_data_private = np.zeros((0, 0, 5), dtype=np.float32)
if len(private_df):
    model_on_test_data_private = build_model(seq_len=130, pred_len=130)
    model_on_test_data_private.load_weights(ckpt_path)
    pred_test_data_private = model_on_test_data_private.predict(
        private_inputs, batch_size=64, verbose=0
    )
    print("pred_test_data_private:", pred_test_data_private.shape)




## === cell 29
def format_predictions(public_preds, private_preds):
    preds = []
    for df, preds_ in [(public_df, public_preds), (private_df, private_preds)]:
        if len(df) == 0:
            continue
        for i, uid in enumerate(df.id):
            single_pred = preds_[i]
            single_df = pd.DataFrame(single_pred, columns=target_cols)
            single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
            preds.append(single_df)
    return (
        pd.concat(preds, axis=0).reset_index(drop=True)
        if len(preds)
        else pd.DataFrame(columns=target_cols + ["id_seqpos"])
    )


lstm_preds = format_predictions(pred_test_data_public, pred_test_data_private)
lstm_preds.head()



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/3790000852.py in <cell line: 0>()
     16 
     17 
---> 18 lstm_preds = format_predictions(pred_test_data_public, pred_test_data_private)
     19 lstm_preds.head()
     20 

/tmp/ipykernel_11/3790000852.py in format_predictions(public_preds, private_preds)
      5             continue
      6         for i, uid in enumerate(df.id):
----> 7             single_pred = preds_[i]
      8             single_df = pd.DataFrame(single_pred, columns=target_cols)
      9             single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]

IndexError: index 0 is out of bounds for axis 0 with size 0

## === cell 30
submission = submission_format[["id_seqpos"]].merge(
    lstm_preds, how="left", on="id_seqpos"
)

for c in target_cols:
    if c not in submission.columns:
        submission[c] = 0.0
submission[target_cols] = submission[target_cols].astype(np.float32).fillna(0.0)

submission = submission[["id_seqpos"] + target_cols]

print("submission shape:", submission.shape)
submission.head()



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2739489852.py in <cell line: 0>()
      1 submission = submission_format[["id_seqpos"]].merge(
----> 2     lstm_preds, how="left", on="id_seqpos"
      3 )
      4 
      5 for c in target_cols:

NameError: name 'lstm_preds' is not defined

## === cell 31
os.chdir("/kaggle/working/")
submission.to_csv("submission.csv", index=False)
print("Wrote:", os.path.abspath("submission.csv"))
print(pd.read_csv("submission.csv").head())

## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2019401606.py in <cell line: 0>()
      1 os.chdir("/kaggle/working/")
----> 2 submission.to_csv("submission.csv", index=False)
      3 print("Wrote:", os.path.abspath("submission.csv"))
      4 print(pd.read_csv("submission.csv").head())

NameError: name 'submission' is not defined
