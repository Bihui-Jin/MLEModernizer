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
Predict which chatbot response a user will prefer in a competition between two chatbots.

## Metric
Log loss with "eps=auto"

## Submission Format
For each id in the test set, you must predict the probability for each target class. The file should contain a header and have the following format:

```
 id,winner_model_a,winner_model_b,winner_tie
 136060,0.33,0,33,0.33
 211333,0.33,0,33,0.33
 1233961,0.33,0,33,0.33
 etc
```

## Dataset
**train.csv**

- `id` - A unique identifier for the row.
- `model_[a/b]` - The identity of model_[a/b]. Included in train.csv but not test.csv.
- `prompt` - The prompt that was given as an input (to both models).
- `response_[a/b]` - The response from model_[a/b] to the given prompt.
- `winner_model_[a/b/tie]` - Binary columns marking the judge's selection. The ground truth target column.

**test.csv**

- `id`
- `prompt`
- `response_[a/b]`

**sample_submission.csv** A submission file in the correct format.

- `id`
- `winner_model_[a/b/tie]` - This is what is predicted from the test set.

# 2. Python version

3.12

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
plotly==5.24.1
plotly-express==0.4.1
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
            description.md (98 lines)
            sample_submission.csv (5749 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (5749 lines)
            test.csv.zip (5.9 MB)
            train.csv (51730 lines)
            train.csv.zip (53.4 MB)
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
        input/
            description.md (98 lines)
            sample_submission.csv (5749 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (5749 lines)
            test.csv.zip (5.9 MB)
            train.csv (51730 lines)
            train.csv.zip (53.4 MB)
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
        working/
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
```

-> data/lmsys-chatbot-arena/sample_submission.csv has 5748 rows and 4 columns.
The columns are: id, winner_model_a, winner_model_b, winner_tie

-> data/lmsys-chatbot-arena/test.csv has 5748 rows and 4 columns.
The columns are: id, prompt, response_a, response_b

-> data/lmsys-chatbot-arena/train.csv has 51729 rows and 9 columns.
The columns are: id, model_a, model_b, prompt, response_a, response_b, winner_model_a, winner_model_b, winner_tie

-> data/sample_submission.csv has 5748 rows and 4 columns.
The columns are: id, winner_model_a, winner_model_b, winner_tie

-> data/test.csv has 5748 rows and 4 columns.
The columns are: id, prompt, response_a, response_b

-> data/train.csv has 51729 rows and 9 columns.
The columns are: id, model_a, model_b, prompt, response_a, response_b, winner_model_a, winner_model_b, winner_tie

-> (stopped after 10 files for performance)

# 5. Target score

1.1289065843520338

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras_nlp
import keras
import tensorflow as tf

import numpy as np
import pandas as pd
from tqdm import tqdm
import json

import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.express as px




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_55/1832410002.py in <cell line: 0>()
      5 os.environ["KERAS_BACKEND"] = "tensorflow"
      6 
----> 7 import keras_nlp
      8 import keras
      9 import tensorflow as tf

/usr/local/lib/python3.11/dist-packages/keras_nlp/__init__.py in <module>
      2 
      3 # Add everything in /api/ to the module search path.
----> 4 import keras_hub
      5 
      6 # Import everything from /api/ into keras.

/usr/local/lib/python3.11/dist-packages/keras_hub/__init__.py in <module>
      8 
      9 # Import everything from /api/ into keras.
---> 10 from keras_hub.api import *  # noqa: F403
     11 from keras_hub.api import __version__  # Import * ignores names start with "_".
     12 

/usr/local/lib/python3.11/dist-packages/keras_hub/api/__init__.py in <module>
      5 """
      6 
----> 7 from keras_hub.api import bounding_box
      8 from keras_hub.api import layers
      9 from keras_hub.api import metrics

/usr/local/lib/python3.11/dist-packages/keras_hub/api/bounding_box/__init__.py in <module>
      5 """
      6 
----> 7 from keras_hub.src.bounding_box.converters import convert_format
      8 from keras_hub.src.bounding_box.formats import CENTER_XYWH
      9 from keras_hub.src.bounding_box.formats import REL_XYWH

/usr/local/lib/python3.11/dist-packages/keras_hub/src/bounding_box/converters.py in <module>
      1 """Converter functions for working with bounding box formats."""
      2 
----> 3 import keras
      4 from keras import ops
      5 

/usr/local/lib/python3.11/dist-packages/keras/__init__.py in <module>
      1 # DO NOT EDIT. Generated by api_gen.sh
----> 2 from keras.api import DTypePolicy
      3 from keras.api import FloatDTypePolicy
      4 from keras.api import Function
      5 from keras.api import Initializer

/usr/local/lib/python3.11/dist-packages/keras/api/__init__.py in <module>
      6 
      7 
----> 8 from keras.api import activations
      9 from keras.api import applications
     10 from keras.api import backend

/usr/local/lib/python3.11/dist-packages/keras/api/activations/__init__.py in <module>
      5 """
      6 
----> 7 from keras.src.activations import deserialize
      8 from keras.src.activations import get
      9 from keras.src.activations import serialize

/usr/local/lib/python3.11/dist-packages/keras/src/__init__.py in <module>
----> 1 from keras.src import activations
      2 from keras.src import applications
      3 from keras.src import backend
      4 from keras.src import constraints
      5 from keras.src import datasets

/usr/local/lib/python3.11/dist-packages/keras/src/activations/__init__.py in <module>
      1 import types
      2 
----> 3 from keras.src.activations.activations import celu
      4 from keras.src.activations.activations import elu
      5 from keras.src.activations.activations import exponential

/usr/local/lib/python3.11/dist-packages/keras/src/activations/activations.py in <module>
----> 1 from keras.src import backend
      2 from keras.src import ops
      3 from keras.src.api_export import keras_export
      4 
      5 

/usr/local/lib/python3.11/dist-packages/keras/src/backend/__init__.py in <module>
      8 
      9 from keras.src.api_export import keras_export
---> 10 from keras.src.backend.common.dtypes import result_type
     11 from keras.src.backend.common.keras_tensor import KerasTensor
     12 from keras.src.backend.common.keras_tensor import any_symbolic_tensors

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/__init__.py in <module>
      1 from keras.src.backend.common import backend_utils
----> 2 from keras.src.backend.common.dtypes import result_type
      3 from keras.src.backend.common.variables import AutocastScope
      4 from keras.src.backend.common.variables import Variable as KerasVariable
      5 from keras.src.backend.common.variables import get_autocast_scope

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/dtypes.py in <module>
      3 from keras.src.api_export import keras_export
      4 from keras.src.backend import config
----> 5 from keras.src.backend.common.variables import standardize_dtype
      6 
      7 BOOL_TYPES = ("bool",)

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/variables.py in <module>
      9 from keras.src.backend.common.stateless_scope import get_stateless_scope
     10 from keras.src.backend.common.stateless_scope import in_stateless_scope
---> 11 from keras.src.utils.module_utils import tensorflow as tf
     12 from keras.src.utils.naming import auto_name
     13 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/__init__.py in <module>
----> 1 from keras.src.utils.audio_dataset_utils import audio_dataset_from_directory
      2 from keras.src.utils.dataset_utils import split_dataset
      3 from keras.src.utils.file_utils import get_file
      4 from keras.src.utils.image_dataset_utils import image_dataset_from_directory
      5 from keras.src.utils.image_utils import array_to_img

/usr/local/lib/python3.11/dist-packages/keras/src/utils/audio_dataset_utils.py in <module>
      2 
      3 from keras.src.api_export import keras_export
----> 4 from keras.src.utils import dataset_utils
      5 from keras.src.utils.module_utils import tensorflow as tf
      6 from keras.src.utils.module_utils import tensorflow_io as tfio

/usr/local/lib/python3.11/dist-packages/keras/src/utils/dataset_utils.py in <module>
      7 import numpy as np
      8 
----> 9 from keras.src import tree
     10 from keras.src.api_export import keras_export
     11 from keras.src.utils import io_utils

/usr/local/lib/python3.11/dist-packages/keras/src/tree/__init__.py in <module>
----> 1 from keras.src.tree.tree_api import assert_same_paths
      2 from keras.src.tree.tree_api import assert_same_structure
      3 from keras.src.tree.tree_api import flatten
      4 from keras.src.tree.tree_api import flatten_with_path
      5 from keras.src.tree.tree_api import is_nested

/usr/local/lib/python3.11/dist-packages/keras/src/tree/tree_api.py in <module>
      6 
      7 if optree.available:
----> 8     from keras.src.tree import optree_impl as tree_impl
      9 elif dmtree.available:
     10     from keras.src.tree import dmtree_impl as tree_impl

/usr/local/lib/python3.11/dist-packages/keras/src/tree/optree_impl.py in <module>
     11 # Register backend-specific node classes
     12 if backend() == "tensorflow":
---> 13     from tensorflow.python.trackable.data_structures import ListWrapper
     14     from tensorflow.python.trackable.data_structures import _DictWrapper
     15 

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

## === cell 1
print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)
print("KerasNLP:", keras_nlp.__version__)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3341192671.py in <cell line: 0>()
----> 1 print("TensorFlow:", tf.__version__)
      2 print("Keras:", keras.__version__)
      3 print("KerasNLP:", keras_nlp.__version__)
      4 
      5 

NameError: name 'tf' is not defined

## === cell 2
class CFG:
    seed = 69  # Random seed
    preset = "bert_tiny_en_uncased"  # Name of pretrained models
    sequence_length = 512  # Input sequence length
    epochs = 8  # Training epochs
    batch_size = 16  # Batch size
    scheduler = "cosine"  # Learning rate scheduler
    label2name = {0: "winner_model_a", 1: "winner_model_b", 2: "winner_tie"}
    name2label = {v: k for k, v in label2name.items()}
    class_labels = list(label2name.keys())
    class_names = list(label2name.values())




## === cell 3
keras.utils.set_random_seed(CFG.seed)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/853911830.py in <cell line: 0>()
----> 1 keras.utils.set_random_seed(CFG.seed)
      2 
      3 

NameError: name 'keras' is not defined

## === cell 4
keras.mixed_precision.set_global_policy("mixed_float16")




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1039363140.py in <cell line: 0>()
----> 1 keras.mixed_precision.set_global_policy("mixed_float16")
      2 
      3 

NameError: name 'keras' is not defined

## === cell 5
BASE_PATH = "/kaggle/input/lmsys-chatbot-arena"




## === cell 6
df = pd.read_csv(f"{BASE_PATH}/train.csv", on_bad_lines="skip", engine="python")

df["prompt"] = df.prompt.map(lambda x: eval(x.lower())[0])
df["response_a"] = df.response_a.map(lambda x: eval(x.replace("null", "''").lower())[0])
df["response_b"] = df.response_b.map(lambda x: eval(x.replace("null", "''").lower())[0])

df["class_name"] = df[["winner_model_a", "winner_model_b", "winner_tie"]].idxmax(axis=1)
df["class_label"] = df.class_name.map(CFG.name2label)

df.head()




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1467947559.py in <cell line: 0>()
----> 1 df = pd.read_csv(f"{BASE_PATH}/train.csv", on_bad_lines="skip", engine="python")
      2 
      3 df["prompt"] = df.prompt.map(lambda x: eval(x.lower())[0])
      4 df["response_a"] = df.response_a.map(lambda x: eval(x.replace("null", "''").lower())[0])
      5 df["response_b"] = df.response_b.map(lambda x: eval(x.replace("null", "''").lower())[0])

NameError: name 'pd' is not defined

## === cell 7
test_df = pd.read_csv(f"{BASE_PATH}/test.csv")

test_df["prompt"] = test_df.prompt.map(lambda x: eval(x.lower())[0])
test_df["response_a"] = test_df.response_a.map(
    lambda x: eval(x.replace("null", "''").lower())[0]
)
test_df["response_b"] = test_df.response_b.map(
    lambda x: eval(x.replace("null", "''").lower())[0]
)

test_df.head()




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1993944652.py in <cell line: 0>()
----> 1 test_df = pd.read_csv(f"{BASE_PATH}/test.csv")
      2 
      3 test_df["prompt"] = test_df.prompt.map(lambda x: eval(x.lower())[0])
      4 test_df["response_a"] = test_df.response_a.map(
      5     lambda x: eval(x.replace("null", "''").lower())[0]

NameError: name 'pd' is not defined

## === cell 8
def make_pairs(row):
    row["encode_fail"] = False
    try:
        prompt = row.prompt.encode("utf-8").decode("utf-8")
    except:
        prompt = ""
        row["encode_fail"] = True

    try:
        response_a = row.response_a.encode("utf-8").decode("utf-8")
    except:
        response_a = ""
        row["encode_fail"] = True

    try:
        response_b = row.response_b.encode("utf-8").decode("utf-8")
    except:
        response_b = ""
        row["encode_fail"] = True

    row["options"] = [
        f"Prompt: {prompt}\n\nResponse: {response_a}",
        f"Prompt: {prompt}\n\nResponse: {response_b}",
    ]
    return row




## === cell 9
df = df.apply(make_pairs, axis=1)  # Apply the make_pairs function to each row in df
display(df.head())

test_df = test_df.apply(
    make_pairs, axis=1
)  # Apply the make_pairs function to each row in test_df
display(test_df.head())




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2621572240.py in <cell line: 0>()
----> 1 df = df.apply(make_pairs, axis=1)  # Apply the make_pairs function to each row in df
      2 display(df.head())
      3 
      4 test_df = test_df.apply(
      5     make_pairs, axis=1

NameError: name 'df' is not defined

## === cell 10
df.encode_fail.value_counts(normalize=False)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1399931581.py in <cell line: 0>()
----> 1 df.encode_fail.value_counts(normalize=False)
      2 
      3 

NameError: name 'df' is not defined

## === cell 11
model_df = pd.concat([df.model_a, df.model_b])
counts = model_df.value_counts().reset_index()
counts.columns = ["LLM", "Count"]

fig = px.bar(
    counts,
    x="LLM",
    y="Count",
    title="Distribution of LLMs",
    color="Count",
    color_continuous_scale="viridis",
)

fig.update_layout(xaxis_tickangle=-45)  # Rotate x-axis labels for better readability

fig.show()




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/792446468.py in <cell line: 0>()
----> 1 model_df = pd.concat([df.model_a, df.model_b])
      2 counts = model_df.value_counts().reset_index()
      3 counts.columns = ["LLM", "Count"]
      4 
      5 fig = px.bar(

NameError: name 'pd' is not defined

## === cell 12
counts = df["class_name"].value_counts().reset_index()
counts.columns = ["Winner", "Win Count"]

fig = px.bar(
    counts,
    x="Winner",
    y="Win Count",
    title="Winner distribution for Train Data",
    labels={"Winner": "Winner", "Win Count": "Win Count"},
    color="Winner",
    color_continuous_scale="viridis",
)

fig.update_layout(xaxis_title="Winner", yaxis_title="Win Count")

fig.show()




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2446821111.py in <cell line: 0>()
----> 1 counts = df["class_name"].value_counts().reset_index()
      2 counts.columns = ["Winner", "Win Count"]
      3 
      4 fig = px.bar(
      5     counts,

NameError: name 'df' is not defined

## === cell 13
from sklearn.model_selection import train_test_split  # Import package

train_df, valid_df = train_test_split(
    df, test_size=0.2, stratify=df["class_label"], random_state=CFG.seed
)




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2821046690.py in <cell line: 0>()
      2 
      3 train_df, valid_df = train_test_split(
----> 4     df, test_size=0.2, stratify=df["class_label"], random_state=CFG.seed
      5 )
      6 

NameError: name 'df' is not defined

## === cell 14
preprocessor = keras_nlp.models.BertPreprocessor.from_preset(
    preset=CFG.preset,  # Name of the model
    sequence_length=CFG.sequence_length,  # Max sequence length, will be padded if shorter
)




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1413191932.py in <cell line: 0>()
----> 1 preprocessor = keras_nlp.models.BertPreprocessor.from_preset(
      2     preset=CFG.preset,  # Name of the model
      3     sequence_length=CFG.sequence_length,  # Max sequence length, will be padded if shorter
      4 )
      5 

NameError: name 'keras_nlp' is not defined

## === cell 15
outs = preprocessor(df.options.iloc[0])  # Process options for the first row

for k, v in outs.items():
    print(k, ":", v.shape)




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3524402361.py in <cell line: 0>()
----> 1 outs = preprocessor(df.options.iloc[0])  # Process options for the first row
      2 
      3 for k, v in outs.items():
      4     print(k, ":", v.shape)
      5 

NameError: name 'preprocessor' is not defined

## === cell 16
def preprocess_fn(text, label=None):
    text = preprocessor(text)  # Preprocess text
    return (
        (text, label) if label is not None else text
    )  # Return processed text and label if available




## === cell 17
def build_dataset(texts, labels=None, batch_size=32, cache=True, shuffle=True):
    AUTO = tf.data.AUTOTUNE  # AUTOTUNE option
    slices = (
        (texts,)
        if labels is None
        else (texts, keras.utils.to_categorical(labels, num_classes=3))
    )  # Create slices
    ds = tf.data.Dataset.from_tensor_slices(slices)  # Create dataset from slices
    ds = ds.cache() if cache else ds  # Cache dataset if enabled
    ds = ds.map(preprocess_fn, num_parallel_calls=AUTO)  # Map preprocessing function
    opt = tf.data.Options()  # Create dataset options
    if shuffle:
        ds = ds.shuffle(shuffle, seed=CFG.seed)  # Shuffle dataset if enabled
        opt.experimental_deterministic = False
    ds = ds.with_options(opt)  # Set dataset options
    ds = ds.batch(batch_size, drop_remainder=False)  # Batch dataset
    ds = ds.prefetch(AUTO)  # Prefetch next batch
    return ds  # Return the built dataset




## === cell 18
train_texts = train_df.options.tolist()  # Extract training texts
train_labels = train_df.class_label.tolist()  # Extract training labels
train_ds = build_dataset(
    train_texts, train_labels, batch_size=CFG.batch_size, shuffle=True
)

valid_texts = valid_df.options.tolist()  # Extract validation texts
valid_labels = valid_df.class_label.tolist()  # Extract validation labels
valid_ds = build_dataset(
    valid_texts, valid_labels, batch_size=CFG.batch_size, shuffle=False
)




## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3942298163.py in <cell line: 0>()
----> 1 train_texts = train_df.options.tolist()  # Extract training texts
      2 train_labels = train_df.class_label.tolist()  # Extract training labels
      3 train_ds = build_dataset(
      4     train_texts, train_labels, batch_size=CFG.batch_size, shuffle=True
      5 )

NameError: name 'train_df' is not defined

## === cell 19
import math


def get_lr_callback(batch_size=8, mode="cos", epochs=10, plot=False):
    lr_start, lr_max, lr_min = 1.0e-6, 0.6e-6 * batch_size, 1e-6
    lr_ramp_ep, lr_sus_ep, lr_decay = 2, 0, 0.8

    def lrfn(epoch):  # Learning rate update function
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
        elif mode == "exp":
            lr = (lr_max - lr_min) * lr_decay ** (
                epoch - lr_ramp_ep - lr_sus_ep
            ) + lr_min
        elif mode == "step":
            lr = lr_max * lr_decay ** ((epoch - lr_ramp_ep - lr_sus_ep) // 2)
        elif mode == "cos":
            decay_total_epochs, decay_epoch_index = (
                epochs - lr_ramp_ep - lr_sus_ep + 3,
                epoch - lr_ramp_ep - lr_sus_ep,
            )
            phase = math.pi * decay_epoch_index / decay_total_epochs
            lr = (lr_max - lr_min) * 0.5 * (1 + math.cos(phase)) + lr_min
        return lr

    if plot:  # Plot lr curve if plot is True
        plt.figure(figsize=(10, 5))
        plt.plot(
            np.arange(epochs), [lrfn(epoch) for epoch in np.arange(epochs)], marker="o"
        )
        plt.xlabel("epoch")
        plt.ylabel("lr")
        plt.title("LR Scheduler")
        plt.show()

    return keras.callbacks.LearningRateScheduler(
        lrfn, verbose=False
    )  # Create lr callback




## === cell 20
lr_cb = get_lr_callback(
    CFG.batch_size, plot=False
)  # Disable plot for non‑interactive runs




## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/801005052.py in <cell line: 0>()
----> 1 lr_cb = get_lr_callback(
      2     CFG.batch_size, plot=False
      3 )  # Disable plot for non‑interactive runs
      4 
      5 

/tmp/ipykernel_55/2733645324.py in get_lr_callback(batch_size, mode, epochs, plot)
     36         plt.show()
     37 
---> 38     return keras.callbacks.LearningRateScheduler(
     39         lrfn, verbose=False
     40     )  # Create lr callback

NameError: name 'keras' is not defined

## === cell 21
ckpt_cb = keras.callbacks.ModelCheckpoint(
    f"best_model.weights.h5",
    monitor="val_log_loss",
    save_best_only=True,
    save_weights_only=True,
    mode="min",
)  # Get Model checkpoint callback




## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/851011975.py in <cell line: 0>()
----> 1 ckpt_cb = keras.callbacks.ModelCheckpoint(
      2     f"best_model.weights.h5",
      3     monitor="val_log_loss",
      4     save_best_only=True,
      5     save_weights_only=True,

NameError: name 'keras' is not defined

## === cell 22
log_loss = keras.metrics.CategoricalCrossentropy(name="log_loss")




## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4271598739.py in <cell line: 0>()
----> 1 log_loss = keras.metrics.CategoricalCrossentropy(name="log_loss")
      2 
      3 

NameError: name 'keras' is not defined

## === cell 23
inputs = {
    "token_ids": keras.Input(shape=(2, None), dtype=tf.int32, name="token_ids"),
    "padding_mask": keras.Input(shape=(2, None), dtype=tf.int32, name="padding_mask"),
    "segment_ids": keras.Input(shape=(2, None), dtype=tf.int32, name="segment_ids"),
}
backbone = keras_nlp.models.BertBackbone.from_preset(
    CFG.preset,
)

response_a = {k: v[:, 0, :] for k, v in inputs.items()}
embed_a = backbone(response_a)

response_b = {k: v[:, 1, :] for k, v in inputs.items()}
embed_b = backbone(response_b)

embeds = keras.layers.Concatenate(axis=-1)(
    [embed_a["sequence_output"], embed_b["sequence_output"]]
)
embeds = keras.layers.GlobalAveragePooling1D()(embeds)
outputs = keras.layers.Dense(3, activation="softmax", name="classifier")(embeds)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(5e-6),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.02),
    metrics=[
        log_loss,
        keras.metrics.CategoricalAccuracy(name="accuracy"),
    ],
)




## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1733231522.py in <cell line: 0>()
      1 inputs = {
----> 2     "token_ids": keras.Input(shape=(2, None), dtype=tf.int32, name="token_ids"),
      3     "padding_mask": keras.Input(shape=(2, None), dtype=tf.int32, name="padding_mask"),
      4     "segment_ids": keras.Input(shape=(2, None), dtype=tf.int32, name="segment_ids"),
      5 }

NameError: name 'keras' is not defined

## === cell 24
model.summary()




## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/775241066.py in <cell line: 0>()
----> 1 model.summary()
      2 
      3 

NameError: name 'model' is not defined

## === cell 25
keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)




## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2604168601.py in <cell line: 0>()
----> 1 keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)
      2 
      3 

NameError: name 'keras' is not defined

## === cell 26
history = model.fit(
    train_ds, epochs=CFG.epochs, validation_data=valid_ds, callbacks=[lr_cb, ckpt_cb]
)




## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4122565215.py in <cell line: 0>()
----> 1 history = model.fit(
      2     train_ds, epochs=CFG.epochs, validation_data=valid_ds, callbacks=[lr_cb, ckpt_cb]
      3 )
      4 
      5 

NameError: name 'model' is not defined

## === cell 27
model.load_weights("best_model.weights.h5")




## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1273657460.py in <cell line: 0>()
----> 1 model.load_weights("best_model.weights.h5")
      2 
      3 

NameError: name 'model' is not defined

## === cell 28
test_texts = test_df.options.tolist()
test_ds = build_dataset(
    test_texts, batch_size=min(len(test_df), CFG.batch_size), shuffle=False
)

test_preds = model.predict(test_ds, verbose=1).astype(np.float32)

row_sums = np.clip(test_preds.sum(axis=1, keepdims=True), a_min=1e-12, a_max=None)
test_preds = test_preds / row_sums

preds_df = pd.DataFrame(test_preds, columns=CFG.class_names)
sub_df = pd.concat([test_df[["id"]].reset_index(drop=True), preds_df], axis=1)

sub_df.to_csv("submission.csv", index=False)

sub_df.head()

## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1698725080.py in <cell line: 0>()
----> 1 test_texts = test_df.options.tolist()
      2 test_ds = build_dataset(
      3     test_texts, batch_size=min(len(test_df), CFG.batch_size), shuffle=False
      4 )
      5 

NameError: name 'test_df' is not defined
