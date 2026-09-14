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
Given pairs of phrases (an `anchor` and a `target` phrase), build a model to rate how similar they are.  

## Metric
Pearson correlation coefficient.

## Submission Format
For each `id` (representing a pair of phrases) in the test set, you must predict the similarity `score`. The file should contain a header and have the following format:

```
id,score
4112d61851461f60,0
09e418c93a776564,0.25
36baf228038e314b,1
etc.

```

## Dataset
The scores are in the 0-1 range with increments of 0.25 with the following meanings:

- **1.0** - Very close match. This is typically an exact match except possibly for differences in conjugation, quantity (e.g. singular vs. plural), and addition or removal of stopwords (e.g. "the", "and", "or").
- **0.75** - Close synonym, e.g. "mobile phone" vs. "cellphone". This also includes abbreviations, e.g. "TCP" -> "transmission control protocol".
- **0.5** - Synonyms which don't have the same meaning (same function, same properties). This includes broad-narrow (hyponym) and narrow-broad (hypernym) matches.
- **0.25** - Somewhat related, e.g. the two phrases are in the same high level domain but are not synonyms. This also includes antonyms.
- **0.0** - Unrelated.

Files
-----

- **train.csv** - the training set, containing phrases, contexts, and their similarity scores
- **test.csv** - the test set set, identical in structure to the training set but without the score
- **sample_submission.csv** - a sample submission file in the correct format

Columns
-------

- `id` - a unique identifier for a pair of phrases
- `anchor` - the first phrase
- `target` - the second phrase
- `context` - the [CPC classification (version 2021.05)](https://en.wikipedia.org/wiki/Cooperative_Patent_Classification), which indicates the subject within which the similarity is to be scored
- `score` - the similarity. This is sourced from a combination of one or more manual expert ratings.

# 2. Python version

3.12

# 3. Installed packages

datasets==4.4.1
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow-datasets==4.9.9
transformers==4.53.3
vega-datasets==0.9.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        input/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        working/
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
```

-> data/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> data/us-patent-phrase-to-phrase-matching/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/us-patent-phrase-to-phrase-matching/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/us-patent-phrase-to-phrase-matching/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> input/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> (stopped after 10 files for performance)

# 5. Target score

0.7837437211583778

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved nan) has done: 'The timeout is overwhelmingly dominated by fine-tuning DeBERTa for 4 epochs on ~24k training rows; tokenization and pandas work are minor by comparison. To finish within 600 seconds without changing the model or training semantics, the biggest safe win is to enable gradient checkpointing (same forward pass math, much lower memory and typically higher throughput on constrained GPUs) and to turn on PyTorch TF32 matmul on Ampere+ GPUs (negligible FP differences, usually faster). I also remove unnecessary dataset columns earlier and avoid computing/storing the `length` field via a Python list loop by using the tokenizer’s built-in `return_length=True` (equivalent) to reduce CPU overhead during preprocessing. Finally, I ensure `use_cache=False` during training (required for checkpointing and avoids overhead), keep determinism/seeds intact, and keep all file paths and core logic unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import random

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")

random.seed(42)
np.random.seed(42)

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed,
    DataCollatorWithPadding,
)

set_seed(42)

try:
    ncpu = os.cpu_count() or 1
    torch.set_num_threads(min(8, ncpu))
    torch.set_num_interop_threads(1)
except Exception:
    pass

if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

try:
    torch.utils.checkpoint.use_reentrant = False  # for older torch
except Exception:
    pass
try:
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/627813086.py in <cell line: 0>()
     14 
     15 import torch
---> 16 from transformers import (
     17     AutoModelForSequenceClassification,
     18     AutoTokenizer,

/usr/local/lib/python3.11/dist-packages/transformers/utils/import_utils.py in __getattr__(self, name)
   2152         elif name in self._class_to_module.keys():
   2153             try:
-> 2154                 module = self._get_module(self._class_to_module[name])
   2155                 value = getattr(module, name)
   2156             except (ModuleNotFoundError, RuntimeError) as e:

/usr/local/lib/python3.11/dist-packages/transformers/utils/import_utils.py in _get_module(self, module_name)
   2182             return importlib.import_module("." + module_name, self.__name__)
   2183         except Exception as e:
-> 2184             raise e
   2185 
   2186     def __reduce__(self):

/usr/local/lib/python3.11/dist-packages/transformers/utils/import_utils.py in _get_module(self, module_name)
   2180     def _get_module(self, module_name: str):
   2181         try:
-> 2182             return importlib.import_module("." + module_name, self.__name__)
   2183         except Exception as e:
   2184             raise e

/usr/lib/python3.11/importlib/__init__.py in import_module(name, package)
    124                 break
    125             level += 1
--> 126     return _bootstrap._gcd_import(name[level:], package, level)
    127 
    128 

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in <module>
     40 # Integrations must be imported before ML frameworks:
     41 # ruff: isort: off
---> 42 from .integrations import (
     43     get_reporting_integration_callbacks,
     44 )

/usr/local/lib/python3.11/dist-packages/transformers/utils/import_utils.py in __getattr__(self, name)
   2152         elif name in self._class_to_module.keys():
   2153             try:
-> 2154                 module = self._get_module(self._class_to_module[name])
   2155                 value = getattr(module, name)
   2156             except (ModuleNotFoundError, RuntimeError) as e:

/usr/local/lib/python3.11/dist-packages/transformers/utils/import_utils.py in _get_module(self, module_name)
   2182             return importlib.import_module("." + module_name, self.__name__)
   2183         except Exception as e:
-> 2184             raise e
   2185 
   2186     def __reduce__(self):

/usr/local/lib/python3.11/dist-packages/transformers/utils/import_utils.py in _get_module(self, module_name)
   2180     def _get_module(self, module_name: str):
   2181         try:
-> 2182             return importlib.import_module("." + module_name, self.__name__)
   2183         except Exception as e:
   2184             raise e

/usr/lib/python3.11/importlib/__init__.py in import_module(name, package)
    124                 break
    125             level += 1
--> 126     return _bootstrap._gcd_import(name[level:], package, level)
    127 
    128 

/usr/local/lib/python3.11/dist-packages/transformers/integrations/integration_utils.py in <module>
     35 import packaging.version
     36 
---> 37 from .. import PreTrainedModel, TFPreTrainedModel, TrainingArguments
     38 from .. import __version__ as version
     39 from ..utils import (

/usr/local/lib/python3.11/dist-packages/transformers/utils/import_utils.py in __getattr__(self, name)
   2152         elif name in self._class_to_module.keys():
   2153             try:
-> 2154                 module = self._get_module(self._class_to_module[name])
   2155                 value = getattr(module, name)
   2156             except (ModuleNotFoundError, RuntimeError) as e:

/usr/local/lib/python3.11/dist-packages/transformers/utils/import_utils.py in _get_module(self, module_name)
   2182             return importlib.import_module("." + module_name, self.__name__)
   2183         except Exception as e:
-> 2184             raise e
   2185 
   2186     def __reduce__(self):

/usr/local/lib/python3.11/dist-packages/transformers/utils/import_utils.py in _get_module(self, module_name)
   2180     def _get_module(self, module_name: str):
   2181         try:
-> 2182             return importlib.import_module("." + module_name, self.__name__)
   2183         except Exception as e:
   2184             raise e

/usr/lib/python3.11/importlib/__init__.py in import_module(name, package)
    124                 break
    125             level += 1
--> 126     return _bootstrap._gcd_import(name[level:], package, level)
    127 
    128 

/usr/local/lib/python3.11/dist-packages/transformers/modeling_utils.py in <module>
     71     verify_tp_plan,
     72 )
---> 73 from .loss.loss_utils import LOSS_MAPPING
     74 from .pytorch_utils import (  # noqa: F401
     75     Conv1D,

/usr/local/lib/python3.11/dist-packages/transformers/loss/loss_utils.py in <module>
     19 from torch.nn import BCEWithLogitsLoss, MSELoss
     20 
---> 21 from .loss_d_fine import DFineForObjectDetectionLoss
     22 from .loss_deformable_detr import DeformableDetrForObjectDetectionLoss, DeformableDetrForSegmentationLoss
     23 from .loss_for_object_detection import ForObjectDetectionLoss, ForSegmentationLoss

/usr/local/lib/python3.11/dist-packages/transformers/loss/loss_d_fine.py in <module>
     19 
     20 from ..utils import is_vision_available
---> 21 from .loss_for_object_detection import (
     22     box_iou,
     23 )

/usr/local/lib/python3.11/dist-packages/transformers/loss/loss_for_object_detection.py in <module>
     30 
     31 if is_vision_available():
---> 32     from transformers.image_transforms import center_to_corners_format
     33 
     34 

/usr/local/lib/python3.11/dist-packages/transformers/image_transforms.py in <module>
     46 
     47 if is_tf_available():
---> 48     import tensorflow as tf
     49 
     50 if is_flax_available():

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
def corr(x, y):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if len(x) < 2:
        return 0.0
    return float(np.corrcoef(x, y)[0][1])




## === cell 2
def corr_d(eval_pred):
    preds, labels = eval_pred
    preds = np.asarray(preds).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    return {"pearson": corr(preds, labels)}




## === cell 3
path = "/kaggle/input/us-patent-phrase-to-phrase-matching/"
train_data = pd.read_csv(path + "train.csv")
print(train_data.head())
print(train_data.shape)



## === cell 4
test_data = pd.read_csv(path + "test.csv")
print(test_data.head())
print(test_data.shape)



## === cell 5
train_data["input"] = (
    "TEXT1: "
    + train_data["context"].astype(str)
    + "; TEXT2: "
    + train_data["target"].astype(str)
    + "; ANC1: "
    + train_data["anchor"].astype(str)
)



## === cell 6
model_nm = "microsoft/deberta-v3-small"
tokenizer = AutoTokenizer.from_pretrained(model_nm, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)

try:
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
except TypeError:
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass
except Exception:
    pass

try:
    model.config.use_cache = False
except Exception:
    pass



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3234749533.py in <cell line: 0>()
      1 model_nm = "microsoft/deberta-v3-small"
----> 2 tokenizer = AutoTokenizer.from_pretrained(model_nm, use_fast=True)
      3 model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)
      4 
      5 try:

NameError: name 'AutoTokenizer' is not defined

## === cell 7
_ = model.config



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/387220790.py in <cell line: 0>()
----> 1 _ = model.config
      2 

NameError: name 'model' is not defined

## === cell 8
pass



## === cell 9
from torch.utils.data import (
    Dataset,
)  # kept to preserve original imports if referenced elsewhere



## === cell 10
from sklearn.model_selection import train_test_split

tr_df, va_df = train_test_split(train_data, test_size=0.25, random_state=42)



## === cell 11
from datasets import Dataset as HFDataset


def _tokenize_batch(batch):
    return tokenizer(
        batch["input"],
        truncation=True,
        max_length=256,
        padding=False,
        return_length=True,
    )


tr_hf = HFDataset.from_pandas(tr_df.reset_index(drop=True), preserve_index=False)
va_hf = HFDataset.from_pandas(va_df.reset_index(drop=True), preserve_index=False)

tr_hf = tr_hf.remove_columns(
    [c for c in tr_hf.column_names if c not in ("input", "score")]
)
va_hf = va_hf.remove_columns(
    [c for c in va_hf.column_names if c not in ("input", "score")]
)

cache_dir = "/kaggle/working/hf_cache_deberta_v3_small_maxlen256"
os.makedirs(cache_dir, exist_ok=True)

ncpu = os.cpu_count() or 1
num_proc = 1 if torch.cuda.is_available() else min(4, max(1, ncpu // 2))

tr_hf = tr_hf.map(
    _tokenize_batch,
    batched=True,
    num_proc=num_proc,
    cache_file_name=os.path.join(cache_dir, "tr_tok.arrow"),
    remove_columns=["input"],
    desc="Tokenizing train",
)
va_hf = va_hf.map(
    _tokenize_batch,
    batched=True,
    num_proc=num_proc,
    cache_file_name=os.path.join(cache_dir, "va_tok.arrow"),
    remove_columns=["input"],
    desc="Tokenizing valid",
)

tr_hf = tr_hf.rename_column("score", "labels")
va_hf = va_hf.rename_column("score", "labels")

format_cols = ["input_ids", "attention_mask", "labels", "length"]
if "token_type_ids" in tr_hf.column_names:
    format_cols.insert(2, "token_type_ids")

tr_hf.set_format(type="torch", columns=format_cols)
va_hf.set_format(type="torch", columns=format_cols)

train_ds = tr_hf
valid_ds = va_hf
print("Train/valid sizes:", len(train_ds), len(valid_ds))
print("Columns:", train_ds.column_names)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
RemoteTraceback                           Traceback (most recent call last)
RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/multiprocess/pool.py", line 125, in worker
    result = (True, func(*args, **kwds))
                    ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/datasets/utils/py_utils.py", line 586, in _write_generator_to_queue
    for i, result in enumerate(func(**kwargs)):
  File "/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py", line 3697, in _map_single
    for i, batch in iter_outputs(shard_iterable):
  File "/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py", line 3647, in iter_outputs
    yield i, apply_function(example, i, offset=offset)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py", line 3570, in apply_function
    processed_inputs = function(*fn_args, *additional_args, **fn_kwargs)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/514035108.py", line 9, in _tokenize_batch
    return tokenizer(
           ^^^^^^^^^
NameError: name 'tokenizer' is not defined
"""

The above exception was the direct cause of the following exception:

NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/514035108.py in <cell line: 0>()
     35 num_proc = 1 if torch.cuda.is_available() else min(4, max(1, ncpu // 2))
     36 
---> 37 tr_hf = tr_hf.map(
     38     _tokenize_batch,
     39     batched=True,

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in wrapper(*args, **kwargs)
    560         }
    561         # apply actual function
--> 562         out: Union["Dataset", "DatasetDict"] = func(self, *args, **kwargs)
    563         datasets: list["Dataset"] = list(out.values()) if isinstance(out, dict) else [out]
    564         # re-apply format to the output

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in map(self, function, with_indices, with_rank, input_columns, batched, batch_size, drop_last_batch, remove_columns, keep_in_memory, load_from_cache_file, cache_file_name, writer_batch_size, features, disable_nullable, fn_kwargs, num_proc, suffix_template, new_fingerprint, desc, try_original_type)
   3330                         logger.info(f"Spawning {num_proc} processes")
   3331 
-> 3332                         for rank, done, content in iflatmap_unordered(
   3333                             pool, Dataset._map_single, kwargs_iterable=unprocessed_kwargs_per_job
   3334                         ):

/usr/local/lib/python3.11/dist-packages/datasets/utils/py_utils.py in iflatmap_unordered(pool, func, kwargs_iterable)
    624             if not pool_changed:
    625                 # we get the result in case there's an error to raise
--> 626                 [async_result.get(timeout=0.05) for async_result in async_results]
    627 
    628 

/usr/local/lib/python3.11/dist-packages/datasets/utils/py_utils.py in <listcomp>(.0)
    624             if not pool_changed:
    625                 # we get the result in case there's an error to raise
--> 626                 [async_result.get(timeout=0.05) for async_result in async_results]
    627 
    628 

/usr/local/lib/python3.11/dist-packages/multiprocess/pool.py in get(self, timeout)
    772             return self._value
    773         else:
--> 774             raise self._value
    775 
    776     def _set(self, i, obj):

NameError: name 'tokenizer' is not defined

## === cell 12
eval_df = test_data.copy()

eval_df["input"] = (
    "TEXT1: "
    + eval_df["context"].astype(str)
    + "; TEXT2: "
    + eval_df["target"].astype(str)
    + "; ANC1: "
    + eval_df["anchor"].astype(str)
)

eval_hf = HFDataset.from_pandas(eval_df.reset_index(drop=True), preserve_index=False)
eval_hf = eval_hf.remove_columns(
    [c for c in eval_hf.column_names if c not in ("input", "id")]
)

eval_hf = eval_hf.map(
    _tokenize_batch,
    batched=True,
    num_proc=num_proc,
    cache_file_name=os.path.join(cache_dir, "te_tok.arrow"),
    remove_columns=["input"],
    desc="Tokenizing test",
)

eval_format_cols = ["input_ids", "attention_mask", "length"]
if "token_type_ids" in eval_hf.column_names:
    eval_format_cols.insert(2, "token_type_ids")

eval_hf.set_format(type="torch", columns=eval_format_cols)

eval_ds = eval_hf
print(eval_df.head())
print("Eval columns:", eval_ds.column_names)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
RemoteTraceback                           Traceback (most recent call last)
RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/multiprocess/pool.py", line 125, in worker
    result = (True, func(*args, **kwds))
                    ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/datasets/utils/py_utils.py", line 586, in _write_generator_to_queue
    for i, result in enumerate(func(**kwargs)):
  File "/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py", line 3697, in _map_single
    for i, batch in iter_outputs(shard_iterable):
  File "/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py", line 3647, in iter_outputs
    yield i, apply_function(example, i, offset=offset)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py", line 3570, in apply_function
    processed_inputs = function(*fn_args, *additional_args, **fn_kwargs)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/514035108.py", line 9, in _tokenize_batch
    return tokenizer(
           ^^^^^^^^^
NameError: name 'tokenizer' is not defined
"""

The above exception was the direct cause of the following exception:

NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/443530970.py in <cell line: 0>()
     16 )
     17 
---> 18 eval_hf = eval_hf.map(
     19     _tokenize_batch,
     20     batched=True,

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in wrapper(*args, **kwargs)
    560         }
    561         # apply actual function
--> 562         out: Union["Dataset", "DatasetDict"] = func(self, *args, **kwargs)
    563         datasets: list["Dataset"] = list(out.values()) if isinstance(out, dict) else [out]
    564         # re-apply format to the output

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in map(self, function, with_indices, with_rank, input_columns, batched, batch_size, drop_last_batch, remove_columns, keep_in_memory, load_from_cache_file, cache_file_name, writer_batch_size, features, disable_nullable, fn_kwargs, num_proc, suffix_template, new_fingerprint, desc, try_original_type)
   3330                         logger.info(f"Spawning {num_proc} processes")
   3331 
-> 3332                         for rank, done, content in iflatmap_unordered(
   3333                             pool, Dataset._map_single, kwargs_iterable=unprocessed_kwargs_per_job
   3334                         ):

/usr/local/lib/python3.11/dist-packages/datasets/utils/py_utils.py in iflatmap_unordered(pool, func, kwargs_iterable)
    624             if not pool_changed:
    625                 # we get the result in case there's an error to raise
--> 626                 [async_result.get(timeout=0.05) for async_result in async_results]
    627 
    628 

/usr/local/lib/python3.11/dist-packages/datasets/utils/py_utils.py in <listcomp>(.0)
    624             if not pool_changed:
    625                 # we get the result in case there's an error to raise
--> 626                 [async_result.get(timeout=0.05) for async_result in async_results]
    627 
    628 

/usr/local/lib/python3.11/dist-packages/multiprocess/pool.py in get(self, timeout)
    772             return self._value
    773         else:
--> 774             raise self._value
    775 
    776     def _set(self, i, obj):

NameError: name 'tokenizer' is not defined

## === cell 13
bs = 32
epochs = 4
lr = 8e-5



## === cell 14
use_fp16 = torch.cuda.is_available()

if torch.cuda.is_available():
    num_workers = 0
else:
    num_workers = min(4, max(0, (os.cpu_count() or 1) // 2))

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer, pad_to_multiple_of=8 if use_fp16 else None
)

args = TrainingArguments(
    output_dir="outputs",
    learning_rate=lr,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    fp16=use_fp16,
    eval_strategy="epoch",
    per_device_train_batch_size=bs,
    per_device_eval_batch_size=bs * 2,
    num_train_epochs=epochs,
    weight_decay=0.01,
    report_to="none",
    save_strategy="no",
    logging_strategy="steps",
    logging_steps=50,
    dataloader_num_workers=num_workers,
    dataloader_pin_memory=torch.cuda.is_available(),
    group_by_length=True,
    length_column_name="length",
    remove_unused_columns=True,
    disable_tqdm=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=corr_d,
)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/891553000.py in <cell line: 0>()
      7     num_workers = min(4, max(0, (os.cpu_count() or 1) // 2))
      8 
----> 9 data_collator = DataCollatorWithPadding(
     10     tokenizer=tokenizer, pad_to_multiple_of=8 if use_fp16 else None
     11 )

NameError: name 'DataCollatorWithPadding' is not defined

## === cell 15
trainer.train()



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3352579090.py in <cell line: 0>()
----> 1 trainer.train()
      2 

NameError: name 'trainer' is not defined

## === cell 16
preds = trainer.predict(eval_ds).predictions
preds = np.asarray(preds, dtype=np.float32).reshape(-1)
preds = np.clip(preds, 0, 1)

score = np.where(
    preds >= 0.875,
    1.0,
    np.where(
        preds >= 0.625,
        0.75,
        np.where(
            preds >= 0.375,
            0.5,
            np.where(preds >= 0.125, 0.25, 0.0),
        ),
    ),
).astype(np.float32)

submission = pd.DataFrame({"id": eval_df["id"].values, "score": score})

assert submission.shape[0] == test_data.shape[0], "Row count mismatch vs test.csv"
assert submission.columns.tolist() == ["id", "score"], "Wrong submission columns"
assert submission["id"].notna().all(), "NaN ids in submission"
assert submission["score"].notna().all(), "NaN scores in submission"

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Columns:", submission.columns.tolist())

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1358525066.py in <cell line: 0>()
----> 1 preds = trainer.predict(eval_ds).predictions
      2 preds = np.asarray(preds, dtype=np.float32).reshape(-1)
      3 preds = np.clip(preds, 0, 1)
      4 
      5 score = np.where(

NameError: name 'trainer' is not defined
