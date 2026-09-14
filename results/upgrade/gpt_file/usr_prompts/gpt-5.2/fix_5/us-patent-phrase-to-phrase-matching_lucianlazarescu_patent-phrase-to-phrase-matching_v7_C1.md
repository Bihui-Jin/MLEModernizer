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

0.7853653726936117

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import random
import numpy as np
import pandas as pd
import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


seed_everything(42)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/2400882946.py in <cell line: 0>()
     14 
     15 from transformers import AutoModelForSequenceClassification, AutoTokenizer
---> 16 from transformers import TrainingArguments, Trainer
     17 
     18 

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
    return np.corrcoef(x, y)[0][1]




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
_ = train_data.target.value_counts().head()
_




## === cell 5
_ = train_data.anchor.value_counts().head()
_




## === cell 6
_ = train_data.score.describe()
_




## === cell 7
train_data["section"] = train_data.context.str[0]
_ = train_data.section.value_counts()
_.head()




## === cell 8
train_data["sectok"] = "[" + train_data.section + "]"
sectoks = list(train_data.sectok.unique())
sectoks[:10], len(sectoks)




## === cell 9
test_data = pd.read_csv(path + "test.csv")
print(test_data.head())
print(test_data.shape)




## === cell 10
test_data["section"] = test_data.context.str[0]
_ = test_data.section.value_counts()
_.head()




## === cell 11
model_nm = "microsoft/deberta-v3-small"

tokenizer = AutoTokenizer.from_pretrained(model_nm, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/3026088090.py in <cell line: 0>()
      1 model_nm = "microsoft/deberta-v3-small"
      2 
----> 3 tokenizer = AutoTokenizer.from_pretrained(model_nm, use_fast=True)
      4 model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)
      5 

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/tokenization_auto.py in from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs)
   1067 
   1068             if tokenizer_class_fast and (use_fast or tokenizer_class_py is None):
-> 1069                 return tokenizer_class_fast.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
   1070             else:
   1071                 if tokenizer_class_py is not None:

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in from_pretrained(cls, pretrained_model_name_or_path, cache_dir, force_download, local_files_only, token, revision, trust_remote_code, *init_inputs, **kwargs)
   2012                 logger.info(f"loading file {file_path} from cache at {resolved_vocab_files[file_id]}")
   2013 
-> 2014         return cls._from_pretrained(
   2015             resolved_vocab_files,
   2016             pretrained_model_name_or_path,

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in _from_pretrained(cls, resolved_vocab_files, pretrained_model_name_or_path, init_configuration, token, cache_dir, local_files_only, _commit_hash, _is_local, trust_remote_code, *init_inputs, **kwargs)
   2258         # Instantiate the tokenizer.
   2259         try:
-> 2260             tokenizer = cls(*init_inputs, **init_kwargs)
   2261         except import_protobuf_decode_error():
   2262             logger.info(

/usr/local/lib/python3.11/dist-packages/transformers/models/deberta_v2/tokenization_deberta_v2_fast.py in __init__(self, vocab_file, tokenizer_file, do_lower_case, split_by_punct, bos_token, eos_token, unk_token, sep_token, pad_token, cls_token, mask_token, **kwargs)
    101         **kwargs,
    102     ) -> None:
--> 103         super().__init__(
    104             vocab_file,
    105             tokenizer_file=tokenizer_file,

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_fast.py in __init__(self, *args, **kwargs)
    118         elif slow_tokenizer:
    119             # We need to convert a slow tokenizer to build the backend
--> 120             fast_tokenizer = convert_slow_tokenizer(slow_tokenizer)
    121         elif gguf_file is not None:
    122             # We need to convert a slow tokenizer to build the backend

/usr/local/lib/python3.11/dist-packages/transformers/convert_slow_tokenizer.py in convert_slow_tokenizer(transformer_tokenizer, from_tiktoken)
   1727     if tokenizer_class_name in SLOW_TO_FAST_CONVERTERS and not from_tiktoken:
   1728         converter_class = SLOW_TO_FAST_CONVERTERS[tokenizer_class_name]
-> 1729         return converter_class(transformer_tokenizer).converted()
   1730 
   1731     else:

/usr/local/lib/python3.11/dist-packages/transformers/convert_slow_tokenizer.py in __init__(self, *args)
    554 
    555         # from .utils import sentencepiece_model_pb2 as model_pb2
--> 556         model_pb2 = import_protobuf()
    557 
    558         m = model_pb2.ModelProto()

/usr/local/lib/python3.11/dist-packages/transformers/convert_slow_tokenizer.py in import_protobuf(error_message)
     35 def import_protobuf(error_message=""):
     36     if is_sentencepiece_available():
---> 37         from sentencepiece import sentencepiece_model_pb2
     38 
     39         return sentencepiece_model_pb2

/usr/local/lib/python3.11/dist-packages/sentencepiece/sentencepiece_model_pb2.py in <module>
      3 # source: sentencepiece_model.proto
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

## === cell 12
sep = " [s] "




## === cell 13
def prepare_data(df: pd.DataFrame) -> pd.Series:
    out = (
        df["sectok"]
        + sep
        + df["context"].astype(str)
        + sep
        + df["anchor"].astype(str).str.lower()
        + sep
        + df["target"].astype(str)
    )
    return out




## === cell 14
tokenizer.add_special_tokens({"additional_special_tokens": sectoks})
train_data["input"] = prepare_data(train_data)




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1994742280.py in <cell line: 0>()
----> 1 tokenizer.add_special_tokens({"additional_special_tokens": sectoks})
      2 train_data["input"] = prepare_data(train_data)
      3 
      4 

NameError: name 'tokenizer' is not defined

## === cell 15
model.resize_token_embeddings(len(tokenizer))




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/629679691.py in <cell line: 0>()
----> 1 model.resize_token_embeddings(len(tokenizer))
      2 
      3 

NameError: name 'model' is not defined

## === cell 16
def tok_func_texts(texts):
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=128,
    )




## === cell 17
from torch.utils.data import Dataset


class PatentPairsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, has_labels: bool = True):
        self.df = df.reset_index(drop=True)
        self.has_labels = has_labels

        texts = self.df["input"].astype(str).tolist()
        enc = tok_func_texts(texts)

        input_ids = np.asarray(enc["input_ids"], dtype=np.int64)
        attention_mask = np.asarray(enc["attention_mask"], dtype=np.int64)

        self.input_ids = torch.from_numpy(input_ids)
        self.attention_mask = torch.from_numpy(attention_mask)

        self.token_type_ids = None
        if "token_type_ids" in enc:
            token_type_ids = np.asarray(enc["token_type_ids"], dtype=np.int64)
            self.token_type_ids = torch.from_numpy(token_type_ids)

        self.labels = None
        if has_labels:
            labels = self.df["score"].to_numpy(dtype=np.float32, copy=False)
            self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
        if self.token_type_ids is not None:
            item["token_type_ids"] = self.token_type_ids[idx]
        if self.has_labels:
            item["labels"] = self.labels[idx]
        return item




## === cell 18
tmp_ds = PatentPairsDataset(train_data.iloc[:5], tokenizer, has_labels=True)
for i in range(2):
    print({k: (v.shape, v.dtype) for k, v in tmp_ds[i].items()})




## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3646034495.py in <cell line: 0>()
----> 1 tmp_ds = PatentPairsDataset(train_data.iloc[:5], tokenizer, has_labels=True)
      2 for i in range(2):
      3     print({k: (v.shape, v.dtype) for k, v in tmp_ds[i].items()})
      4 
      5 

NameError: name 'tokenizer' is not defined

## === cell 19
val_frac = 0.25
train_df = train_data.sample(frac=1.0, random_state=42).reset_index(drop=True)
n_val = int(len(train_df) * val_frac)
valid_df = train_df.iloc[:n_val].reset_index(drop=True)
train_df2 = train_df.iloc[n_val:].reset_index(drop=True)

train_ds = PatentPairsDataset(train_df2, tokenizer, has_labels=True)
valid_ds = PatentPairsDataset(valid_df, tokenizer, has_labels=True)

len(train_ds), len(valid_ds)




## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3158329363.py in <cell line: 0>()
      5 train_df2 = train_df.iloc[n_val:].reset_index(drop=True)
      6 
----> 7 train_ds = PatentPairsDataset(train_df2, tokenizer, has_labels=True)
      8 valid_ds = PatentPairsDataset(valid_df, tokenizer, has_labels=True)
      9 

NameError: name 'tokenizer' is not defined

## === cell 20
eval_df = test_data.copy()
eval_df["sectok"] = "[" + eval_df["section"] + "]"
eval_df["input"] = prepare_data(eval_df)
eval_ds = PatentPairsDataset(eval_df, tokenizer, has_labels=False)




## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3873566521.py in <cell line: 0>()
      2 eval_df["sectok"] = "[" + eval_df["section"] + "]"
      3 eval_df["input"] = prepare_data(eval_df)
----> 4 eval_ds = PatentPairsDataset(eval_df, tokenizer, has_labels=False)
      5 
      6 

NameError: name 'tokenizer' is not defined

## === cell 21
bs = 32
epochs = 4
lr = 8e-5




## === cell 22
use_fp16 = torch.cuda.is_available()

args = TrainingArguments(
    output_dir="outputs",
    learning_rate=lr,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    fp16=use_fp16,
    eval_strategy="no",  # was "epoch"
    per_device_train_batch_size=bs,
    per_device_eval_batch_size=bs * 2,
    num_train_epochs=epochs,
    weight_decay=0.01,
    report_to="none",
    save_strategy="no",
    logging_strategy="steps",
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,  # kept for post-train evaluation
    tokenizer=tokenizer,
    compute_metrics=corr_d,
)




## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1040389386.py in <cell line: 0>()
      4 # which is often the biggest time cost after training itself. Evaluating once at the end preserves
      5 # training logic/epochs and final model state; it only removes repeated metric computation.
----> 6 args = TrainingArguments(
      7     output_dir="outputs",
      8     learning_rate=lr,

NameError: name 'TrainingArguments' is not defined

## === cell 23
trainer.train()




## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/134302066.py in <cell line: 0>()
----> 1 trainer.train()
      2 
      3 

NameError: name 'trainer' is not defined

## === cell 24
_ = trainer.evaluate(eval_dataset=valid_ds)




## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1995679588.py in <cell line: 0>()
      1 # SPEED: compute the same Pearson metric once after training (same dataset/metric), instead of every epoch.
----> 2 _ = trainer.evaluate(eval_dataset=valid_ds)
      3 
      4 

NameError: name 'trainer' is not defined

## === cell 25
preds = trainer.predict(eval_ds).predictions.astype(float).reshape(-1)
preds = np.clip(preds, 0, 1)




## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2904832045.py in <cell line: 0>()
----> 1 preds = trainer.predict(eval_ds).predictions.astype(float).reshape(-1)
      2 preds = np.clip(preds, 0, 1)
      3 
      4 

NameError: name 'trainer' is not defined

## === cell 26
score = np.zeros_like(preds, dtype=float)
score[preds >= 0.125] = 0.25
score[preds >= 0.375] = 0.50
score[preds >= 0.625] = 0.75
score[preds >= 0.875] = 1.00




## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3434218438.py in <cell line: 0>()
----> 1 score = np.zeros_like(preds, dtype=float)
      2 score[preds >= 0.125] = 0.25
      3 score[preds >= 0.375] = 0.50
      4 score[preds >= 0.625] = 0.75
      5 score[preds >= 0.875] = 1.00

NameError: name 'preds' is not defined

## === cell 27
score = np.asarray(score, dtype=float).reshape(-1)
assert len(score) == len(eval_df), (len(score), len(eval_df))

submission = pd.DataFrame({"id": eval_df["id"].values, "score": score})
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("submission.csv path:", os.path.abspath("submission.csv"))

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1961967216.py in <cell line: 0>()
----> 1 score = np.asarray(score, dtype=float).reshape(-1)
      2 assert len(score) == len(eval_df), (len(score), len(eval_df))
      3 
      4 submission = pd.DataFrame({"id": eval_df["id"].values, "score": score})
      5 submission.to_csv("submission.csv", index=False)

NameError: name 'score' is not defined
