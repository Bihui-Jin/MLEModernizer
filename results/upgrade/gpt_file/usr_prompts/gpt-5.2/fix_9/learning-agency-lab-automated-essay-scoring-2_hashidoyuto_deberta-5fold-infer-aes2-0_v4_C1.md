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
Predict the score of student essays.

## Metric
Quadratic weighted kappa.

## Submission Format
For each `essay_id` in the test set, you must predict the corresponding `score` (between 1-6, see [rubric](https://storage.googleapis.com/kaggle-forum-message-attachments/2733927/20538/Rubric_%20Holistic%20Essay%20Scoring.pdf) for more details). The file should contain a header and have the following format:

```
essay_id,score
000d118,3
000fe60,3
001ab80,4
...
```

## Dataset
- **train.csv** - Essays and scores to be used as training data.
    - `essay_id` - The unique ID of the essay
    - `full_text` - The full essay response
    - `score` - Holistic score of the essay on a 1-6 scale
- **test.csv** - The essays to be used as test data. Contains the same fields as `train.csv`, aside from exclusion of `score`.
- **sample_submission.csv** - A submission file in the correct format.
    - `essay_id` - The unique ID of the essay
    - `score` - The predicted holistic score of the essay on a 1-6 scale

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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow-datasets==4.9.9
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
transformers==4.53.3
vega-datasets==0.9.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        input/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        working/
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
```

-> data/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/learning-agency-lab-automated-essay-scoring-2/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/learning-agency-lab-automated-essay-scoring-2/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> data/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> (stopped after 10 files for performance)

# 5. Target score

0.8053301059475746

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")




## === cell 1
import random
import glob
import warnings
import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding

warnings.simplefilter("ignore")




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/3760248102.py in <cell line: 0>()
      7 
      8 from transformers import AutoTokenizer, AutoModelForSequenceClassification
----> 9 from transformers import DataCollatorWithPadding
     10 
     11 warnings.simplefilter("ignore")

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

/usr/local/lib/python3.11/dist-packages/transformers/data/__init__.py in <module>
     27 )
     28 from .metrics import glue_compute_metrics, xnli_compute_metrics
---> 29 from .processors import (
     30     DataProcessor,
     31     InputExample,

/usr/local/lib/python3.11/dist-packages/transformers/data/processors/__init__.py in <module>
     13 # limitations under the License.
     14 
---> 15 from .glue import glue_convert_examples_to_features, glue_output_modes, glue_processors, glue_tasks_num_labels
     16 from .squad import SquadExample, SquadFeatures, SquadV1Processor, SquadV2Processor, squad_convert_examples_to_features
     17 from .utils import DataProcessor, InputExample, InputFeatures, SingleSentenceClassificationProcessor

/usr/local/lib/python3.11/dist-packages/transformers/data/processors/glue.py in <module>
     28 
     29 if is_tf_available():
---> 30     import tensorflow as tf
     31 
     32 logger = logging.get_logger(__name__)

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
class PATHS:
    test_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
    test_path_alt = (
        "/kaggle/data/learning-agency-lab-automated-essay-scoring-2/test.csv"
    )
    model_dir = "/kaggle/input/groupkfold-deberta-aes2-0/"




## === cell 3
class CFG:
    max_length = 512
    num_labels = 6




## === cell 4
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


seed_everything(42)




## === cell 5
class Tokenize(object):
    def __init__(self, test, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.test = test

    def __call__(self):
        texts = self.test["full_text"].tolist()
        enc = self.tokenizer(
            texts,
            truncation=True,
            max_length=CFG.max_length,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors=None,
            padding=False,
            batch_size=2048,  # uses fast tokenizer batching; does not change results
        )
        return enc, self.tokenizer




## === cell 6
if os.path.exists(PATHS.test_path):
    test = pd.read_csv(PATHS.test_path)
elif os.path.exists(PATHS.test_path_alt):
    test = pd.read_csv(PATHS.test_path_alt)
else:
    raise FileNotFoundError(
        f"Could not find test.csv at {PATHS.test_path} or {PATHS.test_path_alt}"
    )

model_paths = []
if os.path.exists(PATHS.model_dir):
    cand = glob.glob(os.path.join(PATHS.model_dir, "*fold*"))
    cand = [p for p in cand if os.path.isdir(p)]
    cand.sort()
    model_paths = cand

if len(model_paths) == 0:
    fallback_model = "microsoft/deberta-v3-base"
    model_paths = [fallback_model]

model_paths




## === cell 7
use_fp16 = torch.cuda.is_available()




## === cell 8
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenize = Tokenize(test, model_paths[0])
tokenized_test, tokenizer = tokenize()


class EncodedDataset(Dataset):
    def __init__(self, encodings):
        self.enc = encodings
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.n = len(self.input_ids)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


per_device_eval_bs = 32 if torch.cuda.is_available() else 8
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

num_workers = 4 if torch.cuda.is_available() else 0
dl = DataLoader(
    EncodedDataset(tokenized_test),
    batch_size=per_device_eval_bs,
    shuffle=False,
    collate_fn=data_collator,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)


def predict_logits(model, dataloader):
    model.eval()
    all_logits = []
    with torch.inference_mode():
        for batch in dataloader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            if use_fp16 and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(**batch)
            else:
                out = model(**batch)
            all_logits.append(out.logits.detach().cpu().numpy())
    return np.concatenate(all_logits, axis=0)


n_test = len(test)
final_pred_logits = None
n_models = 0

for i, model_path in enumerate(model_paths):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=CFG.num_labels
    ).to(device)

    pre_preds = predict_logits(model, dl)  # (n_test, num_labels)

    if final_pred_logits is None:
        if (
            pre_preds.ndim != 2
            or pre_preds.shape[0] != n_test
            or pre_preds.shape[1] != CFG.num_labels
        ):
            raise ValueError(
                f"Unexpected prediction shape at model {i}: {pre_preds.shape}"
            )
        final_pred_logits = np.zeros_like(pre_preds, dtype=np.float64)

    final_pred_logits += pre_preds.astype(np.float64, copy=False)
    n_models += 1

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if n_models == 0 or final_pred_logits is None:
    raise RuntimeError(
        "No predictions were produced; check model_paths and inference pipeline."
    )

n_models, final_pred_logits.shape




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/2078234497.py in <cell line: 0>()
      3 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      4 
----> 5 tokenize = Tokenize(test, model_paths[0])
      6 tokenized_test, tokenizer = tokenize()
      7 

/tmp/ipykernel_11/3120516793.py in __init__(self, test, model_path)
      1 class Tokenize(object):
      2     def __init__(self, test, model_path):
----> 3         self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
      4         self.test = test
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

## === cell 9
final_pred_logits /= n_models
final_pred = final_pred_logits.argmax(axis=1) + 1  # class 0-5 -> score 1-6

submission = pd.DataFrame(
    {"essay_id": test["essay_id"].values, "score": final_pred.astype(int)}
)
submission.to_csv("submission.csv", index=False)

submission.shape, submission.dtypes, submission.head()

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/705188068.py in <cell line: 0>()
----> 1 final_pred_logits /= n_models
      2 final_pred = final_pred_logits.argmax(axis=1) + 1  # class 0-5 -> score 1-6
      3 
      4 submission = pd.DataFrame(
      5     {"essay_id": test["essay_id"].values, "score": final_pred.astype(int)}

NameError: name 'final_pred_logits' is not defined
