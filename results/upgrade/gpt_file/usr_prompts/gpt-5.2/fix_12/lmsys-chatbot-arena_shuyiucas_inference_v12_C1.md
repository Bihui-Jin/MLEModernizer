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

3.14

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
peft==0.16.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1
transformers==4.53.3

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

1.3105775659942334

# 6. Current score

1.09861

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.09861) has done: 'I fix the two root runtime blockers so the notebook can load a local HF model/LoRA on Kaggle without triggering HuggingFace repo-id validation and without crashing on protobuf. Specifically, I (1) remove the protobuf env override that causes the `MessageFactory.GetPrototype` error, and (2) make model/adapter path resolution robust by searching common Kaggle input locations and passing an explicit local directory that contains `config.json`. After the model loads, the rest of your scoring and probability mapping logic stays unchanged; I only add a safe fallback to produce a valid submission (uniform probabilities) if the model files are not present so a `.csv` is always written. This is score-neutral when the model loads correctly, and it unblocks end-to-end execution and submission creation.'
- What this solution (achieved 1.09861) has done: 'I fix the immediate runtime blocker (`MessageFactory.GetPrototype`), which is caused by an incompatible protobuf backend being imported by the Transformers stack in this Kaggle image. The minimal, score-neutral fix is to force the pure-Python protobuf implementation *before* importing `transformers`/`peft`, then restart the imports cleanly. I also add a tiny safety guard so `get_score()` is never called when the model/tokenizer failed to load (to ensure a valid `submission.csv` is always written). No model architecture, inference logic, or probability mapping is changed, so your score behavior should remain consistent with the current approach.'
- What this solution (achieved 1.09861) has done: 'I fix the protobuf/transformers crash by removing the forced pure-Python protobuf setting (which triggers the `MessageFactory.GetPrototype` mismatch in this environment) and instead forcing the faster `upb` backend before importing `transformers`/`peft`. I keep your model/LoRA loading and scoring logic unchanged, only adjusting the import order and environment variables so the notebook runs end-to-end and actually uses the model (avoiding the uniform-probability fallback that would worsen score). I also add a small, score-neutral safety normalization to ensure probabilities are finite and properly normalized for every row. The output still be a valid `submission.csv` with the required columns.'
- What this solution (achieved 1.09861) has done: 'I fix the protobuf crash that prevents any submission from being generated by ensuring the compatible protobuf backend is selected before importing `transformers`/`peft`, and by removing the conflicting protobuf env toggles that lead to `MessageFactory.GetPrototype` errors in this Kaggle image. I keep your model/LoRA loading and inference logic unchanged, only adjusting import order and adding a safe fallback so `submission.csv` is always written. Since your current score (1.09861, lower is better) is already better than the target (1.3106), I not make score-improving changes; the patch is intended to be score-neutral aside from restoring the ability to run end-to-end with the actual model when available. The output remain in the required format with probabilities normalized per row.'
- What this solution (achieved 1.09861) has done: 'I fix the protobuf/Transformers import crash by removing the forced pure-Python protobuf override and instead forcing the compatible `upb` backend before importing `transformers`/`peft`. This is the direct root cause of the `MessageFactory.GetPrototype` AttributeError and is score-neutral (it only changes backend selection so the same model inference can run). I keep your model/LoRA loading, scoring, and probability mapping unchanged, and keep the existing fallback that writes uniform probabilities if the model cannot be loaded. The result run end-to-end and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 1.09861) has done: 'I fix the protobuf crash by removing the unsupported `upb` override (it triggers the `MessageFactory.GetPrototype` error) and instead forcing the safe pure-Python protobuf implementation before importing `transformers/peft`. This is a runtime-only change and does not alter your model/inference/probability mapping logic, so it should be score-neutral aside from actually allowing the model to load and run. I also renumber the notebook cells to start at 1 (your format currently starts at cell 0) and keep the existing fallback that writes a valid `submission.csv` if the model cannot be loaded. No changes are made to the scoring mapping (`calculate_probs`) to avoid moving your already-better-than-target score.'
- What this solution (achieved 1.09861) has done: 'I fix the immediate runtime crash (`AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`) by removing the forced pure-Python protobuf override that triggers an incompatible protobuf backend in this Kaggle image, and instead forcing the default/fast C++ (`upb`) backend before importing Transformers/PEFT. This is a runtime-only change (no model/inference/probability logic changes), so it should keep your score behavior essentially the same while allowing the model to actually load and run (avoiding the uniform-probability fallback that would hurt score). I also renumber the notebook cells to start at 1 to match the required format, keeping paths and core logic intact. The script still always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 1.09861) has done: 'I fix the runtime crash caused by forcing an unsupported protobuf backend: `upb` is not a valid value for `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` and triggers the `MessageFactory.GetPrototype` error when Transformers imports protobuf. The minimal, score-neutral fix is to stop overriding protobuf (and optionally force the safe `cpp` implementation) *before* importing `transformers/peft`, keeping your model/LoRA loading and inference logic unchanged. I also renumber the cells to start at 1 (to match the required format) without changing computation. This should restore end-to-end execution and allow the model to load, preserving your current score behavior (already better than target) rather than falling back to uniform probabilities.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import logging as transformers_logging
from peft import PeftModel

transformers_logging.set_verbosity_error()

model_id = "/kaggle/input/qwen2.5/transformers/3b/1"
adapter_id = "/kaggle/input/qwen2-5-3b-lora/qwen25_arena_rm/qwen25_arena_rm"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _find_dir_with_file(
    base_path: str, filename: str, max_depth: int = 4
) -> str | None:
    """Search for a directory under base_path that contains filename (bounded depth)."""
    if not base_path or not os.path.exists(base_path):
        return None
    if os.path.isdir(base_path) and os.path.isfile(os.path.join(base_path, filename)):
        return base_path

    base_path = os.path.abspath(base_path)
    base_depth = base_path.rstrip(os.sep).count(os.sep)

    for root, dirs, files in os.walk(base_path):
        depth = root.rstrip(os.sep).count(os.sep) - base_depth
        if depth > max_depth:
            dirs[:] = []
            continue
        if filename in files:
            return root
    return None


def _resolve_local_hf_dir(path: str, required_file: str = "config.json") -> str:
    """
    Return a local directory that contains required_file.
    This avoids HFValidationError by ensuring from_pretrained gets a real local folder.
    """
    hit = _find_dir_with_file(path, required_file, max_depth=4)
    if hit:
        return hit

    candidates = [
        path,
        os.path.join(path, "transformers"),
        os.path.join(path, "model"),
        os.path.join(path, "model", "transformers"),
        "/kaggle/input/" + path.strip("/").split("/")[-1] if path else path,
    ]
    for c in candidates:
        hit = _find_dir_with_file(c, required_file, max_depth=4)
        if hit:
            return hit

    base_name = os.path.basename(path.rstrip("/")) if path else ""
    if base_name:
        for prefix in [
            "/kaggle/input",
            "/kaggle/data",
            "/kaggle/input/lmsys-chatbot-arena",
        ]:
            if os.path.isdir(prefix):
                cand = os.path.join(prefix, base_name)
                hit = _find_dir_with_file(cand, required_file, max_depth=4)
                if hit:
                    return hit

    return path


model_dir = _resolve_local_hf_dir(model_id, required_file="config.json")
adapter_dir = _resolve_local_hf_dir(adapter_id, required_file="adapter_config.json")

print("DEVICE:", DEVICE)
print(
    "Resolved model_dir:",
    model_dir,
    "exists:",
    os.path.isdir(model_dir),
    "has config:",
    os.path.isfile(os.path.join(model_dir, "config.json")),
)
print(
    "Resolved adapter_dir:",
    adapter_dir,
    "exists:",
    os.path.isdir(adapter_dir),
    "has adapter_config:",
    os.path.isfile(os.path.join(adapter_dir, "adapter_config.json")),
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_56/1685276765.py in <cell line: 0>()
     16 from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
     17 from transformers import logging as transformers_logging
---> 18 from peft import PeftModel
     19 
     20 transformers_logging.set_verbosity_error()

/usr/local/lib/python3.11/dist-packages/peft/__init__.py in <module>
     15 __version__ = "0.16.0"
     16 
---> 17 from .auto import (
     18     MODEL_TYPE_TO_PEFT_MODEL_MAPPING,
     19     AutoPeftModel,

/usr/local/lib/python3.11/dist-packages/peft/auto.py in <module>
     29 )
     30 
---> 31 from .config import PeftConfig
     32 from .peft_model import (
     33     PeftModel,

/usr/local/lib/python3.11/dist-packages/peft/config.py in <module>
     22 from transformers.utils import PushToHubMixin, http_user_agent
     23 
---> 24 from .utils import CONFIG_NAME, PeftType, TaskType
     25 
     26 

/usr/local/lib/python3.11/dist-packages/peft/utils/__init__.py in <module>
     15 from .integrations import map_cache_to_layer_device_map
     16 from .loftq_utils import replace_lora_weights_loftq
---> 17 from .other import (
     18     CONFIG_NAME,
     19     INCLUDE_LINEAR_LAYERS_SHORTHAND,

/usr/local/lib/python3.11/dist-packages/peft/utils/other.py in <module>
     33 from packaging import version
     34 from safetensors.torch import storage_ptr, storage_size
---> 35 from transformers import PreTrainedModel
     36 
     37 from ..import_utils import is_auto_gptq_available, is_gptqmodel_available, is_torch_tpu_available

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
tokenizer = None
model = None

try:
    if not (
        os.path.isdir(model_dir)
        and os.path.isfile(os.path.join(model_dir, "config.json"))
    ):
        raise FileNotFoundError(f"Model directory missing config.json: {model_dir}")
    if not (
        os.path.isdir(adapter_dir)
        and os.path.isfile(os.path.join(adapter_dir, "adapter_config.json"))
    ):
        raise FileNotFoundError(
            f"Adapter directory missing adapter_config.json: {adapter_dir}"
        )

    config = AutoConfig.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        config=config,
        trust_remote_code=True,
        local_files_only=True,
    )

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        config=config,
        num_labels=1,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True,
        local_files_only=True,
    )

    if not hasattr(base_model, "prepare_inputs_for_generation"):
        base_model.prepare_inputs_for_generation = lambda *args, **kwargs: None

    model = PeftModel.from_pretrained(base_model, adapter_dir, local_files_only=True)
    model.eval()

    if DEVICE != "cuda":
        model.to(DEVICE)

    print("Loaded model + adapter successfully.")
except Exception as e:
    print(
        "WARNING: Failed to load model/adapter; will create a valid fallback submission."
    )
    print("Load error:", repr(e))
    tokenizer = None
    model = None




## === cell 2
def calculate_probs(s_a, s_b, epsilon=0.6, temperature=0.7):
    """
    Difference-to-probability mapping with a tie component.
    Returns (winner_model_a, winner_model_b, winner_tie) that sum to 1.
    """
    diff = (s_a - s_b) / temperature

    p_a_raw = 1.0 / (1.0 + np.exp(-diff))
    p_b_raw = 1.0 - p_a_raw

    p_tie = np.exp(-np.abs(diff) / epsilon) * 0.25  # max tie probability at 0.25

    total = p_a_raw * (1.0 - p_tie) + p_b_raw * (1.0 - p_tie) + p_tie
    winner_a = (p_a_raw * (1.0 - p_tie)) / total
    winner_b = (p_b_raw * (1.0 - p_tie)) / total
    winner_tie = p_tie / total

    winner_a = float(np.clip(winner_a, 1e-8, 1.0))
    winner_b = float(np.clip(winner_b, 1e-8, 1.0))
    winner_tie = float(np.clip(winner_tie, 1e-8, 1.0))
    s = winner_a + winner_b + winner_tie
    winner_a, winner_b, winner_tie = winner_a / s, winner_b / s, winner_tie / s

    return winner_a, winner_b, winner_tie


@torch.no_grad()
def get_score(prompt, response, max_length=2048):
    if tokenizer is None or model is None:
        raise RuntimeError("Model/tokenizer not loaded; cannot score.")
    text = (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>"
    )
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    logits = model(**inputs).logits
    return float(logits[0].item())




## === cell 3
test_path = "/kaggle/input/lmsys-chatbot-arena/test.csv"
if not os.path.exists(test_path):
    test_path = "/kaggle/data/lmsys-chatbot-arena/test.csv"
if not os.path.exists(test_path):
    test_path = "/kaggle/data/test.csv"
if not os.path.exists(test_path):
    test_path = "/kaggle/input/test.csv"

test_df = pd.read_csv(test_path)

results = []

if (tokenizer is None) or (model is None):
    for _, row in test_df.iterrows():
        results.append(
            {
                "id": row["id"],
                "winner_model_a": 1.0 / 3.0,
                "winner_model_b": 1.0 / 3.0,
                "winner_tie": 1.0 / 3.0,
            }
        )
else:
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), disable=True):
        s_a = get_score(row["prompt"], row["response_a"])
        s_b = get_score(row["prompt"], row["response_b"])
        prob_a, prob_b, prob_tie = calculate_probs(s_a, s_b)
        results.append(
            {
                "id": row["id"],
                "winner_model_a": prob_a,
                "winner_model_b": prob_b,
                "winner_tie": prob_tie,
            }
        )

sub = pd.DataFrame(results)
sub = sub[["id", "winner_model_a", "winner_model_b", "winner_tie"]]

for c in ["winner_model_a", "winner_model_b", "winner_tie"]:
    sub[c] = pd.to_numeric(sub[c], errors="coerce").astype("float64")

sub[["winner_model_a", "winner_model_b", "winner_tie"]] = sub[
    ["winner_model_a", "winner_model_b", "winner_tie"]
].replace([np.inf, -np.inf], np.nan)

sub[["winner_model_a", "winner_model_b", "winner_tie"]] = sub[
    ["winner_model_a", "winner_model_b", "winner_tie"]
].fillna(1.0 / 3.0)

row_sums = sub[["winner_model_a", "winner_model_b", "winner_tie"]].sum(axis=1).values
row_sums = np.where(row_sums <= 0, 1.0, row_sums)

sub[["winner_model_a", "winner_model_b", "winner_tie"]] = sub[
    ["winner_model_a", "winner_model_b", "winner_tie"]
].div(row_sums, axis=0)

sub.to_csv("submission.csv", index=False)
print("finished, wrote submission.csv with shape:", sub.shape)
print(sub.head())
