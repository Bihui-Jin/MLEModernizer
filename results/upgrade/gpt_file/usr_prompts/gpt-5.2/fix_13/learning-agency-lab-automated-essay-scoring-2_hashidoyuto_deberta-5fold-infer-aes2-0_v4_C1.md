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

# 8. Previous improvement plans

- What this solution (achieved -0.01867) has done: 'The import crash comes from `transformers` trying to import TensorFlow and protobuf C-extensions that aren’t compatible in this environment; we prevent TF/JAX imports and force pure-Python protobuf before importing `transformers`. `DataCollatorWithPadding` is also triggering the problematic `transformers.data` import path, so we replace it with a tiny local padding collator that keeps the same batching semantics. We also make tokenizer loading robust by preferring local fold directories (if present) and otherwise falling back to a model that can load without sentencepiece conversion, so inference completes. Finally, we ensure `submission.csv` is always written with the required `essay_id,score` columns.'
- What this solution (achieved -0.01867) has done: 'We fix the `protobuf`/`transformers` import crash that triggers `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` by forcing the pure-Python protobuf implementation *before any protobuf/transformers import* and by ensuring TensorFlow/JAX are disabled early. Then we keep your existing inference logic unchanged, but add a robust fallback that avoids loading a broken model directory and guarantees we still produce a valid `submission.csv`. Finally, we keep the output formatting aligned with the competition (integer scores 1–6 with `essay_id,score` columns), so you always get a valid submission file.'
- What this solution (achieved -0.01867) has done: 'We fix the `protobuf`/`transformers` crash by forcing the pure-Python protobuf implementation *and* disabling TF/JAX/Flax before any `transformers` import, and by proactively removing any already-imported `google.protobuf` modules that may have been loaded with the wrong backend. We also make the model loading robust by falling back to a safe, locally-available HF model if a fold directory fails to load (so the pipeline always completes and writes `submission.csv`). To move the score up toward the target, we keep the same inference logic but prefer the provided DeBERTa fold checkpoints when they can be loaded; the fallback only triggers if loading fails. Finally, we ensure the submission has the required `essay_id,score` columns and scores are clipped to 1–6.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import sys
import types

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        sys.modules.pop(m, None)

for _mod in ("tensorflow", "jax", "jaxlib", "flax"):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)



## === cell 1
import random
import glob
import warnings
import numpy as np
import pandas as pd
import torch

warnings.simplefilter("ignore")

from transformers import AutoTokenizer, AutoModelForSequenceClassification




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/801666952.py in <cell line: 0>()
      8 warnings.simplefilter("ignore")
      9 
---> 10 from transformers import AutoTokenizer, AutoModelForSequenceClassification
     11 
     12 

/usr/local/lib/python3.11/dist-packages/transformers/__init__.py in <module>
     25 
     26 # Check the dependencies satisfy the minimal versions required.
---> 27 from . import dependency_versions_check
     28 from .utils import (
     29     OptionalDependencyNotAvailable,

/usr/local/lib/python3.11/dist-packages/transformers/dependency_versions_check.py in <module>
     14 
     15 from .dependency_versions_table import deps
---> 16 from .utils.versions import require_version, require_version_core
     17 
     18 

/usr/local/lib/python3.11/dist-packages/transformers/utils/__init__.py in <module>
     22 
     23 from .. import __version__
---> 24 from .args_doc import (
     25     ClassAttrs,
     26     ClassDocstring,

/usr/local/lib/python3.11/dist-packages/transformers/utils/args_doc.py in <module>
     28     _prepare_output_docstrings,
     29 )
---> 30 from .generic import ModelOutput
     31 
     32 

/usr/local/lib/python3.11/dist-packages/transformers/utils/generic.py in <module>
     32 from packaging import version
     33 
---> 34 from .import_utils import (
     35     get_torch_version,
     36     is_flax_available,

/usr/local/lib/python3.11/dist-packages/transformers/utils/import_utils.py in <module>
    250         # Note: _is_package_available("tensorflow") fails for tensorflow-cpu. Please test any changes to the line below
    251         # with tensorflow-cpu to make sure it still works!
--> 252         _tf_available = importlib.util.find_spec("tensorflow") is not None
    253         if _tf_available:
    254             candidates = (

/usr/lib/python3.11/importlib/util.py in find_spec(name, package)

ValueError: tensorflow.__spec__ is None

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
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
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
    model_paths = ["distilbert-base-uncased"]

model_paths



## === cell 7
from torch.utils.data import DataLoader, Dataset

use_fp16 = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    tokenize = Tokenize(test, model_paths[0])
    tokenized_test, tokenizer = tokenize()
except Exception as e:
    print(
        f"[WARN] Tokenizer load failed for '{model_paths[0]}': {type(e).__name__}: {e}"
    )
    safe_tok = "distilbert-base-uncased"
    tokenize = Tokenize(test, safe_tok)
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


class SimplePadCollator:
    def __init__(self, tokenizer):
        self.tok = tokenizer

    def __call__(self, features):
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attention_mask = [
            torch.tensor(f["attention_mask"], dtype=torch.long) for f in features
        ]

        pad_id = self.tok.pad_token_id
        if pad_id is None:
            pad_id = 0

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=pad_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        return {"input_ids": input_ids, "attention_mask": attention_mask}


per_device_eval_bs = 32 if torch.cuda.is_available() else 8
data_collator = SimplePadCollator(tokenizer=tokenizer)

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
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=CFG.num_labels
        ).to(device)
    except Exception as e:
        print(f"[WARN] Model load failed for '{model_path}': {type(e).__name__}: {e}")
        continue

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
    fallback_model_name = "distilbert-base-uncased"
    print(
        f"[WARN] All provided models failed; falling back to '{fallback_model_name}'."
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        fallback_model_name, num_labels=CFG.num_labels
    ).to(device)
    final_pred_logits = predict_logits(model, dl).astype(np.float64, copy=False)
    n_models = 1
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

n_models, final_pred_logits.shape



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/923195273.py in <cell line: 0>()
      7 try:
----> 8     tokenize = Tokenize(test, model_paths[0])
      9     tokenized_test, tokenizer = tokenize()

/tmp/ipykernel_11/1905866603.py in __init__(self, test, model_path)
      3         # Keep logic: use_fast=False as provided (matches original semantics).
----> 4         self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
      5         self.test = test

NameError: name 'AutoTokenizer' is not defined

During handling of the above exception, another exception occurred:

NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/923195273.py in <cell line: 0>()
     13     )
     14     safe_tok = "distilbert-base-uncased"
---> 15     tokenize = Tokenize(test, safe_tok)
     16     tokenized_test, tokenizer = tokenize()
     17 

/tmp/ipykernel_11/1905866603.py in __init__(self, test, model_path)
      2     def __init__(self, test, model_path):
      3         # Keep logic: use_fast=False as provided (matches original semantics).
----> 4         self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
      5         self.test = test
      6 

NameError: name 'AutoTokenizer' is not defined

## === cell 8
final_pred_logits /= n_models
final_pred = final_pred_logits.argmax(axis=1) + 1  # class 0-5 -> score 1-6
final_pred = np.clip(final_pred, 1, 6)

submission = pd.DataFrame(
    {"essay_id": test["essay_id"].values, "score": final_pred.astype(int)}
)
submission.to_csv("submission.csv", index=False)

submission.shape, submission.dtypes, submission.head()

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/193418455.py in <cell line: 0>()
----> 1 final_pred_logits /= n_models
      2 final_pred = final_pred_logits.argmax(axis=1) + 1  # class 0-5 -> score 1-6
      3 final_pred = np.clip(final_pred, 1, 6)
      4 
      5 submission = pd.DataFrame(

NameError: name 'final_pred_logits' is not defined
