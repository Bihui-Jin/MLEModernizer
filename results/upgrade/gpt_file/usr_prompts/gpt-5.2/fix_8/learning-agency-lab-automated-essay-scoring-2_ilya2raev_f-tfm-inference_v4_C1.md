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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
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

0.8021729294158836

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -0.01486) has done: 'I fix the protobuf-related import crash by forcing the pure-Python protobuf runtime before importing `transformers/datasets`, which avoids the `MessageFactory.GetPrototype` error in this environment. Next, I prevent the fallback DistilBERT path from crashing by capping `MAX_LENGTH` to the tokenizer/model’s maximum supported length (512 for DistilBERT), eliminating the tensor size mismatch. I also make inference deterministic/stable and memory-safe by enabling `torch.inference_mode()` and `fp16` only when on CUDA (score-neutral). Finally, I ensure `preds` is always defined and that a valid `submission.csv` is written with the required `essay_id,score` columns.'
- What this solution (achieved -0.01867) has done: 'I fix the protobuf import crash by forcing the pure-Python protobuf runtime *and* ensuring it is applied early enough, plus applying a small compatibility patch for the `MessageFactory.GetPrototype` attribute that `transformers/datasets` can hit in some Kaggle images. I also make the fallback model path use the correct head (6 labels) so it doesn’t output a 2-class classification head by default, which is the main reason your QWK score is extremely low. Finally, I keep your core inference/ensembling logic intact while ensuring the submission is generated deterministically and with valid 1–6 integer scores.'
- What this solution (achieved -0.03141) has done: 'I fix the protobuf compatibility patch so it doesn’t crash on import in this Kaggle image by applying the `MessageFactory.GetPrototype` shim at the *instance* level as well (some builds expose it only per-instance), and by doing it before importing `datasets/transformers`. Then I keep your model/inference logic intact, but correct the post-processing for the 6-class classification head: instead of `argmax` over logits, convert logits to an expected score via softmax-weighted mean (a minimal calibration-aligned change that typically improves QWK for ordinal labels). Finally, I keep the same submission merge/alignment logic and ensure `submission.csv` is always written with valid 1–6 integer scores.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from google.protobuf.message_factory import MessageFactory  # type: ignore

    if hasattr(MessageFactory, "GetMessageClass") and not hasattr(
        MessageFactory, "GetPrototype"
    ):

        def _shim_getprototype(self, descriptor):  # type: ignore
            return self.GetMessageClass(descriptor)

        MessageFactory.GetPrototype = _shim_getprototype  # type: ignore[attr-defined]

        try:
            _mf = MessageFactory()
            if not hasattr(_mf, "GetPrototype"):
                setattr(_mf, "GetPrototype", _shim_getprototype.__get__(_mf, MessageFactory))  # type: ignore
        except Exception:
            pass
except Exception:
    pass

import random
import numpy as np
import pandas as pd

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
MODEL_ROOT = "/kaggle/input/f-tfm-small"
MAX_LENGTH = 3072
BATCH_SIZE = 2
N_FOLDS = 5

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




## === cell 2
def _resolve_competition_dir() -> str:
    candidates = [
        "/kaggle/input/learning-agency-lab-automated-essay-scoring-2",
        "/kaggle/data/learning-agency-lab-automated-essay-scoring-2",
        "/kaggle/input",
        "/kaggle/data",
    ]
    for base in candidates:
        if (
            os.path.isdir(base)
            and os.path.exists(os.path.join(base, "train.csv"))
            and os.path.exists(os.path.join(base, "test.csv"))
        ):
            return base

    for base in ["/kaggle/input", "/kaggle/data"]:
        if os.path.isdir(base):
            for name in os.listdir(base):
                p = os.path.join(base, name)
                if (
                    os.path.isdir(p)
                    and os.path.exists(os.path.join(p, "train.csv"))
                    and os.path.exists(os.path.join(p, "test.csv"))
                    and os.path.exists(os.path.join(p, "sample_submission.csv"))
                ):
                    return p

    raise FileNotFoundError(
        "Could not locate competition dataset directory with train.csv and test.csv under /kaggle/input or /kaggle/data."
    )


COMP_DIR = _resolve_competition_dir()

df_train = pd.read_csv(os.path.join(COMP_DIR, "train.csv"))
df_test = pd.read_csv(os.path.join(COMP_DIR, "test.csv"))
sample_submission = pd.read_csv(os.path.join(COMP_DIR, "sample_submission.csv"))

assert {"essay_id", "full_text"}.issubset(df_test.columns)
assert {"essay_id", "full_text", "score"}.issubset(df_train.columns)

print("COMP_DIR:", COMP_DIR)
print(
    "train:", df_train.shape, "test:", df_test.shape, "sample:", sample_submission.shape
)




## === cell 3
def _resolve_model_root(model_root: str) -> str:
    if os.path.isdir(model_root):
        return model_root

    base = "/kaggle/input"
    if os.path.isdir(base):
        for name in os.listdir(base):
            p = os.path.join(base, name)
            if os.path.isdir(p) and os.path.basename(p) == os.path.basename(model_root):
                return p
    return model_root  # keep original; downstream will handle fallback


MODEL_ROOT = _resolve_model_root(MODEL_ROOT)


def _find_fold_dir(fold: int) -> str:
    """
    Prefer the originally intended folder pattern.
    If not found, fall back to scanning MODEL_ROOT for any directory containing the fold id.
    """
    direct = os.path.join(MODEL_ROOT, f"f-tfm-small_AES2_fold_{fold}")
    if os.path.isdir(direct):
        return direct

    candidates = []
    if os.path.isdir(MODEL_ROOT):
        for name in os.listdir(MODEL_ROOT):
            p = os.path.join(MODEL_ROOT, name)
            if os.path.isdir(p) and (f"fold_{fold}" in name or f"fold{fold}" in name):
                candidates.append(p)
    candidates = sorted(candidates)
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"Could not locate model directory for fold={fold}. "
        f"Tried: {direct} and scanning under {MODEL_ROOT}. "
        f"MODEL_ROOT exists={os.path.isdir(MODEL_ROOT)}"
    )


FALLBACK_MODEL = "distilbert-base-uncased"

use_fallback = False
tokenizer_dir = None
try:
    tokenizer_dir = _find_fold_dir(0)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, local_files_only=True, use_fast=True
    )
except FileNotFoundError as e:
    print("WARNING:", str(e))
    print(f"Falling back to tokenizer/model: {FALLBACK_MODEL}")
    use_fallback = True
    tokenizer = AutoTokenizer.from_pretrained(
        FALLBACK_MODEL, local_files_only=True, use_fast=True
    )

tok_max = getattr(tokenizer, "model_max_length", None)
if tok_max is None or tok_max > 100000:
    tok_max = 512
EVAL_MAX_LENGTH = int(min(MAX_LENGTH, tok_max))


def tokenize(batch):
    return tokenizer(batch["full_text"], max_length=EVAL_MAX_LENGTH, truncation=True)


print("MODEL_ROOT:", MODEL_ROOT)
print("tokenizer_dir:", tokenizer_dir)
print("use_fallback:", use_fallback)
print(
    "MAX_LENGTH:",
    MAX_LENGTH,
    "EVAL_MAX_LENGTH:",
    EVAL_MAX_LENGTH,
    "tokenizer.model_max_length:",
    tokenizer.model_max_length,
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/656390419.py in <cell line: 0>()
     47 try:
---> 48     tokenizer_dir = _find_fold_dir(0)
     49     tokenizer = AutoTokenizer.from_pretrained(

/tmp/ipykernel_11/656390419.py in _find_fold_dir(fold)
     35 
---> 36     raise FileNotFoundError(
     37         f"Could not locate model directory for fold={fold}. "

FileNotFoundError: Could not locate model directory for fold=0. Tried: /kaggle/input/f-tfm-small/f-tfm-small_AES2_fold_0 and scanning under /kaggle/input/f-tfm-small. MODEL_ROOT exists=False

During handling of the above exception, another exception occurred:

LocalEntryNotFoundError                   Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    469             # This is slightly better for only 1 file
--> 470             hf_hub_download(
    471                 path_or_repo_id,

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    113 
--> 114         return fn(*args, **kwargs)
    115 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in hf_hub_download(repo_id, filename, subfolder, repo_type, revision, library_name, library_version, cache_dir, local_dir, user_agent, force_download, proxies, etag_timeout, token, local_files_only, headers, endpoint, resume_download, force_filename, local_dir_use_symlinks)
   1006     else:
-> 1007         return _hf_hub_download_to_cache_dir(
   1008             # Destination

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in _hf_hub_download_to_cache_dir(cache_dir, repo_id, filename, repo_type, revision, endpoint, etag_timeout, headers, proxies, token, local_files_only, force_download)
   1113         # Otherwise, raise appropriate error
-> 1114         _raise_on_head_call_error(head_call_error, force_download, local_files_only)
   1115 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in _raise_on_head_call_error(head_call_error, force_download, local_files_only)
   1645     if local_files_only:
-> 1646         raise LocalEntryNotFoundError(
   1647             "Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable"

LocalEntryNotFoundError: Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable hf.co look-ups and downloads online, set 'local_files_only' to False.

The above exception was the direct cause of the following exception:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/656390419.py in <cell line: 0>()
     55     print(f"Falling back to tokenizer/model: {FALLBACK_MODEL}")
     56     use_fallback = True
---> 57     tokenizer = AutoTokenizer.from_pretrained(
     58         FALLBACK_MODEL, local_files_only=True, use_fast=True
     59     )

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/tokenization_auto.py in from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs)
   1001                     config = AutoConfig.for_model(**config_dict)
   1002                 else:
-> 1003                     config = AutoConfig.from_pretrained(
   1004                         pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
   1005                     )

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/configuration_auto.py in from_pretrained(cls, pretrained_model_name_or_path, **kwargs)
   1195         code_revision = kwargs.pop("code_revision", None)
   1196 
-> 1197         config_dict, unused_kwargs = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
   1198         has_remote_code = "auto_map" in config_dict and "AutoConfig" in config_dict["auto_map"]
   1199         has_local_code = "model_type" in config_dict and config_dict["model_type"] in CONFIG_MAPPING

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
    606         original_kwargs = copy.deepcopy(kwargs)
    607         # Get config dict associated with the base config file
--> 608         config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
    609         if config_dict is None:
    610             return {}, kwargs

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in _get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
    665             try:
    666                 # Load from local folder or from cache or download from model Hub and cache
--> 667                 resolved_config_file = cached_file(
    668                     pretrained_model_name_or_path,
    669                     configuration_file,

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_file(path_or_repo_id, filename, **kwargs)
    310     ```
    311     """
--> 312     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    313     file = file[0] if file is not None else file
    314     return file

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    541             # even when `local_files_only` is True, in which case raising for connections errors only would not make sense)
    542             elif _raise_exceptions_for_missing_entries:
--> 543                 raise OSError(
    544                     f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load the files, and couldn't find them in the"
    545                     f" cached files.\nCheck your internet connection or see how to run the library in offline mode at"

OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.

## === cell 4
test_ds = Dataset.from_pandas(df_test[["essay_id", "full_text"]], preserve_index=False)
test_ds = test_ds.map(tokenize, batched=True, desc="Tokenizing test")
test_ds = test_ds.remove_columns(["essay_id", "full_text"])

use_cuda = torch.cuda.is_available()
args = TrainingArguments(
    output_dir="./_tmp_pred",
    per_device_eval_batch_size=BATCH_SIZE,
    report_to="none",
    dataloader_num_workers=0,
    fp16=use_cuda,  # inference only
)

predictions = []
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

with torch.inference_mode():
    if use_fallback:
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                FALLBACK_MODEL,
                local_files_only=True,
                num_labels=6,
                ignore_mismatched_sizes=True,
            )
        except Exception as e:
            raise RuntimeError(
                "Fallback model could not be loaded locally (no internet in Kaggle). "
                "Please ensure your intended MODEL_ROOT exists under /kaggle/input or that "
                f"'{FALLBACK_MODEL}' is available in the local cache."
            ) from e

        trainer = Trainer(
            model=model, args=args, data_collator=data_collator, tokenizer=tokenizer
        )
        fold_logits = trainer.predict(test_ds).predictions
        predictions.append(fold_logits)
        print(f"Fallback model logits shape={fold_logits.shape} from {FALLBACK_MODEL}")
    else:
        for fold in range(N_FOLDS):
            model_dir = _find_fold_dir(fold)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_dir, local_files_only=True
            )

            trainer = Trainer(
                model=model,
                args=args,
                data_collator=data_collator,
                tokenizer=tokenizer,
            )

            fold_logits = trainer.predict(test_ds).predictions
            predictions.append(fold_logits)
            print(f"Fold {fold}: logits shape={fold_logits.shape} from {model_dir}")

preds = np.mean(np.stack(predictions, axis=0), axis=0)
print("Ensembled preds shape:", preds.shape)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/298306327.py in <cell line: 0>()
      1 test_ds = Dataset.from_pandas(df_test[["essay_id", "full_text"]], preserve_index=False)
----> 2 test_ds = test_ds.map(tokenize, batched=True, desc="Tokenizing test")
      3 test_ds = test_ds.remove_columns(["essay_id", "full_text"])
      4 
      5 use_cuda = torch.cuda.is_available()

NameError: name 'tokenize' is not defined

## === cell 5
if preds.ndim == 2 and preds.shape[1] == 1:
    raw = preds[:, 0]
    scores = (np.clip(raw, 0, 5).round(0) + 1).astype(np.int32)
else:
    if preds.ndim != 2:
        cls = np.argmax(preds, axis=1)
        scores = (cls + 1).astype(np.int32)
        scores = np.clip(scores, 1, 6).astype(np.int32)
    elif preds.shape[1] != 6:
        cls = np.argmax(preds, axis=1)
        scores = (cls + 1).astype(np.int32)
        scores = np.clip(scores, 1, 6).astype(np.int32)
    else:
        x = preds - preds.max(axis=1, keepdims=True)
        p = np.exp(x)
        p = p / p.sum(axis=1, keepdims=True)

        classes = np.arange(1, 7, dtype=np.float32)[None, :]
        exp_score = (p * classes).sum(axis=1)

        scores = np.rint(exp_score).astype(np.int32)
        scores = np.clip(scores, 1, 6).astype(np.int32)

sub = pd.DataFrame({"essay_id": df_test["essay_id"].values, "score": scores})
sub = sample_submission[["essay_id"]].merge(sub, on="essay_id", how="left")
assert (
    sub["score"].isna().sum() == 0
), "Missing predictions for some essay_id after merge."
sub["score"] = sub["score"].astype(np.int32)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print(
    f"Wrote submission.csv with shape={sub.shape} to {os.path.abspath('submission.csv')}"
)
print("Score distribution:\n", sub["score"].value_counts().sort_index())

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3867322558.py in <cell line: 0>()
      1 # Fix: enforce expected 6-class behavior for ordinal scoring postprocess (score-neutral if already correct).
----> 2 if preds.ndim == 2 and preds.shape[1] == 1:
      3     raw = preds[:, 0]
      4     scores = (np.clip(raw, 0, 5).round(0) + 1).astype(np.int32)
      5 else:

NameError: name 'preds' is not defined
