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

3.14

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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.8019899540067545

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -0.12599) has done: 'I fix the two root causes preventing an end-to-end run: (1) the protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation before importing `transformers`, and (2) the Hugging Face loader failing because the provided `MODEL_PATH` isn’t a valid local `from_pretrained` directory in this environment. To keep core logic intact while ensuring a submission is produced, I add a robust local-path resolver and a safe fallback to a public DeBERTa model if the Kaggle input model folder is missing/unloadable. I also make sure the prediction code always returns a correctly formatted `submission.csv` with `essay_id,score` aligned to the test set order. No changes are made to the chunking/prediction aggregation logic beyond making it run reliably.'
- What this solution (achieved 0.0) has done: 'You’re currently crashing at import time due to an incompatibility between `transformers` and the installed protobuf runtime (`MessageFactory.GetPrototype`), so the first fix is to force a protobuf version that works in Kaggle by uninstalling `protobuf`/`googleapis-common-protos` and installing a compatible protobuf (3.20.x) before importing `transformers`. Next, your predictions are likely badly calibrated for QWK because you’re using a 6-class classifier head with `argmax+1` on a base model that isn’t your intended fine-tuned checkpoint; to move the score toward the target, we keep the same inference logic but make the local model-path resolver actually find and load the Kaggle-provided checkpoint reliably (and avoid falling back to an unfine-tuned public model). Finally, we ensure the written file is exactly `submission.csv` with `essay_id,score` aligned to the test order and all scores clipped to 1–6.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with an “invalid submission” or a badly misaligned prediction scale, so the smallest safe improvement is to guarantee the model you load is the intended fine-tuned local checkpoint and that the written `submission.csv` exactly matches the required schema and row order. I keep your chunking + mean-aggregation + rounding core logic unchanged, but I (1) strengthen the local model directory resolver to correctly handle Kaggle dataset layouts (including `snapshots/...`), and (2) enforce `id2label/label2id`-aware mapping so we don’t accidentally shift labels with `+1` when the checkpoint already encodes labels 1–6. Finally, I add a lightweight sanity check that the submission has exactly `len(test)` rows, the right columns, and scores are in [1,6], so you don’t end up with a 0.0 due to formatting/alignment issues.'
- What this solution (achieved 0.0) has done: 'I fix the import-time protobuf crash by enforcing the pure-Python protobuf implementation *before* any `transformers` import, and by guarding against stale `google.protobuf` modules already loaded. Then I make model loading robust: it try to load your provided local checkpoint, but if that path truly doesn’t exist in this notebook’s inputs it fall back to a known public DeBERTa model so the pipeline still completes and writes a valid `submission.csv`. Finally, I keep your chunking, aggregation, and rounding logic unchanged, and add a small safety check to ensure the submission is aligned to `test.csv` order with valid score bounds (1–6). These changes are primarily to ensure an end-to-end run and a valid submission; score improvement depends on whether your fine-tuned checkpoint is actually available.'
- What this solution (achieved -0.12508) has done: 'We fix the import-time protobuf crash that stops the notebook at cell 1 by forcing the pure-Python protobuf implementation before any `transformers` import and by defensively patching `google.protobuf`’s `MessageFactory.GetPrototype` when the runtime lacks it. This is a correctness/stability fix that enables the rest of your pipeline (model load → chunking → inference → CSV write) to run end-to-end and produce a valid `submission.csv`. We keep your model inference/chunking/aggregation logic unchanged, and we also make the fallback model load “local-only first” to avoid internet dependency in Kaggle. Finally, we preserve the required submission schema and alignment checks so you don’t get another 0.0 from an invalid/misaligned file.'
- What this solution (achieved 0.0) has done: 'We fix the import-time protobuf crash by applying a safe, version-agnostic monkeypatch to `google.protobuf.message_factory.MessageFactory.GetPrototype` (and the module-level `GetPrototype`) before importing `transformers`, so the notebook runs end-to-end. We also make sure the script can always load a model locally (no internet) by searching the provided Kaggle input tree for a valid HF checkpoint if `MODEL_PATH` is missing, while keeping your inference/chunking/aggregation logic unchanged. Finally, we keep the submission formatting/alignment checks, ensuring a valid `submission.csv` is always written.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_DOWNLOADS", "1")

TRAIN_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
TEST_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
MODEL_PATH = "/kaggle/input/aes2-debertav3large/transformers/default/1"
OUTPUT_PATH = "/kaggle/working/"

MAX_LEN = 512
OVERLAP = 64
MIN_CHUNK_RATIO = 0.3  # Skip if this chunk < RATIO * MAX_LEN
BATCH_SIZE = 2
NUM_LABELS = 6

import sys

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        del sys.modules[m]

try:
    from google.protobuf import message_factory as _message_factory

    def _ensure_getprototype_on_factory_instance():
        if not hasattr(_message_factory.MessageFactory, "GetPrototype"):
            if hasattr(_message_factory.MessageFactory, "GetMessageClass"):

                def _GetPrototype(self, descriptor):
                    return self.GetMessageClass(descriptor)

                _message_factory.MessageFactory.GetPrototype = _GetPrototype
            else:
                def _GetPrototype(self, descriptor):
                    raise AttributeError(
                        "GetPrototype is not available in this protobuf runtime."
                    )

                _message_factory.MessageFactory.GetPrototype = _GetPrototype

        if not hasattr(_message_factory, "GetPrototype"):
            if hasattr(_message_factory, "GetMessageClass"):

                def _mf_GetPrototype(descriptor):
                    return _message_factory.GetMessageClass(descriptor)

                _message_factory.GetPrototype = _mf_GetPrototype

        gen = getattr(_message_factory, "_GENERATED_FACTORY", None)
        if gen is not None and not hasattr(gen, "GetPrototype"):
            try:
                gen.GetPrototype = _message_factory.MessageFactory.GetPrototype.__get__(
                    gen, gen.__class__
                )
            except Exception:
                pass

    _ensure_getprototype_on_factory_instance()
except Exception:
    pass

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    DebertaV2Tokenizer,
    DebertaV2ForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")




## === cell 1
def _resolve_local_model_dir(model_path: str) -> str | None:
    """
    Ensure we reliably find the actual local fine-tuned HF checkpoint directory.
    Handles common Kaggle layouts including:
      - transformers/default/1
      - snapshots/<hash> (from HF cache-like exports)
    """
    if not model_path:
        return None

    candidates = []
    for p in [
        model_path,
        os.path.dirname(model_path),
        os.path.dirname(os.path.dirname(model_path)),
        os.path.dirname(os.path.dirname(os.path.dirname(model_path))),
    ]:
        if p and os.path.isdir(p):
            candidates.append(p)

    probes = [
        model_path,
        os.path.join(model_path, "transformers", "default", "1"),
        os.path.join(model_path, "transformers", "default"),
        os.path.join(model_path, "transformers"),
        os.path.join(model_path, "1"),
    ]
    for p in probes:
        if p and os.path.isdir(p):
            candidates.append(p)

    seen = set()
    uniq = []
    for c in candidates:
        c = os.path.abspath(c)
        if c not in seen:
            seen.add(c)
            uniq.append(c)

    def looks_like_hf_dir(d: str) -> bool:
        if not os.path.isdir(d):
            return False
        has_config = os.path.isfile(os.path.join(d, "config.json"))
        has_weights = any(
            os.path.isfile(os.path.join(d, fn))
            for fn in ["pytorch_model.bin", "model.safetensors"]
        )
        return has_config and has_weights

    for c in uniq:
        if looks_like_hf_dir(c):
            return c

    for c in uniq:
        snap_root = os.path.join(c, "snapshots")
        if os.path.isdir(snap_root):
            try:
                for sub in sorted(os.listdir(snap_root)):
                    d = os.path.join(snap_root, sub)
                    if looks_like_hf_dir(d):
                        return d
            except Exception:
                pass

        try:
            for root, dirs, files in os.walk(c):
                rel_depth = os.path.relpath(root, c).count(os.sep)
                if rel_depth > 8:
                    dirs[:] = []
                    continue
                if "config.json" in files and (
                    "pytorch_model.bin" in files or "model.safetensors" in files
                ):
                    return root
        except Exception:
            continue

    return None


def _search_kaggle_input_for_checkpoint() -> str | None:
    root = "/kaggle/input"
    if not os.path.isdir(root):
        return None

    def looks_like_hf_dir(d: str) -> bool:
        if not os.path.isdir(d):
            return False
        if not os.path.isfile(os.path.join(d, "config.json")):
            return False
        if not any(
            os.path.isfile(os.path.join(d, fn))
            for fn in ["pytorch_model.bin", "model.safetensors"]
        ):
            return False
        return True

    preferred_tokens = ["aes", "essay", "deberta", "automated", "scoring", "lab"]
    try:
        for top in sorted(os.listdir(root)):
            top_path = os.path.join(root, top)
            if not os.path.isdir(top_path):
                continue
            low = top.lower()
            if not any(t in low for t in preferred_tokens):
                continue
            for dirpath, dirnames, filenames in os.walk(top_path):
                rel_depth = os.path.relpath(dirpath, top_path).count(os.sep)
                if rel_depth > 10:
                    dirnames[:] = []
                    continue
                if "config.json" in filenames and (
                    "pytorch_model.bin" in filenames or "model.safetensors" in filenames
                ):
                    if looks_like_hf_dir(dirpath):
                        return dirpath
    except Exception:
        pass

    try:
        for top in sorted(os.listdir(root)):
            top_path = os.path.join(root, top)
            if not os.path.isdir(top_path):
                continue
            for dirpath, dirnames, filenames in os.walk(top_path):
                rel_depth = os.path.relpath(dirpath, top_path).count(os.sep)
                if rel_depth > 8:
                    dirnames[:] = []
                    continue
                if "config.json" in filenames and (
                    "pytorch_model.bin" in filenames or "model.safetensors" in filenames
                ):
                    if looks_like_hf_dir(dirpath):
                        return dirpath
    except Exception:
        pass

    return None


def _load_tokenizer_and_model(model_path: str):
    """
    Preference order:
      1) load resolved local fine-tuned checkpoint (best for score, no internet),
      2) if model_path is bad/missing, search /kaggle/input for a checkpoint,
      3) last resort: fall back to a public model ONLY if already cached locally.
    """
    resolved = _resolve_local_model_dir(model_path)
    if resolved is None:
        alt = _search_kaggle_input_for_checkpoint()
        if alt is not None:
            print(
                f"[INFO] MODEL_PATH not usable; found alternate local checkpoint: {alt}"
            )
            resolved = alt

    if resolved is not None:
        try:
            tok = AutoTokenizer.from_pretrained(
                resolved, local_files_only=True, use_fast=True
            )
            mdl = AutoModelForSequenceClassification.from_pretrained(
                resolved, local_files_only=True
            )
            print(f"Loaded local model via Auto* from: {resolved}")
            return tok, mdl
        except Exception as e_auto:
            print(f"[WARN] Local Auto* load failed ({resolved}): {repr(e_auto)}")

        try:
            tok = DebertaV2Tokenizer.from_pretrained(resolved, local_files_only=True)
            mdl = DebertaV2ForSequenceClassification.from_pretrained(
                resolved, local_files_only=True
            )
            print(f"Loaded local model via DebertaV2* from: {resolved}")
            return tok, mdl
        except Exception as e_deb:
            print(f"[WARN] Local DebertaV2* load failed ({resolved}): {repr(e_deb)}")

    fallback_name = "microsoft/deberta-v3-large"
    print(
        f"[WARN] Could not load local checkpoint. Falling back to cached: {fallback_name}"
    )
    tok = AutoTokenizer.from_pretrained(
        fallback_name, use_fast=True, local_files_only=True
    )
    mdl = AutoModelForSequenceClassification.from_pretrained(
        fallback_name, num_labels=NUM_LABELS, local_files_only=True
    )
    return tok, mdl


tokenizer, model = _load_tokenizer_and_model(MODEL_PATH)

if hasattr(model, "config") and getattr(model.config, "num_labels", None) is not None:
    NUM_LABELS = int(model.config.num_labels)
print(f"Using NUM_LABELS={NUM_LABELS}")




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
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
/tmp/ipykernel_56/691984886.py in <cell line: 0>()
    191 
    192 
--> 193 tokenizer, model = _load_tokenizer_and_model(MODEL_PATH)
    194 
    195 if hasattr(model, "config") and getattr(model.config, "num_labels", None) is not None:

/tmp/ipykernel_56/691984886.py in _load_tokenizer_and_model(model_path)
    182         f"[WARN] Could not load local checkpoint. Falling back to cached: {fallback_name}"
    183     )
--> 184     tok = AutoTokenizer.from_pretrained(
    185         fallback_name, use_fast=True, local_files_only=True
    186     )

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

## === cell 2
def plot_confusion_matrix(preds, labels, title="Confusion Matrix"):
    preds = np.array(preds)
    labels = np.array(labels)

    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Greens", values_format="d")
    plt.title(title)
    plt.show()




## === cell 3
def compute_metrics(preds, labels, device="cuda"):
    if len(preds.shape) > 1:
        preds = np.argmax(preds, axis=-1)
    qwk_score = cohen_kappa_score(labels, preds, weights="quadratic")
    acc_score = accuracy_score(labels, preds)
    return {"quadratic_weighted_kappa": qwk_score, "accuracy": acc_score}




## === cell 4
def _infer_label_mapping(mdl, num_labels: int):
    """
    Avoid incorrect '+1' shifting when the checkpoint's id2label already maps
    indices to 1..6 or otherwise non-0-based labels. If mapping is unclear, fall back to +1.
    Returns a function: idx_array -> score_array (int).
    """
    cfg = getattr(mdl, "config", None)
    id2label = getattr(cfg, "id2label", None) if cfg is not None else None

    def default_map(idxs: np.ndarray) -> np.ndarray:
        return idxs.astype(int) + 1

    if not isinstance(id2label, dict) or len(id2label) != num_labels:
        return default_map

    try:
        keys = sorted(int(k) for k in id2label.keys())
        labels = [id2label[str(k)] if str(k) in id2label else id2label[k] for k in keys]

        numeric = []
        for lab in labels:
            if isinstance(lab, (int, np.integer)):
                numeric.append(int(lab))
            elif isinstance(lab, str) and lab.strip().isdigit():
                numeric.append(int(lab.strip()))
            else:
                numeric = []
                break

        if len(numeric) == num_labels and sorted(numeric) == list(
            range(1, num_labels + 1)
        ):
            mapping = {k: v for k, v in zip(keys, numeric)}

            def mapped(idxs: np.ndarray) -> np.ndarray:
                return np.vectorize(lambda z: mapping.get(int(z), int(z) + 1))(
                    idxs
                ).astype(int)

            print("[INFO] Using config.id2label-based mapping (no forced +1 shift).")
            return mapped
    except Exception:
        pass

    return default_map


def predict_essay_score(model, dataset, num_labels, batch_size, device):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    model.eval()

    map_fn = _infer_label_mapping(model, num_labels)

    preds, essay_ids_all = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            essay_ids = batch["essay_id"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            pred_class = torch.argmax(logits, axis=1).cpu().numpy()
            pred_labels = map_fn(pred_class)

            preds.extend(pred_labels)
            essay_ids_all.extend(essay_ids)

    preds = np.array(preds)
    essay_ids_all = np.array(essay_ids_all)

    chunk_results = pd.DataFrame({"essay_id": essay_ids_all, "raw_prediction": preds})

    aggregated_predictions = chunk_results.groupby("essay_id")["raw_prediction"].mean()
    y_pred_aggregated = np.rint(aggregated_predictions.values)
    final_scores = np.clip(y_pred_aggregated, 1, num_labels).astype(int)

    final_results_df = pd.DataFrame(
        {"essay_id": aggregated_predictions.index.values, "score": final_scores}
    )

    return final_results_df, preds




## === cell 5
class TestEssayDataset(Dataset):
    def __init__(
        self,
        df,
        tokenizer,
        max_len: int = 512,
        overlap: int = 128,
        min_chunk_ratio: float = 0.3,
    ):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.overlap = overlap
        self.min_chunk_ratio = min_chunk_ratio

        num_special_tokens = self.tokenizer.num_special_tokens_to_add(pair=False)
        self.max_content_len = max_len - num_special_tokens

        for _, row in df.iterrows():
            essay_id = row["essay_id"]
            text = "[A] " + row["full_text"]
            tokens = tokenizer(text, add_special_tokens=False)["input_ids"]

            step = self.max_content_len - overlap
            start = 0

            while start < len(tokens):
                end = start + self.max_content_len
                chunk_content = tokens[start:end]
                if len(chunk_content) < self.max_content_len * self.min_chunk_ratio:
                    break

                processed_tokens = tokenizer.build_inputs_with_special_tokens(
                    chunk_content
                )
                padding_len = self.max_len - len(processed_tokens)
                if padding_len > 0:
                    processed_tokens += [tokenizer.pad_token_id] * padding_len

                self.samples.append((processed_tokens, essay_id))
                start += step

                if end >= len(tokens):
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_ids, essay_id = self.samples[idx]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "essay_id": essay_id,
        }




## === cell 6
df_test = pd.read_csv(TEST_PATH)

test_dataset = TestEssayDataset(df_test, tokenizer, MAX_LEN, OVERLAP, MIN_CHUNK_RATIO)

test_results, _ = predict_essay_score(
    dataset=test_dataset,
    model=model,
    num_labels=NUM_LABELS,
    batch_size=BATCH_SIZE,
    device=device,
)

test_results = df_test[["essay_id"]].merge(test_results, on="essay_id", how="left")
test_results["score"] = test_results["score"].fillna(3).astype(int)
test_results["score"] = test_results["score"].clip(1, NUM_LABELS)

test_results = test_results[["essay_id", "score"]]
assert len(test_results) == len(df_test), "Submission row count mismatch with test.csv"
assert list(test_results.columns) == [
    "essay_id",
    "score",
], "Submission columns incorrect"

print("Submission head:")
print(test_results.head())
print(f"Rows: {len(test_results)} (expected {len(df_test)})")
print("Score value counts:")
print(test_results["score"].value_counts().sort_index())



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/1922158759.py in <cell line: 0>()
      1 df_test = pd.read_csv(TEST_PATH)
      2 
----> 3 test_dataset = TestEssayDataset(df_test, tokenizer, MAX_LEN, OVERLAP, MIN_CHUNK_RATIO)
      4 
      5 test_results, _ = predict_essay_score(

NameError: name 'tokenizer' is not defined

## === cell 7
sub_path = os.path.join(OUTPUT_PATH, "submission.csv")
test_results.to_csv(sub_path, index=False)

check = pd.read_csv(sub_path)
assert len(check) == len(df_test), "Written CSV row count mismatch"
assert list(check.columns) == ["essay_id", "score"], "Written CSV columns mismatch"
assert (
    check["score"].between(1, NUM_LABELS).all()
), "Written CSV has out-of-range scores"

print(f"Wrote submission to: {sub_path}")
print(check.head())

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/1966351938.py in <cell line: 0>()
      1 sub_path = os.path.join(OUTPUT_PATH, "submission.csv")
----> 2 test_results.to_csv(sub_path, index=False)
      3 
      4 check = pd.read_csv(sub_path)
      5 assert len(check) == len(df_test), "Written CSV row count mismatch"

NameError: name 'test_results' is not defined
