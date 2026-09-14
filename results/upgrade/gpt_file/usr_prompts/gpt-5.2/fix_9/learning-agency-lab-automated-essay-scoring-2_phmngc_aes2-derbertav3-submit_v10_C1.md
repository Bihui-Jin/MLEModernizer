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

0.7924246284831522

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the failure to locate the local HuggingFace model by (1) falling back to a compatible pretrained model that can be downloaded in Kaggle (so the notebook runs even when the attached model dataset is missing), and (2) making the resolver also search the competition dataset folders under `/kaggle/input/learning-agency-lab-automated-essay-scoring-2/`. I keep your chunking/overlap inference logic and aggregation exactly the same, only making the model loading robust so later cells can run. Finally, I ensure a valid `submission.csv` is always written with the required `essay_id,score` columns and correct row count/order.'
- What this solution (achieved 0.0) has done: 'I fix the `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` by switching the protobuf runtime back to the C++/upb implementation (your current env var forces the pure-Python protobuf backend that triggers this Transformers import/model load crash in some Kaggle images). I also make the model-loading more robust in offline Kaggle runs by (a) preferring any attached local HF model and (b) only attempting an online fallback if explicitly possible; if not, we fail fast with a clear message instead of producing a broken 0-score submission. Finally, I keep your chunking/inference/aggregation exactly the same and ensure `submission.csv` is always written with correct columns and row order.'
- What this solution (achieved 0.00335) has done: 'I fix the Transformers import/model-load crash causing `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` by forcing the safe protobuf Python implementation *before* importing `transformers`, which is the minimal change needed to unblock execution in this Kaggle image. I also make model loading deterministic/offline-safe by preferring the provided local `MODEL_PATH` and, if that fails, scanning `/kaggle/input` for any attached HF model directory; only then it try the remote fallback (which likely won’t work without internet). The chunking/overlap inference, aggregation, and submission formatting remain unchanged so the core logic and evaluation semantics are preserved. Finally, the script always write `/kaggle/working/submission.csv` with the required `essay_id,score` columns and correct row count/order.'
- What this solution (achieved 0.0) has done: 'I fix the Transformers/protobuf crash by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` *and* disabling the problematic `google.protobuf` C-descriptor path before importing `transformers`, then force a clean import so the environment variable actually takes effect. I also make model loading more robust and score-improving by preferring the competition’s commonly-attached local DeBERTa model dataset (if present) and by ensuring we load the fine-tuned checkpoint correctly (rather than accidentally constructing a fresh random head). Finally, I keep your chunking/overlap inference and aggregation unchanged, and ensure we always write `/kaggle/working/submission.csv` with the required `essay_id,score` columns and correct row order.'

# 9. Code solution

## === cell 0
TRAIN_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
TEST_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
MODEL_PATH = "/kaggle/input/aes2-debertav3-large/transformers/default/1"
OUTPUT_PATH = "/kaggle/working/"

MAX_LEN = 512
OVERLAP = 64
MIN_CHUNK_RATIO = 0.3  # Skip if this chunk < RATIO * MAX_LEN
BATCH_SIZE = 2
NUM_LABELS = 6



## === cell 1
import os

for k in [
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION",
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION",
]:
    if k in os.environ:
        del os.environ[k]
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys

for m in [
    "transformers",
    "google.protobuf",
    "google.protobuf.descriptor_pool",
    "google.protobuf.message_factory",
]:
    if m in sys.modules:
        del sys.modules[m]

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings

warnings.filterwarnings("ignore", category=UserWarning)




## === cell 2
def resolve_local_model_dir(path: str) -> str:
    if (
        path
        and os.path.isdir(path)
        and os.path.isfile(os.path.join(path, "config.json"))
    ):
        return path

    if path and os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            if "config.json" in files:
                return root

    raise FileNotFoundError(
        f"Could not find a local HuggingFace model folder with config.json under: {path}"
    )


def find_any_hf_model_dir(search_roots) -> str:
    if isinstance(search_roots, str):
        search_roots = [search_roots]

    preferred = []
    for sr in search_roots:
        if os.path.isdir(sr):
            for d in sorted(os.listdir(sr)):
                if any(
                    k in d.lower()
                    for k in ["deberta", "aes", "essay", "model", "transformer"]
                ):
                    preferred.append(os.path.join(sr, d))

    scan_roots = preferred + [r for r in search_roots if os.path.isdir(r)]
    visited = set()

    for sr in scan_roots:
        if not os.path.isdir(sr) or sr in visited:
            continue
        visited.add(sr)
        for root, dirs, files in os.walk(sr):
            if "config.json" in files:
                return root

    raise FileNotFoundError(
        f"Could not find any local HuggingFace model folder with config.json under: {search_roots}. "
        f"Check that the model dataset is attached."
    )


fallback_model_id = "microsoft/deberta-v3-base"

resolved_model_dir = None
use_remote_model_id = None

try:
    resolved_model_dir = resolve_local_model_dir(MODEL_PATH)
except FileNotFoundError as e:
    print("MODEL_PATH not usable:", e)
    try:
        resolved_model_dir = find_any_hf_model_dir(
            [
                "/kaggle/input/aes2-debertav3-large",
                "/kaggle/input",
                "/kaggle/input/learning-agency-lab-automated-essay-scoring-2",
                "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/learning-agency-lab-automated-essay-scoring-2",
            ]
        )
    except FileNotFoundError as e2:
        print("No local HF model found:", e2)
        use_remote_model_id = fallback_model_id

if use_remote_model_id is None:
    print("Resolved local model dir:", resolved_model_dir)
    print(
        "Has config.json:",
        os.path.isfile(os.path.join(resolved_model_dir, "config.json")),
    )
else:
    print("Using remote model id:", use_remote_model_id)



## === cell 3
model_source = (
    use_remote_model_id if use_remote_model_id is not None else resolved_model_dir
)
local_only = use_remote_model_id is None

if use_remote_model_id is not None:
    local_only = True

try:
    config = AutoConfig.from_pretrained(model_source, local_files_only=local_only)
except Exception:
    config = None

try:
    tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=local_only)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    if (
        config is not None
        and hasattr(config, "num_labels")
        and int(config.num_labels) == int(NUM_LABELS)
    ):
        model = AutoModelForSequenceClassification.from_pretrained(
            model_source,
            config=config,
            local_files_only=local_only,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_source,
            num_labels=NUM_LABELS,
            local_files_only=local_only,
        )
except Exception as e:
    raise RuntimeError(
        "Failed to load the HuggingFace model/tokenizer. "
        "Ensure the intended local model dataset is attached and MODEL_PATH points to a valid HF directory "
        "(must contain config.json). If you intended to use a remote model id, note Kaggle likely has no internet.\n"
        f"Model source: {model_source}\n"
        f"local_files_only={local_only}\n"
        f"Error: {repr(e)}"
    )

if len(tokenizer) != model.get_input_embeddings().num_embeddings:
    model.resize_token_embeddings(len(tokenizer))




## --- ERROR in cell 3, traceback:
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
/tmp/ipykernel_55/4245693638.py in <cell line: 0>()
     16 try:
---> 17     tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=local_only)
     18     if tokenizer.pad_token is None:

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/tokenization_auto.py in from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs)
   1002                 else:
-> 1003                     config = AutoConfig.from_pretrained(
   1004                         pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/configuration_auto.py in from_pretrained(cls, pretrained_model_name_or_path, **kwargs)
   1196 
-> 1197         config_dict, unused_kwargs = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
   1198         has_remote_code = "auto_map" in config_dict and "AutoConfig" in config_dict["auto_map"]

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
    607         # Get config dict associated with the base config file
--> 608         config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
    609         if config_dict is None:

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in _get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
    666                 # Load from local folder or from cache or download from model Hub and cache
--> 667                 resolved_config_file = cached_file(
    668                     pretrained_model_name_or_path,

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_file(path_or_repo_id, filename, **kwargs)
    311     """
--> 312     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    313     file = file[0] if file is not None else file

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    542             elif _raise_exceptions_for_missing_entries:
--> 543                 raise OSError(
    544                     f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load the files, and couldn't find them in the"

OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.

During handling of the above exception, another exception occurred:

RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/4245693638.py in <cell line: 0>()
     39         )
     40 except Exception as e:
---> 41     raise RuntimeError(
     42         "Failed to load the HuggingFace model/tokenizer. "
     43         "Ensure the intended local model dataset is attached and MODEL_PATH points to a valid HF directory "

RuntimeError: Failed to load the HuggingFace model/tokenizer. Ensure the intended local model dataset is attached and MODEL_PATH points to a valid HF directory (must contain config.json). If you intended to use a remote model id, note Kaggle likely has no internet.
Model source: microsoft/deberta-v3-base
local_files_only=True
Error: OSError("We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.\nCheck your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.")

## === cell 4
def plot_confusion_matrix(preds, labels, title="Confusion Matrix"):
    preds = np.array(preds)
    labels = np.array(labels)

    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Greens", values_format="d")
    plt.title(title)
    plt.show()




## === cell 5
def compute_metrics(preds, labels, device="cuda"):
    if len(preds.shape) > 1:
        preds = np.argmax(preds, axis=-1)
    qwk_score = cohen_kappa_score(labels, preds, weights="quadratic")
    acc_score = accuracy_score(labels, preds)
    return {"quadratic_weighted_kappa": qwk_score, "accuracy": acc_score}




## === cell 6
def predict_essay_score(model, dataset, num_labels, batch_size, device):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    model.eval()

    preds, essay_ids_all = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            essay_ids = batch["essay_id"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            pred_labels = torch.argmax(logits, axis=1).cpu().numpy() + 1

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




## === cell 7
class TestEssayDataset(Dataset):
    def __init__(
        self,
        df,
        tokenizer,
        max_len: int = 512,
        overlap: int = 64,
        min_chunk_ratio: float = 0.3,
    ):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.overlap = overlap
        self.min_chunk_ratio = min_chunk_ratio

        self.num_special = tokenizer.num_special_tokens_to_add(pair=False)

        tag_tokens = tokenizer("[A]", add_special_tokens=False)["input_ids"]
        len_tag = len(tag_tokens)

        self.max_content_len = max_len - self.num_special - len_tag
        if self.max_content_len <= 0:
            raise ValueError(
                f"max_content_len must be > 0, got {self.max_content_len}. "
                f"Check MAX_LEN/num_special/tag length."
            )

        for _, row in df.iterrows():
            essay_id = row["essay_id"]
            base_tokens = tokenizer(row["full_text"], add_special_tokens=False)[
                "input_ids"
            ]

            step = self.max_content_len - overlap
            if step <= 0:
                raise ValueError(
                    f"Step must be > 0, got {step}. Reduce OVERLAP or increase MAX_LEN."
                )

            start = 0
            while start < len(base_tokens):
                end = start + self.max_content_len
                chunk = base_tokens[start:end]

                if (
                    len(chunk) < self.max_content_len * self.min_chunk_ratio
                    and start != 0
                ):
                    break

                chunk_with_tag = tag_tokens + chunk
                processed = tokenizer.build_inputs_with_special_tokens(chunk_with_tag)

                if len(processed) > self.max_len:
                    processed = processed[: self.max_len]

                pad_len = self.max_len - len(processed)
                if pad_len > 0:
                    processed += [tokenizer.pad_token_id] * pad_len

                self.samples.append((processed, essay_id))

                start += step
                if end >= len(base_tokens):
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




## === cell 8
df_test = pd.read_csv(TEST_PATH)
test_dataset = TestEssayDataset(df_test, tokenizer, MAX_LEN, OVERLAP, MIN_CHUNK_RATIO)

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    model = model.half()

test_results, _ = predict_essay_score(
    dataset=test_dataset,
    model=model,
    num_labels=NUM_LABELS,
    batch_size=BATCH_SIZE,
    device=device,
)

test_results = df_test[["essay_id"]].merge(test_results, on="essay_id", how="left")
test_results["score"] = test_results["score"].fillna(3).astype(int)

print("Submission preview:")
print(test_results.head())
print("Rows:", len(test_results), "Cols:", list(test_results.columns))
print("Any missing scores:", test_results["score"].isna().sum())
print("Score value counts (head):")
print(test_results["score"].value_counts().head(10))



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1857134501.py in <cell line: 0>()
      1 df_test = pd.read_csv(TEST_PATH)
----> 2 test_dataset = TestEssayDataset(df_test, tokenizer, MAX_LEN, OVERLAP, MIN_CHUNK_RATIO)
      3 
      4 device = "cuda" if torch.cuda.is_available() else "cpu"
      5 

NameError: name 'tokenizer' is not defined

## === cell 9
submission_path = os.path.join(OUTPUT_PATH, "submission.csv")
test_results.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(pd.read_csv(submission_path).head())
print("File size (bytes):", os.path.getsize(submission_path))

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2659102024.py in <cell line: 0>()
      1 submission_path = os.path.join(OUTPUT_PATH, "submission.csv")
----> 2 test_results.to_csv(submission_path, index=False)
      3 
      4 print("Wrote:", submission_path)
      5 print(pd.read_csv(submission_path).head())

NameError: name 'test_results' is not defined
