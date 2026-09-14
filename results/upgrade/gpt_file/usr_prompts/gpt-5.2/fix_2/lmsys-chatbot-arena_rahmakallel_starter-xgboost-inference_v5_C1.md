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
xgboost==2.0.3

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

1.0817042583042464

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import torch

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))




## === cell 1
class lmsysdataset:
    def __init__(self, data, target=None, tokenizer=None):
        self.data = data.reset_index(drop=True)
        self.target = None if target is None else target.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def _normalize_text(self, x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        if not isinstance(x, str):
            x = str(x)
        s = x.strip()
        if s.startswith("[") and s.endswith("]") and '","' in s:
            return " ".join([t.strip('"') for t in s.strip("[]").split('","')])
        return s

    def __getitem__(self, idx):
        prompt = self._normalize_text(self.data.iloc[idx]["prompt"])
        response_a = self._normalize_text(self.data.iloc[idx]["response_a"])
        response_b = self._normalize_text(self.data.iloc[idx]["response_b"])

        if self.target is not None:
            y = torch.tensor(
                [
                    self.target.iloc[idx]["winner_model_a"],
                    self.target.iloc[idx]["winner_model_b"],
                    self.target.iloc[idx]["winner_tie"],
                ],
                dtype=torch.float32,
            )
        else:
            y = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

        text_a = f"prompt: {prompt} model_a: {response_a}"
        text_b = f"prompt: {prompt} model_b: {response_b}"

        if self.tokenizer is not None:
            encoding_a = self.tokenizer.encode_plus(
                text_a,
                truncation=True,
                padding="max_length",
                max_length=1024,
                return_tensors="pt",
            )
            encoding_b = self.tokenizer.encode_plus(
                text_b,
                truncation=True,
                padding="max_length",
                max_length=1024,
                return_tensors="pt",
            )

            input_ids_a = encoding_a["input_ids"].squeeze(0)
            attention_mask_a = encoding_a["attention_mask"].squeeze(0)

            input_ids_b = encoding_b["input_ids"].squeeze(0)
            attention_mask_b = encoding_b["attention_mask"].squeeze(0)

            return input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, y
        else:
            return text_a, text_b, y




## === cell 2
from transformers import AutoTokenizer, AutoModel
import xgboost as xgb

TEST_PATH = "/kaggle/input/lmsys-chatbot-arena/test.csv"
df = pd.read_csv(TEST_PATH)
data = df[["prompt", "response_a", "response_b"]]

tokenizer_candidates = [
    "/kaggle/input/lmsys-gte/gte_tokenizer",
    "/kaggle/input/lmsys-gte",
]
model_candidates = [
    "/kaggle/input/base_custom_gte/transformers/default/1",
    "/kaggle/input/base_custom_gte",
]


def _load_local_tokenizer(candidates):
    for p in candidates:
        if os.path.isdir(p):
            try:
                return AutoTokenizer.from_pretrained(p, local_files_only=True)
            except Exception:
                pass
    return AutoTokenizer.from_pretrained("thenlper/gte-base", local_files_only=True)


def _load_local_model(candidates):
    for p in candidates:
        if os.path.isdir(p):
            try:
                return AutoModel.from_pretrained(
                    p, trust_remote_code=True, local_files_only=True
                )
            except Exception:
                pass
    return AutoModel.from_pretrained("thenlper/gte-base", local_files_only=True)


tokenizer = _load_local_tokenizer(tokenizer_candidates)
model = _load_local_model(model_candidates)

xgb_model = xgb.XGBClassifier()
xgb_model.load_model("/kaggle/input/lmsys-xgb/xgb_model.json")

dataset = lmsysdataset(data, tokenizer=tokenizer)




## --- ERROR in cell 2, traceback:
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
/tmp/ipykernel_55/1683469199.py in <cell line: 0>()
     42 
     43 
---> 44 tokenizer = _load_local_tokenizer(tokenizer_candidates)
     45 model = _load_local_model(model_candidates)
     46 

/tmp/ipykernel_55/1683469199.py in _load_local_tokenizer(candidates)
     27                 pass
     28     # Fallback (should exist in Kaggle base images cache in many notebooks; if not, it will error clearly)
---> 29     return AutoTokenizer.from_pretrained("thenlper/gte-base", local_files_only=True)
     30 
     31 

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

## === cell 3
def mean_pooling(last_hidden_state, attention_mask):
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    )
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask
    return mean_embeddings


def process_batch(model, batch, device):
    input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, labels = batch
    input_ids_a = input_ids_a.to(device)
    attention_mask_a = attention_mask_a.to(device)
    input_ids_b = input_ids_b.to(device)
    attention_mask_b = attention_mask_b.to(device)

    with torch.no_grad():
        outputs_a = model(input_ids_a, attention_mask=attention_mask_a)
        outputs_b = model(input_ids_b, attention_mask=attention_mask_b)

    lhs_a = outputs_a.last_hidden_state
    lhs_b = outputs_b.last_hidden_state

    embeddings_a = mean_pooling(lhs_a, attention_mask_a)
    embeddings_b = mean_pooling(lhs_b, attention_mask_b)

    labels = labels.detach().cpu().numpy()
    embeddings_a = embeddings_a.detach().cpu().numpy()
    embeddings_b = embeddings_b.detach().cpu().numpy()

    return embeddings_a, embeddings_b, labels




## === cell 4
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

emb_a_list = []
emb_b_list = []

for batch in dataloader:
    emb_a, emb_b, _ = process_batch(model, batch, device)
    emb_a_list.append(emb_a)
    emb_b_list.append(emb_b)

emb_a_all = np.vstack(emb_a_list)
emb_b_all = np.vstack(emb_b_list)

X = np.concatenate([emb_a_all, emb_b_all], axis=1)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3783851264.py in <cell line: 0>()
      2 
      3 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
----> 4 model.to(device)
      5 model.eval()
      6 

NameError: name 'model' is not defined

## === cell 5
y_pred = xgb_model.predict_proba(X)

if y_pred.ndim != 2 or y_pred.shape[1] != 3:
    raise ValueError(f"Expected predict_proba to return (n,3), got {y_pred.shape}")



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/776915032.py in <cell line: 0>()
----> 1 y_pred = xgb_model.predict_proba(X)
      2 
      3 # Safety: ensure proper shape (n, 3)
      4 if y_pred.ndim != 2 or y_pred.shape[1] != 3:
      5     raise ValueError(f"Expected predict_proba to return (n,3), got {y_pred.shape}")

NameError: name 'xgb_model' is not defined

## === cell 6
submission = pd.DataFrame(
    {
        "id": df["id"].values,
        "winner_model_a": y_pred[:, 0],
        "winner_model_b": y_pred[:, 1],
        "winner_tie": y_pred[:, 2],
    }
)
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4200293556.py in <cell line: 0>()
      2     {
      3         "id": df["id"].values,
----> 4         "winner_model_a": y_pred[:, 0],
      5         "winner_model_b": y_pred[:, 1],
      6         "winner_tie": y_pred[:, 2],

NameError: name 'y_pred' is not defined
