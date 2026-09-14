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

geopandas==0.14.4
joblib==1.5.2
lightning-utilities==0.15.2
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

0.7681975217610972

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import glob
import gc
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel

import pytorch_lightning as L

from sklearn.cluster import KMeans
from sklearn.metrics import cohen_kappa_score

warnings.filterwarnings("ignore")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)



## === cell 1
DATA_DIR = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"
TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
SAMPLE_SUB_PATH = f"{DATA_DIR}/sample_submission.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

test_df = test_df.rename(columns={"full_text": "text"})
train_df["score"] = train_df["score"] - 1  # keep original logic
train_df = train_df.rename(columns={"full_text": "text", "score": "labels"})

assert "essay_id" in test_df.columns and "text" in test_df.columns
assert (
    "essay_id" in train_df.columns
    and "text" in train_df.columns
    and "labels" in train_df.columns
)




## === cell 2
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


class EmbeddingsDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=1024):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.df.loc[idx, "text"],
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        return enc


def find_local_transformers_model_dir():
    """
    Try to locate a locally-available transformers model directory inside /kaggle/input.
    Minimal heuristic: find folders containing config.json.
    """
    candidates = []
    for base in ["/kaggle/input", "/kaggle/data"]:
        if not os.path.isdir(base):
            continue
        for cfg in glob.glob(os.path.join(base, "**", "config.json"), recursive=True):
            candidates.append(os.path.dirname(cfg))
    for c in candidates:
        lc = c.lower()
        if "deberta" in lc:
            return c
    return candidates[0] if candidates else None


MODEL_DIR = find_local_transformers_model_dir()
FALLBACK_MODEL_NAME = "microsoft/deberta-v3-small"

tokenizer = None
model = None
try:
    if MODEL_DIR is not None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
        model = AutoModel.from_pretrained(MODEL_DIR, local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            FALLBACK_MODEL_NAME, local_files_only=True
        )
        model = AutoModel.from_pretrained(FALLBACK_MODEL_NAME, local_files_only=True)
except Exception as e:
    raise RuntimeError(
        f"Could not load a local transformers model. "
        f"Found MODEL_DIR={MODEL_DIR}. Error: {repr(e)}"
    )

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()




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
/tmp/ipykernel_55/773135117.py in <cell line: 0>()
     57     else:
---> 58         tokenizer = AutoTokenizer.from_pretrained(
     59             FALLBACK_MODEL_NAME, local_files_only=True

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
/tmp/ipykernel_55/773135117.py in <cell line: 0>()
     61         model = AutoModel.from_pretrained(FALLBACK_MODEL_NAME, local_files_only=True)
     62 except Exception as e:
---> 63     raise RuntimeError(
     64         f"Could not load a local transformers model. "
     65         f"Found MODEL_DIR={MODEL_DIR}. Error: {repr(e)}"

RuntimeError: Could not load a local transformers model. Found MODEL_DIR=None. Error: OSError("We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.\nCheck your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.")

## === cell 3
@torch.no_grad()
def compute_embeddings(df, tokenizer, model, batch_size=32, max_length=1024):
    ds = EmbeddingsDataset(df, tokenizer, max_length=max_length)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    embs = []

    for batch in tqdm(dl, total=len(dl), desc="Embedding"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        pooled = mean_pooling(outputs.last_hidden_state, batch["attention_mask"])
        embs.append(pooled.detach().cpu().numpy())

    return np.vstack(embs)


train_emb = compute_embeddings(
    train_df[["text"]], tokenizer, model, batch_size=32, max_length=1024
)
test_emb = compute_embeddings(
    test_df[["text"]], tokenizer, model, batch_size=32, max_length=1024
)

del model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2673625316.py in <cell line: 0>()
     21 
     22 # Compute embeddings for both train and test to replace missing external .npy files.
---> 23 train_emb = compute_embeddings(
     24     train_df[["text"]], tokenizer, model, batch_size=32, max_length=1024
     25 )

/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py in decorate_context(*args, **kwargs)
    114     def decorate_context(*args, **kwargs):
    115         with ctx_factory():
--> 116             return func(*args, **kwargs)
    117 
    118     return decorate_context

/tmp/ipykernel_55/2673625316.py in compute_embeddings(df, tokenizer, model, batch_size, max_length)
     11     embs = []
     12 
---> 13     for batch in tqdm(dl, total=len(dl), desc="Embedding"):
     14         batch = {k: v.to(device) for k, v in batch.items()}
     15         outputs = model(**batch)

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

TypeError: Caught TypeError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/tmp/ipykernel_55/773135117.py", line 17, in __getitem__
    enc = self.tokenizer(
          ^^^^^^^^^^^^^^^
TypeError: 'NoneType' object is not callable


## === cell 4
n_tasks = 8  # small and stable; acts as a proxy for the original task clustering
kmeans = KMeans(n_clusters=n_tasks, random_state=42, n_init=10)
train_df["task"] = kmeans.fit_predict(train_emb)
test_df["task"] = kmeans.predict(test_emb)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3623404492.py in <cell line: 0>()
      3 n_tasks = 8  # small and stable; acts as a proxy for the original task clustering
      4 kmeans = KMeans(n_clusters=n_tasks, random_state=42, n_init=10)
----> 5 train_df["task"] = kmeans.fit_predict(train_emb)
      6 test_df["task"] = kmeans.predict(test_emb)
      7 

NameError: name 'train_emb' is not defined

## === cell 5
class CustomDataset(Dataset):
    def __init__(self, data, tasks, train_data, train_labels, train_task, n_sample=1):
        self.data = data
        self.tasks = tasks
        self.train_data = train_data
        self.train_labels = train_labels
        self.train_task = train_task

        self.n_sample = n_sample

        self.train_labels = self.train_labels.astype(int).reshape(-1)

        self.train_num_labels = len(np.unique(self.train_labels))
        self.train_label_indices = {
            i: np.where(self.train_labels == i)[0] for i in range(self.train_num_labels)
        }

        self.train_num_tasks = len(np.unique(self.train_task))
        self.train_task_indices = {
            i: np.where(self.train_task == i)[0] for i in range(self.train_num_tasks)
        }

        self.pairs = self.make_pairs()

    def make_pairs(self):
        pairs = np.empty((0, 2), dtype=int)
        data_len = len(self.data)

        task_indices_sets = {
            task: set(indices) for task, indices in self.train_task_indices.items()
        }
        label_candidates = {
            label: np.array(list(set(indices) & set(range(len(self.train_data)))))
            for label, indices in self.train_label_indices.items()
        }

        print("Making essay pairs...")
        for i in tqdm(range(data_len), total=data_len):
            e1_task = int(self.tasks[i])
            task_set = task_indices_sets.get(e1_task, set(range(len(self.train_data))))

            for label, candidates in label_candidates.items():
                if len(task_set) == 0:
                    continue
                valid_mask = np.isin(
                    candidates, np.fromiter(task_set, dtype=int, count=len(task_set))
                )
                valid_candidates = candidates[valid_mask]
                if len(valid_candidates) > 0:
                    e2 = np.random.choice(valid_candidates, self.n_sample, replace=True)
                    for j in e2:
                        pairs = np.vstack((pairs, np.array([i, j], dtype=int)))

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        e1 = int(self.pairs[idx][0])
        e2 = int(self.pairs[idx][1])

        x_e1 = self.data[e1]
        x_e2 = self.train_data[e2]
        y_e2 = float(self.train_labels[e2])

        output = {
            "x_e1": torch.tensor(x_e1, dtype=torch.float32),
            "x_e2": torch.tensor(x_e2, dtype=torch.float32),
            "relative_score": torch.tensor(0.0, dtype=torch.float32),
            "y_e1": torch.tensor(0.0, dtype=torch.float32),
            "y_e2": torch.tensor(y_e2, dtype=torch.float32),
        }
        return output




## === cell 6
class Projector(nn.Module):
    def __init__(self, d, dropout=0.2):
        super().__init__()
        self.p = dropout

        self.linear_1 = nn.Linear(d, int(d / 2))
        self.bn_1 = nn.BatchNorm1d(int(d / 2))

        self.linear_2 = nn.Linear(int(d / 2), d)
        self.bn_2 = nn.BatchNorm1d(d)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.bn_1(x)
        x = F.tanh(x)
        x = F.dropout(x, p=self.p, training=self.training)

        x = self.linear_2(x)
        x = self.bn_2(x)
        x = F.tanh(x)
        x = F.dropout(x, p=self.p, training=self.training)
        return x


class AESNet(L.LightningModule):
    def __init__(self, input_dim=768, lr=1e-5):
        super(AESNet, self).__init__()
        self.save_hyperparameters()

        self.lr = lr

        self.feature_extractor = nn.Sequential(
            Projector(input_dim),
            nn.Linear(input_dim, input_dim),
        )

        self.relative_score_head = nn.Linear(input_dim, 1, bias=False)
        self.mse_loss = nn.MSELoss()

    def _scale_back(self, x, min_x=-5, max_x=5):
        return x * (max_x - min_x) + min_x

    def forward(self, e1, e2, y_e2):
        x_1 = self.feature_extractor(e1)
        f1 = e1 + x_1
        f1 = F.normalize(f1, dim=-1)

        x_2 = self.feature_extractor(e2)
        f2 = e2 + x_2
        f2 = F.normalize(f2, dim=-1)

        dv = f1 - f2
        relative_score_hat = torch.sigmoid(self.relative_score_head(dv))
        y_e1_hat = relative_score_hat + y_e2
        return relative_score_hat, y_e1_hat

    def training_step(self, batch, batch_idx):
        relative_score_hat, _ = self.forward(
            batch["x_e1"], batch["x_e2"], batch["y_e2"]
        )
        loss = self.mse_loss(relative_score_hat, batch["relative_score"])
        self.log("train loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        relative_score_hat, _ = self.forward(
            batch["x_e1"], batch["x_e2"], batch["y_e2"]
        )
        loss = self.mse_loss(relative_score_hat, batch["relative_score"])

        rs_hat = self._scale_back(relative_score_hat)
        rs = self._scale_back(batch["relative_score"])

        qwk = cohen_kappa_score(
            rs.detach().cpu().numpy().round(0).astype(int),
            rs_hat.detach().cpu().numpy().round(0).astype(int),
            weights="quadratic",
        )

        self.log("validation loss", loss, prog_bar=True)
        self.log("validation qwk", qwk, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        relative_score_hat, y_e1_hat = self.forward(
            batch["x_e1"], batch["x_e2"], batch["y_e2"]
        )
        relative_score_hat = self._scale_back(relative_score_hat)
        y_e1_hat = relative_score_hat + batch["y_e2"]
        return y_e1_hat

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.6,
            total_iters=self.trainer.max_epochs,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }




## === cell 7
test_ds = CustomDataset(
    test_emb,
    test_df["task"].values,
    train_emb,
    train_df["labels"].values,
    train_df["task"].values,
)

test_dl = DataLoader(
    test_ds, batch_size=2, shuffle=False, num_workers=2, drop_last=False
)

ckpt_paths = glob.glob("/kaggle/input/**/models/**/*.ckpt", recursive=True)
test_pred = np.zeros((len(test_df), 1), dtype=np.float32)
model_count = 0

if len(ckpt_paths) == 0:
    print(
        "No .ckpt files found under /kaggle/input/**/models/**/*.ckpt; using randomly initialized model."
    )
    input_dim = test_emb.shape[1]
    model = AESNet(input_dim=input_dim, lr=1e-5)
    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )
    output = trainer.predict(model=model, dataloaders=test_dl)
    output = torch.vstack(output).detach().cpu().numpy()

    out_df = pd.DataFrame(
        {"essay_idx": test_ds.pairs[:, 0], "pred_label": output.flatten()}
    )
    out_df = out_df.groupby("essay_idx", sort=True).agg({"pred_label": "mean"})
    preds = out_df["pred_label"].values.reshape(-1, 1)

    test_pred += preds
    model_count = 1
else:
    print(f"Found {len(ckpt_paths)} checkpoint(s). Predicting ensemble...")
    for path in ckpt_paths:
        print(path)
        model = AESNet.load_from_checkpoint(path)
        trainer = L.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
        )

        output = trainer.predict(model=model, dataloaders=test_dl)
        output = torch.vstack(output).detach().cpu().numpy()

        out_df = pd.DataFrame(
            {"essay_idx": test_ds.pairs[:, 0], "pred_label": output.flatten()}
        )
        out_df = out_df.groupby("essay_idx", sort=True).agg({"pred_label": "mean"})
        preds = out_df["pred_label"].values.reshape(-1, 1)

        test_pred += preds
        model_count += 1

test_pred = test_pred / max(model_count, 1)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4145663423.py in <cell line: 0>()
      1 # Build test dataset/dataloader exactly as the original pipeline intends, using computed embeddings.
      2 test_ds = CustomDataset(
----> 3     test_emb,
      4     test_df["task"].values,
      5     train_emb,

NameError: name 'test_emb' is not defined

## === cell 8
test_pred = np.round(test_pred, 0).astype(int) + 1
test_pred = np.clip(test_pred, 1, 6)

submission_df = pd.DataFrame(
    {"essay_id": test_df["essay_id"].values, "score": test_pred.flatten()}
)

if len(sample_sub) == len(submission_df) and set(sample_sub["essay_id"]) == set(
    submission_df["essay_id"]
):
    submission_df = sample_sub[["essay_id"]].merge(
        submission_df, on="essay_id", how="left"
    )
else:
    submission_df = submission_df.sort_values("essay_id").reset_index(drop=True)

submission_df.to_csv("submission.csv", index=False)
print(submission_df.head())
print("Wrote submission.csv with shape:", submission_df.shape)

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2345296881.py in <cell line: 0>()
      1 # Convert back to competition scores: original code did round + 1 after training used score-1 labels.
----> 2 test_pred = np.round(test_pred, 0).astype(int) + 1
      3 test_pred = np.clip(test_pred, 1, 6)
      4 
      5 submission_df = pd.DataFrame(

NameError: name 'test_pred' is not defined
