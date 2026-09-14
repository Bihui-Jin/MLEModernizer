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
from torch.utils.data import Dataset


class lmsysdataset(Dataset):
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


def _find_local_hf_model_dirs(root="/kaggle/input", max_dirs=2000):
    hits = []
    scanned = 0
    for dirpath, dirnames, filenames in os.walk(root):
        scanned += 1
        if scanned > max_dirs:
            break
        if "config.json" in filenames:
            hits.append(dirpath)
    return hits


tokenizer_candidates = [
    "/kaggle/input/lmsys-gte/gte_tokenizer",
    "/kaggle/input/lmsys-gte",
]
model_candidates = [
    "/kaggle/input/base_custom_gte/transformers/default/1",
    "/kaggle/input/base_custom_gte",
]


def _try_load_tokenizer(path):
    return AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)


def _try_load_model(path):
    return AutoModel.from_pretrained(
        path, trust_remote_code=True, local_files_only=True
    )


def _load_local_tokenizer(candidates):
    for p in candidates:
        if os.path.isdir(p):
            try:
                return _try_load_tokenizer(p)
            except Exception:
                pass

    for p in _find_local_hf_model_dirs("/kaggle/input"):
        try:
            return _try_load_tokenizer(p)
        except Exception:
            continue

    raise RuntimeError(
        "Could not find any local tokenizer to load (offline). "
        "Please attach a dataset with a HuggingFace tokenizer/model directory under /kaggle/input."
    )


def _load_local_model(candidates):
    for p in candidates:
        if os.path.isdir(p):
            try:
                return _try_load_model(p)
            except Exception:
                pass

    for p in _find_local_hf_model_dirs("/kaggle/input"):
        try:
            return _try_load_model(p)
        except Exception:
            continue

    raise RuntimeError(
        "Could not find any local transformer model to load (offline). "
        "Please attach a dataset with a HuggingFace model directory under /kaggle/input."
    )


tokenizer = _load_local_tokenizer(tokenizer_candidates)
model = _load_local_model(model_candidates)

xgb_model = xgb.XGBClassifier()
xgb_paths = [
    "/kaggle/input/lmsys-xgb/xgb_model.json",
    "/kaggle/input/lmsys-xgb/model.json",
    "/kaggle/input/lmsys-xgb/xgb.json",
]
_loaded = False
for p in xgb_paths:
    if os.path.exists(p):
        xgb_model.load_model(p)
        _loaded = True
        break
if not _loaded:
    raise FileNotFoundError(
        f"Could not find XGBoost model JSON in {xgb_paths}. Please attach the dataset containing xgb_model.json."
    )

dataset = lmsysdataset(data, tokenizer=tokenizer)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2274541461.py in <cell line: 0>()
     83 
     84 
---> 85 tokenizer = _load_local_tokenizer(tokenizer_candidates)
     86 model = _load_local_model(model_candidates)
     87 

/tmp/ipykernel_55/2274541461.py in _load_local_tokenizer(candidates)
     56             continue
     57 
---> 58     raise RuntimeError(
     59         "Could not find any local tokenizer to load (offline). "
     60         "Please attach a dataset with a HuggingFace tokenizer/model directory under /kaggle/input."

RuntimeError: Could not find any local tokenizer to load (offline). Please attach a dataset with a HuggingFace tokenizer/model directory under /kaggle/input.

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

print("Embedding shapes:", emb_a_all.shape, emb_b_all.shape, "X:", X.shape)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2260344963.py in <cell line: 0>()
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

y_pred = np.asarray(y_pred, dtype=np.float64)
y_pred = np.clip(y_pred, 1e-15, 1.0)
y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)

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

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2139724180.py in <cell line: 0>()
----> 1 y_pred = xgb_model.predict_proba(X)
      2 
      3 if y_pred.ndim != 2 or y_pred.shape[1] != 3:
      4     raise ValueError(f"Expected predict_proba to return (n,3), got {y_pred.shape}")
      5 

NameError: name 'xgb_model' is not defined
