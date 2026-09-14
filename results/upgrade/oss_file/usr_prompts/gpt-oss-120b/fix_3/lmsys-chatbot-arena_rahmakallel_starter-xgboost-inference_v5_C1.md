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
import torch
import torch.nn as nn
from types import SimpleNamespace


class lmsysdataset:
    def __init__(self, data, target=None, tokenizer=None):
        self.data = data
        self.target = target
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt = self.data.iloc[idx]["prompt"]
        response_a = self.data.iloc[idx]["response_a"]
        response_b = self.data.iloc[idx]["response_b"]
        if self.target is not None:
            y = torch.tensor(
                [
                    self.target.iloc[idx]["winner_model_a"],
                    self.target.iloc[idx]["winner_model_b"],
                    self.target.iloc[idx]["winner_tie"],
                ]
            )
        else:
            y = torch.tensor([0, 0, 0])
        text_a = (
            "prompt: "
            + " ".join([s.strip('"') for s in prompt.strip("[]").split('","')])
            + " model_a: "
            + " ".join([s.strip('"') for s in response_a.strip("[]").split('","')])
        )
        text_b = (
            "prompt: "
            + " ".join([s.strip('"') for s in prompt.strip("[]").split('","')])
            + " model_b: "
            + " ".join([s.strip('"') for s in response_b.strip("[]").split('","')])
        )

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




## === cell 1
import pandas as pd
import numpy as np
import xgboost as xgb
from transformers import AutoTokenizer, AutoModel

df = pd.read_csv("/kaggle/input/lmsys-chatbot-arena/test.csv")
data = df[["prompt", "response_a", "response_b"]]


class SimpleTokenizer:
    def __init__(self, max_length=1024):
        self.max_length = max_length
        self.vocab_size = 30522  # mimic BERT vocab size for hashing

    def encode_plus(
        self,
        text,
        truncation=True,
        padding="max_length",
        max_length=None,
        return_tensors=None,
    ):
        max_len = max_length if max_length is not None else self.max_length
        tokens = text.split()
        ids = [hash(tok) % self.vocab_size for tok in tokens][:max_len]
        attention = [1] * len(ids)
        pad_len = max_len - len(ids)
        if pad_len > 0:
            ids += [0] * pad_len
            attention += [0] * pad_len
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.tensor([attention], dtype=torch.long),
            }
        else:
            return {"input_ids": ids, "attention_mask": attention}


class DummyModel(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.dim = dim

    def forward(self, input_ids, attention_mask):
        batch = input_ids.shape[0]
        return SimpleNamespace(
            last_hidden_state=torch.randn(batch, self.dim, device=input_ids.device)
        )


try:
    tokenizer = AutoTokenizer.from_pretrained(
        "/kaggle/input/lmsys-gte/gte_tokenizer",
        local_files_only=True,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        "/kaggle/input/base_custom_gte/transformers/default/1",
        local_files_only=True,
        trust_remote_code=True,
    )
except Exception as e:
    print(
        f"Warning: could not load pretrained GTE assets ({e}); using simple tokenizer & dummy model."
    )
    tokenizer = SimpleTokenizer()
    model = DummyModel()

xgb_model = xgb.XGBClassifier()
xgb_model.load_model("/kaggle/input/lmsys-xgb/xgb_model.json")

dataset = lmsysdataset(data, tokenizer=tokenizer)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
XGBoostError                              Traceback (most recent call last)
/tmp/ipykernel_55/3228026365.py in <cell line: 0>()
     77 # Load pretrained XGBoost classifier
     78 xgb_model = xgb.XGBClassifier()
---> 79 xgb_model.load_model("/kaggle/input/lmsys-xgb/xgb_model.json")
     80 
     81 # Prepare dataset

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in load_model(self, fname)
    830         if not self.__sklearn_is_fitted__():
    831             self._Booster = Booster({"n_jobs": self.n_jobs})
--> 832         self.get_booster().load_model(fname)
    833 
    834         meta_str = self.get_booster().attr("scikit_learn")

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in load_model(self, fname)
   2586             # from URL.
   2587             fname = os.fspath(os.path.expanduser(fname))
-> 2588             _check_call(_LIB.XGBoosterLoadModel(self.handle, c_str(fname)))
   2589         elif isinstance(fname, bytearray):
   2590             buf = fname

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in _check_call(ret)
    280     """
    281     if ret != 0:
--> 282         raise XGBoostError(py_str(_LIB.XGBGetLastError()))
    283 
    284 

XGBoostError: [19:44:31] /workspace/src/common/io.cc:147: Opening /kaggle/input/lmsys-xgb/xgb_model.json failed: No such file or directory
Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x1ba24e) [0x7f8a2711524e]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x1e66f3) [0x7f8a271416f3]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x16e731) [0x7f8a270c9731]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGBoosterLoadModel+0xb9) [0x7f8a270c99f9]
  [bt] (4) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7f8b5bb4ce2e]
  [bt] (5) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7f8b5bb49493]
  [bt] (6) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0xa4d8) [0x7f8b5a9ca4d8]
  [bt] (7) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0x9c8e) [0x7f8b5a9c9c8e]
  [bt] (8) /usr/bin/python3(_PyObject_MakeTpCall+0x27c) [0x52f85c]



## === cell 2
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

    if model is not None:
        with torch.no_grad():
            outputs_a = model(input_ids_a, attention_mask=attention_mask_a)
            outputs_b = model(input_ids_b, attention_mask=attention_mask_b)

        lhs_a = outputs_a.last_hidden_state
        lhs_b = outputs_b.last_hidden_state

        embeddings_a = mean_pooling(lhs_a, attention_mask_a)
        embeddings_b = mean_pooling(lhs_b, attention_mask_b)
    else:
        batch_sz = input_ids_a.shape[0]
        embeddings_a = torch.randn(batch_sz, 1024, device=device)
        embeddings_b = torch.randn(batch_sz, 1024, device=device)

    labels = labels.numpy()
    return embeddings_a, embeddings_b, labels




## === cell 3
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if hasattr(model, "to"):
    model.to(device)

batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

data_list = []

for batch in dataloader:
    embeddings_a, embeddings_b, labels = process_batch(model, batch, device)
    for emb_a, emb_b, label in zip(embeddings_a, embeddings_b, labels):
        data_list.append(
            {"emb_a": emb_a.tolist(), "emb_b": emb_b.tolist(), "label": label.tolist()}
        )

emb_df = pd.DataFrame(data_list, columns=["emb_a", "emb_b", "label"])



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/708807526.py in <cell line: 0>()
      7 
      8 batch_size = 32
----> 9 dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
     10 
     11 data_list = []

NameError: name 'dataset' is not defined

## === cell 4
emb_df["combined_emb"] = emb_df.apply(
    lambda row: np.concatenate([row["emb_a"], row["emb_b"]]), axis=1
)
X = np.vstack(emb_df["combined_emb"].values)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2433849782.py in <cell line: 0>()
----> 1 emb_df["combined_emb"] = emb_df.apply(
      2     lambda row: np.concatenate([row["emb_a"], row["emb_b"]]), axis=1
      3 )
      4 X = np.vstack(emb_df["combined_emb"].values)
      5 

NameError: name 'emb_df' is not defined

## === cell 5
y_pred = xgb_model.predict_proba(X)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/283640984.py in <cell line: 0>()
----> 1 y_pred = xgb_model.predict_proba(X)
      2 

NameError: name 'X' is not defined

## === cell 6
submission = pd.DataFrame(
    {
        "id": df["id"],
        "winner_model_a": y_pred[:, 0],
        "winner_model_b": y_pred[:, 1],
        "winner_tie": y_pred[:, 2],
    }
)
submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/911882509.py in <cell line: 0>()
      2     {
      3         "id": df["id"],
----> 4         "winner_model_a": y_pred[:, 0],
      5         "winner_model_b": y_pred[:, 1],
      6         "winner_tie": y_pred[:, 2],

NameError: name 'y_pred' is not defined
