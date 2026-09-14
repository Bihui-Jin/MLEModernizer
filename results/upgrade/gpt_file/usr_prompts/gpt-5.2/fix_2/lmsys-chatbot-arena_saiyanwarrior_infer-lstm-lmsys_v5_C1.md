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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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

1.1056421633619875

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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from tqdm import tqdm

BASE_INPUT = "/kaggle/input/lmsys-chatbot-arena"
train_path = os.path.join(BASE_INPUT, "train.csv")
test_path = os.path.join(BASE_INPUT, "test.csv")
sample_path = os.path.join(BASE_INPUT, "sample_submission.csv")

train = pd.read_csv(train_path)
final_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

print("train:", train.shape, "test:", final_df.shape, "sample:", sample_sub.shape)
print("train cols:", list(train.columns))
print("test cols:", list(final_df.columns))



## === cell 1
final_df["prompt"] = final_df["prompt"].fillna("")
final_df["response_a"] = final_df["response_a"].fillna("")
final_df["response_b"] = final_df["response_b"].fillna("")

train["prompt"] = train["prompt"].fillna("")
train["response_a"] = train["response_a"].fillna("")
train["response_b"] = train["response_b"].fillna("")

final_df["text"] = (
    "User prompt: "
    + final_df["prompt"]
    + "\n\nModel A :\n"
    + final_df["response_a"]
    + "\n\n--------\n\nModel B:\n"
    + final_df["response_b"]
)

train["text"] = (
    "User prompt: "
    + train["prompt"]
    + "\n\nModel A :\n"
    + train["response_a"]
    + "\n\n--------\n\nModel B:\n"
    + train["response_b"]
)

print(final_df["text"].iloc[0][:500])



## === cell 2
final_texts = final_df["text"].values
train_texts = train["text"].values

print("n_train_texts:", len(train_texts), "n_test_texts:", len(final_texts))



## === cell 3
batch_size = 8
num_classes = 3

train_tokenized = [t.split() for t in train_texts]
final_tokenized = [t.split() for t in final_texts]

vocab = {"<pad>": 0, "<unk>": 1}
for word in Counter(w for sent in train_tokenized for w in sent):
    vocab[word] = len(vocab)
for word in Counter(w for sent in final_tokenized for w in sent):
    vocab[word] = len(vocab)


def encode(sentence_tokens):
    return torch.tensor([vocab.get(w, 1) for w in sentence_tokens], dtype=torch.long)


print("vocab_size:", len(vocab))




## === cell 4
class TextDataset(Dataset):
    def __init__(self, tokenized_texts):
        self.tokens = tokenized_texts

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return (encode(self.tokens[idx]),)


def collate_fn(batch):
    sequences = [item[0] for item in batch]
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    return sequences_padded




## === cell 5
class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=64,
        hidden_dim=128,
        hidden_dim2=64,
        num_classes=num_classes,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        output, (h_n, c_n) = self.lstm(x)
        output = self.fc(h_n[-1])  # use final hidden state
        logits = self.fc2(output)
        return logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)



## === cell 6
ckpt_path = "/kaggle/input/training-v0-lmsys/lstm_classifier_0.pth"

model_loaded = LSTMClassifier(vocab_size=len(vocab))
state = torch.load(ckpt_path, map_location="cpu")

if (
    isinstance(state, dict)
    and "state_dict" in state
    and isinstance(state["state_dict"], dict)
):
    state = state["state_dict"]

if isinstance(state, dict):
    sample_key = next(iter(state.keys()))
    if sample_key.startswith("model."):
        state = {k.replace("model.", "", 1): v for k, v in state.items()}
    if sample_key.startswith("module."):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

model_loaded.load_state_dict(state, strict=True)
model_loaded.to(device)
model_loaded.eval()

print("Checkpoint loaded from:", ckpt_path)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/4289318193.py in <cell line: 0>()
      3 
      4 model_loaded = LSTMClassifier(vocab_size=len(vocab))
----> 5 state = torch.load(ckpt_path, map_location="cpu")
      6 
      7 # Handle potential wrappers (e.g., {"state_dict": ...})

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in load(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)
   1423         pickle_load_args["encoding"] = "utf-8"
   1424 
-> 1425     with _open_file_like(f, "rb") as opened_file:
   1426         if _is_zipfile(opened_file):
   1427             # The zipfile reader is going to advance the current file position.

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _open_file_like(name_or_buffer, mode)
    749 def _open_file_like(name_or_buffer, mode):
    750     if _is_path(name_or_buffer):
--> 751         return _open_file(name_or_buffer, mode)
    752     else:
    753         if "w" in mode:

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in __init__(self, name, mode)
    730 class _open_file(_opener):
    731     def __init__(self, name, mode):
--> 732         super().__init__(open(name, mode))
    733 
    734     def __exit__(self, *args):

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/training-v0-lmsys/lstm_classifier_0.pth'

## === cell 7
test_ds = TextDataset(final_tokenized)
test_loader = DataLoader(
    test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)

all_probs = []
with torch.no_grad():
    for x in tqdm(test_loader, desc="Infer"):
        x = x.to(device)
        logits = model_loaded(x)
        probs = F.softmax(logits, dim=1)
        all_probs.append(probs.detach().cpu().numpy())

probs = np.vstack(all_probs)
assert probs.shape == (len(final_df), 3), f"Unexpected probs shape: {probs.shape}"

eps = 1e-12
probs = np.clip(probs, eps, 1.0)
probs = probs / probs.sum(axis=1, keepdims=True)

print(
    "probs min/max:",
    probs.min(),
    probs.max(),
    "row_sums:",
    probs.sum(axis=1).min(),
    probs.sum(axis=1).max(),
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3705630275.py in <cell line: 0>()
      9     for x in tqdm(test_loader, desc="Infer"):
     10         x = x.to(device)
---> 11         logits = model_loaded(x)
     12         probs = F.softmax(logits, dim=1)
     13         all_probs.append(probs.detach().cpu().numpy())

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/tmp/ipykernel_55/3091832804.py in forward(self, x)
     16 
     17     def forward(self, x):
---> 18         x = self.embedding(x)
     19         output, (h_n, c_n) = self.lstm(x)
     20         output = self.fc(h_n[-1])  # use final hidden state

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/sparse.py in forward(self, input)
    188 
    189     def forward(self, input: Tensor) -> Tensor:
--> 190         return F.embedding(
    191             input,
    192             self.weight,

/usr/local/lib/python3.11/dist-packages/torch/nn/functional.py in embedding(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
   2549         # remove once script supports set_grad_enabled
   2550         _no_grad_embedding_renorm_(weight, input, max_norm, norm_type)
-> 2551     return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)
   2552 
   2553 

RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0! (when checking argument for argument index in method wrapper_CUDA__index_select)

## === cell 8
sub = pd.DataFrame(
    {
        "id": final_df["id"].values,
        "winner_model_a": probs[:, 0],
        "winner_model_b": probs[:, 1],
        "winner_tie": probs[:, 2],
    }
)

sub["id"] = sub["id"].astype(sample_sub["id"].dtype, copy=False)

required_cols = ["id", "winner_model_a", "winner_model_b", "winner_tie"]
assert list(sub.columns) == required_cols, f"Submission columns mismatch: {sub.columns}"
assert len(sub) == len(
    sample_sub
), f"Row count mismatch: {len(sub)} vs {len(sample_sub)}"

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/43153894.py in <cell line: 0>()
      3     {
      4         "id": final_df["id"].values,
----> 5         "winner_model_a": probs[:, 0],
      6         "winner_model_b": probs[:, 1],
      7         "winner_tie": probs[:, 2],

NameError: name 'probs' is not defined
