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

1.0986028982898544

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import pandas as pd
import numpy as np

TEST_PATH = "/kaggle/input/lmsys-chatbot-arena/test.csv"
TRAIN_PATH = "/kaggle/input/lmsys-chatbot-arena/train.csv"

final_df = pd.read_csv(TEST_PATH)
train_df = pd.read_csv(TRAIN_PATH)

final_df["text"] = (
    "User prompt: "
    + final_df["prompt"]
    + "\n\nModel A :\n"
    + final_df["response_a"]
    + "\n\n--------\n\nModel B:\n"
    + final_df["response_b"]
)

train_df["text"] = (
    "User prompt: "
    + train_df["prompt"]
    + "\n\nModel A :\n"
    + train_df["response_a"]
    + "\n\n--------\n\nModel B:\n"
    + train_df["response_b"]
)



## === cell 1
all_sentences = pd.concat([train_df["text"], final_df["text"]])
tokenized = [sentence.split() for sentence in all_sentences]

vocab = {"<pad>": 0, "<unk>": 1}
for word in Counter(w for sent in tokenized for w in sent):
    if word not in vocab:
        vocab[word] = len(vocab)

vocab_size = len(vocab)


def encode(tokens):
    """Convert list of tokens to tensor of token ids."""
    return torch.tensor(
        [vocab.get(t, vocab["<unk>"]) for t in tokens], dtype=torch.long
    )




## === cell 2
class ChatArenaDataset(Dataset):
    def __init__(self, df):
        self.texts = df["text"].values
        self.labels = df[["winner_model_a", "winner_model_b", "winner_tie"]].values
        self.int_labels = np.argmax(self.labels, axis=1).astype(np.int64)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts[idx].split()
        encoded = encode(tokens)
        label = self.int_labels[idx]
        return encoded, label


def collate_fn(batch):
    seqs, labs = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    padded = torch.nn.utils.rnn.pad_sequence(
        seqs, batch_first=True, padding_value=vocab["<pad>"]
    )
    return padded, torch.tensor(labs, dtype=torch.long), lengths


train_dataset = ChatArenaDataset(train_df)
train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn
)




## === cell 3
class GRUClassifier(nn.Module):
    def __init__(
        self, vocab_size, embed_dim=128, hidden_dim=64, hidden_dim2=32, num_classes=3
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab["<pad>"])
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x, lengths=None):
        embed = self.embedding(x)
        if lengths is not None:
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                embed, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, h_n = self.gru(packed)
        else:
            _, h_n = self.gru(embed)
        out = self.fc(h_n[-1])
        logits = self.fc2(out)
        return logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GRUClassifier(vocab_size=vocab_size).to(device)



## === cell 4
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
for epoch in range(1):  # 1 epoch
    total_loss = 0.0
    for batch_x, batch_y, batch_len in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_len = batch_len.to(device)

        optimizer.zero_grad()
        logits = model(batch_x, batch_len)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_x.size(0)

    avg_loss = total_loss / len(train_dataset)
    print(f"Epoch {epoch+1}, Avg loss: {avg_loss:.4f}")



## === cell 5
model.eval()
test_texts = final_df["text"].values
test_tokenized = [t.split() for t in test_texts]

test_encoded = [encode(tokens) for tokens in test_tokenized]
test_padded = torch.nn.utils.rnn.pad_sequence(
    test_encoded, batch_first=True, padding_value=vocab["<pad>"]
)
test_lengths = torch.tensor([len(seq) for seq in test_encoded], dtype=torch.long)

with torch.no_grad():
    logits = model(test_padded.to(device), test_lengths.to(device))
    probs = F.softmax(logits, dim=1).cpu().numpy()

final_df["winner_model_a"] = probs[:, 0]
final_df["winner_model_b"] = probs[:, 1]
final_df["winner_tie"] = probs[:, 2]



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
OutOfMemoryError                          Traceback (most recent call last)
/tmp/ipykernel_55/1827465059.py in <cell line: 0>()
     12 
     13 with torch.no_grad():
---> 14     logits = model(test_padded.to(device), test_lengths.to(device))
     15     probs = F.softmax(logits, dim=1).cpu().numpy()
     16 

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

/tmp/ipykernel_55/2014412730.py in forward(self, x, lengths)
     14         if lengths is not None:
     15             # Pack padded sequence for efficiency
---> 16             packed = torch.nn.utils.rnn.pack_padded_sequence(
     17                 embed, lengths.cpu(), batch_first=True, enforce_sorted=False
     18             )

/usr/local/lib/python3.11/dist-packages/torch/nn/utils/rnn.py in pack_padded_sequence(input, lengths, batch_first, enforce_sorted)
    339         input = input.index_select(batch_dim, sorted_indices)
    340 
--> 341     data, batch_sizes = _VF._pack_padded_sequence(input, lengths, batch_first)
    342     return _packed_sequence_init(data, batch_sizes, sorted_indices, None)
    343 

OutOfMemoryError: CUDA out of memory. Tried to allocate 1.31 GiB. GPU 0 has a total capacity of 47.53 GiB of which 1.09 GiB is free. Process 2483754 has 46.43 GiB memory in use. Of the allocated memory 45.61 GiB is allocated by PyTorch, and 509.17 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

## === cell 6
submission_path = "submission.csv"
final_df[["id", "winner_model_a", "winner_model_b", "winner_tie"]].to_csv(
    submission_path, index=False
)
print(f"Submission written to {submission_path}")

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/1193769506.py in <cell line: 0>()
      1 # Write submission file (correct filename and column order)
      2 submission_path = "submission.csv"
----> 3 final_df[["id", "winner_model_a", "winner_model_b", "winner_tie"]].to_csv(
      4     submission_path, index=False
      5 )

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['winner_model_a', 'winner_model_b', 'winner_tie'] not in index"
