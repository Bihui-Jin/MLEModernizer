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

0.66238

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import torch
from sklearn.metrics import cohen_kappa_score
from torch import nn, optim
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
from torch.nn import functional as F
from collections import Counter



## === cell 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)




## === cell 3
class CustomDatasetIterator:
    def __init__(self, dataset):
        self.dataset = dataset
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.dataset):
            raise StopIteration
        else:
            sample = self.dataset[self.index]
            self.index += 1
            return sample


class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sample = (row["essay_id"], row["full_text"], row["score"])
        if self.transform:
            sample = self.transform(sample)
        return sample




## === cell 4
def tokenizer(text):
    return text.lower().split()


train_csv_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
dataset = CustomDataset(train_csv_path)

counter = Counter()
for _, text, _ in dataset:
    counter.update(tokenizer(text))

vocab = {"<pad>": 0, "<unk>": 1}
for token, freq in counter.items():
    if token not in vocab:
        vocab[token] = len(vocab)

vocab_size = len(vocab)


def text_pipeline(x):
    return [vocab.get(tok, vocab["<unk>"]) for tok in tokenizer(x)]


def label_pipeline(x):
    return int(x) - 1




## === cell 5
def collate_batch(batch):
    label_list, text_list = [], []
    for _, _text, _label in batch:
        label_list.append(label_pipeline(_label))
        tokenized_text = text_pipeline(_text)
        text_list.append(torch.tensor(tokenized_text, dtype=torch.int64))

    label_tensor = torch.tensor(label_list, dtype=torch.int64)
    padded_text = pad_sequence(
        text_list, batch_first=True, padding_value=vocab["<pad>"]
    )
    return label_tensor.to(device), padded_text.to(device)




## === cell 6
train_size = int(0.85 * len(dataset))  # 85% of the dataset for training
val_size = len(dataset) - train_size
indices = list(range(len(dataset)))
train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)




## === cell 7
class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)  # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T)
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x




## === cell 8
class EssayScorer(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim, hidden_dim, output_size, num_layers=2
    ):
        super(EssayScorer, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=vocab["<pad>"]
        )
        self.embedding_dropout = SpatialDropout(0.3)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim * 2, output_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.embedding_dropout(embedded)

        lstm_out, _ = self.lstm(embedded)
        lstm_out2, _ = self.lstm2(lstm_out)

        avg_pool = torch.mean(lstm_out2, dim=1)
        max_pool, _ = torch.max(lstm_out2, dim=1)

        h_conc = torch.cat((max_pool, avg_pool), dim=1)  # shape (batch, 2*hidden_dim)
        logits = self.fc(h_conc)  # shape (batch, output_size)
        return logits




## === cell 9
num_classes = 6  # scores 1‑6 → 0‑5 after label_pipeline
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
model = EssayScorer(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, num_classes)
model.to(device)



## === cell 10
LR = 0.01
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = CrossEntropyLoss()
EPOCHS = 15
BATCH_SIZE = 32

train_dataloader = DataLoader(
    dataset, batch_size=BATCH_SIZE, collate_fn=collate_batch, sampler=train_sampler
)
valid_dataloader = DataLoader(
    dataset, batch_size=BATCH_SIZE, collate_fn=collate_batch, sampler=val_sampler
)




## === cell 11
def train(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for scores, essays in dataloader:
        optimizer.zero_grad()
        outputs = model(essays)
        loss = criterion(outputs, scores)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)


def evaluate(model, dataloader):
    model.eval()
    pred_scores = []
    true_scores = []
    with torch.no_grad():
        for scores, essays in dataloader:
            outputs = model(essays)
            pred_scores.extend(outputs.cpu().numpy())
            true_scores.extend(scores.cpu().numpy())
    kappa = cohen_kappa_score(
        true_scores, np.argmax(pred_scores, axis=1), weights="quadratic"
    )
    return kappa




## === cell 12
do_train = True

if do_train:
    for epoch in range(EPOCHS):
        train_loss = train(model, train_dataloader, criterion, optimizer)
        val_kappa = evaluate(model, valid_dataloader)
        print(
            f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Val Kappa: {val_kappa:.4f}"
        )

    torch.save(model.state_dict(), "/kaggle/working/score_model.pth")
else:
    model.load_state_dict(torch.load("/kaggle/working/score_model.pth"))




## === cell 13
def inference(model, dataloader):
    model.eval()
    pred_scores = []
    idx = []
    with torch.no_grad():
        for ids_batch, essays in dataloader:
            outputs = model(essays)
            pred_scores.extend(outputs.cpu().numpy())
            idx.extend(ids_batch)
    return idx, pred_scores


def collate_testbatch(batch):
    id_list, text_list = [], []
    for _id, _text in batch:
        id_list.append(_id)
        tokenized = text_pipeline(_text)
        text_list.append(torch.tensor(tokenized, dtype=torch.int64))
    padded_text = pad_sequence(
        text_list, batch_first=True, padding_value=vocab["<pad>"]
    )
    return id_list, padded_text.to(device)




## === cell 14
test_dataset = CustomDataset(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
)
test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_testbatch
)

ids, preds = inference(model, test_dataloader)
preds = np.argmax(preds, axis=1) + 1  # convert back to 1‑6 range
submission = pd.DataFrame({"essay_id": ids, "score": preds})
print(f"Submission shape: {submission.shape}")
submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'score'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_56/3455216816.py in <cell line: 0>()
      6 )
      7 
----> 8 ids, preds = inference(model, test_dataloader)
      9 preds = np.argmax(preds, axis=1) + 1  # convert back to 1‑6 range
     10 submission = pd.DataFrame({"essay_id": ids, "score": preds})

/tmp/ipykernel_56/2893753194.py in inference(model, dataloader)
      4     idx = []
      5     with torch.no_grad():
----> 6         for ids_batch, essays in dataloader:
      7             outputs = model(essays)
      8             pred_scores.extend(outputs.cpu().numpy())

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
    762     def _next_data(self):
    763         index = self._next_index()  # may raise StopIteration
--> 764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
    766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in <listcomp>(.0)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/tmp/ipykernel_56/2822043676.py in __getitem__(self, idx)
     27         # returns (essay_id, full_text, score) as a tuple of Python objects
     28         row = self.data.iloc[idx]
---> 29         sample = (row["essay_id"], row["full_text"], row["score"])
     30         if self.transform:
     31             sample = self.transform(sample)

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in __getitem__(self, key)
   1119 
   1120         elif key_is_scalar:
-> 1121             return self._get_value(key)
   1122 
   1123         # Convert generator to list before going through hashable part

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in _get_value(self, label, takeable)
   1235 
   1236         # Similar to Index.get_value, but we do not fall back to positional
-> 1237         loc = self.index.get_loc(label)
   1238 
   1239         if is_integer(loc):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'score'
