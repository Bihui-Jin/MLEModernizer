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

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import torch
from sklearn.metrics import cohen_kappa_score
from torch import nn, optim
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn import functional as F


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1088013400.py in <cell line: 0>()
      5 from torch.nn.utils.rnn import pad_sequence
      6 from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
----> 7 from torchtext.data import get_tokenizer
      8 from torchtext.vocab import build_vocab_from_iterator
      9 from torch.nn import functional as F

ModuleNotFoundError: No module named 'torchtext'

## === cell 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        sample = self.data.iloc[idx, :].values
        if self.transform:
            sample = self.transform(sample)
        return sample


## === cell 4
dataset = CustomDataset('/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv')
tokenizer = get_tokenizer("basic_english")
def get_tokens(data_iter):
    for _, text, _ in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(get_tokens(dataset), specials=[""])
vocab.set_default_index(vocab[""])
vocab_size = len(vocab)

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2219636759.py in <cell line: 0>()
      1 dataset = CustomDataset('/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv')
----> 2 tokenizer = get_tokenizer("basic_english")
      3 def get_tokens(data_iter):
      4     for _, text, _ in data_iter:
      5         yield tokenizer(text)

NameError: name 'get_tokenizer' is not defined

## === cell 5
def collate_batch(batch):
    label_list, text_list = [], []
    for _, _text, _label in batch:
        label_list.append(label_pipeline(_label))
        tokenized_text = text_pipeline(_text)
        text_list.append(tokenized_text)

    label_list = torch.tensor(label_list, dtype=torch.int64)
    padded_text = pad_sequence([torch.tensor(text, dtype=torch.int64) for text in text_list], batch_first=True)
    text_list = padded_text
    return label_list.to(device), text_list.to(device)


## === cell 6
train_size = int(0.85 * len(dataset))  # 85% of the dataset for training
val_size = len(dataset) - train_size  # Remaining 20% for validation
indices = list(range(len(dataset)))
train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)


## === cell 7
class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x


## === cell 8
class EssayScorer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size, num_layers=2):
        super(EssayScorer, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dropout = SpatialDropout(0.3)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        self.linear1 = nn.Linear(2*hidden_dim, 2*hidden_dim)
        self.linear2 = nn.Linear(2*hidden_dim, 2*hidden_dim)

        self.linear_out = nn.Linear(2*hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim*2, output_size)
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
        avg_pool = torch.mean(lstm_out2, 1)
        max_pool, _ = torch.max(lstm_out2, 1)

        h_conc = torch.cat((max_pool, avg_pool), 1)
        h_conc_linear1  = F.relu(self.linear1(h_conc))
        h_conc_linear2  = F.relu(self.linear2(h_conc))

        hidden = h_conc + h_conc_linear1 + h_conc_linear2
        result = self.linear_out(hidden)
        out = self.fc(hidden)
        out = torch.cat([result, out],1)
        return out


## === cell 9
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

data_iter = CustomDatasetIterator(dataset)
num_classes = len(set(label for (_, text, label) in data_iter))

EMBEDDING_DIM = 64
HIDDEN_DIM = 128
model = EssayScorer(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, num_classes)
model.to(device)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2526468551.py in <cell line: 0>()
      7 EMBEDDING_DIM = 64
      8 HIDDEN_DIM = 128
----> 9 model = EssayScorer(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, num_classes)
     10 model.to(device)

NameError: name 'vocab_size' is not defined

## === cell 10
LR=0.01
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = CrossEntropyLoss()
EPOCHS = 15
BATCH_SIZE = 32

train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_batch, sampler=train_sampler)
valid_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_batch, sampler=val_sampler)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/923648646.py in <cell line: 0>()
      1 LR=0.01
      2 # optimizer = optim.SGD(model.parameters(), lr=LR)
----> 3 optimizer = optim.Adam(model.parameters(), lr=0.01)
      4 criterion = CrossEntropyLoss()
      5 # scheduler = optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)

NameError: name 'model' is not defined

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
    kappa = cohen_kappa_score(true_scores, np.argmax(pred_scores, axis=1), weights='quadratic')
    return kappa


## === cell 12
do_train = True

if do_train:
    for epoch in range(EPOCHS):
        train_loss = train(model, train_dataloader, criterion, optimizer)
        val_kappa = evaluate(model, valid_dataloader)
        print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Val Kappa: {val_kappa:.4f}")

    torch.save(model.state_dict(), "/kaggle/working/score_model.pth")
else:
    model.load_state_dict(torch.load("/kaggle/working/score_model.pth"))


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3825475774.py in <cell line: 0>()
      2 
      3 if do_train:
----> 4     for epoch in range(EPOCHS):
      5         train_loss = train(model, train_dataloader, criterion, optimizer)
      6         val_kappa = evaluate(model, valid_dataloader)

NameError: name 'EPOCHS' is not defined

## === cell 13
def inference(model, dataloader):
    model.eval()
    pred_scores = []
    idx = []
    with torch.no_grad():
        for id, essays in dataloader:
            outputs = model(essays)
            pred_scores.extend(outputs.cpu().numpy())
            idx.extend(id)
    return idx, pred_scores

def collate_testbatch(batch):
    id_list, text_list = [], []
    for _id, _text in batch:
        id_list.append(_id)
        tokenized_text = text_pipeline(_text)
        text_list.append(tokenized_text)
    padded_text = pad_sequence([torch.tensor(text, dtype=torch.int64) for text in text_list], batch_first=True)
    text_list = padded_text
    return id_list, text_list.to(device)


## === cell 14
test_dataset = CustomDataset("/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv")
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_testbatch)


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3157693221.py in <cell line: 0>()
      1 test_dataset = CustomDataset("/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv")
----> 2 test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_testbatch)

NameError: name 'BATCH_SIZE' is not defined

## === cell 15
ids, preds = inference(model, test_dataloader)

preds = np.argmax(preds, axis=1)
submission = pd.DataFrame()
submission["essay_id"] = ids
submission["score"] = preds
submission["score"] = submission["score"] + 1
print(f"Submission shape: {submission.shape}")
submission.to_csv("submission.csv", index=False)


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2042800273.py in <cell line: 0>()
----> 1 ids, preds = inference(model, test_dataloader)
      2 
      3 preds = np.argmax(preds, axis=1)
      4 submission = pd.DataFrame()
      5 submission["essay_id"] = ids

NameError: name 'model' is not defined
