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
Predict the word or phrase from tweets that exemplifies the labelled sentiment.

## Metric
Word-level Jaccard score.

## Submission Format
For each ID in the test set, you must predict the string that best supports the sentiment for the tweet in question. Note that the selected text _needs_ to be **quoted** and **complete** (include punctuation, etc. - the above code splits ONLY on whitespace) to work correctly. The file should contain a header and have the following format:
```
textID,selected_text
2,"very good"
5,"I don't care"
6,"bad"
8,"it was, yes"
etc.
```

## Dataset
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

- `textID` - unique ID for each piece of text
- `text` - the text of the tweet
- `sentiment` - the general sentiment of the tweet
- `selected_text` - [train only] the text that supports the tweet's sentiment

# 2. Python version

3.11

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
scipy==1.15.3
seaborn==0.12.2
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
            description.md (130 lines)
            sample_submission.csv (2750 lines)
            sample_submission.csv.zip (18.6 kB)
            test.csv (2750 lines)
            test.csv.zip (114.6 kB)
            train.csv (24733 lines)
            train.csv.zip (1.2 MB)
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
        input/
            description.md (130 lines)
            sample_submission.csv (2750 lines)
            sample_submission.csv.zip (18.6 kB)
            test.csv (2750 lines)
            test.csv.zip (114.6 kB)
            train.csv (24733 lines)
            train.csv.zip (1.2 MB)
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
        working/
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
```

-> data/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> data/test.csv has 2749 rows and 3 columns.
The columns are: textID, text, sentiment

-> data/train.csv has 24732 rows and 4 columns.
The columns are: textID, text, selected_text, sentiment

-> data/tweet-sentiment-extraction/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> data/tweet-sentiment-extraction/test.csv has 2749 rows and 3 columns.
The columns are: textID, text, sentiment

-> data/tweet-sentiment-extraction/train.csv has 24732 rows and 4 columns.
The columns are: textID, text, selected_text, sentiment

-> input/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> (stopped after 10 files for performance)

# 5. Target score

0.5516448616981506

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, gc, random, time, math, re, string
import tqdm, numpy as np, pandas as pd, scipy as sp, matplotlib.pyplot as plt, seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (
    train_test_split,
    KFold,
    GroupKFold,
    StratifiedKFold,
    StratifiedGroupKFold,
)
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from transformers import AutoTokenizer, AutoModel, AutoConfig, DataCollatorWithPadding
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
class CFG:
    DEBUG = False
    TRAIN = True
    N_FOLDS = 5
    TRAIN_FOLDS = [i for i in range(N_FOLDS)]
    SEED = 42
    TEST_BATCHSIZE = 100
    MAX_LENGTH = 128
    MODEL_NAME = "microsoft/deberta-v3-base"
    FC_DROPOUT = [0.1, 0.2, 0.3, 0.4, 0.5]




## === cell 2
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(CFG.SEED)



## === cell 3
test_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
test_df.head()



## === cell 4
CFG.TOKENIZER = AutoTokenizer.from_pretrained(CFG.MODEL_NAME, use_fast=True)

collate_fn = DataCollatorWithPadding(
    CFG.TOKENIZER, padding="longest", return_tensors="pt"
)


class QADataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = " ".join(str(self.df.text.iloc[idx]).split())
        inputs = CFG.TOKENIZER(
            text,
            add_special_tokens=True,
            max_length=CFG.MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        tok_text_tokens = CFG.TOKENIZER.convert_ids_to_tokens(input_ids.tolist())
        sentiment = [1, 0, 0]
        if self.df.sentiment.iloc[idx] == "positive":
            sentiment = [0, 0, 1]
        elif self.df.sentiment.iloc[idx] == "negative":
            sentiment = [0, 1, 0]
        return {
            "input_ids": input_ids.long(),
            "mask": attention_mask.long(),
            "text_tokens": " ".join(tok_text_tokens),
            "sentiment": torch.tensor(sentiment, dtype=torch.long),
            "orig_text": self.df.text.iloc[idx],
            "orig_sentiment": self.df.sentiment.iloc[idx],
        }




## === cell 5
class QAModel(nn.Module):
    def __init__(self, config_path=None, pretrained=False):
        super().__init__()
        if config_path is None:
            self.config = AutoConfig.from_pretrained(
                CFG.MODEL_NAME, output_hidden_states=True
            )
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.backbone = AutoModel.from_pretrained(
                CFG.MODEL_NAME, config=self.config
            )
        else:
            self.backbone = AutoModel.from_config(self.config)
        self.fc_dropout = nn.ModuleList([nn.Dropout(p) for p in CFG.FC_DROPOUT])
        self.attention_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.GELU(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1),
        )
        self.fc = nn.Linear(self.config.hidden_size, 2)
        self.apply(self.initialize_parameters)

    def initialize_parameters(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight.data, mean=0.0, std=1.0)

    def forward(self, input_ids, mask):
        embeddings = self.backbone(
            input_ids=input_ids, attention_mask=mask
        ).last_hidden_state
        logits = self.fc(embeddings)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits




## === cell 6
def test_fn(dataloader, model):
    model.eval()
    fin_output_start, fin_output_end, fin_mask = [], [], []
    fin_text_tokens, fin_orig_text, fin_orig_sentiment = [], [], []
    with torch.no_grad():
        for data in tqdm(dataloader, total=len(dataloader)):
            input_ids = data["input_ids"].to(device)
            mask = data["mask"].to(device)
            start_logits, end_logits = model(input_ids, mask)
            fin_output_start.append(start_logits.cpu())
            fin_output_end.append(end_logits.cpu())
            fin_mask.append(mask.cpu())
            fin_text_tokens.extend(data["text_tokens"])
            fin_orig_text.extend(data["orig_text"])
            fin_orig_sentiment.extend(data["orig_sentiment"])
    fin_output_start = torch.vstack(fin_output_start)
    fin_output_end = torch.vstack(fin_output_end)
    fin_mask = torch.vstack(fin_mask)
    return (
        fin_output_start,
        fin_output_end,
        fin_mask,
        fin_text_tokens,
        fin_orig_text,
        fin_orig_sentiment,
    )




## === cell 7
test_dataset = QADataset(test_df)
test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.TEST_BATCHSIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_fn,
)



## === cell 8
for idx, fold in enumerate(CFG.TRAIN_FOLDS):
    try:
        model = QAModel(
            config_path="/kaggle/input/tweet-sentiment-extraction/config.pth",
            pretrained=True,
        )
        model.load_state_dict(
            torch.load(
                f"/kaggle/input/tweet-sentiment-extraction/QAbert{fold}.pth",
                map_location=device,
            )
        )
    except Exception as e:
        model = QAModel(pretrained=True)
    model.to(device)
    if idx == 0:
        (
            fin_output_start,
            fin_output_end,
            fin_mask,
            fin_text_tokens,
            fin_orig_text,
            fin_orig_sentiment,
        ) = test_fn(test_loader, model)
    else:
        a, b, c, _, _, _ = test_fn(test_loader, model)
        fin_output_start = fin_output_start + a
        fin_output_end = fin_output_end + b
        fin_mask = fin_mask + c



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_54/2450708413.py in <cell line: 0>()
     25             fin_orig_text,
     26             fin_orig_sentiment,
---> 27         ) = test_fn(test_loader, model)
     28     else:
     29         a, b, c, _, _, _ = test_fn(test_loader, model)

/tmp/ipykernel_54/3219200427.py in test_fn(dataloader, model)
      4     fin_text_tokens, fin_orig_text, fin_orig_sentiment = [], [], []
      5     with torch.no_grad():
----> 6         for data in tqdm(dataloader, total=len(dataloader)):
      7             input_ids = data["input_ids"].to(device)
      8             mask = data["mask"].to(device)

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

ValueError: Caught ValueError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py", line 767, in convert_to_tensors
    tensor = as_tensor(value)
             ^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py", line 729, in as_tensor
    return torch.tensor(value)
           ^^^^^^^^^^^^^^^^^^^
ValueError: too many dimensions 'str'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 55, in fetch
    return self.collate_fn(data)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/data/data_collator.py", line 272, in __call__
    batch = pad_without_fast_tokenizer_warning(
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/data/data_collator.py", line 67, in pad_without_fast_tokenizer_warning
    padded = tokenizer.pad(*pad_args, **pad_kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py", line 3374, in pad
    return BatchEncoding(batch_outputs, tensor_type=return_tensors)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py", line 240, in __init__
    self.convert_to_tensors(tensor_type=tensor_type, prepend_batch_axis=prepend_batch_axis)
  File "/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py", line 783, in convert_to_tensors
    raise ValueError(
ValueError: Unable to create tensor, you should probably activate truncation and/or padding with 'padding=True' 'truncation=True' to have batched tensors with the same length. Perhaps your features (`text_tokens` in this case) have excessive nesting (inputs type `list` where type `int` is expected).


## === cell 9
softmax = torch.nn.Softmax(dim=1)
folds = len(CFG.TRAIN_FOLDS)
fin_output_start = softmax(fin_output_start / folds)
fin_output_end = softmax(fin_output_end / folds)
fin_mask = fin_mask / folds  # keep as float for later element‑wise multiplication



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/3503567772.py in <cell line: 0>()
      2 softmax = torch.nn.Softmax(dim=1)
      3 folds = len(CFG.TRAIN_FOLDS)
----> 4 fin_output_start = softmax(fin_output_start / folds)
      5 fin_output_end = softmax(fin_output_end / folds)
      6 fin_mask = fin_mask / folds  # keep as float for later element‑wise multiplication

NameError: name 'fin_output_start' is not defined

## === cell 10
final_outputs = []
for j in range(len(fin_text_tokens)):
    text_token = fin_text_tokens[j]
    mask = fin_mask[j]
    mask_start = fin_output_start[j] * mask
    mask_end = fin_output_end[j] * mask
    idx_start = int(torch.argmax(mask_start).item())
    idx_end = int(torch.argmax(mask_end).item())
    if idx_end < idx_start:
        idx_end = idx_start
    span_mask = [0] * len(mask)
    for pos in range(idx_start, idx_end + 1):
        span_mask[pos] = 1
    output_tokens = [
        tok for i, tok in enumerate(text_token.split()) if span_mask[i] == 1
    ]
    output_tokens = [tok for tok in output_tokens if tok not in ("[CLS]", "[SEP]")]
    final_output = ""
    for ot in output_tokens:
        if ot.startswith("▁"):
            final_output += " " + ot[1:]
        elif len(ot) == 1 and ot in string.punctuation:
            final_output += ot
        else:
            final_output += " " + ot
    final_output = final_output.strip()
    final_outputs.append(final_output)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/1638377187.py in <cell line: 0>()
      1 final_outputs = []
----> 2 for j in range(len(fin_text_tokens)):
      3     text_token = fin_text_tokens[j]
      4     mask = fin_mask[j]
      5     # Apply mask to logits

NameError: name 'fin_text_tokens' is not defined

## === cell 11
sub = pd.DataFrame({"textID": test_df["textID"], "selected_text": final_outputs})
sub.head()



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_54/3482958895.py in <cell line: 0>()
----> 1 sub = pd.DataFrame({"textID": test_df["textID"], "selected_text": final_outputs})
      2 sub.head()
      3 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    776         elif isinstance(data, dict):
    777             # GH#38939 de facto copy defaults to False only in non-dict cases
--> 778             mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy, typ=manager)
    779         elif isinstance(data, ma.MaskedArray):
    780             from numpy.ma import mrecords

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in dict_to_mgr(data, index, columns, dtype, typ, copy)
    501             arrays = [x.copy() if hasattr(x, "dtype") else x for x in arrays]
    502 
--> 503     return arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ, consolidate=copy)
    504 
    505 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in arrays_to_mgr(arrays, columns, index, dtype, verify_integrity, typ, consolidate)
    112         # figure out the index, if necessary
    113         if index is None:
--> 114             index = _extract_index(arrays)
    115         else:
    116             index = ensure_index(index)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in _extract_index(data)
    688                     f"length {len(index)}"
    689                 )
--> 690                 raise ValueError(msg)
    691         else:
    692             index = default_index(lengths[0])

ValueError: array length 0 does not match index length 2749

## === cell 12
sub.to_csv("submission.csv", index=False)

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/352017882.py in <cell line: 0>()
----> 1 sub.to_csv("submission.csv", index=False)

NameError: name 'sub' is not defined
