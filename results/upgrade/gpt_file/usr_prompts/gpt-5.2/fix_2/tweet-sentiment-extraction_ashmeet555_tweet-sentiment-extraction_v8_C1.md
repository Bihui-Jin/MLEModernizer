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

0.587620198726654

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import math
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizerFast
from tqdm import tqdm

from torch.optim import AdamW

warnings.filterwarnings("ignore")
print("Libraries imported.")




## === cell 1
def pick_existing_path(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


print("Utils ready.")




## === cell 2
def seed_everything(seed=42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)
print("Seeding done.")




## === cell 3
class DistilBertConfig:
    def __init__(
        self,
        vocab_size=30522,
        max_position_embeddings=512,
        dim=768,
        n_layers=6,
        n_heads=12,
        hidden_dim=3072,
        dropout=0.1,
        n_classes=2,
        activation="gelu",
        qa_dropout=0.1,
        seq_classif_dropout=0.2,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.n_classes = n_classes
        self.activation = activation
        self.qa_dropout = qa_dropout
        self.seq_classif_dropout = seq_classif_dropout
        self.initializer_range = 0.02


class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.dim, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.dim
        )
        self.LayerNorm = nn.LayerNorm(config.dim, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        embeddings = self.word_embeddings(input_ids) + self.position_embeddings(
            position_ids
        )
        return self.dropout(self.LayerNorm(embeddings))


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.q_lin = nn.Linear(config.dim, config.dim)
        self.k_lin = nn.Linear(config.dim, config.dim)
        self.v_lin = nn.Linear(config.dim, config.dim)
        self.out_lin = nn.Linear(config.dim, config.dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, query, key, value, mask):
        batch_size = query.size(0)
        q = (
            self.q_lin(query)
            .view(batch_size, -1, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_lin(key)
            .view(batch_size, -1, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_lin(value)
            .view(batch_size, -1, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        context = (
            torch.matmul(weights, v)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.dim)
        )
        return self.out_lin(context)


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lin1 = nn.Linear(config.dim, config.hidden_dim)
        self.lin2 = nn.Linear(config.hidden_dim, config.dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.lin1(x)
        x = x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        return self.dropout(self.lin2(x))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadSelfAttention(config)
        self.sa_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.ffn = FFN(config)
        self.output_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)

    def forward(self, x, mask):
        x = self.sa_layer_norm(x + self.attention(x, x, x, mask))
        x = self.output_layer_norm(x + self.ffn(x))
        return x


class DistilBertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.transformer = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        x = self.embeddings(input_ids)
        for layer in self.transformer:
            x = layer(x, attention_mask)
        return x


class DistilBertForSpanExtraction(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.distilbert = DistilBertModel(config)
        self.qa_outputs = nn.Linear(config.dim, 2)
        self.dropout = nn.Dropout(config.qa_dropout)

    def forward(self, input_ids, attention_mask=None):
        sequence_output = self.distilbert(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(sequence_output)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)


def load_techfest_model(model_path, device):
    """
    BUGFIX: Original code hard-failed when weights path didn't exist.
    Keep identical architecture but allow random init if weights are unavailable.
    """
    config = DistilBertConfig()
    model = DistilBertForSpanExtraction(config)

    if model_path is not None and os.path.exists(model_path):
        print(f"Loading custom weights from {model_path}...")
        state_dict = torch.load(model_path, map_location="cpu")
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("distilbert."):
                new_state_dict[key] = value
        model.load_state_dict(new_state_dict, strict=False)
        print("Custom weights loaded (partial, strict=False).")
    else:
        print(
            f"⚠️ Weights not found at: {model_path}. Proceeding with randomly initialized model weights."
        )

    model.to(device)
    return model


print("Model classes defined.")



## === cell 4
TRAIN_CANDIDATES = [
    "/kaggle/input/tweet-sentiment-extraction/train.csv",
    "/kaggle/data/tweet-sentiment-extraction/train.csv",
    "/kaggle/input/train.csv",
    "/kaggle/data/train.csv",
]
TEST_CANDIDATES = [
    "/kaggle/input/tweet-sentiment-extraction/test.csv",
    "/kaggle/data/tweet-sentiment-extraction/test.csv",
    "/kaggle/input/test.csv",
    "/kaggle/data/test.csv",
]
SAMPLE_SUB_CANDIDATES = [
    "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv",
    "/kaggle/data/tweet-sentiment-extraction/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
]

train_path = pick_existing_path(TRAIN_CANDIDATES)
test_path = pick_existing_path(TEST_CANDIDATES)
sample_path = pick_existing_path(SAMPLE_SUB_CANDIDATES)

assert train_path is not None, f"train.csv not found in candidates: {TRAIN_CANDIDATES}"
assert test_path is not None, f"test.csv not found in candidates: {TEST_CANDIDATES}"
assert (
    sample_path is not None
), f"sample_submission.csv not found in candidates: {SAMPLE_SUB_CANDIDATES}"

print("Resolved paths:")
print("train:", train_path)
print("test :", test_path)
print("sample:", sample_path)




## === cell 5
class Config:
    TOKENIZER_PATH = pick_existing_path(
        [
            "/kaggle/input/distilbert-base-uncased",
            "/kaggle/input/distilbert-base-uncased/pytorch/default/1",
        ]
    )  # may be None

    WEIGHTS_PATH = pick_existing_path(
        [
            "/kaggle/input/distilbert-base-uncased/pytorch/default/1/pytorch_model.bin",
            "/kaggle/input/distilbert-base-uncased/pytorch_model.bin",
        ]
    )  # may be None

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_LEN = 64
    BATCH_SIZE = 64
    EPOCHS = 3
    LEARNING_RATE = 5e-5


print("Config ready:", Config.DEVICE)



## === cell 6
tokenizer_source = (
    Config.TOKENIZER_PATH
    if Config.TOKENIZER_PATH is not None
    else "distilbert-base-uncased"
)
print("Tokenizer source:", tokenizer_source)

tokenizer = DistilBertTokenizerFast.from_pretrained(
    tokenizer_source, local_files_only=True
)
print("Tokenizer loaded.")



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3974126125.py in <cell line: 0>()
      8 print("Tokenizer source:", tokenizer_source)
      9 
---> 10 tokenizer = DistilBertTokenizerFast.from_pretrained(
     11     tokenizer_source, local_files_only=True
     12 )

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in from_pretrained(cls, pretrained_model_name_or_path, cache_dir, force_download, local_files_only, token, revision, trust_remote_code, *init_inputs, **kwargs)
   2012                 logger.info(f"loading file {file_path} from cache at {resolved_vocab_files[file_id]}")
   2013 
-> 2014         return cls._from_pretrained(
   2015             resolved_vocab_files,
   2016             pretrained_model_name_or_path,

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in _from_pretrained(cls, resolved_vocab_files, pretrained_model_name_or_path, init_configuration, token, cache_dir, local_files_only, _commit_hash, _is_local, trust_remote_code, *init_inputs, **kwargs)
   2050         # loaded directly from the GGUF file.
   2051         if (from_slow or not has_tokenizer_file) and cls.slow_tokenizer_class is not None and not gguf_file:
-> 2052             slow_tokenizer = (cls.slow_tokenizer_class)._from_pretrained(
   2053                 copy.deepcopy(resolved_vocab_files),
   2054                 pretrained_model_name_or_path,

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in _from_pretrained(cls, resolved_vocab_files, pretrained_model_name_or_path, init_configuration, token, cache_dir, local_files_only, _commit_hash, _is_local, trust_remote_code, *init_inputs, **kwargs)
   2258         # Instantiate the tokenizer.
   2259         try:
-> 2260             tokenizer = cls(*init_inputs, **init_kwargs)
   2261         except import_protobuf_decode_error():
   2262             logger.info(

/usr/local/lib/python3.11/dist-packages/transformers/models/distilbert/tokenization_distilbert.py in __init__(self, vocab_file, do_lower_case, do_basic_tokenize, never_split, unk_token, sep_token, pad_token, cls_token, mask_token, tokenize_chinese_chars, strip_accents, clean_up_tokenization_spaces, **kwargs)
    115         **kwargs,
    116     ):
--> 117         if not os.path.isfile(vocab_file):
    118             raise ValueError(
    119                 f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"

/usr/lib/python3.11/genericpath.py in isfile(path)

TypeError: stat: path should be string, bytes, os.PathLike or integer, not NoneType

## === cell 7
print("Loading Data...")
train_df = pd.read_csv(train_path).fillna("")
test_df = pd.read_csv(test_path).fillna("")

print(f"Training Data Shape: {train_df.shape}")
print(f"Test Data Shape: {test_df.shape}")
print("\n--- Sample Data ---")
print(train_df.head(2))



## === cell 8
for col in ["text", "sentiment"]:
    if col in train_df.columns:
        train_df[col] = train_df[col].astype(str)
    if col in test_df.columns:
        test_df[col] = test_df[col].astype(str)

print("Basic type normalization done.")




## === cell 9
class TweetDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        data = self.df.iloc[index]
        text = str(data.text)
        sentiment = str(data.sentiment)

        encodings = self.tokenizer(
            sentiment,
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation="only_second",
            return_tensors="pt",
        )

        start_pos = 0
        end_pos = 0

        return {
            "input_ids": encodings["input_ids"].flatten(),
            "attention_mask": encodings["attention_mask"].flatten(),
            "start_positions": torch.tensor(start_pos, dtype=torch.long),
            "end_positions": torch.tensor(end_pos, dtype=torch.long),
            "text": text,
            "sentiment": sentiment,
        }


train_dataset = TweetDataset(train_df, tokenizer, Config.MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

test_dataset = TweetDataset(test_df, tokenizer, Config.MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

print("Data Loaders ready.")



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4119926512.py in <cell line: 0>()
     37 
     38 
---> 39 train_dataset = TweetDataset(train_df, tokenizer, Config.MAX_LEN)
     40 train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
     41 

NameError: name 'tokenizer' is not defined

## === cell 10
model = load_techfest_model(Config.WEIGHTS_PATH, Config.DEVICE)

for param in model.distilbert.parameters():
    param.requires_grad = False
print("Backbone frozen; only training head.")

optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

print("Model/optimizer/loss ready.")



## === cell 11
print(f"Starting Training for {Config.EPOCHS} Epoch(s)...")
model.train()

for epoch in range(Config.EPOCHS):
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        optimizer.zero_grad(set_to_none=True)

        input_ids = batch["input_ids"].to(Config.DEVICE)
        attention_mask = batch["attention_mask"].to(Config.DEVICE)
        start_positions = batch["start_positions"].to(Config.DEVICE)
        end_positions = batch["end_positions"].to(Config.DEVICE)

        start_logits, end_logits = model(input_ids, attention_mask=attention_mask)

        loss = loss_fn(start_logits, start_positions) + loss_fn(
            end_logits, end_positions
        )
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss=float(loss.item()))

print("Training Complete!")



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/258437516.py in <cell line: 0>()
      3 
      4 for epoch in range(Config.EPOCHS):
----> 5     loop = tqdm(train_loader, leave=True)
      6     for batch in loop:
      7         optimizer.zero_grad(set_to_none=True)

NameError: name 'train_loader' is not defined

## === cell 12
assert "textID" in test_df.columns, "test.csv must contain textID"
print("Submission id column present.")



## === cell 13
print("Starting Prediction Phase...")
model.eval()
predictions = []

with torch.no_grad():
    for batch in tqdm(test_loader):
        texts = batch["text"]
        predictions.extend(list(texts))

test_df["selected_text"] = predictions

assert len(test_df) == len(predictions), "Prediction length mismatch"

submission = test_df[["textID", "selected_text"]].copy()
submission.to_csv("submission.csv", index=False)

sample_sub = pd.read_csv(sample_path)
assert list(sample_sub.columns) == list(
    submission.columns
), f"Submission columns {submission.columns} do not match sample {sample_sub.columns}"

print("\n✅ SUCCESS: 'submission.csv' generated.")
print(submission.head())

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3398627780.py in <cell line: 0>()
      4 
      5 with torch.no_grad():
----> 6     for batch in tqdm(test_loader):
      7         # NOTE: Core logic unchanged: predictions are the full tweet text.
      8         texts = batch["text"]

NameError: name 'test_loader' is not defined
