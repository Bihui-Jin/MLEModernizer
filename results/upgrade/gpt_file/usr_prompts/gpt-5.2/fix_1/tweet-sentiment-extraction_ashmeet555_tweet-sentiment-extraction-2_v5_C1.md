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

## === cell 6
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math
import json
import logging
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizerFast
from torch.optim import AdamW # We still need Tokenizer & Optimizer
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
print("Libraries imported.")

## === cell 8
class Config:
    TOKENIZER_PATH = '/kaggle/input/distilbert-base-uncased/pytorch/default/1'
    
    WEIGHTS_PATH = '/kaggle/input/distilbert-base-uncased/pytorch/default/1/pytorch_model.bin'
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    MAX_LEN = 64 # Maximum number of tokens per input sequence.
    BATCH_SIZE = 64 # Number of samples processed in one training step.
    EPOCHS = 3    # Number of training passes over the dataset.
    LEARNING_RATE = 5e-5

## === cell 10
print("Loading Data...")
train_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv').fillna('')
test_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv').fillna('')

if Config.DEVICE.type == 'cpu':
    print("⚠️ CPU DETECTED: Subsampling data to finish in <5 mins")
    train_df = train_df.sample(frac=0.2, random_state=42).reset_index(drop=True)

print(f"Training Data Shape: {train_df.shape}")
print(f"Test Data Shape: {test_df.shape}")
print("\n--- Sample Data ---")
print(train_df.head(2))

import matplotlib.pyplot as plt
train_df['sentiment'].value_counts().plot(kind='bar')
plt.title("Sentiment Distribution in Training Data")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()
train_df['tweet_length'] = train_df['text'].apply(len)

plt.hist(train_df['tweet_length'], bins=50)
plt.title("Tweet Length Distribution")
plt.xlabel("Number of Characters")
plt.ylabel("Frequency")
plt.show()

## === cell 12
class TweetDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        data = self.df.iloc[index]
        text = str(data.text)
        sentiment = str(data.sentiment)
        
        encodings = self.tokenizer.encode_plus(
            sentiment, 
            text, 
            max_length=self.max_len, 
            padding="max_length",      # Old API for padding
            truncation="only_second",    # Truncate tweet if too long, keep sentiment
            return_tensors="pt"
        )
        
        start_pos = 0
        end_pos = 0

        return {
            'input_ids': encodings['input_ids'].flatten(),
            'attention_mask': encodings['attention_mask'].flatten(),
            'start_positions': torch.tensor(start_pos, dtype=torch.long),
            'end_positions': torch.tensor(end_pos, dtype=torch.long),
            'text': text,
            'sentiment': sentiment
        }

tokenizer = DistilBertTokenizerFast.from_pretrained(Config.TOKENIZER_PATH, local_files_only=True)

train_dataset = TweetDataset(train_df, tokenizer, Config.MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

test_dataset = TweetDataset(test_df, tokenizer, Config.MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)

print("Data Loaders ready.")

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")

sample_batch = next(iter(train_loader))
print("\nSample batch keys:", sample_batch.keys())
print("Input IDs shape:", sample_batch['input_ids'].shape)
print("Attention mask shape:", sample_batch['attention_mask'].shape)
print("Start positions shape:", sample_batch['start_positions'].shape)
print("End positions shape:", sample_batch['end_positions'].shape)

print("\nSample text:", sample_batch['text'][0])
print("Sample sentiment:", sample_batch['sentiment'][0])

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
HFValidationError                         Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    469             # This is slightly better for only 1 file
--> 470             hf_hub_download(
    471                 path_or_repo_id,

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    153     if repo_id.count("/") > 1:
--> 154         raise HFValidationError(
    155             "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/distilbert-base-uncased/pytorch/default/1'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

HFValidationError                         Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in from_pretrained(cls, pretrained_model_name_or_path, cache_dir, force_download, local_files_only, token, revision, trust_remote_code, *init_inputs, **kwargs)
   1911                     try:
-> 1912                         resolved_config_file = cached_file(
   1913                             pretrained_model_name_or_path,

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_file(path_or_repo_id, filename, **kwargs)
    311     """
--> 312     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    313     file = file[0] if file is not None else file

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    521         # Now we try to recover if we can find all files correctly in the cache
--> 522         resolved_files = [
    523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in <listcomp>(.0)
    522         resolved_files = [
--> 523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames
    524         ]

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in _get_cache_file_to_return(path_or_repo_id, full_filename, cache_dir, revision)
    139     # We try to see if we have a cached version (not up to date):
--> 140     resolved_file = try_to_load_from_cache(path_or_repo_id, full_filename, cache_dir=cache_dir, revision=revision)
    141     if resolved_file is not None and resolved_file != _CACHED_NO_EXIST:

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    153     if repo_id.count("/") > 1:
--> 154         raise HFValidationError(
    155             "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/distilbert-base-uncased/pytorch/default/1'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/383771820.py in <cell line: 0>()
     40 
     41 # Initialize Tokenizer (Offline Mode)
---> 42 tokenizer = DistilBertTokenizerFast.from_pretrained(Config.TOKENIZER_PATH, local_files_only=True)
     43 
     44 # Create DataLoaders

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in from_pretrained(cls, pretrained_model_name_or_path, cache_dir, force_download, local_files_only, token, revision, trust_remote_code, *init_inputs, **kwargs)
   1930                     except Exception:
   1931                         # For any other exception, we throw a generic error.
-> 1932                         raise OSError(
   1933                             f"Can't load tokenizer for '{pretrained_model_name_or_path}'. If you were trying to load it from "
   1934                             "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "

OSError: Can't load tokenizer for '/kaggle/input/distilbert-base-uncased/pytorch/default/1'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '/kaggle/input/distilbert-base-uncased/pytorch/default/1' is the correct path to a directory containing all relevant files for a DistilBertTokenizerFast tokenizer.

## === cell 14
class DistilBertConfig:
    """Stores all model hyperparameters such as:
    hidden size
    number of layers
    number of attention heads
    dropout values
    """
    def __init__(self, vocab_size=30522, max_position_embeddings=512, dim=768, 
                 n_layers=6, n_heads=12, hidden_dim=3072, dropout=0.1, 
                 n_classes=2, activation="gelu", qa_dropout=0.1, seq_classif_dropout=0.2):
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
    """
    Embedding Layer
    Converts tokens into vectors
    Adds positional embeddings
    Applies layer normalization and dropout
    """
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.dim, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.dim)
        self.LayerNorm = nn.LayerNorm(config.dim, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        embeddings = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)
        return self.dropout(self.LayerNorm(embeddings))

class MultiHeadSelfAttention(nn.Module):
    """
    Core transformer mechanism
    Allows each token to attend to other tokens in the sequence
    """
    
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
        q = self.q_lin(query).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_lin(key).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_lin(value).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(weights, v).transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        return self.out_lin(context)

class FFN(nn.Module):
    """
    Applies non-linearity (GELU)
    Processes attention outputs
    """
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
    """
    Combines attention + FFN
    Uses residual connections and layer normalization
    """
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
    """
    Stack of transformer blocks
    Outputs contextual embeddings for each token
    """
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.transformer = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

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

    """
    This helper function:
    Loads pretrained DistilBERT weights
    Maps them to the manual architecture
    Moves the model to CPU or GPU
    📌 We use:
    Pretrained weights for stability
    Custom architecture for learning and flexibility
    """
def load_techfest_model(model_path, device):
    print(f"Loading custom weights from {model_path}...")
    config = DistilBertConfig()
    model = DistilBertForSpanExtraction(config)
    
    state_dict = torch.load(model_path, map_location='cpu')
    
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("distilbert."):
            new_state_dict[key] = value
            
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    return model

## === cell 16
import time
import torch

model = load_techfest_model(Config.WEIGHTS_PATH, Config.DEVICE)

for param in model.distilbert.parameters():
    param.requires_grad = False
print("❄️ Backbone Frozen! Only training head.")

optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)

loss_fn = nn.CrossEntropyLoss()

model.train()
for batch in train_loader:
    input_ids = batch['input_ids'].to(Config.DEVICE)
    attention_mask = batch['attention_mask'].to(Config.DEVICE)
    _ = model(input_ids, attention_mask=attention_mask)
    break

if Config.DEVICE == "cuda":
    torch.cuda.synchronize()

start_time = time.time()

print(f"Starting Training for {Config.EPOCHS} Epoch(s)...")

for epoch in range(Config.EPOCHS):
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(Config.DEVICE)
        attention_mask = batch['attention_mask'].to(Config.DEVICE)
        start_positions = batch['start_positions'].to(Config.DEVICE)
        end_positions = batch['end_positions'].to(Config.DEVICE)

        start_logits, end_logits = model(
            input_ids,
            attention_mask=attention_mask
        )

        loss = (
            loss_fn(start_logits, start_positions)
            + loss_fn(end_logits, end_positions)
        )

        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss=loss.item())

if Config.DEVICE == "cuda":
    torch.cuda.synchronize()

training_time = time.time() - start_time

print("Training Complete!")
print("Training Time (seconds):", round(training_time, 4))


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1395729149.py in <cell line: 0>()
      3 
      4 # 1. Load Model
----> 5 model = load_techfest_model(Config.WEIGHTS_PATH, Config.DEVICE)
      6 
      7 # 2. Freeze Backbone

/tmp/ipykernel_11/354009153.py in load_techfest_model(model_path, device)
    155 
    156     # Load weights
--> 157     state_dict = torch.load(model_path, map_location='cpu')
    158 
    159     # Fix key names to match our manual model

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/distilbert-base-uncased/pytorch/default/1/pytorch_model.bin'

## === cell 18
import time


print("Starting Prediction Phase...")

start_infer = time.time()

model.eval()
predictions = []

with torch.no_grad():
    for batch in tqdm(test_loader):

        input_ids = batch['input_ids'].to(Config.DEVICE)
        attention_mask = batch['attention_mask'].to(Config.DEVICE)

        start_logits, end_logits = model(
            input_ids,
            attention_mask=attention_mask
        )

        start_logits = start_logits.cpu().numpy()
        end_logits = end_logits.cpu().numpy()

        start_preds = np.argmax(start_logits, axis=1)
        end_preds = np.argmax(end_logits, axis=1)

        texts = batch['text']
        sentiments = batch['sentiment']

        for i in range(len(texts)):
            if sentiments[i] == 'neutral':
                predictions.append(texts[i])
            else:
                predictions.append(texts[i])

inference_time = time.time() - start_infer

model_size = sum(p.numel() for p in model.parameters())

test_df['selected_text'] = predictions
submission = test_df[['textID', 'selected_text']]
submission.to_csv('submission.csv', index=False)

print("\n✅ SUCCESS: 'submission.csv' generated.")

print("Inference Time (s):", round(inference_time, 4))
print("Training Time (seconds):", round(training_time, 4))
print("Model Parameters:", model_size)
print("Model Size (MB):", round(model_size * 4 / (1024 ** 2), 2))


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2329715333.py in <cell line: 0>()
      9 start_infer = time.time()
     10 
---> 11 model.eval()
     12 predictions = []
     13 

NameError: name 'model' is not defined
