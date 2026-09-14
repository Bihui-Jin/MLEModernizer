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

0.67291

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.45364) has done: 'The fix updates the imports so that `AdamW` is taken from `torch.optim` (the current transformers version no longer provides it) and correctly imports `AutoModelForQuestionAnswering`. The rest of the pipeline remains unchanged, allowing the model to train and generate a valid `submission.csv` file.'
- What this solution (achieved 0.46236) has done: 'I fix the import error by using `DistilBertForQuestionAnswering` instead of the generic `AutoModelForQuestionAnswering`, and adjust the data‑loading logic so that token offsets are mapped only to the tweet text (second sequence). This corrects the training labels and improves span prediction, moving the Jaccard score closer to the target while keeping the core model and training loop unchanged.'
- What this solution (achieved 0.67291) has done: 'I added an environment‑variable fix before importing transformers to avoid the protobuf `MessageFactory` error, and corrected the QA model’s forward‑call usage: the model now returns a `QuestionAnsweringModelOutput` from which we directly read `loss`, `start_logits`, and `end_logits`. This removes the dimension mismatch in the loss computation and aligns the prediction phase with the same output handling. The rest of the pipeline is unchanged, so the script now runs end‑to‑end and writes a valid `submission.csv` while nudging the Jaccard score toward the target.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math
import json
import logging
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering
from torch.optim import AdamW
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
print("Libraries imported.")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
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


def load_techfest_model(model_path, device):
    """
    Load a pretrained DistilBert model for QA from HuggingFace.
    The custom weight path is ignored because it does not exist in the environment.
    """
    print("Loading pretrained DistilBertForQuestionAnswering from 🤗 Hub...")
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
    model.to(device)
    return model




## === cell 2
class Config:
    TOKENIZER_PATH = "distilbert-base-uncased"
    WEIGHTS_PATH = None
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_LEN = 64
    BATCH_SIZE = 64
    EPOCHS = 3
    LEARNING_RATE = 5e-5




## === cell 3
print("Loading Data...")
train_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv").fillna("")
test_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv").fillna("")

if Config.DEVICE.type == "cpu":
    print("⚠️ CPU DETECTED: Subsampling data to finish in <5 mins")
    train_df = train_df.sample(frac=0.2, random_state=42).reset_index(drop=True)

print(f"Training Data Shape: {train_df.shape}")
print(f"Test Data Shape: {test_df.shape}")
print("\n--- Sample Data ---")
print(train_df.head(2))




## === cell 4
class TweetDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        data = self.df.iloc[index]
        text = str(data.text)
        sentiment = str(data.sentiment)
        selected = str(data.selected_text) if not self.is_test else ""

        encodings = self.tokenizer.encode_plus(
            sentiment,
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation="only_second",
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].flatten()
        attention_mask = encodings["attention_mask"].flatten()
        offset_mapping = encodings["offset_mapping"].squeeze(0).numpy()
        sequence_ids = encodings.sequence_ids()

        start_pos = 0
        end_pos = 0

        if not self.is_test and selected:
            start_char = text.find(selected)
            end_char = start_char + len(selected)

            token_start_index = 0
            token_end_index = 0

            for idx, (seq_id, (s, e)) in enumerate(zip(sequence_ids, offset_mapping)):
                if seq_id != 1:  # only consider tokens from the tweet text part
                    continue
                if s <= start_char < e:
                    token_start_index = idx
                if s < end_char <= e:
                    token_end_index = idx

            start_pos = token_start_index
            end_pos = token_end_index

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "start_positions": torch.tensor(start_pos, dtype=torch.long),
            "end_positions": torch.tensor(end_pos, dtype=torch.long),
            "text": text,
            "sentiment": sentiment,
        }


tokenizer = DistilBertTokenizerFast.from_pretrained(
    Config.TOKENIZER_PATH, local_files_only=False
)

train_dataset = TweetDataset(train_df, tokenizer, Config.MAX_LEN, is_test=False)
train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

test_dataset = TweetDataset(test_df, tokenizer, Config.MAX_LEN, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)

print("Data Loaders ready.")




## === cell 5
model = load_techfest_model(Config.WEIGHTS_PATH, Config.DEVICE)

optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)

print(f"Starting Training for {Config.EPOCHS} Epoch(s)...")
model.train()

for epoch in range(Config.EPOCHS):
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(Config.DEVICE)
        attention_mask = batch["attention_mask"].to(Config.DEVICE)
        start_positions = batch["start_positions"].to(Config.DEVICE)
        end_positions = batch["end_positions"].to(Config.DEVICE)

        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
        )
        loss = outputs.loss  # model already computes the sum of start and end losses
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss=loss.item())

print("Training Complete!")




## === cell 6
print("Starting Prediction Phase...")
model.eval()
predictions = []

with torch.no_grad():
    for batch in tqdm(test_loader):
        input_ids = batch["input_ids"].to(Config.DEVICE)
        attention_mask = batch["attention_mask"].to(Config.DEVICE)
        texts = batch["text"]
        sentiments = batch["sentiment"]

        outputs = model(input_ids, attention_mask=attention_mask)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        start_idxs = torch.argmax(start_logits, dim=1).cpu().numpy()
        end_idxs = torch.argmax(end_logits, dim=1).cpu().numpy()

        for i in range(len(texts)):
            if sentiments[i] == "neutral":
                predictions.append(texts[i])
            else:
                start = start_idxs[i]
                end = end_idxs[i]
                if end < start:
                    end = start
                end = end + 1  # inclusive slice
                token_ids = input_ids[i][start:end]
                selected_text = tokenizer.decode(
                    token_ids, skip_special_tokens=True
                ).strip()
                predictions.append(selected_text)

test_df["selected_text"] = predictions
submission = test_df[["textID", "selected_text"]]
submission.to_csv("submission.csv", index=False)
print("\n✅ SUCCESS: 'submission.csv' generated.")
