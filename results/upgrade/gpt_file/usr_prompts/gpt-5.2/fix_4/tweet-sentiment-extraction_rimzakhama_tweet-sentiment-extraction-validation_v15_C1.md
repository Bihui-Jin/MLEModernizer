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

3.8

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
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tokenizers==0.21.2
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

0.6186519861221313

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import string
from tqdm import tqdm

import transformers
from transformers import BertTokenizerFast, BertModel


class config:
    MAX_LEN = 141
    TRAIN_BATCH_SIZE = 40
    VALID_BATCH_SIZE = 16
    EPOCHS = 10

    BERT_PATH = "bert-base-uncased"
    MODEL_PATH = "model.bin"
    TRAINING_FILE = "/kaggle/input/tweet-sentiment-extraction/train.csv"

    TOKENIZER = None


def _init_tokenizer_offline():
    try:
        tok = BertTokenizerFast.from_pretrained(config.BERT_PATH, local_files_only=True)
        return tok
    except Exception as e:
        raise RuntimeError(
            f"Could not initialize tokenizer offline for '{config.BERT_PATH}'. "
            "Ensure the model is available in the Kaggle environment cache or as an input dataset."
        ) from e


config.TOKENIZER = _init_tokenizer_offline()


class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = BertModel.from_pretrained(config.BERT_PATH, local_files_only=True)
        self.l0 = nn.Linear(768, 2)

    def forward(self, ids, mask, token_type_ids):
        out = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        sequence_output = out.last_hidden_state
        logits = self.l0(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BERTBaseUncased().to(device)
model = nn.DataParallel(model)

weights_path = "/kaggle/input/tweet-sentiment-extraction-training/model.bin"
if os.path.exists(weights_path):
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
else:
    print(
        f"WARNING: pretrained competition weights not found at {weights_path}. Using base bert weights."
    )

model.eval()




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, tweet, sentiment, selected_text):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.max_len = config.MAX_LEN
        self.tokenizer = config.TOKENIZER

        if self.tokenizer is None or not hasattr(self.tokenizer, "encode"):
            raise ValueError(
                "config.TOKENIZER is not initialized correctly (missing .encode)."
            )

    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):
        tweet = str(self.tweet[item])
        tweet = " ".join(tweet.split())

        selected_text = str(self.selected_text[item])
        selected_text = " ".join(selected_text.split())

        len_sel_text = len(selected_text)

        idx0 = -1
        idx1 = -1
        if len_sel_text > 0 and len(tweet) > 0:
            for ind in (i for i, e in enumerate(tweet) if e == selected_text[0]):
                if tweet[ind : ind + len_sel_text] == selected_text:
                    idx0 = ind
                    idx1 = ind + len_sel_text - 1
                    break

        char_targets = [0] * len(tweet)
        if idx0 != -1 and idx1 != -1:
            for j in range(idx0, idx1 + 1):
                if tweet[j] != " ":
                    char_targets[j] = 1

        tok_tweet = self.tokenizer.encode(tweet)

        enc = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            return_offsets_mapping=True,
            max_length=self.max_len,
            truncation=True,
            padding=False,
        )

        tok_tweet_ids = enc["input_ids"]
        tok_tweet_tokens = self.tokenizer.convert_ids_to_tokens(tok_tweet_ids)
        offsets = enc["offset_mapping"]
        tok_tweet_offsets = offsets[1:-1]

        targets = [0] * (len(tok_tweet_tokens) - 2)
        for j, (offset1, offset2) in enumerate(tok_tweet_offsets):
            if offset2 > offset1 and sum(char_targets[offset1:offset2]) > 0:
                targets[j] = 1

        targets = [0] + targets + [0]
        targets_start = [0] * len(targets)
        targets_end = [0] * len(targets)

        non_zero = np.nonzero(targets)[0]
        if len(non_zero) > 0:
            targets_start[non_zero[0]] = 1
            targets_end[non_zero[-1]] = 1

        mask = [1] * len(tok_tweet_ids)
        token_type_ids = [0] * len(tok_tweet_ids)

        padding_len = self.max_len - len(tok_tweet_ids)
        if padding_len < 0:
            tok_tweet_ids = tok_tweet_ids[: self.max_len]
            mask = mask[: self.max_len]
            token_type_ids = token_type_ids[: self.max_len]
            targets = targets[: self.max_len]
            targets_start = targets_start[: self.max_len]
            targets_end = targets_end[: self.max_len]
            padding_len = 0
            tok_tweet_tokens = tok_tweet_tokens[: self.max_len]
        else:
            tok_tweet_ids = tok_tweet_ids + [0] * padding_len
            mask = mask + [0] * padding_len
            token_type_ids = token_type_ids + [0] * padding_len
            targets = targets + [0] * padding_len
            targets_start = targets_start + [0] * padding_len
            targets_end = targets_end + [0] * padding_len

        sentiment = [1, 0, 0]
        if self.sentiment[item] == "positive":
            sentiment = [0, 0, 1]
        if self.sentiment[item] == "negative":
            sentiment = [0, 1, 0]

        return {
            "ids": torch.tensor(tok_tweet_ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.long),
            "targets_start": torch.tensor(targets_start, dtype=torch.long),
            "targets_end": torch.tensor(targets_end, dtype=torch.long),
            "padding_len": torch.tensor(padding_len, dtype=torch.long),
            "tweet_tokens": " ".join(tok_tweet_tokens),
            "orig_tweet": self.tweet[item],
            "sentiment": torch.tensor(sentiment, dtype=torch.long),
            "orig_sentiment": self.sentiment[item],
            "orig_selected_text": self.selected_text[item],
        }




## === cell 2
df_test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
df_test.loc[:, "selected_text"] = df_test.text.values

test_dataset = TweetDataset(
    tweet=df_test.text.values,
    sentiment=df_test.sentiment.values,
    selected_text=df_test.selected_text.values,
)

valid_data_loader = torch.utils.data.DataLoader(
    test_dataset, shuffle=False, batch_size=config.VALID_BATCH_SIZE, num_workers=0
)

final_col = []
fin_output_start = []
fin_output_end = []
fin_padding_lens = []
fin_tweet_tokens = []
fin_orig_sentiment = []
fin_orig_selected_text = []
fin_orig_tweet = []

with torch.no_grad():
    for bi, d in enumerate(valid_data_loader):
        ids = d["ids"].to(device, dtype=torch.long)
        token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)
        mask = d["mask"].to(device, dtype=torch.long)

        tweet_tokens = d["tweet_tokens"]
        padding_len = d["padding_len"]
        orig_sentiment = d["orig_sentiment"]
        orig_selected_text = d["orig_selected_text"]
        orig_tweet = d["orig_tweet"]

        o1, o2 = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

        fin_output_start.append(torch.sigmoid(o1).cpu().detach().numpy())
        fin_output_end.append(torch.sigmoid(o2).cpu().detach().numpy())
        fin_padding_lens.extend(padding_len.cpu().detach().numpy().tolist())

        fin_tweet_tokens.extend(tweet_tokens)
        fin_orig_sentiment.extend(list(orig_sentiment))
        fin_orig_selected_text.extend(list(orig_selected_text))
        fin_orig_tweet.extend(list(orig_tweet))

fin_output_start = np.vstack(fin_output_start)
fin_output_end = np.vstack(fin_output_end)

threshold = 0.2
for j in range(len(fin_tweet_tokens)):
    tweet_tokens = fin_tweet_tokens[j]
    padding_len = fin_padding_lens[j]
    original_tweet = fin_orig_tweet[j]
    sentiment = fin_orig_sentiment[j]

    if padding_len > 0:
        mask_start = fin_output_start[j, :][:-padding_len] >= threshold
        mask_end = fin_output_end[j, :][:-padding_len] >= threshold
    else:
        mask_start = fin_output_start[j, :] >= threshold
        mask_end = fin_output_end[j, :] >= threshold

    mask = [0] * len(mask_start)
    idx_start = np.nonzero(mask_start)[0]
    idx_end = np.nonzero(mask_end)[0]

    if len(idx_start) > 0:
        idx_start = int(idx_start[0])
        if len(idx_end) > 0:
            idx_end = int(idx_end[0])
        else:
            idx_end = idx_start
    else:
        idx_start = 0
        idx_end = 0

    if idx_end < idx_start:
        idx_end = idx_start

    for mj in range(idx_start, min(idx_end + 1, len(mask))):
        mask[mj] = 1

    output_tokens = [
        x for p, x in enumerate(tweet_tokens.split()) if p < len(mask) and mask[p] == 1
    ]
    output_tokens = [x for x in output_tokens if x not in ("[CLS]", "[SEP]")]

    final_output = ""
    for ot in output_tokens:
        if ot.startswith("##"):
            final_output = final_output + ot[2:]
        elif len(ot) == 1 and ot in string.punctuation:
            final_output = final_output + ot
        else:
            final_output = final_output + " " + ot
    final_output = final_output.strip()

    if sentiment == "neutral" or len(str(original_tweet).split()) < 4:
        final_output = str(original_tweet)

    final_col.append(final_output)

submission = pd.read_csv(
    "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"
)
submission["selected_text"] = final_col

assert len(submission) == len(
    df_test
), f"Submission rows {len(submission)} != test rows {len(df_test)}"
assert list(submission.columns) == [
    "textID",
    "selected_text",
], f"Unexpected submission columns: {submission.columns.tolist()}"

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3490348603.py in <cell line: 0>()
      2 df_test.loc[:, "selected_text"] = df_test.text.values
      3 
----> 4 test_dataset = TweetDataset(
      5     tweet=df_test.text.values,
      6     sentiment=df_test.sentiment.values,

/tmp/ipykernel_55/1816181981.py in __init__(self, tweet, sentiment, selected_text)
      8 
      9         if self.tokenizer is None or not hasattr(self.tokenizer, "encode"):
---> 10             raise ValueError(
     11                 "config.TOKENIZER is not initialized correctly (missing .encode)."
     12             )

ValueError: config.TOKENIZER is not initialized correctly (missing .encode).
