# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.41598

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.41598) has done: 'I fix the dtype mismatch that occurs when the pretrained BERT model cannot be loaded. The fallback creates a zero tensor using the same dtype as the input IDs (Long), but the linear layer expects Float tensors. By creating the fallback tensor with a float dtype the model can run and produce a valid prediction file, allowing the pipeline to generate the required `submission.csv` without altering the core logic.'
- What this solution (achieved 0.41598) has done: 'I replace the fixed‑threshold logic that turns model logits into start/end positions with a simple arg‑max selection (choosing the highest‑probability token for start and end). This usually improves the Jaccard score because it avoids missing the correct span when the threshold is too low or high. The rest of the pipeline, model loading and CSV handling, remains unchanged, preserving the core logic while moving the score closer to the target.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd

try:
    import tokenizers
except Exception:
    tokenizers = None

import torch
import torch.nn as nn
from tqdm import tqdm
import string
import csv

try:
    import transformers
except Exception:
    transformers = None


from sklearn import model_selection
from torch.optim import AdamW


class config:
    MAX_LEN = 141
    TRAIN_BATCH_SIZE = 40  # 8 #??
    VALID_BATCH_SIZE = 16  # 4 #??
    EPOCHS = 10
    BERT_PATH = "../input/bertbaseuncased/"
    MODEL_PATH = "model.bin"
    TRAINING_FILE = "../input/tweet-sentiment-extraction/train.csv"
    TOKENIZER = None


class SimpleTokenizer:
    def __init__(self):
        pass

    def encode(self, text):
        tokens = text.split()
        ids = list(range(len(tokens)))
        offsets = []
        pos = 0
        for tok in tokens:
            start = text.find(tok, pos)
            end = start + len(tok)
            offsets.append((start, end))
            pos = end
        offsets = [(0, 0)] + offsets + [(len(text), len(text))]
        ids = [101] + ids + [102]  # dummy CLS/SEP ids
        return type(
            "EncodeResult",
            (),
            {
                "tokens": ["[CLS]"] + tokens + ["[SEP]"],
                "ids": ids,
                "offsets": offsets,
            },
        )


try:
    vocab_path = os.path.join(config.BERT_PATH, "vocab.txt")
    if os.path.isfile(vocab_path) and tokenizers is not None:
        config.TOKENIZER = tokenizers.BertWordPieceTokenizer(vocab_path, lowercase=True)
    else:
        raise FileNotFoundError
except Exception:
    try:
        if transformers is not None:
            config.TOKENIZER = transformers.AutoTokenizer.from_pretrained(
                "bert-base-uncased", use_fast=True
            )
        else:
            raise ImportError
    except Exception:
        config.TOKENIZER = SimpleTokenizer()


class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        try:
            if transformers is not None and os.path.isdir(config.BERT_PATH):
                self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
            else:
                raise ImportError
        except Exception as e:
            print(
                f"Warning: could not load pretrained BERT model ({e}); using zero‑tensor fallback."
            )
            self.bert = None
        self.l0 = nn.Linear(768, 2)

    def forward(self, ids, mask, token_type_ids):
        if self.bert is not None:
            sequence_output, pooled_output = self.bert(
                ids, attention_mask=mask, token_type_ids=token_type_ids
            )
        else:
            batch_size, seq_len = ids.shape
            sequence_output = torch.zeros(
                (batch_size, seq_len, 768), device=ids.device, dtype=torch.float32
            )
        logits = self.l0(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTBaseUncased()
model.to(device)
model = nn.DataParallel(model)

pretrained_path = "../input/tweet-sentiment-extraction-training/model.bin"
if os.path.isfile(pretrained_path):
    try:
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
    except Exception:
        print("Failed to load pretrained weights – using uninitialised model.")
else:
    print("Pretrained model.bin not found – using randomly initialized model.")

model.eval()


class TweetDataset:
    """initialization function"""

    def __init__(self, tweet, sentiment, selected_text):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.max_len = config.MAX_LEN
        self.tokenizer = config.TOKENIZER

    def __len__(self):
        return len(self.tweet)

    def _encode(self, text):
        """Unified tokenisation handling for different tokenizer types."""
        result = self.tokenizer.encode(text)

        if (
            hasattr(result, "tokens")
            and hasattr(result, "ids")
            and hasattr(result, "offsets")
        ):
            tokens = result.tokens
            ids = result.ids
            offsets = result.offsets
        elif isinstance(result, list):
            ids = result
            if hasattr(self.tokenizer, "convert_ids_to_tokens"):
                tokens = self.tokenizer.convert_ids_to_tokens(ids)
            else:
                tokens = ["[UNK]"] * len(ids)
            simple = SimpleTokenizer()
            simple_res = simple.encode(text)
            offsets = simple_res.offsets
        else:
            simple = SimpleTokenizer()
            simple_res = simple.encode(text)
            tokens = simple_res.tokens
            ids = simple_res.ids
            offsets = simple_res.offsets

        return tokens, ids, offsets

    def __getitem__(self, item):
        tweet = str(self.tweet[item])
        tweet = " ".join(tweet.split())

        selected_text = str(self.selected_text[item])
        selected_text = " ".join(selected_text.split())

        len_sel_text = len(selected_text)

        idx0 = -1
        idx1 = -1
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

        tok_tweet_tokens, tok_tweet_ids, tok_tweet_offsets = self._encode(tweet)

        tok_tweet_offsets = tok_tweet_offsets[1:-1]

        targets = [0] * (len(tok_tweet_tokens) - 2)
        for j, (offset1, offset2) in enumerate(tok_tweet_offsets):
            if sum(char_targets[offset1:offset2]) > 0:
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
        ids = tok_tweet_ids + [0] * padding_len
        mask = mask + [0] * padding_len
        token_type_ids = token_type_ids + [0] * padding_len
        targets = targets + [0] * padding_len
        targets_start = targets_start + [0] * padding_len
        targets_end = targets_end + [0] * padding_len

        sentiment = [1, 0, 0]  # neutral
        if self.sentiment[item] == "positive":
            sentiment = [0, 0, 1]
        if self.sentiment[item] == "negative":
            sentiment = [0, 1, 0]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
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


df_test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
df_test.loc[:, "selected_text"] = df_test.text.values  # placeholder

test_dataset = TweetDataset(
    tweet=df_test.text.values,
    sentiment=df_test.sentiment.values,
    selected_text=df_test.selected_text.values,
)

valid_data_loader = torch.utils.data.DataLoader(
    test_dataset, shuffle=False, batch_size=config.VALID_BATCH_SIZE, num_workers=1
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

        o1, o2 = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

        fin_output_start.append(torch.sigmoid(o1).cpu().numpy())
        fin_output_end.append(torch.sigmoid(o2).cpu().numpy())
        fin_padding_lens.extend(d["padding_len"].cpu().numpy().tolist())
        fin_tweet_tokens.extend(d["tweet_tokens"])
        fin_orig_sentiment.extend(d["orig_sentiment"])
        fin_orig_selected_text.extend(d["orig_selected_text"])
        fin_orig_tweet.extend(d["orig_tweet"])

    fin_output_start = np.vstack(fin_output_start)
    fin_output_end = np.vstack(fin_output_end)

    for j in range(len(fin_tweet_tokens)):
        tweet_tokens = fin_tweet_tokens[j]
        padding_len = fin_padding_lens[j]
        original_tweet = fin_orig_tweet[j]
        sentiment = fin_orig_sentiment[j]

        seq_len = fin_output_start.shape[1] - padding_len
        start_probs = fin_output_start[j, :seq_len]
        end_probs = fin_output_end[j, :seq_len]

        idx_start = int(np.argmax(start_probs))
        idx_end = int(np.argmax(end_probs))
        if idx_end < idx_start:
            idx_end = idx_start

        mask = [0] * seq_len
        for mj in range(idx_start, idx_end + 1):
            mask[mj] = 1

        output_tokens = [x for p, x in enumerate(tweet_tokens.split()) if mask[p] == 1]
        output_tokens = [x for x in output_tokens if x not in ("[CLS]", "[SEP]")]

        final_output = ""
        for ot in output_tokens:
            if ot.startswith("##"):
                final_output += ot[2:]
            elif len(ot) == 1 and ot in string.punctuation:
                final_output += ot
            else:
                final_output += " " + ot
        final_output = final_output.strip()
        if sentiment == "neutral" or len(original_tweet.split()) < 4:
            final_output = original_tweet

        final_col.append(final_output)

submission_path = "../input/tweet-sentiment-extraction/sample_submission.csv"
if not os.path.isfile(submission_path):
    submission_path = "sample_submission.csv"  # fallback to working directory

submission = pd.read_csv(submission_path)
submission.selected_text = final_col
submission.to_csv("submission.csv", index=False)

print(submission.head())
