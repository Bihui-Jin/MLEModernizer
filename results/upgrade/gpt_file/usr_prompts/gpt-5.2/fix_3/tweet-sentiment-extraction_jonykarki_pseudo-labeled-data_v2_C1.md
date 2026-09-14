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
joblib==1.5.2
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

0.6645270586013794

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the unavailable `fairseq/fastBPE` dependency (it cannot be imported in this environment) and replace it with the locally-available `tokenizers` ByteLevelBPETokenizer built from the provided BERTweet `bpe.codes`/`dict.txt`, keeping the same overall inference approach. I also fix the Hugging Face config/model loading to use local paths correctly with this transformers version, and make the Roberta forward call compatible (use `output_hidden_states=True` and read `hidden_states`). Finally, I prevent DataLoader worker crashes by using `num_workers=0`, ensure deterministic inference (remove random span jitter), clamp span indices safely, and write a submission CSV whose row count exactly matches `test.csv`.'
- What this solution (achieved 0.0) has done: 'I fix the missing BERTweet asset paths by automatically locating the `BERTweet_base_transformers` directory inside the provided `/kaggle/input` tree instead of assuming `/kaggle/input/bertweet-model/` exists. I also resolve the transformers config/model loading error by explicitly pointing `from_pretrained` to a real local directory containing `config.json`, which avoids Hugging Face “repo id” validation. The `AttributeError: 'MessageFactory'...` is typically due to an incompatible `protobuf` implementation imported indirectly; I prevent that by forcing transformers to avoid optional protobuf usage and by importing transformers after setting the relevant environment variables. Finally, I ensure `BOS_ID/EOS_ID/PAD_ID` are defined before the dataset is used, so inference runs and a correctly-sized `submission.csv` is written.'

# 9. Code solution

## === cell 0
import os
import re
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn

os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import transformers
from tokenizers import ByteLevelBPETokenizer

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
LEARNING_RATE = 6e-5
MAX_LEN = 126
TRAIN_BATCH_SIZE = 35
VALID_BATCH_SIZE = 32
EPOCHS = 3

INPUT_PATH = "/kaggle/input/"

TEST_CSV = "/kaggle/input/tweet-sentiment-extraction/test.csv"
SAMPLE_SUB = "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"


def find_bertweet_dir(root="/kaggle/input"):
    """
    Bugfix: the dataset layout may not include /kaggle/input/bertweet-model/.
    We search for a local directory containing the expected BERTweet assets.
    """
    root = Path(root)
    candidates = []
    for p in root.rglob("BERTweet_base_transformers"):
        if p.is_dir():
            score = 0
            if (p / "config.json").exists():
                score += 10
            if (p / "pytorch_model.bin").exists():
                score += 8
            if (p / "dict.txt").exists():
                score += 5
            if (p / "bpe.codes").exists():
                score += 5
            candidates.append((score, p))
    if not candidates:
        raise FileNotFoundError(
            "Could not find 'BERTweet_base_transformers' under /kaggle/input. "
            "Please ensure the BERTweet model dataset is attached."
        )
    candidates.sort(key=lambda x: (-x[0], str(x[1])))
    return str(candidates[0][1])


BERTWEET_DIR = find_bertweet_dir(INPUT_PATH)

dict_path = Path(BERTWEET_DIR) / "dict.txt"
codes_path = Path(BERTWEET_DIR) / "bpe.codes"
config_path = Path(BERTWEET_DIR) / "config.json"

assert dict_path.exists(), f"Missing {dict_path}"
assert codes_path.exists(), f"Missing {codes_path}"
assert config_path.exists(), f"Missing {config_path}"

tmp_vocab_json = Path("/kaggle/working/bertweet_vocab.json")

token_to_id = {}
with open(dict_path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split()
        if not parts:
            continue
        tok = parts[0]
        if tok not in token_to_id:
            token_to_id[tok] = len(token_to_id)

with open(tmp_vocab_json, "w", encoding="utf-8") as f:
    json.dump(token_to_id, f, ensure_ascii=False)

bpe_tokenizer = ByteLevelBPETokenizer(
    vocab=str(tmp_vocab_json),
    merges=str(codes_path),
    add_prefix_space=True,
)

id_to_token = {i: t for t, i in token_to_id.items()}
PAD_ID = int(token_to_id.get("<pad>", 1))
BOS_ID = int(token_to_id.get("<s>", 0))
EOS_ID = int(token_to_id.get("</s>", 2))


def bpe_encode_to_ids(text: str):
    text = " " + " ".join(str(text).split())
    enc = bpe_tokenizer.encode(text)
    return enc.ids


def bpe_decode_from_ids(ids):
    s = bpe_tokenizer.decode(ids)
    s = s.replace("<s>", "").replace("</s>", "")
    return s


print("Using BERTWEET_DIR:", BERTWEET_DIR)
print("Special IDs:", {"BOS_ID": BOS_ID, "EOS_ID": EOS_ID, "PAD_ID": PAD_ID})




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2809952650.py in <cell line: 0>()
     40 
     41 
---> 42 BERTWEET_DIR = find_bertweet_dir(INPUT_PATH)
     43 
     44 dict_path = Path(BERTWEET_DIR) / "dict.txt"

/tmp/ipykernel_55/2809952650.py in find_bertweet_dir(root)
     32             candidates.append((score, p))
     33     if not candidates:
---> 34         raise FileNotFoundError(
     35             "Could not find 'BERTweet_base_transformers' under /kaggle/input. "
     36             "Please ensure the BERTweet model dataset is attached."

FileNotFoundError: Could not find 'BERTweet_base_transformers' under /kaggle/input. Please ensure the BERTweet model dataset is attached.

## === cell 2
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, tweets, sentiments, selected_texts):
        self.tweets = [" " + " ".join(str(tweet).split()) for tweet in tweets]
        self.sentiments = [
            " " + " ".join(str(sentiment).split()) for sentiment in sentiments
        ]
        self.selected_texts = [
            " " + " ".join(str(selected_text).split())
            for selected_text in selected_texts
        ]
        self.max_len = MAX_LEN

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, item):
        tweet_ids = [BOS_ID] + bpe_encode_to_ids(self.tweets[item]) + [EOS_ID]

        sent = self.sentiments[item].strip()
        sent_ids_core = bpe_encode_to_ids(sent)
        if sent != "neutral":
            sentiment_ids = [EOS_ID] + sent_ids_core + sent_ids_core + [EOS_ID]
        else:
            sentiment_ids = [EOS_ID] + sent_ids_core + [EOS_ID]

        enc_tweet_sentiment = tweet_ids + sentiment_ids

        enc_tweet_sentiment = enc_tweet_sentiment[: self.max_len]
        attention_mask = [1] * len(enc_tweet_sentiment)
        padding_len = self.max_len - len(enc_tweet_sentiment)

        input_ids = enc_tweet_sentiment + ([PAD_ID] * padding_len)
        attention_mask = attention_mask + ([0] * padding_len)
        token_type_ids = [0] * self.max_len

        start_index, end_index = 0, 0
        sel_ids = bpe_encode_to_ids(self.selected_texts[item])
        if len(sel_ids) > 0:
            for j in (i for i, e in enumerate(enc_tweet_sentiment) if e == sel_ids[0]):
                if enc_tweet_sentiment[j : j + len(sel_ids)] == sel_ids:
                    start_index = j
                    end_index = j + len(sel_ids)
                    break

        return {
            "ids": torch.tensor(input_ids, dtype=torch.long),
            "mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets_start": torch.tensor(start_index, dtype=torch.long),
            "targets_end": torch.tensor(end_index, dtype=torch.long),
            "orig_tweet": self.tweets[item],
            "orig_selected": self.selected_texts[item],
            "sentiment": self.sentiments[item],
        }




## === cell 3
class TweetModel(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(TweetModel, self).__init__(conf)
        self.roberta = transformers.RobertaModel.from_pretrained(
            BERTWEET_DIR,
            config=conf,
            local_files_only=True,
        )
        self.drop_out = nn.Dropout(0.1)
        self.activation = nn.LeakyReLU()
        self.l0 = nn.Linear(768 * 2, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)

    def forward(self, ids, mask, token_type_ids):
        out = self.roberta(
            input_ids=ids,
            attention_mask=mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = out.hidden_states  # tuple(layer0..layerN)
        hs = torch.cat((hidden_states[-1], hidden_states[-2]), dim=-1)
        hs = self.drop_out(hs)
        logits = self.l0(hs)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
df_test = pd.read_csv(TEST_CSV)
df_test.loc[:, "selected_text"] = df_test["text"].values



## === cell 5
model_config = transformers.RobertaConfig.from_pretrained(
    BERTWEET_DIR,
    local_files_only=True,
)
model_config.output_hidden_states = True



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3699134933.py in <cell line: 0>()
      1 # Bugfix: ensure transformers treats this as a local directory (must contain config.json).
      2 model_config = transformers.RobertaConfig.from_pretrained(
----> 3     BERTWEET_DIR,
      4     local_files_only=True,
      5 )

NameError: name 'BERTWEET_DIR' is not defined

## === cell 6
test_dataset = TweetDataset(
    tweets=df_test.text.values,
    sentiments=df_test.sentiment.values,
    selected_texts=df_test.selected_text.values,
)

data_loader = torch.utils.data.DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=VALID_BATCH_SIZE,
    num_workers=0,
)



## === cell 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_paths = [f"{INPUT_PATH}twitroberta/model_{i}.bin" for i in range(8)]
models = []
for mp in model_paths:
    if not Path(mp).exists():
        raise FileNotFoundError(f"Missing model checkpoint: {mp}")
    m = TweetModel(conf=model_config).to(device)
    state = torch.load(mp, map_location="cpu")
    m.load_state_dict(state, strict=True)
    m.eval()
    models.append(m)

print(f"Loaded {len(models)} models onto {device}.")



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1421656772.py in <cell line: 0>()
      5 for mp in model_paths:
      6     if not Path(mp).exists():
----> 7         raise FileNotFoundError(f"Missing model checkpoint: {mp}")
      8     m = TweetModel(conf=model_config).to(device)
      9     state = torch.load(mp, map_location="cpu")

FileNotFoundError: Missing model checkpoint: /kaggle/input/twitroberta/model_0.bin

## === cell 8
final_output = []

with torch.no_grad():
    tk0 = tqdm(data_loader, total=len(data_loader))
    for _, d in enumerate(tk0):
        ids = d["ids"].to(device, dtype=torch.long)
        token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)
        mask = d["mask"].to(device, dtype=torch.long)

        orig_tweet = d["orig_tweet"]

        outs_start = []
        outs_end = []
        for m in models:
            s, e = m(ids=ids, mask=mask, token_type_ids=token_type_ids)
            outs_start.append(s)
            outs_end.append(e)

        outputs_start = torch.stack(outs_start, dim=0).mean(dim=0)
        outputs_end = torch.stack(outs_end, dim=0).mean(dim=0)

        outputs_start = torch.softmax(outputs_start, dim=1).cpu().numpy()
        outputs_end = torch.softmax(outputs_end, dim=1).cpu().numpy()

        for i, tweet in enumerate(orig_tweet):
            a = int(np.argmax(outputs_start[i]))
            b = int(np.argmax(outputs_end[i]))

            a = max(0, min(a, MAX_LEN - 1))
            b = max(0, min(b, MAX_LEN))

            if a > b:
                selected_text = tweet
            else:
                tweet_ids = [BOS_ID] + bpe_encode_to_ids(tweet) + [EOS_ID]
                select_ids = tweet_ids[a:b] if b > a else []
                selected_text = bpe_decode_from_ids(select_ids).strip()
                if selected_text == "":
                    selected_text = tweet

            final_output.append(selected_text)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/417393075.py in <cell line: 0>()
      3 with torch.no_grad():
      4     tk0 = tqdm(data_loader, total=len(data_loader))
----> 5     for _, d in enumerate(tk0):
      6         ids = d["ids"].to(device, dtype=torch.long)
      7         token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)

/usr/local/lib/python3.11/dist-packages/tqdm/notebook.py in __iter__(self)
    248         try:
    249             it = super().__iter__()
--> 250             for obj in it:
    251                 # return super(tqdm...) will not catch exception
    252                 yield obj

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

/tmp/ipykernel_55/404148934.py in __getitem__(self, item)
     15 
     16     def __getitem__(self, item):
---> 17         tweet_ids = [BOS_ID] + bpe_encode_to_ids(self.tweets[item]) + [EOS_ID]
     18 
     19         sent = self.sentiments[item].strip()

NameError: name 'BOS_ID' is not defined

## === cell 9
sub = pd.read_csv(SAMPLE_SUB)
if len(final_output) != len(sub):
    final_output = final_output[: len(sub)]
    if len(final_output) < len(sub):
        final_output = final_output + [""] * (len(sub) - len(final_output))

sub.loc[:, "selected_text"] = final_output
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)



## === cell 10
sub.sample(20, random_state=SEED)
