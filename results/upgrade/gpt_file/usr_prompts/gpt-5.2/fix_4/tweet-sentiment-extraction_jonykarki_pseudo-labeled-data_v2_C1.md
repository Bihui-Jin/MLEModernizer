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

0.59324

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the unavailable `fairseq/fastBPE` dependency (it cannot be imported in this environment) and replace it with the locally-available `tokenizers` ByteLevelBPETokenizer built from the provided BERTweet `bpe.codes`/`dict.txt`, keeping the same overall inference approach. I also fix the Hugging Face config/model loading to use local paths correctly with this transformers version, and make the Roberta forward call compatible (use `output_hidden_states=True` and read `hidden_states`). Finally, I prevent DataLoader worker crashes by using `num_workers=0`, ensure deterministic inference (remove random span jitter), clamp span indices safely, and write a submission CSV whose row count exactly matches `test.csv`.'
- What this solution (achieved 0.0) has done: 'I fix the missing BERTweet asset paths by automatically locating the `BERTweet_base_transformers` directory inside the provided `/kaggle/input` tree instead of assuming `/kaggle/input/bertweet-model/` exists. I also resolve the transformers config/model loading error by explicitly pointing `from_pretrained` to a real local directory containing `config.json`, which avoids Hugging Face “repo id” validation. The `AttributeError: 'MessageFactory'...` is typically due to an incompatible `protobuf` implementation imported indirectly; I prevent that by forcing transformers to avoid optional protobuf usage and by importing transformers after setting the relevant environment variables. Finally, I ensure `BOS_ID/EOS_ID/PAD_ID` are defined before the dataset is used, so inference runs and a correctly-sized `submission.csv` is written.'
- What this solution (achieved 0.59324) has done: 'Your run fails early because the expected BERTweet asset directory and the `twitroberta` checkpoints are not present in this Kaggle environment, so `BERTWEET_DIR`, special token IDs, and the model ensemble never get created. To make the notebook run end-to-end and produce a valid `submission.csv`, I keep your pipeline structure but add a robust fallback that (a) uses the provided `sample_submission.csv`/`test.csv` and (b) generates a safe baseline prediction (`selected_text = text`, with the common neutral heuristic) when model files are missing. I also avoid importing/using transformers protobuf-dependent code paths when we’re not actually loading a model, which eliminates the `MessageFactory` crash. This move your score from 0.0 (no valid submission) to a non-zero baseline and ensure a correct, quoted CSV output.'

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
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

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
    Robust search for a local directory containing expected BERTweet assets.
    If not found, returns None (so we can fall back to a safe baseline).
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
        return None
    candidates.sort(key=lambda x: (-x[0], str(x[1])))
    return str(candidates[0][1])


BERTWEET_DIR = find_bertweet_dir(INPUT_PATH)

PAD_ID, BOS_ID, EOS_ID = 1, 0, 2

bpe_tokenizer = None
bpe_encode_to_ids = None
bpe_decode_from_ids = None

if BERTWEET_DIR is not None:
    dict_path = Path(BERTWEET_DIR) / "dict.txt"
    codes_path = Path(BERTWEET_DIR) / "bpe.codes"
    config_path = Path(BERTWEET_DIR) / "config.json"

    if dict_path.exists() and codes_path.exists() and config_path.exists():
        from tokenizers import ByteLevelBPETokenizer

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

        PAD_ID = int(token_to_id.get("<pad>", PAD_ID))
        BOS_ID = int(token_to_id.get("<s>", BOS_ID))
        EOS_ID = int(token_to_id.get("</s>", EOS_ID))

        def _bpe_encode_to_ids(text: str):
            text = " " + " ".join(str(text).split())
            enc = bpe_tokenizer.encode(text)
            return enc.ids

        def _bpe_decode_from_ids(ids):
            s = bpe_tokenizer.decode(ids)
            s = s.replace("<s>", "").replace("</s>", "")
            return s

        bpe_encode_to_ids = _bpe_encode_to_ids
        bpe_decode_from_ids = _bpe_decode_from_ids

print("Using BERTWEET_DIR:", BERTWEET_DIR)
print("Special IDs:", {"BOS_ID": BOS_ID, "EOS_ID": EOS_ID, "PAD_ID": PAD_ID})




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

        if bpe_encode_to_ids is None:
            raise RuntimeError(
                "BPE tokenizer is not available (BERTweet assets not found). "
                "Cannot build TweetDataset for model inference."
            )

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
transformers = None


def maybe_import_transformers():
    global transformers
    if transformers is None:
        import transformers as _tf

        transformers = _tf
    return transformers




## === cell 4
TweetModel = None
model_config = None


def build_model_config_and_class():
    tf = maybe_import_transformers()

    class _TweetModel(tf.BertPreTrainedModel):
        def __init__(self, conf):
            super(_TweetModel, self).__init__(conf)
            self.roberta = tf.RobertaModel.from_pretrained(
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
            hidden_states = out.hidden_states
            hs = torch.cat((hidden_states[-1], hidden_states[-2]), dim=-1)
            hs = self.drop_out(hs)
            logits = self.l0(hs)
            start_logits, end_logits = logits.split(1, dim=-1)
            return start_logits.squeeze(-1), end_logits.squeeze(-1)

    conf = tf.RobertaConfig.from_pretrained(
        BERTWEET_DIR,
        local_files_only=True,
    )
    conf.output_hidden_states = True
    return conf, _TweetModel




## === cell 5
df_test = pd.read_csv(TEST_CSV)
df_test.loc[:, "selected_text"] = df_test["text"].values




## === cell 6
model_paths = [f"{INPUT_PATH}twitroberta/model_{i}.bin" for i in range(8)]
all_ckpts_exist = all(Path(mp).exists() for mp in model_paths)

can_run_model_inference = (
    (BERTWEET_DIR is not None) and (bpe_encode_to_ids is not None) and all_ckpts_exist
)

print("all_ckpts_exist:", all_ckpts_exist)
print("can_run_model_inference:", can_run_model_inference)

final_output = []

if not can_run_model_inference:
    final_output = []
    for t, s in zip(
        df_test["text"].astype(str).values, df_test["sentiment"].astype(str).values
    ):
        t_clean = " ".join(t.split()).strip()
        if s.strip().lower() == "neutral":
            final_output.append(t_clean)
        else:
            final_output.append(t_clean)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config, TweetModel = build_model_config_and_class()

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

    models = []
    for mp in model_paths:
        m = TweetModel(conf=model_config).to(device)
        state = torch.load(mp, map_location="cpu")
        m.load_state_dict(state, strict=True)
        m.eval()
        models.append(m)

    print(f"Loaded {len(models)} models onto {device}.")

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




## === cell 7
sub = pd.read_csv(SAMPLE_SUB)

if len(final_output) != len(sub):
    final_output = final_output[: len(sub)]
    if len(final_output) < len(sub):
        final_output = final_output + [""] * (len(sub) - len(final_output))

sub.loc[:, "selected_text"] = ["" if pd.isna(x) else str(x) for x in final_output]

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())




## === cell 8
sub.sample(20, random_state=SEED)
