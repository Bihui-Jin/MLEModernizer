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

0.7075726985931396

# 6. Current score

0.47694

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.59324) has done: 'I remove the failing `fairseq/fastBPE` dependency (it isn’t available) and replace it with a compatible local Hugging Face tokenizer/model loader for the provided BERTweet files, keeping the same start/end-span inference logic. I also fix the Transformers API usage: load config/model from a local directory instead of passing a config.json path, and update the forward pass to use `hidden_states` correctly. To prevent DataLoader worker crashes due to missing globals, I set `num_workers=0` and ensure `bpe`/`vocab` equivalents are initialized before dataset creation. Finally, I guarantee the submission length matches `sample_submission.csv` and write `submission.csv` with the required columns.'
- What this solution (achieved 0.59324) has done: 'I fix the environment-breaking import error by pinning `protobuf` to the pure-Python implementation before importing `transformers`, which resolves the `MessageFactory.GetPrototype` crash. Then I make the BERTweet directory discovery robust by searching under `/kaggle/input/**/BERTweet_base_transformers` (since the current hardcoded `/kaggle/input/bertweet-model/` path doesn’t exist), so `RobertaTokenizerFast` and `RobertaConfig` load from the actual local folder. Finally, I ensure `bpe/vocab/model_config` are always defined before the dataset/loader/model code runs, so the later `NameError`s disappear, and the script reliably writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.59324) has done: 'I fix the early crash caused by an incompatible protobuf/transformers combination by importing `google.protobuf` safely and, if necessary, pinning the pure-Python protobuf implementation before importing `transformers`. Then I make model/tokenizer discovery robust for this dataset-only environment by falling back to a RoBERTa tokenizer/model (`roberta-base`) when the local BERTweet folder is not present, while keeping the same span start/end inference logic and weights loading (using the provided `twitroberta/model_*.bin`). Finally, I ensure `BERTWEET_DIR`, `tokenizer`, `bpe`, `vocab`, and `model_config` are always defined before the dataset/model code runs so the DataLoader and inference don’t hit `NameError`, and I still write a valid `submission.csv` with the required columns and length.'
- What this solution (achieved 0.47862) has done: 'I fix the early `protobuf` crash by forcing the pure-Python protobuf implementation before any `google.protobuf`/`transformers` import, which is the root cause of the `MessageFactory.GetPrototype` error in this environment. Then I fix the missing weights issue by searching for the `twitroberta` directory under `/kaggle/input/**/` and, if no local `model_*.bin` files are present (common in dataset-only setups), fall back to using the loaded base RoBERTa/BERTweet model directly (score likely drop, but it run end-to-end and produce a valid submission). Finally, I keep your exact start/end-span inference logic and ensure the submission is written as `submission.csv` with the correct columns and row count.'
- What this solution (achieved 0.52475) has done: 'I fix the immediate crash caused by an incompatible compiled protobuf implementation by forcing the pure-Python protobuf backend *and* preventing `transformers` from importing protobuf-based components before that setting takes effect. Then I fix a model-class mismatch bug (using `BertPreTrainedModel` with a RoBERTa backbone) which can silently break loading and worsen predictions, while keeping your exact span start/end inference logic unchanged. Finally, I make the weight loading more robust by allowing non-strict loading (to tolerate minor key-prefix differences) so available `model_*.bin` weights actually get used, which should move the score upward toward the target without changing the architecture or decoding procedure.'
- What this solution (achieved 0.47215) has done: 'You’re crashing before any modeling because `transformers` is importing a compiled protobuf backend that’s incompatible here, so the environment variable alone isn’t taking effect early enough. I force the pure-Python protobuf implementation *before* importing anything from `google.protobuf`/`transformers` by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2` and clearing any already-imported protobuf modules, which fixes the `MessageFactory.GetPrototype` AttributeError. This change is score-neutral but unblocks execution so your existing ensemble and span-decoding logic can run and produce `submission.csv`. I keep all model/dataset/inference logic unchanged.'
- What this solution (achieved 0.42468) has done: 'I fix the early `protobuf`/`transformers` crash that prevents the notebook from running by ensuring the pure-Python protobuf backend is selected *before* `transformers` is imported, and by removing any already-imported protobuf modules in a safer way. This is score-neutral but required to unblock execution and generate a valid submission. I also make the input CSV path resolution robust to both `/kaggle/input/...` and `/kaggle/data/...` layouts without changing any modeling logic. Everything else (dataset encoding, model architecture, weight loading, ensembling, and span decoding) is kept identical.'
- What this solution (achieved 0.55045) has done: 'We fix the immediate runtime crash by forcing the pure-Python protobuf implementation *before* any `google.protobuf`/`transformers` import and by proactively removing any already-imported protobuf modules, which addresses the `MessageFactory.GetPrototype` incompatibility in this environment. Next, we keep your exact model/inference logic but ensure the tokenizer/backbone selection is robust: prefer a local BERTweet directory if present, otherwise fall back to `roberta-base` without breaking. Finally, we preserve the existing decoding/submission logic and guarantee `submission.csv` is written with the correct columns and row count.'
- What this solution (achieved 0.39411) has done: 'I fix the crash in the first cell caused by an incompatible protobuf runtime by forcing the pure-Python protobuf implementation early and also disabling the C++ protobuf path via an additional environment variable, which is the usual root cause of `MessageFactory.GetPrototype` in Kaggle images. I keep the exact model architecture, span decoding, and ensembling logic unchanged, only adding this import-time safeguard so the notebook runs end-to-end reliably. To help nudge score upward (toward your target) without changing core logic, I also make sure we actually use the intended local BERTweet/twitroberta assets if present under both `/kaggle/input` and `/kaggle/data` (some runs were silently falling back to `roberta-base`). The output submission writing stays identical, producing a valid `submission.csv` with required columns and row count.'
- What this solution (achieved 0.39905) has done: 'We fix the immediate crash by forcing the pure-Python protobuf implementation *before* any `transformers` import in a way that works reliably in this Kaggle image (the current env var approach isn’t taking effect early enough). This is score-neutral but unblocks the whole pipeline so it can run end-to-end and write `submission.csv`. Then, without changing your model/span-decoding logic, we also ensure the code prefers the local competition-provided BERTweet/twitroberta assets under both `/kaggle/input` and `/kaggle/data` and loads any available `model_*.bin` weights (to avoid silently falling back to `roberta-base`, which likely caused the low 0.394 score). Everything else (architecture, ensembling, MAX_LEN, decoding) is kept the same.'
- What this solution (achieved 0.53325) has done: 'We fix the immediate crash in the first cell caused by an incompatible protobuf backend by preventing `google.protobuf` from being imported at all (Transformers can run fine without that explicit import, and it’s the line triggering the `MessageFactory.GetPrototype` error here). Then we keep your exact dataset/model/span-decoding logic intact, only ensuring the code uses the already-set pure-Python protobuf environment variables before importing `transformers`. This is score-neutral but unblocks execution so the ensemble/weights logic can run; if the local `twitroberta/model_*.bin` weights are present, the score should move back up toward your target. Finally, we still write a valid `submission.csv` with the required columns and row count.'
- What this solution (achieved 0.47694) has done: 'I fix the crash happening before any training/inference by preventing the protobuf C++ backend from being used and by avoiding imports that trigger the incompatible `MessageFactory.GetPrototype` path in this Kaggle image. Concretely, we set the environment variables *before* importing `transformers`, and we do not purge `sys.modules` in a way that can leave protobuf in a partially-initialized state. These changes are score-neutral but unblock execution so your existing ensemble/weights + span-decoding logic can run as intended (which should move your score back up toward the target if the local weights are found/loaded). I keep the model, dataset encoding, ensembling, decoding, and submission format unchanged.'

# 9. Code solution

## === cell 0
import os
import sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings

warnings.filterwarnings("ignore")

import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch.nn as nn

import transformers
from transformers import (
    RobertaConfig,
    RobertaModel,
    RobertaTokenizerFast,
    RobertaPreTrainedModel,
)

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
LEARNING_RATE = 6e-5
MAX_LEN = 126
TRAIN_BATCH_SIZE = 35
VALID_BATCH_SIZE = 32
EPOCHS = 3

INPUT_PATH = "/kaggle/input/"
if not os.path.isdir(INPUT_PATH):
    INPUT_PATH = "/kaggle/data/"


def _find_bertweet_dir(base_dir: str) -> str:
    candidates = []
    for root, dirs, files in os.walk(base_dir):
        if os.path.basename(root) == "BERTweet_base_transformers":
            if os.path.exists(os.path.join(root, "config.json")):
                candidates.append(root)
    if not candidates:
        return ""
    candidates = sorted(candidates, key=lambda p: (p.count(os.sep), len(p)))
    return candidates[0]


BERTWEET_DIR = _find_bertweet_dir(INPUT_PATH)
if not BERTWEET_DIR and INPUT_PATH != "/kaggle/data/":
    BERTWEET_DIR = _find_bertweet_dir("/kaggle/data/")

if BERTWEET_DIR:
    MODEL_DIR = BERTWEET_DIR
    print("Using local BERTweet dir:", MODEL_DIR)
else:
    MODEL_DIR = "roberta-base"
    print("BERTweet dir not found; falling back to:", MODEL_DIR)

tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_DIR)


class _BPEWrap:
    def __init__(self, tok):
        self.tok = tok

    def encode(self, text: str) -> str:
        toks = self.tok.tokenize(text)
        return " ".join(toks)

    def decode(self, token_str: str) -> str:
        toks = token_str.split()
        return self.tok.convert_tokens_to_string(toks)


bpe = _BPEWrap(tokenizer)


class _VocabWrap:
    def __init__(self, tok):
        self.tok = tok
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else 1

    def encode_line(self, token_str: str, append_eos=False, add_if_not_exist=False):
        toks = token_str.strip().split()
        ids = self.tok.convert_tokens_to_ids(toks)
        unk = self.tok.unk_token_id
        ids = [unk if (i is None or i < 0) else i for i in ids]
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx: int) -> str:
        return self.tok.convert_ids_to_tokens(int(idx))


vocab = _VocabWrap(tokenizer)




## === cell 2
class TweetDataset:
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
        e_tweet = "<s> " + bpe.encode(self.tweets[item]) + " </s>"
        enc_tweet = (
            vocab.encode_line(e_tweet, append_eos=False, add_if_not_exist=False)
            .long()
            .tolist()
        )

        if self.sentiments[item].strip() != "neutral":
            e_sentiment = (
                "</s> "
                + bpe.encode(self.sentiments[item])
                + " "
                + bpe.encode(self.sentiments[item])
                + " </s>"
            )
        else:
            e_sentiment = "</s> " + bpe.encode(self.sentiments[item]) + " </s>"
        enc_sentiment = (
            vocab.encode_line(e_sentiment, append_eos=False, add_if_not_exist=False)
            .long()
            .tolist()
        )

        enc_tweet_sentiment = enc_tweet + enc_sentiment

        if len(enc_tweet_sentiment) > self.max_len:
            enc_tweet_sentiment = enc_tweet_sentiment[: self.max_len]

        padding_len = self.max_len - len(enc_tweet_sentiment)
        pad_id = vocab.pad_id
        input_ids = enc_tweet_sentiment + ([pad_id] * padding_len)
        attention_mask = ([1] * len(enc_tweet_sentiment)) + ([0] * padding_len)

        start_index, end_index = 0, 0
        token_type_ids = [0] * self.max_len

        e_selected_text_ids = bpe.encode(self.selected_texts[item])
        enc_selected_text_ids = (
            vocab.encode_line(
                e_selected_text_ids, append_eos=False, add_if_not_exist=False
            )
            .long()
            .tolist()
        )

        if len(enc_selected_text_ids) > 0:
            for j in (
                i
                for i, e in enumerate(enc_tweet_sentiment)
                if e == enc_selected_text_ids[0]
            ):
                if (
                    enc_tweet_sentiment[j : j + len(enc_selected_text_ids)]
                    == enc_selected_text_ids
                ):
                    start_index = j
                    end_index = j + (len(enc_selected_text_ids))
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
class TweetModel(RobertaPreTrainedModel):
    def __init__(self, conf):
        super(TweetModel, self).__init__(conf)
        self.roberta = RobertaModel.from_pretrained(MODEL_DIR, config=conf)
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
        hs_last = hidden_states[-1]
        hs_prev = hidden_states[-2]
        cat = torch.cat((hs_last, hs_prev), dim=-1)
        cat = self.drop_out(cat)
        logits = self.l0(cat)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits




## === cell 4
def _resolve_data_file(rel_path: str) -> str:
    p1 = os.path.join("/kaggle/input", rel_path)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join("/kaggle/data", rel_path)
    if os.path.exists(p2):
        return p2
    p3 = os.path.join(INPUT_PATH, rel_path.replace("tweet-sentiment-extraction/", ""))
    return p3


test_path = _resolve_data_file("tweet-sentiment-extraction/test.csv")
df_test = pd.read_csv(test_path)
df_test.loc[:, "selected_text"] = df_test.text.values
df_test.head()



## === cell 5
model_config = RobertaConfig.from_pretrained(MODEL_DIR)
model_config.output_hidden_states = True

test_dataset = TweetDataset(
    tweets=df_test.text.values,
    sentiments=df_test.sentiment.values,
    selected_texts=df_test.selected_text.values,
)

data_loader = torch.utils.data.DataLoader(
    test_dataset, shuffle=False, batch_size=VALID_BATCH_SIZE, num_workers=0
)



## === cell 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


def _discover_weight_dir(base_dir: str) -> str:
    direct = os.path.join(base_dir, "twitroberta")
    if os.path.isdir(direct):
        return direct
    found = []
    for root, dirs, files in os.walk(base_dir):
        if os.path.basename(root) == "twitroberta":
            found.append(root)
    if not found:
        return ""
    return sorted(found, key=lambda p: (p.count(os.sep), len(p)))[0]


def _try_load_state_dict(m: torch.nn.Module, state: dict) -> bool:
    try:
        m.load_state_dict(state, strict=True)
        return True
    except Exception:
        pass

    if isinstance(state, dict):
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]

        if any(k.startswith("module.") for k in state.keys()):
            try:
                m.load_state_dict(
                    {k.replace("module.", "", 1): v for k, v in state.items()},
                    strict=False,
                )
                return True
            except Exception:
                pass

        try:
            m.load_state_dict(state, strict=False)
            return True
        except Exception:
            return False
    return False


def _load_model_from_weights_or_fallback(weight_path: str = ""):
    """
    Keep exact model architecture and inference logic.
    Fix: load available weights robustly; fallback only if missing/unloadable.
    """
    m = TweetModel(conf=model_config).to(device)
    if weight_path and os.path.exists(weight_path):
        state = torch.load(weight_path, map_location="cpu")
        ok = _try_load_state_dict(m, state)
        if ok:
            print("Loaded weights:", os.path.basename(weight_path))
        else:
            print("Found weights but could not fully load; using init/partial load.")
    else:
        print("Weights not found; using fallback (base backbone, random span head).")
    m.eval()
    return m


WEIGHT_DIR = _discover_weight_dir("/kaggle/input")
if not WEIGHT_DIR:
    WEIGHT_DIR = _discover_weight_dir("/kaggle/data")
if not WEIGHT_DIR:
    WEIGHT_DIR = _discover_weight_dir(INPUT_PATH)

print("Using WEIGHT_DIR:", WEIGHT_DIR if WEIGHT_DIR else "(not found)")

existing = []
if WEIGHT_DIR:
    weight_paths = [os.path.join(WEIGHT_DIR, f"model_{i}.bin") for i in range(8)]
    existing = [p for p in weight_paths if os.path.exists(p)]

model1 = _load_model_from_weights_or_fallback(existing[0] if len(existing) > 0 else "")
model2 = _load_model_from_weights_or_fallback(existing[1] if len(existing) > 1 else "")
model3 = _load_model_from_weights_or_fallback(existing[2] if len(existing) > 2 else "")
model4 = _load_model_from_weights_or_fallback(existing[3] if len(existing) > 3 else "")
model5 = _load_model_from_weights_or_fallback(existing[4] if len(existing) > 4 else "")
model6 = _load_model_from_weights_or_fallback(existing[5] if len(existing) > 5 else "")
model7 = _load_model_from_weights_or_fallback(existing[6] if len(existing) > 6 else "")
model8 = _load_model_from_weights_or_fallback(existing[7] if len(existing) > 7 else "")



## === cell 7
final_output = []
with torch.no_grad():
    tk0 = tqdm(data_loader, total=len(data_loader))
    for bi, d in enumerate(tk0):
        ids = d["ids"].to(device, dtype=torch.long)
        token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)
        mask = d["mask"].to(device, dtype=torch.long)

        orig_tweet = d["orig_tweet"]

        outputs_start1, outputs_end1 = model1(
            ids=ids, mask=mask, token_type_ids=token_type_ids
        )
        outputs_start2, outputs_end2 = model2(
            ids=ids, mask=mask, token_type_ids=token_type_ids
        )
        outputs_start3, outputs_end3 = model3(
            ids=ids, mask=mask, token_type_ids=token_type_ids
        )
        outputs_start4, outputs_end4 = model4(
            ids=ids, mask=mask, token_type_ids=token_type_ids
        )
        outputs_start5, outputs_end5 = model5(
            ids=ids, mask=mask, token_type_ids=token_type_ids
        )
        outputs_start6, outputs_end6 = model6(
            ids=ids, mask=mask, token_type_ids=token_type_ids
        )
        outputs_start7, outputs_end7 = model7(
            ids=ids, mask=mask, token_type_ids=token_type_ids
        )
        outputs_start8, outputs_end8 = model8(
            ids=ids, mask=mask, token_type_ids=token_type_ids
        )

        outputs_start = (
            outputs_start1
            + outputs_start2
            + outputs_start3
            + outputs_start4
            + outputs_start5
            + outputs_start7
            + outputs_start6
            + outputs_start8
        ) / 8

        outputs_end = (
            outputs_end1
            + outputs_end2
            + outputs_end3
            + outputs_end4
            + outputs_end5
            + outputs_end6
            + outputs_end7
            + outputs_end8
        ) / 8

        outputs_start = torch.softmax(outputs_start, dim=1).cpu().numpy()
        outputs_end = torch.softmax(outputs_end, dim=1).cpu().numpy()

        for i, tweet in enumerate(orig_tweet):
            a = int(np.argmax(outputs_start[i]))
            b = int(np.argmax(outputs_end[i]))

            if a > b:
                selected_text = tweet
            else:
                n_tweet = "<s> " + bpe.encode(tweet) + " </s>"
                nn_tweet = (
                    vocab.encode_line(n_tweet, append_eos=False, add_if_not_exist=False)
                    .long()
                    .tolist()
                )
                nn_tweet = nn_tweet[:MAX_LEN]
                select_ids = nn_tweet[a:b]
                if len(select_ids) == 0:
                    selected_text = tweet
                else:
                    selected_text = bpe.decode(" ".join([vocab[t] for t in select_ids]))
                    selected_text = selected_text.replace("<s>", "").replace("</s>", "")
                    if selected_text.strip() == "":
                        selected_text = tweet

            final_output.append(selected_text)

len(final_output), final_output[:3]



## === cell 8
sample_path = _resolve_data_file("tweet-sentiment-extraction/sample_submission.csv")
sub = pd.read_csv(sample_path)

if len(final_output) != len(sub):
    final_output = (final_output[: len(sub)] + df_test.text.tolist())[: len(sub)]

sub.loc[:, "selected_text"] = final_output
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())



## === cell 9
assert list(sub.columns) == ["textID", "selected_text"]
assert sub["selected_text"].isna().sum() == 0
sub.head()
