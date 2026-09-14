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

0.6118680238723755

# 6. Current score

0.47686

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.41598) has done: 'I fix the pipeline so it can run in this Kaggle environment by removing the hard dependency on an external `bert-base-uncased` folder and `model.bin`, which are not present in your input paths. Instead, the code load `bert-base-uncased` weights/tokenizer directly via `transformers` (cached on Kaggle) and keep the same span-extraction model architecture and inference logic. I also fix the `TEST_CSV`/`SAMPLE_SUB` `NameError` by ensuring path discovery always defines them, and I make the dataset compatible with `BertTokenizerFast` while preserving the existing offset/wordpiece reconstruction behavior. Finally, the script always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.41598) has done: 'I fix the runtime crash caused by an incompatibility between `transformers` and the installed `protobuf` version by forcing the pure-Python protobuf implementation before importing `transformers`. Then I load the model weights from a local `model.bin`/`pytorch_model.bin` if present in the Kaggle input tree (preserving the exact same architecture and inference), otherwise fall back to the base `bert-base-uncased` weights as your current code does. This should both make the notebook run end-to-end reliably and (when the trained checkpoint exists) move the score up toward your target without changing the core span-extraction logic. Finally, I keep the submission writing unchanged and ensure `submission.csv` is produced with the correct columns.'
- What this solution (achieved 0.41598) has done: 'I fix the crash happening when importing/using `transformers` by setting the additional environment flags that make `protobuf` behave compatibly in Kaggle’s environment, and I delay importing `transformers` until after those flags are set. This is a runtime-only fix (no model/logic change) that unblocks end-to-end execution and ensures `submission.csv` is produced. I also keep the checkpoint auto-discovery behavior as-is so that if a trained checkpoint exists in the input tree it load and your score can move upward toward the target; otherwise it safely fall back to base `bert-base-uncased` as your current code intends. No changes are made to the architecture, tokenization strategy, span masking logic, or thresholding.'
- What this solution (achieved 0.41598) has done: 'I fix the `transformers` import crash caused by the `protobuf` / `transformers` incompatibility (the `MessageFactory.GetPrototype` error) by forcing the pure-Python protobuf implementation and additionally disabling the compiled protobuf backend before `transformers` is imported. I also make the environment setup happen as early as possible (in the very first cell) to ensure it takes effect reliably in Kaggle. No model architecture, tokenization logic, inference masking, or submission formatting be changed—this is a runtime/stability fix only. After this, the notebook should run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.41598) has done: 'I fix the `transformers` import crash caused by an incompatible protobuf backend by forcing the pure-Python protobuf implementation *and* preventing `transformers` from importing TensorFlow/JAX/Flax codepaths that trigger the failing protobuf symbols in this environment. This is a runtime-only change that preserves your model/inference logic and should unblock cell 1 end-to-end. I also make the `DataLoader` use `num_workers=0` to avoid occasional Kaggle worker spawning issues with tokenizers, without changing results. The rest of the architecture, thresholding, token reconstruction, and submission writing remain unchanged so the score impact should come only from successfully running with the intended `transformers` model weights/checkpoint loading.'
- What this solution (achieved 0.41598) has done: 'I fix the `transformers` import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf runtime *and* proactively using the compatible “python” protobuf API implementation before importing `transformers`. Then I ensure the model wrapper is called correctly when wrapped in `DataParallel` by using positional arguments (avoids occasional keyword-arg forwarding issues across PyTorch versions) without changing the model architecture or inference semantics. Finally, I keep all paths and submission formatting intact so the notebook reliably runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.41598) has done: 'I fix the runtime crash happening at `import transformers`/tokenizer load by forcing a protobuf/runtime combination that is compatible with this Kaggle image and by preventing optional TF/Flax/JAX imports that can trigger the failing protobuf symbols. These changes are environment-only and do not alter your model architecture, tokenization parameters, span logic, thresholding, or submission formatting. After that, the script run end-to-end and write a valid `submission.csv` with the required columns. Score should improve only if a trained checkpoint is actually found and successfully loaded; otherwise it behave the same as your current base-weights fallback.'
- What this solution (achieved 0.41598) has done: 'We fix the `transformers` import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation *and* monkey-patching `google.protobuf.message_factory.MessageFactory.GetPrototype` to call `GetMessageClass` when missing, before importing `transformers`. This is a runtime/stability fix that preserves your model architecture, tokenization, and inference semantics. With `transformers` successfully imported, the rest of the pipeline run end-to-end and write a valid `submission.csv` with the required columns. Score should improve only if a trained checkpoint is actually found and loaded; otherwise it behave the same as your current base-weights fallback.'
- What this solution (achieved 0.48352) has done: 'Your current score is far below the target, and the biggest limiter is that inference is running with base BERT weights (or a partially mismatched checkpoint), plus the span decoding uses a fixed sigmoid threshold that’s usually much worse than the standard argmax start/end decoding for this competition. I keep your exact model/forward pass and data/tokenization core logic, but (1) make checkpoint loading stricter/safer so we only use a checkpoint when it truly matches the head, otherwise we clearly fall back, and (2) change only the span selection post-processing to the common “argmax with constraint end>=start and max span length” approach, which typically boosts Jaccard substantially without changing training/architecture. I also add the standard neutral/short-text fallback you already have, and keep paths and submission writing identical. These are minimal changes focused on moving your score upward toward ~0.61.'
- What this solution (achieved 0.48968) has done: 'Your score gap to the target is large (0.48352 → 0.61187), so we should make a small but impactful inference-time fix without changing the model/training logic. The biggest issue is that decoding uses `tweet_tokens` built from `convert_ids_to_tokens` and then `.split()`, which breaks alignment whenever a token is `[UNK]` or when spacing differs—this hurts span reconstruction and Jaccard. I keep your model forward pass and argmax constrained decoding, but change decoding to use the tokenizer’s `offset_mapping` to slice directly from the original tweet text (the standard, metric-aligned way for this competition). I also exclude special/pad tokens from candidate spans and keep your neutral/short-text fallback and submission writing unchanged.'
- What this solution (achieved 0.47686) has done: 'We keep your model and constrained argmax decoding exactly as-is, but fix two inference-time issues that commonly suppress Jaccard in this competition: (1) the neutral/short-text fallback is too aggressive (`<4` words), so we restrict it to the standard neutral-only fallback; and (2) the decoding currently allows choosing `[CLS]/[SEP]`-adjacent tokens (offsets `(0,0)`) but also accidentally discards any real token whose true offset starts at 0, so we build a safer “valid token” mask using the attention mask and special token ids instead. These are minimal, metric-aligned post-processing changes that should improve score toward your target without altering architecture, training, or loss. The script still run end-to-end and write `submission.csv` in the required format.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_DISABLE_C_EXTENSIONS"] = "1"

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TRANSFORMERS_NO_JAX"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import numpy as np
import pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        if filename in (
            "train.csv",
            "test.csv",
            "sample_submission.csv",
            "vocab.txt",
            "pytorch_model.bin",
            "model.bin",
        ):
            print(os.path.join(dirname, filename))




## === cell 1
import torch
import torch.nn as nn
import transformers


def _first_existing_path(candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def _find_any_checkpoint_file(
    search_roots, filenames=("model.bin", "pytorch_model.bin")
):
    """
    Minimal helper to locate a trained checkpoint if it exists in the Kaggle input tree.
    This preserves the same model architecture; it only changes which weights are loaded.
    """
    for root in search_roots:
        if not root or not os.path.exists(root):
            continue
        for dirpath, _, files in os.walk(root):
            for fn in filenames:
                if fn in files:
                    return os.path.join(dirpath, fn)
    return None


DATA_ROOT = _first_existing_path(
    [
        "/kaggle/input/tweet-sentiment-extraction",
        "/kaggle/data/tweet-sentiment-extraction",
        "/kaggle/input",
        "/kaggle/data",
    ]
)
if DATA_ROOT is None:
    raise FileNotFoundError("Could not locate Kaggle input data root.")

TRAIN_CSV = _first_existing_path(
    [
        os.path.join(DATA_ROOT, "train.csv"),
        "/kaggle/input/train.csv",
        "/kaggle/data/train.csv",
    ]
)
TEST_CSV = _first_existing_path(
    [
        os.path.join(DATA_ROOT, "test.csv"),
        "/kaggle/input/test.csv",
        "/kaggle/data/test.csv",
    ]
)
SAMPLE_SUB = _first_existing_path(
    [
        os.path.join(DATA_ROOT, "sample_submission.csv"),
        "/kaggle/input/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
    ]
)

if TRAIN_CSV is None or TEST_CSV is None or SAMPLE_SUB is None:
    raise FileNotFoundError(
        f"Missing required files. TRAIN_CSV={TRAIN_CSV}, TEST_CSV={TEST_CSV}, SAMPLE_SUB={SAMPLE_SUB}"
    )


class config:
    MAX_LEN = 141
    TRAIN_BATCH_SIZE = 40
    VALID_BATCH_SIZE = 16
    EPOCHS = 2
    BERT_PATH = "bert-base-uncased"
    TRAINING_FILE = TRAIN_CSV
    TOKENIZER = transformers.BertTokenizerFast.from_pretrained(BERT_PATH)


class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.l0 = nn.Linear(768, 2)

    def forward(self, ids, mask, token_type_ids):
        out = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        sequence_output = (
            out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state
        )
        logits = self.l0(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTBaseUncased().to(device)

ckpt_path = _find_any_checkpoint_file(
    search_roots=[
        "/kaggle/input",
        "/kaggle/data",
        DATA_ROOT,
    ]
)

if ckpt_path is not None:
    state = torch.load(ckpt_path, map_location="cpu")
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = k[7:] if k.startswith("module.") else k
            new_state[nk] = v

        try:
            model.load_state_dict(new_state, strict=True)
            print("Loaded checkpoint (strict):", ckpt_path)
        except Exception as e:
            print("Strict checkpoint load failed, falling back to base weights.")
            print("Checkpoint:", ckpt_path)
            print("Error:", repr(e))
            ckpt_path = None
    else:
        print(
            "Checkpoint found but unrecognized format, using base pretrained weights only:",
            ckpt_path,
        )
        ckpt_path = None
else:
    print(
        "No trained checkpoint found; using base pretrained weights:", config.BERT_PATH
    )

model = nn.DataParallel(model)
model.eval()




## === cell 2
class TweetDataset:
    def __init__(self, tweet, sentiment, selected_text):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.max_len = config.MAX_LEN
        self.tokenizer = config.TOKENIZER

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
        if len_sel_text > 0:
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

        enc = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_token_type_ids=True,
            return_attention_mask=True,
        )

        ids = enc["input_ids"]
        mask = enc["attention_mask"]
        token_type_ids = enc["token_type_ids"]
        offsets = enc["offset_mapping"]

        tokens = self.tokenizer.convert_ids_to_tokens(ids)

        targets = [0] * len(tokens)
        for j, (o1, o2) in enumerate(offsets):
            if o1 == 0 and o2 == 0:
                continue
            if o2 > o1 and sum(char_targets[o1:o2]) > 0:
                targets[j] = 1

        targets_start = [0] * len(targets)
        targets_end = [0] * len(targets)

        non_zero = np.nonzero(targets)[0]
        if len(non_zero) > 0:
            targets_start[non_zero[0]] = 1
            targets_end[non_zero[-1]] = 1

        padding_len = 0
        for m in reversed(mask):
            if m == 0:
                padding_len += 1
            else:
                break

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
            "offsets": torch.tensor(offsets, dtype=torch.long),
            "orig_tweet": tweet,
            "sentiment": torch.tensor(sentiment, dtype=torch.long),
            "orig_sentiment": self.sentiment[item],
            "orig_selected_text": self.selected_text[item],
        }




## === cell 3
df_test = pd.read_csv(TEST_CSV)
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
fin_offsets = []
fin_orig_sentiment = []
fin_orig_tweet = []
fin_ids = []
fin_masks = []

with torch.no_grad():
    for bi, d in enumerate(valid_data_loader):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        padding_len = d["padding_len"]
        offsets = d["offsets"]
        orig_sentiment = d["orig_sentiment"]
        orig_tweet = d["orig_tweet"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)

        o1, o2 = model(ids, mask, token_type_ids)

        fin_output_start.append(o1.cpu().detach().numpy())
        fin_output_end.append(o2.cpu().detach().numpy())
        fin_padding_lens.extend(padding_len.cpu().detach().numpy().tolist())
        fin_offsets.append(offsets.cpu().detach().numpy())
        fin_ids.append(d["ids"].cpu().numpy())
        fin_masks.append(d["mask"].cpu().numpy())

        fin_orig_sentiment.extend(list(orig_sentiment))
        fin_orig_tweet.extend(list(orig_tweet))

fin_output_start = np.vstack(fin_output_start)
fin_output_end = np.vstack(fin_output_end)
fin_offsets = np.vstack(fin_offsets)  # (N, MAX_LEN, 2)
fin_ids = np.vstack(fin_ids)  # (N, MAX_LEN)
fin_masks = np.vstack(fin_masks)  # (N, MAX_LEN)

MAX_ANS_LEN = 30  # keep your constraint

cls_id = int(config.TOKENIZER.cls_token_id)
sep_id = int(config.TOKENIZER.sep_token_id)
pad_id = int(config.TOKENIZER.pad_token_id)

for j in range(len(fin_orig_tweet)):
    original_tweet = str(fin_orig_tweet[j])
    sentiment = fin_orig_sentiment[j]
    padding_len = int(fin_padding_lens[j])

    if padding_len > 0:
        start_logits = fin_output_start[j, :-padding_len]
        end_logits = fin_output_end[j, :-padding_len]
        offsets = fin_offsets[j, :-padding_len, :]
        ids_row = fin_ids[j, :-padding_len]
        mask_row = fin_masks[j, :-padding_len]
    else:
        start_logits = fin_output_start[j, :]
        end_logits = fin_output_end[j, :]
        offsets = fin_offsets[j, :, :]
        ids_row = fin_ids[j, :]
        mask_row = fin_masks[j, :]

    L = min(
        start_logits.shape[0],
        end_logits.shape[0],
        offsets.shape[0],
        ids_row.shape[0],
        mask_row.shape[0],
    )
    if L <= 0:
        final_col.append(original_tweet)
        continue

    start_logits = start_logits[:L]
    end_logits = end_logits[:L]
    offsets = offsets[:L]
    ids_row = ids_row[:L]
    mask_row = mask_row[:L]

    valid = (
        (mask_row == 1)
        & (ids_row != cls_id)
        & (ids_row != sep_id)
        & (ids_row != pad_id)
    )

    best_s, best_e = 0, 0
    best_score = -1e18

    for s in range(L):
        if not valid[s]:
            continue
        e_max = min(L - 1, s + MAX_ANS_LEN)
        best_e_local = None
        best_end_logit = -1e18
        for e in range(s, e_max + 1):
            if not valid[e]:
                continue
            val = float(end_logits[e])
            if val > best_end_logit:
                best_end_logit = val
                best_e_local = e
        if best_e_local is None:
            continue
        score = float(start_logits[s] + end_logits[best_e_local])
        if score > best_score:
            best_score = score
            best_s, best_e = s, best_e_local

    char_start = int(offsets[best_s, 0])
    char_end = int(offsets[best_e, 1])
    if char_end <= char_start:
        final_output = original_tweet
    else:
        final_output = original_tweet[char_start:char_end].strip()
        if final_output == "":
            final_output = original_tweet

    if sentiment == "neutral":
        final_output = str(original_tweet)

    final_col.append(final_output)

submission = pd.read_csv(SAMPLE_SUB)
submission["selected_text"] = final_col[: len(submission)]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Using DATA_ROOT:", DATA_ROOT)
print("Using TRAIN_CSV:", TRAIN_CSV)
print("Using TEST_CSV:", TEST_CSV)
print("Using SAMPLE_SUB:", SAMPLE_SUB)
print("Using pretrained:", config.BERT_PATH)
print("Checkpoint used:", ckpt_path)
