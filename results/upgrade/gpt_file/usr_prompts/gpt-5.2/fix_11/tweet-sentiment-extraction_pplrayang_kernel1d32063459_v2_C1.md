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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
nltk==3.9.2
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

0.7155378460884094

# 6. Current score

0.57486

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.59324) has done: 'I fix the environment/runtime blockers so the notebook can run end-to-end and actually write `submission.csv`. The main issues are (1) a `protobuf`/Transformers import-time crash and (2) hardcoded RoBERTa/tokenizer paths that don’t exist in your provided dataset tree, causing the tokenizer files to be missing and preventing inference (which also causes `predictions` to be undefined). I switch to loading `roberta-base` and its fast tokenizer directly from the installed `transformers` package (offline, no internet), and if the external fold checkpoints aren’t present, I fall back to a safe rule-based submission so you always get a valid CSV. This preserves your core model architecture and inference semantics when checkpoints are available, while ensuring a submission file is produced in all cases.'
- What this solution (achieved 0.59324) has done: 'I fix the import-time crash by forcing the pure-Python protobuf implementation *before* importing `transformers`, and I make the RoBERTa/tokenizer loading robust to Kaggle’s offline environment by first trying local Kaggle model caches and then falling back to `roberta-base` only if available locally. This should restore the intended checkpoint-based inference (which is what gets you closer to the target score) instead of silently falling back to the heuristic. I also fix a determinism/perf setting bug (`cudnn.deterministic` + `benchmark` conflict) without changing training/inference semantics, and I keep the submission writing/format exactly as required.'
- What this solution (achieved 0.57486) has done: 'I fix the import-time protobuf crash that prevents `transformers` from loading by forcing a compatible protobuf runtime setting *and* avoiding the code paths that trigger the `MessageFactory.GetPrototype` issue. Then I make the RoBERTa/tokenizer/model loading robust in Kaggle’s offline environment by trying common local cache/dataset locations first and only using `from_pretrained(..., local_files_only=True)` on those resolved paths. Finally, to move score up toward the target, I prevent the low-scoring “echo the whole tweet” fallback unless no local roberta weights can be found at all; if roberta weights exist but fold checkpoints are missing, we still run the base model (random head) would be worse, so we keep the heuristic, but we improve the heuristic slightly using a minimal, sentiment-aware word-span selection (still rule-based, no architecture/training changes) to better align with Jaccard.'
- What this solution (achieved 0.57486) has done: 'I fix the import-time crash by preventing `transformers` from importing TensorFlow/vision loss modules and by using the pure-Python protobuf implementation (your current `"cpp"` setting is what triggers the `_message` ImportError). Then I make the notebook robust to Kaggle’s offline environment by loading RoBERTa/tokenizer/config strictly from local caches or local dataset folders, and if not present, cleanly falling back to the existing heuristic so a valid `submission.csv` is always produced. Finally, I fix the cascading `NameError`s by ensuring all constants (like `MAX_LEN`, `test_file`, `submission_template`) are defined in the first cell even if tokenizer loading fails.'
- What this solution (achieved 0.57486) has done: 'I fix the import-time crash coming from an incompatible protobuf runtime by removing the unsupported `TRANSFORMERS_NO_PROTOBUF` flag and pinning protobuf to the pure-Python implementation before any `transformers` import. Then I make the tokenizer/model loading robust in Kaggle’s offline environment by trying both the dataset-provided roberta directories and the local HF cache, while keeping your exact model and inference logic unchanged. Finally, I ensure that if model loading still fails for any reason, the code cleanly falls back to the existing heuristic path and always writes a valid `submission.csv` with the correct columns and row count.'
- What this solution (achieved 0.57486) has done: 'I fix the `transformers` import-time crash (`MessageFactory.GetPrototype`) by forcing a protobuf runtime that’s compatible with Kaggle’s environment *before* importing `transformers`, and by adding the standard safe flags that prevent optional TF/vision imports from triggering problematic protobuf code paths. This is a runtime-only fix that keeps your model/inference logic unchanged and should allow the RoBERTa tokenizer/config to load so you can use your fold checkpoints when they exist (which should lift score toward the target). I also make the pretrained-resolver slightly more robust to local HF cache layouts without changing what model is loaded when files are present. Finally, I ensure we always write a valid `submission.csv` even if model loading still fails, preserving the existing heuristic fallback.'
- What this solution (achieved 0.57486) has done: 'I fix the Transformers/protobuf crash (`MessageFactory.GetPrototype`) by pinning a safe protobuf runtime behavior *before* importing `transformers`, and by forcing Transformers to avoid optional integrations that can trigger problematic protobuf code paths in Kaggle. This is a runtime-only change that preserves your model/inference logic, but it should allow the RoBERTa tokenizer/config to load so you can actually use the fold checkpoints (which is the main path to improve score toward your target). I also make the pretrained-directory resolver slightly more robust to local HuggingFace cache layouts while keeping `local_files_only=True` and the same `roberta-base` backbone. Everything else (dataset processing, model architecture, inference averaging, and submission writing) is kept the same.'
- What this solution (achieved 0.57486) has done: 'I fix the `transformers` import-time crash (`MessageFactory.GetPrototype`) by forcing Transformers to use the Python protobuf runtime and by explicitly disabling optional integrations that can trigger the bad protobuf code paths before importing `transformers`. This is a runtime compatibility fix and does not alter your model architecture, weights usage, or inference logic. I also make the tokenizer/config loader slightly more robust by avoiding the failing `google.protobuf` C++ module when present, while keeping `local_files_only=True` (offline-safe). With these changes, the notebook should run end-to-end and (when your fold checkpoints exist) use them for inference, which should lift score toward your target; otherwise it still produce a valid heuristic submission.'

# 9. Code solution

## === cell 0
import os
import warnings

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_KERAS", "1")
os.environ.setdefault("DISABLE_TRANSFORMERS_IMAGE_TRANSFORMS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

import numpy as np
import pandas as pd
import random
import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm


from transformers import RobertaModel, RobertaConfig, RobertaTokenizerFast

warnings.filterwarnings("ignore")


def seed_everything(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed = 42
seed_everything(seed)

batch_size = 32
N = 10
skf = StratifiedKFold(n_splits=N, shuffle=True, random_state=seed)
NUM_WORKERS = 2

test_file = "/kaggle/input/tweet-sentiment-extraction/test.csv"
submission_template = "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"

PRETRAINED_NAME = "roberta-base"
outdir = "/kaggle/input/robertalineardropout/"

MAX_LEN = 96
LINEAR_DROPOUT = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _path_has_roberta_files(path: str) -> bool:
    if not isinstance(path, str) or not path:
        return False
    if not os.path.isdir(path):
        return False
    has_cfg = os.path.isfile(os.path.join(path, "config.json"))
    has_vocab = os.path.isfile(os.path.join(path, "vocab.json"))
    has_merges = os.path.isfile(os.path.join(path, "merges.txt"))
    has_sp = os.path.isfile(os.path.join(path, "sentencepiece.bpe.model"))
    return has_cfg and ((has_vocab and has_merges) or has_sp)


def _try_resolve_pretrained_dir_candidates():
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    candidates = [
        "/kaggle/input/roberta-base",
        "/kaggle/input/roberta",
        "/kaggle/input/tweet-sentiment-extraction/roberta-base",
        "/kaggle/input/tweet-sentiment-extraction/roberta",
        os.path.join(hf_home, "hub", "models--roberta-base", "snapshots"),
        "/kaggle/working/.cache/huggingface/hub/models--roberta-base/snapshots",
        os.path.join(hf_home, "transformers"),
        "/kaggle/working/.cache/huggingface/transformers",
        PRETRAINED_NAME,  # only works if cached locally; used with local_files_only=True
    ]

    expanded = []
    for c in candidates:
        if isinstance(c, str) and c.endswith("snapshots") and os.path.isdir(c):
            try:
                for child in sorted(os.listdir(c)):
                    p = os.path.join(c, child)
                    if os.path.isdir(p):
                        expanded.append(p)
            except Exception:
                pass
        else:
            expanded.append(c)

    preferred = []
    for c in expanded:
        if isinstance(c, str) and _path_has_roberta_files(c):
            preferred.append(c)
    rest = [c for c in expanded if c not in preferred]
    return preferred + rest


def _try_load_tokenizer_and_config():
    candidates = _try_resolve_pretrained_dir_candidates()
    last_err = None
    for name in candidates:
        try:
            tok = RobertaTokenizerFast.from_pretrained(name, local_files_only=True)
            cfg = RobertaConfig.from_pretrained(
                name, output_hidden_states=True, local_files_only=True
            )
            print(f"Loaded tokenizer/config from: {name}")
            return tok, cfg, name
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        "Could not load RoBERTa tokenizer/config from local files (offline). "
        "Please ensure roberta-base exists in the Kaggle environment (either as a dataset input or in the HF cache)."
    ) from last_err


_TOKENIZER_OK = True
try:
    TOKENIZER, _ROBERTA_CONFIG, PRETRAINED_RESOLVED = _try_load_tokenizer_and_config()
except Exception as e:
    _TOKENIZER_OK = False
    TOKENIZER, _ROBERTA_CONFIG, PRETRAINED_RESOLVED = None, None, None
    print("WARNING: RoBERTa tokenizer/config could not be loaded locally.")
    print(f"Reason: {type(e).__name__}: {e}")
    print("Will fall back to heuristic predictions (valid submission, lower score).")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, df, max_len=MAX_LEN):
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
        self.labeled = "selected_text" in df.columns

    def __getitem__(self, index):
        data = {}
        row = self.df.iloc[index]

        ids, masks, tweet, offsets = self.get_input_data(row)
        data["ids"] = ids
        data["masks"] = masks
        data["tweet"] = tweet
        data["offsets"] = offsets

        if self.labeled:
            start_idx, end_idx = self.get_target_idx(row, tweet, offsets)
            data["start_idx"] = start_idx
            data["end_idx"] = end_idx

        return data

    def __len__(self):
        return len(self.df)

    def get_input_data(self, row):
        if TOKENIZER is None:
            raise RuntimeError("TOKENIZER is not available; cannot build model inputs.")

        tweet = " " + " ".join(str(row.text).lower().split())
        sentiment = str(row.sentiment).lower()

        enc = TOKENIZER(
            sentiment,
            tweet,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_offsets_mapping=True,
        )

        ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        masks = torch.tensor(enc["attention_mask"], dtype=torch.long)
        offsets = torch.tensor(enc["offset_mapping"], dtype=torch.long)

        return ids, masks, tweet, offsets

    def get_target_idx(self, row, tweet, offsets):
        selected_text = " " + " ".join(str(row.selected_text).lower().split())

        len_st = len(selected_text) - 1
        idx0 = None
        idx1 = None

        for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
            if " " + tweet[ind : ind + len_st] == selected_text:
                idx0 = ind
                idx1 = ind + len_st - 1
                break

        char_targets = [0] * len(tweet)
        if idx0 is not None and idx1 is not None:
            for ct in range(idx0, idx1 + 1):
                char_targets[ct] = 1

        target_idx = []
        for j, (o1, o2) in enumerate(offsets.tolist()):
            if o1 == o2 == 0:
                continue
            if sum(char_targets[o1:o2]) > 0:
                target_idx.append(j)

        if len(target_idx) == 0:
            return 0, 0

        start_idx = target_idx[0]
        end_idx = target_idx[-1]
        return start_idx, end_idx




## === cell 2
def get_test_loader(df, batch_size=32):
    loader = torch.utils.data.DataLoader(
        TweetDataset(df),
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    return loader




## === cell 3
class TweetModel(nn.Module):
    def __init__(self):
        super(TweetModel, self).__init__()

        if _ROBERTA_CONFIG is None or PRETRAINED_RESOLVED is None:
            raise RuntimeError(
                "RoBERTa config/weights path not available to build TweetModel."
            )

        config = _ROBERTA_CONFIG
        self.roberta = RobertaModel.from_pretrained(
            PRETRAINED_RESOLVED, config=config, local_files_only=True
        )

        self.dropout = nn.Dropout(LINEAR_DROPOUT)
        self.fc = nn.Linear(config.hidden_size, 2)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hs = outputs.hidden_states

        x = torch.stack([hs[-1], hs[-2], hs[-3]])
        x = torch.mean(x, 0)
        x = self.dropout(x)
        x = self.fc(x)

        start_logits, end_logits = x.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits




## === cell 4
def get_selected_text(text, start_idx, end_idx, offsets):
    selected_text = ""
    for ix in range(start_idx, end_idx + 1):
        o1, o2 = int(offsets[ix][0]), int(offsets[ix][1])
        if o1 == o2 == 0:
            continue
        selected_text += text[o1:o2]
        if (ix + 1) < len(offsets) and int(offsets[ix][1]) < int(offsets[ix + 1][0]):
            selected_text += " "
    return selected_text


def jaccard(str1, str2):
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())
    c = a.intersection(b)
    denom = len(a) + len(b) - len(c)
    return float(len(c)) / denom if denom != 0 else 0.0


def compute_jaccard_score(text, start_idx, end_idx, start_logits, end_logits, offsets):
    start_pred = int(np.argmax(start_logits))
    end_pred = int(np.argmax(end_logits))
    if start_pred > end_pred:
        pred = text
    else:
        pred = get_selected_text(text, start_pred, end_pred, offsets)

    true = get_selected_text(text, start_idx, end_idx, offsets)
    return jaccard(true, pred)


_POS_WORDS = {
    "good",
    "great",
    "love",
    "awesome",
    "best",
    "amazing",
    "nice",
    "happy",
    "fantastic",
    "excellent",
    "perfect",
    "wonderful",
    "thanks",
    "thank",
}
_NEG_WORDS = {
    "bad",
    "hate",
    "worst",
    "awful",
    "sad",
    "terrible",
    "horrible",
    "annoying",
    "angry",
    "disappointed",
    "sucks",
    "suck",
    "sorry",
}


def heuristic_select(text: str, sentiment: str) -> str:
    t = str(text)
    s = str(sentiment).lower()
    t_stripped = t.strip()
    if s == "neutral" or len(t_stripped.split()) <= 2:
        return t_stripped

    words = t_stripped.split()
    low_words = [w.strip(".,!?;:\"'()[]{}").lower() for w in words]

    lex = _POS_WORDS if s == "positive" else _NEG_WORDS
    idxs = [i for i, w in enumerate(low_words) if w in lex]
    if not idxs:
        if s == "negative":
            return " ".join(words[max(0, len(words) - 3) :])
        return " ".join(words[: min(3, len(words))])

    i0, i1 = min(idxs), max(idxs)
    i0 = max(0, i0 - 1)
    i1 = min(len(words) - 1, i1 + 1)
    return " ".join(words[i0 : i1 + 1]).strip()




## === cell 5
test_df = pd.read_csv(test_file)
test_df["text"] = test_df["text"].astype(str)

predictions = []
models = []


def _checkpoint_exists(path: str) -> bool:
    try:
        return os.path.isfile(path)
    except Exception:
        return False


can_run_model = _TOKENIZER_OK

available_ckpts = []
if can_run_model:
    print("loading models..")
    for fold in range(skf.n_splits):
        state_path = f"{outdir}roberta_fold{fold+1}.pth"
        if _checkpoint_exists(state_path):
            available_ckpts.append((fold, state_path))
else:
    print("Tokenizer/config unavailable -> skipping model inference.")

if can_run_model and len(available_ckpts) > 0:
    test_loader = get_test_loader(test_df, batch_size=batch_size)

    for fold, state_path in tqdm(available_ckpts):
        model = TweetModel().to(device)
        state = torch.load(state_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        models.append(model)

    for data in tqdm(test_loader):
        ids = data["ids"].to(device)
        masks = data["masks"].to(device)
        tweet = data["tweet"]
        offsets = data["offsets"].cpu().numpy()

        start_logits_list = []
        end_logits_list = []
        for model in models:
            with torch.no_grad():
                out_start, out_end = model(ids, masks)
                start_logits_list.append(torch.softmax(out_start, dim=1).cpu().numpy())
                end_logits_list.append(torch.softmax(out_end, dim=1).cpu().numpy())

        start_logits = np.mean(start_logits_list, axis=0)
        end_logits = np.mean(end_logits_list, axis=0)

        for i in range(ids.size(0)):
            start_pred = int(np.argmax(start_logits[i]))
            end_pred = int(np.argmax(end_logits[i]))
            if start_pred > end_pred:
                pred = tweet[i]
            else:
                pred = get_selected_text(tweet[i], start_pred, end_pred, offsets[i])
            predictions.append(pred)
else:
    if can_run_model:
        print(f"No fold checkpoints found under {outdir}. Using heuristic predictions.")
    print("Generating heuristic predictions...")
    for _, row in test_df.iterrows():
        predictions.append(heuristic_select(row["text"], row["sentiment"]))



## === cell 6
sub_df = pd.read_csv(submission_template)

if len(predictions) != len(sub_df):
    raise ValueError(
        f"Prediction length {len(predictions)} != submission rows {len(sub_df)}"
    )

sub_df["selected_text"] = predictions
sub_df["selected_text"] = sub_df["selected_text"].astype(str)
sub_df["selected_text"] = sub_df["selected_text"].apply(
    lambda x: x.replace("!!!!", "!") if len(x.split()) == 1 else x
)
sub_df["selected_text"] = sub_df["selected_text"].apply(
    lambda x: x.replace("..", ".") if len(x.split()) == 1 else x
)
sub_df["selected_text"] = sub_df["selected_text"].apply(
    lambda x: x.replace("...", ".") if len(x.split()) == 1 else x
)

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print(f"Wrote {sub_path} with shape {sub_df.shape}")
print(sub_df.head(10))
