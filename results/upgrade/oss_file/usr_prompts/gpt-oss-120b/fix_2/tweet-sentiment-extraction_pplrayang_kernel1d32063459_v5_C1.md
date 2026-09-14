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

0.7075039744377136

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, random, warnings, re, string, sys
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from transformers import RobertaModel, RobertaConfig, RobertaTokenizerFast
from tqdm.notebook import tqdm

warnings.filterwarnings("ignore")


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed = 42
seed_everything(seed)

batch_size = 32
N = 10
skf = StratifiedKFold(n_splits=N, shuffle=True, random_state=seed)
NUM_WORKERS = 2

ROBERTA_PATH = "/kaggle/input/robertamodel0524/"
MODEL_CONFIG_PATH = os.path.join(ROBERTA_PATH, "roberta-base-config.json")
MODEL_PATH = os.path.join(ROBERTA_PATH, "roberta-base-pytorch_model.bin")
MODEL_VOCAB_PATH = os.path.join(ROBERTA_PATH, "roberta-base-vocab.json")
MODEL_VOCAB_MERGES_PATH = os.path.join(ROBERTA_PATH, "roberta-base-merges.txt")
outdir = "/kaggle/input/roberta714kernel/"

test_file = "/kaggle/input/tweet-sentiment-extraction/test.csv"
submission_template = "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"

MAX_LEN = 96
LINEAR_DROPOUT = 0.2

CLS_TOK = 0
PAD_TOK = 1
SEP_TOK = 2




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
        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            ROBERTA_PATH,
            add_prefix_space=True,
            max_len=self.max_len,
            vocab_file=MODEL_VOCAB_PATH,
            merges_file=MODEL_VOCAB_MERGES_PATH,
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ids, masks, tweet, offsets = self._prepare_input(row)
        item = {"ids": ids, "masks": masks, "tweet": tweet, "offsets": offsets}
        if self.labeled:
            start_idx, end_idx = self._get_target_idx(row, tweet, offsets)
            item["start_idx"] = start_idx
            item["end_idx"] = end_idx
        return item

    def _prepare_input(self, row):
        tweet = " " + " ".join(row.text.lower().split())
        enc = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=False,
            max_length=self.max_len - 4,  # reserve space for CLS, SEP, sentiment tokens
            truncation=True,
            return_offsets_mapping=True,
        )
        sentiment_ids = self.tokenizer.encode(row.sentiment, add_special_tokens=False)
        ids = (
            [CLS_TOK]
            + sentiment_ids
            + [SEP_TOK, SEP_TOK]
            + enc["input_ids"]
            + [SEP_TOK]
        )
        offsets = [(0, 0)] * 4 + enc["offset_mapping"] + [(0, 0)]

        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [PAD_TOK] * pad_len
            offsets += [(0, 0)] * pad_len

        ids_tensor = torch.tensor(ids, dtype=torch.long)
        masks_tensor = torch.where(
            ids_tensor != PAD_TOK,
            torch.tensor(1, dtype=torch.long),
            torch.tensor(0, dtype=torch.long),
        )
        offsets_tensor = torch.tensor(offsets, dtype=torch.long)

        return ids_tensor, masks_tensor, tweet, offsets_tensor

    def _get_target_idx(self, row, tweet, offsets):
        selected_text = " " + " ".join(row.selected_text.lower().split())
        len_st = len(selected_text) - 1
        idx0 = idx1 = None
        for ind in (i for i, ch in enumerate(tweet) if ch == selected_text[1]):
            if " " + tweet[ind : ind + len_st] == selected_text:
                idx0 = ind
                idx1 = ind + len_st - 1
                break

        char_targets = [0] * len(tweet)
        if idx0 is not None and idx1 is not None:
            for ct in range(idx0, idx1 + 1):
                char_targets[ct] = 1

        target_idx = []
        for j, (off1, off2) in enumerate(offsets):
            if sum(char_targets[off1:off2]) > 0:
                target_idx.append(j)

        start_idx = target_idx[0]
        end_idx = target_idx[-1]
        return start_idx, end_idx




## === cell 2
def get_test_loader(df, batch_size=batch_size):
    return torch.utils.data.DataLoader(
        TweetDataset(df),
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )




## === cell 3
class TweetModel(nn.Module):
    def __init__(self):
        super(TweetModel, self).__init__()
        config = RobertaConfig.from_pretrained(
            MODEL_CONFIG_PATH, output_hidden_states=True
        )
        self.roberta = RobertaModel.from_pretrained(MODEL_PATH, config=config)
        self.dropout = nn.Dropout(LINEAR_DROPOUT)
        self.fc = nn.Linear(config.hidden_size, 2)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0.0)

    def forward(self, input_ids, attention_mask):
        _, _, hs = self.roberta(input_ids, attention_mask)
        x = torch.stack([hs[-1], hs[-2], hs[-3]])  # last 3 hidden states
        x = torch.mean(x, dim=0)
        x = self.dropout(x)
        x = self.fc(x)
        start_logits, end_logits = x.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)




## === cell 4
def get_selected_text(text, start_idx, end_idx, offsets):
    selected = ""
    for ix in range(start_idx, end_idx + 1):
        selected += text[offsets[ix][0] : offsets[ix][1]]
        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
            selected += " "
    return selected


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))




## === cell 5
test_df = pd.read_csv(test_file)
test_df["text"] = test_df["text"].astype(str)
test_loader = get_test_loader(test_df)

predictions = []
models = []

print("Loading models...")
for fold in range(2, skf.n_splits):
    if fold == 3:
        model = TweetModel()
        model.cuda()
        model_path = f"{outdir}roberta_fold{fold+1}.pth"
        try:
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            print(f"Loaded {model_path}")
        except Exception as e:
            print(f"Could not load {model_path}: {e}. Using unfitted model.")
        model.eval()
        models.append(model)

if not models:
    model = TweetModel().cuda().eval()
    models.append(model)

for batch in tqdm(test_loader, desc="Predicting"):
    ids = batch["ids"].cuda()
    masks = batch["masks"].cuda()
    tweets = batch["tweet"]
    offsets = batch["offsets"].numpy()

    start_logits_acc = []
    end_logits_acc = []
    for mdl in models:
        with torch.no_grad():
            start_log, end_log = mdl(ids, masks)
            start_logits_acc.append(torch.softmax(start_log, dim=1).cpu().numpy())
            end_logits_acc.append(torch.softmax(end_log, dim=1).cpu().numpy())

    start_logits = np.mean(start_logits_acc, axis=0)
    end_logits = np.mean(end_logits_acc, axis=0)

    for i in range(len(ids)):
        start_pred = np.argmax(start_logits[i])
        end_pred = np.argmax(end_logits[i])
        if start_pred > end_pred:
            pred = tweets[i]
        else:
            pred = get_selected_text(tweets[i], start_pred, end_pred, offsets[i])
        predictions.append(pred)



## --- ERROR in cell 5, traceback:
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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/robertamodel0524/'. Use `repo_type` argument if needed.

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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/robertamodel0524/'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_55/1019787212.py in <cell line: 0>()
      2 test_df = pd.read_csv(test_file)
      3 test_df["text"] = test_df["text"].astype(str)
----> 4 test_loader = get_test_loader(test_df)
      5 
      6 predictions = []

/tmp/ipykernel_55/2671232600.py in get_test_loader(df, batch_size)
      1 def get_test_loader(df, batch_size=batch_size):
      2     return torch.utils.data.DataLoader(
----> 3         TweetDataset(df),
      4         batch_size=batch_size,
      5         shuffle=False,

/tmp/ipykernel_55/2828290579.py in __init__(self, df, max_len)
      5         self.labeled = "selected_text" in df.columns
      6         # Use the fast tokenizer that works with local vocab files
----> 7         self.tokenizer = RobertaTokenizerFast.from_pretrained(
      8             ROBERTA_PATH,
      9             add_prefix_space=True,

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in from_pretrained(cls, pretrained_model_name_or_path, cache_dir, force_download, local_files_only, token, revision, trust_remote_code, *init_inputs, **kwargs)
   1930                     except Exception:
   1931                         # For any other exception, we throw a generic error.
-> 1932                         raise OSError(
   1933                             f"Can't load tokenizer for '{pretrained_model_name_or_path}'. If you were trying to load it from "
   1934                             "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "

OSError: Can't load tokenizer for '/kaggle/input/robertamodel0524/'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '/kaggle/input/robertamodel0524/' is the correct path to a directory containing all relevant files for a RobertaTokenizerFast tokenizer.

## === cell 6
sub_df = pd.read_csv(submission_template)
sub_df["selected_text"] = predictions
sub_df["selected_text"] = sub_df["selected_text"].apply(
    lambda x: x.replace("!!!!", "!") if len(str(x).split()) == 1 else x
)
sub_df["selected_text"] = sub_df["selected_text"].apply(
    lambda x: x.replace("..", ".") if len(str(x).split()) == 1 else x
)
sub_df["selected_text"] = sub_df["selected_text"].apply(
    lambda x: x.replace("...", ".") if len(str(x).split()) == 1 else x
)
sub_df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2919149951.py in <cell line: 0>()
      1 sub_df = pd.read_csv(submission_template)
----> 2 sub_df["selected_text"] = predictions
      3 # Simple post‑processing to clean obvious punctuation repetitions
      4 sub_df["selected_text"] = sub_df["selected_text"].apply(
      5     lambda x: x.replace("!!!!", "!") if len(str(x).split()) == 1 else x

NameError: name 'predictions' is not defined
