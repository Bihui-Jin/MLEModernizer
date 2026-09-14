# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given pairs of phrases (an `anchor` and a `target` phrase), build a model to rate how similar they are.  

## Metric
Pearson correlation coefficient.

## Submission Format
For each `id` (representing a pair of phrases) in the test set, you must predict the similarity `score`. The file should contain a header and have the following format:

```
id,score
4112d61851461f60,0
09e418c93a776564,0.25
36baf228038e314b,1
etc.

```

## Dataset
The scores are in the 0-1 range with increments of 0.25 with the following meanings:

- **1.0** - Very close match. This is typically an exact match except possibly for differences in conjugation, quantity (e.g. singular vs. plural), and addition or removal of stopwords (e.g. "the", "and", "or").
- **0.75** - Close synonym, e.g. "mobile phone" vs. "cellphone". This also includes abbreviations, e.g. "TCP" -> "transmission control protocol".
- **0.5** - Synonyms which don't have the same meaning (same function, same properties). This includes broad-narrow (hyponym) and narrow-broad (hypernym) matches.
- **0.25** - Somewhat related, e.g. the two phrases are in the same high level domain but are not synonyms. This also includes antonyms.
- **0.0** - Unrelated.

Files
-----

- **train.csv** - the training set, containing phrases, contexts, and their similarity scores
- **test.csv** - the test set set, identical in structure to the training set but without the score
- **sample_submission.csv** - a sample submission file in the correct format

Columns
-------

- `id` - a unique identifier for a pair of phrases
- `anchor` - the first phrase
- `target` - the second phrase
- `context` - the [CPC classification (version 2021.05)](https://en.wikipedia.org/wiki/Cooperative_Patent_Classification), which indicates the subject within which the similarity is to be scored
- `score` - the similarity. This is sourced from a combination of one or more manual expert ratings.

# 2. Python version

3.10

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
scipy==1.15.3
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
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        input/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        working/
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
```

-> data/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> data/us-patent-phrase-to-phrase-matching/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/us-patent-phrase-to-phrase-matching/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/us-patent-phrase-to-phrase-matching/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> input/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import math
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from transformers import (
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

DATA_DIR_CANDIDATES = [
    "/kaggle/input/us-patent-phrase-to-phrase-matching",
    "/kaggle/data/us-patent-phrase-to-phrase-matching",
    "/kaggle/input",
    "/kaggle/data",
]


def first_existing_file(relpath: str):
    for base in DATA_DIR_CANDIDATES:
        p = os.path.join(base, relpath)
        if os.path.exists(p):
            return p
    return None


train_path = first_existing_file("train.csv") or first_existing_file(
    "us-patent-phrase-to-phrase-matching/train.csv"
)
test_path = first_existing_file("test.csv") or first_existing_file(
    "us-patent-phrase-to-phrase-matching/test.csv"
)
sample_path = first_existing_file("sample_submission.csv") or first_existing_file(
    "us-patent-phrase-to-phrase-matching/sample_submission.csv"
)

assert (
    train_path is not None and test_path is not None and sample_path is not None
), "Could not locate competition CSVs."


def find_local_hf_model_dir(preferred: str | None = None):
    cands = []
    if preferred is not None:
        cands.append(preferred)
    cands += [
        "/kaggle/input/roberta-pre/patent_pretrained",
        "../input/roberta-pre/patent_pretrained",
        "/kaggle/input/roberta-base",
        "/kaggle/input/roberta-large",
    ]
    if os.path.isdir("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "config.json" in files:
                cands.append(root)
    for p in cands:
        if p and os.path.isdir(p) and os.path.exists(os.path.join(p, "config.json")):
            return p
    return "roberta-base"


model_n = find_local_hf_model_dir("../input/roberta-pre/patent_pretrained")

max_len = 32
batch_size = 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

warmup_ratio = 0.06
weight_decay = 0.01
gradient_accumulation_steps = 1
num_train_epochs = 2
learning_rate = 2e-5
adam_epsilon = 1e-8

context_mapping = {
    "A": "Human Necessities",
    "B": "Operations and Transport",
    "C": "Chemistry and Metallurgy",
    "D": "Textiles",
    "E": "Fixed Constructions",
    "F": "Mechanical Engineering",
    "G": "Physics",
    "H": "Electricity",
    "Y": "Emerging Cross-Sectional Technologies",
}

_NUM_WORKERS = 0
_PIN_MEMORY = torch.cuda.is_available()

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

torch.set_num_threads(max(1, (os.cpu_count() or 1) // 2))


def _seed_worker(worker_id: int):
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def pearsonr_np(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum()) * np.sqrt((y * y).sum())
    if denom == 0.0:
        return 0.0
    return float((x * y).sum() / denom)




## === cell 1
class PhraseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_ids_t: torch.Tensor,
        attention_mask_t: torch.Tensor,
        score_t: torch.Tensor,
    ):
        self.input_ids = input_ids_t
        self.attention_mask = attention_mask_t
        self.score = score_t

    def __len__(self):
        return self.score.shape[0]

    def __getitem__(self, item):
        return {
            "ids": self.input_ids[item],
            "mask": self.attention_mask[item],
            "score": self.score[item],
        }




## === cell 2
class SimilarPhraseModel(nn.Module):
    def __init__(self, model_name, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.model_name = model_name

        config = AutoConfig.from_pretrained(model_name)
        config.update(
            {
                "output_hidden_states": False,
                "add_pooling_layer": True,
                "num_labels": 1,
            }
        )

        try:
            self.transformer = AutoModel.from_pretrained(
                model_name, config=config, attn_implementation="sdpa"
            )
        except TypeError:
            self.transformer = AutoModel.from_pretrained(model_name, config=config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output = nn.Linear(config.hidden_size, 1)

    def forward(self, ids, mask, score):
        transformer_out = self.transformer(input_ids=ids, attention_mask=mask)
        output = transformer_out.pooler_output
        output = self.dropout(output)
        output = self.output(output)
        loss = nn.MSELoss()(
            output.squeeze(-1), score.squeeze(-1) if score.ndim > 0 else score
        )
        metrics = {}
        return output, loss, metrics




## === cell 3
from torch.utils.data import DataLoader

df = pd.read_csv(train_path)
df["context"] = df["context"].apply(lambda x: context_mapping[str(x)[0]])


def make_folds(df_in: pd.DataFrame, n_splits: int = 10) -> pd.DataFrame:
    df_out = df_in.copy()
    fold_idx = df_out["id"].apply(
        lambda s: (
            int(str(s), 16)
            if all(c in "0123456789abcdef" for c in str(s).lower())
            else abs(hash(str(s)))
        )
        % n_splits
    )
    df_out["kfold"] = fold_idx.astype(int)
    return df_out


df = make_folds(df, n_splits=10)

tokenizer = AutoTokenizer.from_pretrained(model_n)


def batch_tokenize(context_arr, anchor_arr, target_arr, tokenizer, max_len: int):
    text1 = [f"{c} {a}" for c, a in zip(context_arr, anchor_arr)]
    text2 = list(target_arr)
    enc = tokenizer(
        text1,
        text2,
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors=None,
    )
    return np.asarray(enc["input_ids"], dtype=np.int64), np.asarray(
        enc["attention_mask"], dtype=np.int64
    )


all_ids_np, all_mask_np = batch_tokenize(
    df.context.values, df.anchor.values, df.target.values, tokenizer, max_len
)

all_ids_t = torch.from_numpy(np.ascontiguousarray(all_ids_np)).long()
all_mask_t = torch.from_numpy(np.ascontiguousarray(all_mask_np)).long()
all_score_t = (
    torch.from_numpy(df.score.values.astype(np.float32)).unsqueeze(-1).contiguous()
)

best_overall_pearson = -1.0
best_model_path = "my_best_model.pt"


def to_device(batch):
    batch["ids"] = batch["ids"].to(device, non_blocking=True)
    batch["mask"] = batch["mask"].to(device, non_blocking=True)
    batch["score"] = batch["score"].to(device, non_blocking=True)
    return batch


def fast_collate(batch_list):
    ids = torch.stack([b["ids"] for b in batch_list], dim=0)
    mask = torch.stack([b["mask"] for b in batch_list], dim=0)
    score = torch.stack([b["score"] for b in batch_list], dim=0)
    return {"ids": ids, "mask": mask, "score": score}


g = torch.Generator()
g.manual_seed(42)

last_model_state = None

_base_model = SimilarPhraseModel(model_name=model_n, learning_rate=0.001)
_base_state = {k: v.detach().cpu().clone() for k, v in _base_model.state_dict().items()}
del _base_model

_no_decay = ("bias", "LayerNorm.weight")

for fold_ in range(10):
    train_idx = df["kfold"].values != fold_
    valid_idx = ~train_idx

    train_dataset = PhraseDataset(
        input_ids_t=all_ids_t[train_idx],
        attention_mask_t=all_mask_t[train_idx],
        score_t=all_score_t[train_idx],
    )
    valid_dataset = PhraseDataset(
        input_ids_t=all_ids_t[valid_idx],
        attention_mask_t=all_mask_t[valid_idx],
        score_t=all_score_t[valid_idx],
    )

    model = SimilarPhraseModel(model_name=model_n, learning_rate=0.001)
    model.load_state_dict(_base_state, strict=True)
    model.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=_NUM_WORKERS,
        pin_memory=_PIN_MEMORY,
        persistent_workers=False,
        collate_fn=fast_collate,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=_NUM_WORKERS,
        pin_memory=_PIN_MEMORY,
        persistent_workers=False,
        collate_fn=fast_collate,
    )

    t_total = len(train_loader) // gradient_accumulation_steps * num_train_epochs

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in _no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in _no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    warmup_steps = math.ceil(t_total * warmup_ratio)
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    for epoch in range(num_train_epochs):
        model.train()

        for batch in train_loader:
            batch = to_device(batch)
            input_ids = batch["ids"]
            attention_mask = batch["mask"]
            labels = batch["score"]

            output, loss, _ = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        model.eval()

        n_valid = len(valid_dataset)
        valid_pred_t = torch.empty((n_valid,), dtype=torch.float32, device="cpu")
        valid_tgt_t = torch.empty((n_valid,), dtype=torch.float32, device="cpu")
        valid_pos = 0

        with torch.inference_mode():
            for batch in valid_loader:
                batch = to_device(batch)
                input_ids = batch["ids"]
                attention_mask = batch["mask"]
                labels = batch["score"]
                out, _, _ = model(input_ids, attention_mask, labels)

                out_cpu = out.detach().float().cpu().view(-1)
                y_cpu = labels.detach().float().cpu().view(-1)
                bs = out_cpu.numel()
                valid_pred_t[valid_pos : valid_pos + bs].copy_(out_cpu)
                valid_tgt_t[valid_pos : valid_pos + bs].copy_(y_cpu)
                valid_pos += bs

        va_pearson = pearsonr_np(
            valid_pred_t[:valid_pos].numpy(), valid_tgt_t[:valid_pos].numpy()
        )
        print(
            f"Fold {fold_} | Pearson correlation for epoch {epoch} during validation is {{'pearsonr': tensor({va_pearson})}}"
        )

        current_pearson = float(va_pearson)
        last_model_state = model.state_dict()
        if current_pearson > best_overall_pearson:
            torch.save(last_model_state, best_model_path)
            best_overall_pearson = current_pearson
            print(f"New best model saved: pearson={best_overall_pearson:.6f}")

    if not os.path.exists(best_model_path) and last_model_state is not None:
        torch.save(last_model_state, best_model_path)

if not os.path.exists(best_model_path):
    if last_model_state is not None:
        torch.save(last_model_state, best_model_path)
    else:
        raise RuntimeError("Training did not produce any model state to save.")

print(f"Best validation pearson observed: {best_overall_pearson:.6f}")



## === cell 4
df_test = pd.read_csv(test_path)
df_test["context"] = df_test["context"].apply(lambda x: context_mapping[str(x)[0]])


class PhraseTestDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids_t: torch.Tensor, attention_mask_t: torch.Tensor):
        self.input_ids = input_ids_t
        self.attention_mask = attention_mask_t

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, item):
        return {
            "ids": self.input_ids[item],
            "mask": self.attention_mask[item],
        }


class PhraseModelTest(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        config.update(
            {
                "output_hidden_states": False,
                "add_pooling_layer": True,
                "num_labels": 1,
            }
        )
        try:
            self.transformer = AutoModel.from_pretrained(
                model_name, config=config, attn_implementation="sdpa"
            )
        except TypeError:
            self.transformer = AutoModel.from_pretrained(model_name, config=config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output = nn.Linear(config.hidden_size, 1)

    def forward(self, ids, mask):
        transformer_out = self.transformer(input_ids=ids, attention_mask=mask)
        output = transformer_out.pooler_output
        output = self.dropout(output)
        output = self.output(output)
        return output, 0, {}


te_ids, te_mask = batch_tokenize(
    df_test.context.values,
    df_test.anchor.values,
    df_test.target.values,
    tokenizer,
    max_len,
)

te_ids_t = torch.from_numpy(np.ascontiguousarray(te_ids)).long()
te_mask_t = torch.from_numpy(np.ascontiguousarray(te_mask)).long()

test_dataset = PhraseTestDataset(input_ids_t=te_ids_t, attention_mask_t=te_mask_t)


def fast_collate_test(batch_list):
    ids = torch.stack([b["ids"] for b in batch_list], dim=0)
    mask = torch.stack([b["mask"] for b in batch_list], dim=0)
    return {"ids": ids, "mask": mask}


test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=_NUM_WORKERS,
    pin_memory=_PIN_MEMORY,
    persistent_workers=False,
    collate_fn=fast_collate_test,
)


def predict(model_path: str):
    model_test = PhraseModelTest(model_n)
    state = torch.load(model_path, map_location="cpu")
    model_test.load_state_dict(state)
    model_test.to(device)
    model_test.eval()

    n = len(test_dataset)
    pred_t = torch.empty((n,), dtype=torch.float32, device="cpu")
    pos = 0

    with torch.inference_mode():
        for batch in test_loader:
            batch["ids"] = batch["ids"].to(device, non_blocking=True)
            batch["mask"] = batch["mask"].to(device, non_blocking=True)
            out, _, _ = model_test(batch["ids"], batch["mask"])
            out_cpu = out.detach().float().cpu().view(-1)
            bs = out_cpu.numel()
            pred_t[pos : pos + bs].copy_(out_cpu)
            pos += bs

    return pred_t[:pos].numpy()


final_pred = predict(best_model_path)

submission = pd.read_csv(sample_path)
assert "id" in submission.columns and "score" in submission.columns
assert len(submission) == len(final_pred), "Prediction length mismatch with submission."

submission["score"] = np.clip(final_pred, 0.0, 1.0)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Saved to:", os.path.abspath("submission.csv"))
