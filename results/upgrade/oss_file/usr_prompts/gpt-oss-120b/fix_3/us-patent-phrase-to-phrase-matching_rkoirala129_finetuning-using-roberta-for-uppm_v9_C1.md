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

# 5. Target score

0.8100979449941955

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import math, torch, torch.nn as nn, pandas as pd, numpy as np
from scipy import stats
from tqdm import tqdm
from sklearn.model_selection import KFold
from transformers import (
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    AdamW,  # corrected import
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_55/3750780094.py in <cell line: 0>()
      8 from tqdm import tqdm
      9 from sklearn.model_selection import KFold
---> 10 from transformers import (
     11     AutoModel,
     12     AutoConfig,

ImportError: cannot import name 'AdamW' from 'transformers' (/usr/local/lib/python3.11/dist-packages/transformers/__init__.py)

## === cell 1
model_name = (
    "sentence-transformers/all-MiniLM-L6-v2"  # public model that can be loaded locally
)
max_len = 32
batch_size = 32
num_epochs = 2
learning_rate = 2e-5
weight_decay = 0.01
adam_epsilon = 1e-8
warmup_ratio = 0.06
gradient_accumulation_steps = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 2
class PhraseDataset(torch.utils.data.Dataset):
    def __init__(self, anchor, target, context, scores, tokenizer, max_len):
        self.anchor = anchor
        self.target = target
        self.context = context
        self.scores = scores  # may be None for test set
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.anchor)

    def __getitem__(self, idx):
        enc = self.tokenizer.encode_plus(
            self.context[idx] + " " + self.anchor[idx],
            self.target[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )
        item = {
            "ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
        }
        if self.scores is not None:
            item["score"] = torch.tensor(self.scores[idx], dtype=torch.float)
        return item




## === cell 3
class SimilarPhraseModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        config.update({"output_hidden_states": True, "add_pooling_layer": True})
        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out = nn.Linear(config.hidden_size, 1)

    def monitor_metrics(self, preds, targets):
        preds_np = preds.detach().cpu().numpy().ravel()
        t_np = targets.detach().cpu().numpy().ravel()
        corr = stats.pearsonr(preds_np, t_np)[0]
        return {"pearsonr": torch.tensor(corr, device="cpu")}

    def forward(self, ids, mask, score=None):
        tr_out = self.transformer(input_ids=ids, attention_mask=mask)
        pooled = tr_out.pooler_output
        x = self.dropout(pooled)
        out = self.out(x).squeeze(-1)
        loss = None
        metrics = {}
        if score is not None:
            loss = nn.MSELoss()(out, score)
            metrics = self.monitor_metrics(out, score)
        return out, loss, metrics




## === cell 4
train_path = "../input/us-patent-phrase-to-phrase-matching/train.csv"
test_path = "../input/us-patent-phrase-to-phrase-matching/test.csv"
sample_sub_path = "../input/us-patent-phrase-to-phrase-matching/sample_submission.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

context_map = {
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
df_train["context"] = df_train["context"].apply(lambda x: context_map[x[0]])
df_test["context"] = df_test["context"].apply(lambda x: context_map[x[0]])

tokenizer = AutoTokenizer.from_pretrained(model_name)

best_pearson = -1.0
best_state_dict_path = "my_best_model.pt"  # explicit path

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(df_train)):
    print(f"\n=== Fold {fold + 1} ===")
    tr_df = df_train.iloc[train_idx].reset_index(drop=True)
    val_df = df_train.iloc[val_idx].reset_index(drop=True)

    train_ds = PhraseDataset(
        anchor=tr_df["anchor"].values,
        target=tr_df["target"].values,
        context=tr_df["context"].values,
        scores=tr_df["score"].values,
        tokenizer=tokenizer,
        max_len=max_len,
    )
    val_ds = PhraseDataset(
        anchor=val_df["anchor"].values,
        target=val_df["target"].values,
        context=val_df["context"].values,
        scores=val_df["score"].values,
        tokenizer=tokenizer,
        max_len=max_len,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False
    )

    model = SimilarPhraseModel(model_name).to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)

    t_total = len(train_loader) // gradient_accumulation_steps * num_epochs
    warmup_steps = math.ceil(t_total * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Fold{fold+1} Epoch{epoch+1} Train"):
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            scores = batch["score"].to(device)

            _, loss, _ = model(ids, mask, scores)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
        print(
            f"Fold {fold+1} Epoch {epoch+1} train loss: {epoch_loss/len(train_loader):.4f}"
        )

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Fold{fold+1} Epoch{epoch+1} Val"):
                ids = batch["ids"].to(device)
                mask = batch["mask"].to(device)
                scores = batch["score"].to(device)
                preds, _, _ = model(ids, mask, scores)
                all_preds.append(preds.detach().cpu())
                all_labels.append(scores.detach().cpu())
        preds_cat = torch.cat(all_preds)
        labels_cat = torch.cat(all_labels)
        pearson = stats.pearsonr(preds_cat.numpy().ravel(), labels_cat.numpy().ravel())[
            0
        ]
        print(f"Fold {fold+1} Epoch {epoch+1} validation Pearson: {pearson:.4f}")

        if pearson > best_pearson:
            best_pearson = pearson
            torch.save(model.state_dict(), best_state_dict_path)
            print(f"--> New best model saved (Pearson {best_pearson:.4f})")

print(f"\nBest validation Pearson across folds: {best_pearson:.4f}")



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 5
test_ds = PhraseDataset(
    anchor=df_test["anchor"].values,
    target=df_test["target"].values,
    context=df_test["context"].values,
    scores=None,
    tokenizer=tokenizer,
    max_len=max_len,
)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)

model = SimilarPhraseModel(model_name).to(device)
model.load_state_dict(torch.load(best_state_dict_path))
model.eval()

preds = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predict"):
        ids = batch["ids"].to(device)
        mask = batch["mask"].to(device)
        out, _, _ = model(ids, mask)
        preds.append(out.cpu().numpy())

preds = np.array(preds).ravel()
preds = np.clip(preds, 0.0, 1.0)

submission = pd.read_csv(sample_sub_path)
submission["score"] = preds
submission.to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' written successfully.")

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2879717450.py in <cell line: 0>()
     10 
     11 model = SimilarPhraseModel(model_name).to(device)
---> 12 model.load_state_dict(torch.load(best_state_dict_path))
     13 model.eval()
     14 

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in load(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)
   1423         pickle_load_args["encoding"] = "utf-8"
   1424 
-> 1425     with _open_file_like(f, "rb") as opened_file:
   1426         if _is_zipfile(opened_file):
   1427             # The zipfile reader is going to advance the current file position.

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _open_file_like(name_or_buffer, mode)
    749 def _open_file_like(name_or_buffer, mode):
    750     if _is_path(name_or_buffer):
--> 751         return _open_file(name_or_buffer, mode)
    752     else:
    753         if "w" in mode:

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in __init__(self, name, mode)
    730 class _open_file(_opener):
    731     def __init__(self, name, mode):
--> 732         super().__init__(open(name, mode))
    733 
    734     def __exit__(self, *args):

FileNotFoundError: [Errno 2] No such file or directory: 'my_best_model.pt'
