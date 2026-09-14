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
Predict the score of student essays.

## Metric
Quadratic weighted kappa.

## Submission Format
For each `essay_id` in the test set, you must predict the corresponding `score` (between 1-6, see [rubric](https://storage.googleapis.com/kaggle-forum-message-attachments/2733927/20538/Rubric_%20Holistic%20Essay%20Scoring.pdf) for more details). The file should contain a header and have the following format:

```
essay_id,score
000d118,3
000fe60,3
001ab80,4
...
```

## Dataset
- **train.csv** - Essays and scores to be used as training data.
    - `essay_id` - The unique ID of the essay
    - `full_text` - The full essay response
    - `score` - Holistic score of the essay on a 1-6 scale
- **test.csv** - The essays to be used as test data. Contains the same fields as `train.csv`, aside from exclusion of `score`.
- **sample_submission.csv** - A submission file in the correct format.
    - `essay_id` - The unique ID of the essay
    - `score` - The predicted holistic score of the essay on a 1-6 scale

# 2. Python version

3.12

# 3. Installed packages

datasets==4.4.1
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
tensorflow-datasets==4.9.9
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
vega-datasets==0.9.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        input/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        working/
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
```

-> data/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/learning-agency-lab-automated-essay-scoring-2/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/learning-agency-lab-automated-essay-scoring-2/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> data/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> (stopped after 10 files for performance)

# 5. Target score

0.7811435150492567

# 6. Current score

0.00197

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.00197) has done: 'I fix the hard failures by (1) removing the invalid local model path usage and consistently loading `bert-base-uncased` from the Hugging Face cache (offline-safe), (2) avoiding the `datasets`-protobuf crash by not using `load_dataset` at all (reading CSVs with pandas instead), and (3) producing predictions in the correct 1–6 integer range with a proper `submission.csv`. To keep the core logic intact, I preserve your BERT sequence-classification approach but ensure `num_labels=6` and map labels 1–6 ↔ 0–5 correctly. I also replace the `pipeline`-based inference (slow and sometimes label-string brittle) with direct batched model inference for speed and stability within the 600s limit. The PCA visualization cells are kept but made non-blocking and safe so they won’t crash the run.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))



## === cell 1
train_file = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
test_file = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
sample_file = (
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
)

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
sample_df = pd.read_csv(sample_file)

train_df.head()



## === cell 2
len(train_df), len(test_df), sample_df.shape



## === cell 3
train_df.isnull().sum()



## === cell 4
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer



## === cell 5
MODEL_NAME = "bert-base-uncased"

torch.manual_seed(42)
np.random.seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

id2label = {i: str(i + 1) for i in range(6)}
label2id = {str(i + 1): i for i in range(6)}

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=6,
    id2label=id2label,
    label2id=label2id,
)
model.eval()
model.to(device)

device



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 6
pca = PCA(n_components=3)
all_embeddings = []
scores = []



## === cell 7
batch_size = 64
df_filtered = train_df[train_df["score"].isin([1, 6])].copy()

if len(df_filtered) > 0:
    import tqdm

    for i in tqdm.auto.tqdm(range(0, len(df_filtered), batch_size)):
        batch_df = df_filtered.iloc[i : i + batch_size]
        batch_texts = batch_df["full_text"].tolist()
        batch_scores = batch_df["score"].tolist()

        encoded_input = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

        with torch.no_grad():
            outputs = model(**encoded_input, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1]  # (B, T, H)
            embeddings = last_hidden_states.mean(dim=1).detach().cpu().numpy()

        all_embeddings.append(embeddings)
        scores.extend(batch_scores)

    all_embeddings = np.vstack(all_embeddings)
    principal_components = pca.fit_transform(all_embeddings)
else:
    all_embeddings = np.zeros((0, 768), dtype=np.float32)
    principal_components = np.zeros((0, 3), dtype=np.float32)

principal_components.shape



## === cell 8
if principal_components.shape[0] > 0:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        principal_components[:, 0],
        principal_components[:, 1],
        principal_components[:, 2],
        c=scores,
        cmap="viridis",
        alpha=0.7,
    )
    ax.set_title("3D PCA Projection of BERT Embeddings with Scores (scores 1 vs 6)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    fig.colorbar(scatter, label="Score")
    plt.show()



## === cell 9
import gc

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()



## === cell 10
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class EssayDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512, with_labels=True):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_labels = with_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.loc[idx, "full_text"]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.with_labels:
            item["labels"] = torch.tensor(
                int(self.df.loc[idx, "score"]) - 1, dtype=torch.long
            )
        return item




## === cell 11
is_submission = True

output_dir = "/kaggle/working/results"
os.makedirs(output_dir, exist_ok=True)

train_tr, train_va = train_test_split(
    train_df,
    train_size=0.8,
    random_state=42,
    stratify=train_df["score"],
)

train_ds = EssayDataset(train_tr, tokenizer, max_length=256, with_labels=True)
valid_ds = EssayDataset(train_va, tokenizer, max_length=256, with_labels=True)

train_loader = DataLoader(
    train_ds,
    batch_size=16,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
valid_loader = DataLoader(
    valid_ds,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

len(train_ds), len(valid_ds)



## === cell 12
if not is_submission:
    from torch.optim import AdamW

    model.train()
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    num_epochs = 3
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / max(1, len(train_loader))
        print(f"epoch {epoch+1}/{num_epochs} - train_loss: {avg_loss:.4f}")

    model.save_pretrained(os.path.join(output_dir, "bert-base-uncased"))
    tokenizer.save_pretrained(os.path.join(output_dir, "bert-base-uncased"))
    model.eval()



## === cell 13
test_ds = EssayDataset(
    test_df.rename(columns={"full_text": "full_text"}),
    tokenizer,
    max_length=256,
    with_labels=False,
)
test_loader = DataLoader(
    test_ds,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

model.eval()
all_preds = []

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits  # (B,6)
        preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()  # 0..5
        all_preds.append(preds)

all_preds = np.concatenate(all_preds, axis=0)
all_preds.shape, len(test_df)



## === cell 14
pred_scores = (all_preds + 1).astype(int)
pred_scores.min(), pred_scores.max()



## === cell 15
sub = sample_df.copy()
pred_map = dict(zip(test_df["essay_id"].values, pred_scores))
sub["score"] = sub["essay_id"].map(pred_map).astype(int)

sub["score"] = sub["score"].fillna(3).astype(int)

sub.head(), sub.shape



## === cell 16
submission_path = "/kaggle/working/submission.csv"
sub.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission_path, "rows:", len(sub), "cols:", sub.columns.tolist())
