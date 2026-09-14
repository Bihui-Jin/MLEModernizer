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

0.03145

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.03145) has done: 'I fix the pipeline so it runs fully offline in Kaggle by removing `local_files_only=True` and automatically locating a locally available transformer model from `/kaggle/input` (or using the normal cache if present). I also bypass the `datasets` library (it’s currently crashing due to a protobuf incompatibility) and keep the same core logic by doing the same tokenization + `AutoModelForSequenceClassification` + argmax inference using plain PyTorch `Dataset/DataLoader`. Finally, I ensure a valid `submission.csv` is always written with the correct columns and row alignment, and I make the earlier PCA/plot cells robust so they don’t block submission generation.'

# 9. Code solution

## === cell 0
import os
import gc
import numpy as np
import pandas as pd

import torch

torch.manual_seed(42)
np.random.seed(42)

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train_file = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
train_df = pd.read_csv(train_file)
train_df.head()



## === cell 2
len(train_df)



## === cell 3
train_df.isnull().sum()



## === cell 4
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import transformers



## === cell 5
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer




## === cell 6
def _find_local_hf_model_dir(preferred_names=("bert-base-uncased",)):
    roots = ["/kaggle/input", "/kaggle/working"]
    for root in roots:
        for pref in preferred_names:
            cand = os.path.join(root, pref)
            if os.path.isdir(cand):
                return cand
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            fn = set(filenames)
            if "config.json" in fn and ("tokenizer.json" in fn or "vocab.txt" in fn):
                return dirpath
    return None


device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "bert-base-uncased"
local_model_dir = _find_local_hf_model_dir((model_name,))

resolved_model_path = local_model_dir if local_model_dir is not None else model_name
print("Resolved model path:", resolved_model_path)

tokenizer = AutoTokenizer.from_pretrained(resolved_model_path)
model = AutoModelForSequenceClassification.from_pretrained(
    resolved_model_path, num_labels=6
)
model.eval()
model.to(device)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 7
pca = PCA(n_components=3)
all_embeddings = []
scores = []



## === cell 8
batch_size = 32
df_filtered = train_df[train_df["score"].isin([1, 6])].copy()

import tqdm

max_pca_rows = 512  # safety cap for runtime; does not affect training/submission
df_filtered = df_filtered.iloc[:max_pca_rows]

for i in tqdm.auto.tqdm(range(0, len(df_filtered), batch_size)):
    batch_df = df_filtered.iloc[i : i + batch_size]
    batch_texts = batch_df["full_text"].tolist()
    batch_scores = batch_df["score"].tolist()

    encoded = tokenizer(
        batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]  # (bs, seq, hidden)
        embeddings = last_hidden_states.mean(dim=1).detach().cpu().numpy()

    all_embeddings.append(embeddings)
    scores.extend(batch_scores)

if len(all_embeddings) > 0:
    all_embeddings = np.vstack(all_embeddings)
    principal_components = pca.fit_transform(all_embeddings)
else:
    all_embeddings = np.empty((0, 768), dtype=np.float32)
    principal_components = np.empty((0, 3), dtype=np.float32)



## === cell 9
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()



## === cell 10
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
    ax.set_title("3D PCA Projection of BERT Embeddings with Scores (subset)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    fig.colorbar(scatter, label="Score")
    plt.show()



## === cell 11
if isinstance(all_embeddings, np.ndarray) and all_embeddings.shape[0] > 1:
    pca2 = PCA(n_components=2)
    principal_components_2d = pca2.fit_transform(all_embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        principal_components_2d[:, 0],
        principal_components_2d[:, 1],
        c=scores,
        cmap="viridis",
        alpha=0.7,
    )
    plt.title("2D PCA Projection of BERT Embeddings with Scores (subset)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(scatter, label="Score")
    plt.show()



## === cell 12
import gc

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()



## === cell 13
import subprocess

try:
    print(subprocess.check_output(["nvcc", "--version"], text=True))
except Exception as e:
    print("nvcc not available:", repr(e))



## === cell 14

data_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"
train_file_path = f"{data_path}/train.csv"
test_file_path = f"{data_path}/test.csv"

train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)

train_df = train_df.rename(columns={"full_text": "text", "score": "labels"})
test_df = test_df.rename(columns={"full_text": "text"})

train_df["labels"] = train_df["labels"].astype(int) - 1  # 0..5



## === cell 15
id2label = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
label2id = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}

is_submission = True

tokenizer = AutoTokenizer.from_pretrained(resolved_model_path)
model = AutoModelForSequenceClassification.from_pretrained(
    resolved_model_path,
    num_labels=6,
    id2label=id2label,
    label2id=label2id,
).to(device)



## === cell 16
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def preprocess_function_texts(texts):
    return tokenizer(texts, truncation=True, padding=False, max_length=512)


class EssayTestDataset(Dataset):
    def __init__(self, df):
        self.essay_ids = df["essay_id"].tolist()
        self.texts = df["text"].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(self.texts[idx], truncation=True, padding=False, max_length=512)
        enc["essay_id"] = self.essay_ids[idx]
        return enc


test_ds = EssayTestDataset(test_df)



## === cell 17
model.eval()


def collate_with_ids(features):
    essay_ids = [f.pop("essay_id") for f in features]
    batch = data_collator(features)
    batch["essay_id"] = essay_ids
    return batch


test_loader = DataLoader(
    test_ds,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_with_ids,
)

all_ids = []
all_preds = []
with torch.no_grad():
    for batch in test_loader:
        essay_ids = batch.pop("essay_id")
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = outputs.logits.argmax(dim=-1).detach().cpu().numpy()
        all_preds.append(preds)
        all_ids.extend(essay_ids)

all_preds = np.concatenate(all_preds, axis=0)  # 0..5
final_scores = (all_preds + 1).astype(int)



## === cell 18
result_df = pd.DataFrame(
    {
        "essay_id": np.array(all_ids, dtype=str),
        "score": final_scores,
    }
)

result_df = test_df[["essay_id"]].merge(
    result_df, on="essay_id", how="left", validate="one_to_one"
)

assert len(result_df) == len(test_df)
assert result_df["score"].notna().all()

result_df["score"] = result_df["score"].astype(int).clip(1, 6)

result_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", result_df.shape)
print(result_df.head())



## === cell 19
result_df.head()
