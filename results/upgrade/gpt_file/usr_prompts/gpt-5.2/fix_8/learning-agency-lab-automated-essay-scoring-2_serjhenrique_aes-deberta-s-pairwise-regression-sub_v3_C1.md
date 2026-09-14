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

geopandas==0.14.4
joblib==1.5.2
lightning-utilities==0.15.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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

0.7718595915073396

# 6. Current score

-0.01069

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the failing `lightning` install cell and switch the import to `pytorch_lightning`, which is already available in your environment. I also fix the missing external dataset paths by computing DeBERTa embeddings directly from the bundled Kaggle DeBERTa checkpoint inside the competition input folder, so `train_data_emb`, `embedding_features`, and the model/tokenizer load all work offline. Next, I replace the missing `kmeans_task_clf.joblib` dependency with a small in-notebook KMeans fit on the computed train embeddings to keep the same “task clustering” semantics without changing the downstream pairwise/prediction architecture. Finally, I ensure the pipeline completes end-to-end and writes `submission.csv` with the required columns and 1–6 clipped integer scores.'
- What this solution (achieved 0.0) has done: 'I fix the pipeline-breaking Hugging Face load by resolving a valid offline model path from the competition input folder and, if the bundled DeBERTa checkpoint is not present, fall back to a standard DeBERTa-v3-small from the local HF cache (still offline). This unblock embedding computation so downstream KMeans task clustering and the pairwise dataset can be built without `NameError`s. I also make mean pooling numerically safe (avoid division-by-zero) and make the groupby aggregation reindex to all essay indices so the submission always has the correct length/order. These changes are execution/stability fixes; they should move the score upward from 0.0 (currently no valid run) toward the target by producing non-random, embedding-based predictions.'
- What this solution (achieved 0.0) has done: 'I fix the offline Hugging Face model loading failure by resolving a real local DeBERTa folder shipped inside `/kaggle/input` (and, if not present, falling back to any cached HF model without forcing `local_files_only=True` on a non-existent repo). This unblocks embedding computation so downstream KMeans task clustering, pair construction, and Lightning inference can run without `NameError`s. I also add a safe CPU fallback that produces deterministic bag-of-words style embeddings only if no transformer weights are available locally, ensuring the notebook always completes within the time limit and writes `submission.csv`. These changes preserve your core pipeline (embeddings → KMeans tasks → pairwise AESNet inference → aggregation → 1–6 clipping) while moving the score up from 0.0 (no valid run) toward the target by producing meaningful, non-random predictions.'
- What this solution (achieved -0.02067) has done: 'We fix the cell 9 length-mismatch error by ensuring Lightning `predict()` outputs exactly one prediction per pair in `test_ds.pairs`, regardless of whether Lightning returns a list of tensors or a list of per-batch arrays. This is done by safely concatenating predictions with a helper that handles both tensor and numpy outputs, and by trimming to the expected number of pairs if an extra dimension slips in. These changes are execution/stability fixes that preserve your core pipeline (embeddings → KMeans tasks → pairwise AESNet inference → aggregation) and move the score up from 0.0 by producing a valid submission. We also set `persistent_workers` safely to avoid DataLoader issues when `num_workers=0` and keep runtime stable.'
- What this solution (achieved 0.0) has done: 'Your current score is far below the target, so we should improve predictive signal with minimal disruption to your existing pipeline. The biggest issue is that you are (often) running inference with a randomly initialized AESNet because no checkpoints are found, which produces essentially random scores and negative QWK. To move toward the target while preserving your core logic (embeddings → KMeans tasks → pairwise AESNet → aggregation), I add a small, deterministic training step on the provided training set to fit the same AESNet architecture before predicting test pairs. I also fix the relative-score target to match the model’s sigmoid output range by scaling the label-difference into [0,1] (and keep the same scale-back logic at prediction time), which should materially improve QWK without changing the fundamental approach.'
- What this solution (achieved 0.01678) has done: 'We fix the prediction concatenation bug causing the tensor size mismatch by making the output-flattening function handle variable shapes per batch (e.g., (B,1), (B,), or accidental extra dims) by flattening each batch to 1D before concatenation. We also make `compute_embeddings()` DataLoader use `persistent_workers` only when `num_workers>0` to avoid runtime issues across environments. These are execution/stability fixes that preserve your existing pipeline (embeddings → KMeans tasks → pairwise AESNet → aggregation) and should move the score up from 0.0 by producing a valid submission. No model architecture or training loop semantics are changed.'
- What this solution (achieved -0.01069) has done: 'Your current score (0.01678) is far below the target (0.77186), so we should improve predictive signal while keeping the same pipeline (embeddings → KMeans tasks → pairwise AESNet → aggregation). The biggest issue is the inference math: you train the model to predict a *scaled relative difference* in [0,1], but at predict time you “scale back” and then **add it directly to y_e2**, mixing incompatible units; this makes predictions badly miscalibrated. I minimally fix this by converting the predicted relative score back into a label difference in [-5,5] and then adding to y_e2, matching the training target semantics without changing the model or loss. I also make `CustomDataset` pass `y_e2` as float32 consistently and clamp predictions to valid label range before aggregation to reduce extreme outliers that hurt QWK.'

# 9. Code solution

## === cell 0
import os
import glob
import gc
import math
import random
import joblib

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as L

from sklearn.metrics import cohen_kappa_score

from transformers import AutoTokenizer, AutoModel


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)



## === cell 1
DATA_DIR = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"
TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
SAMPLE_SUB_PATH = f"{DATA_DIR}/sample_submission.csv"

train_data = pd.read_csv(TRAIN_PATH)
data = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

data = data.rename(columns={"full_text": "text"})
train_data["score"] = train_data["score"] - 1
train_data = train_data.rename(columns={"full_text": "text", "score": "labels"})

assert "essay_id" in data.columns and "text" in data.columns
assert (
    "essay_id" in train_data.columns
    and "text" in train_data.columns
    and "labels" in train_data.columns
)




## === cell 2
class EmbeddingsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 512):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        txt = self.df.loc[idx, "text"]
        out = self.tokenizer(
            txt,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        return {k: v.squeeze(0) for k, v in out.items()}


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    denom = mask.sum(dim=1).clamp(min=1)[..., None]
    sentence_embeddings = token_embeddings.sum(dim=1) / denom
    return sentence_embeddings


@torch.no_grad()
def compute_embeddings(
    df: pd.DataFrame, tokenizer, model, batch_size: int = 16, max_length: int = 512
):
    ds = EmbeddingsDataset(df, tokenizer, max_length=max_length)
    num_workers = 2
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )
    embs = []
    model.eval()
    device = next(model.parameters()).device
    for batch in tqdm(dl, total=len(dl), desc="Embedding"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        pooled = mean_pooling(outputs.last_hidden_state, batch["attention_mask"])
        embs.append(pooled.detach().cpu().numpy())
    return np.vstack(embs)


def compute_hashed_bow_embeddings(texts, dim: int = 768):
    embs = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(tqdm(texts, total=len(texts), desc="HashedBOW")):
        if not isinstance(t, str):
            t = "" if t is None else str(t)
        for w in t.lower().split():
            h = hash(w)
            idx = h % dim
            sign = 1.0 if (h & 1) == 0 else -1.0
            embs[i, idx] += sign
        norm = np.linalg.norm(embs[i])
        if norm > 0:
            embs[i] /= norm
    return embs




## === cell 3
def resolve_local_model_dir(data_dir: str) -> str | None:
    """
    Search /kaggle/input recursively for a usable local model folder
    (has config.json and some weights). If not found, return None.
    """
    search_roots = [
        data_dir,
        "/kaggle/input",
    ]
    name_hints = [
        "deberta-v3-small",
        "deberta-v3-base",
        "deberta-v3-large",
        "deberta",
    ]

    candidates = []
    for root in search_roots:
        for hint in name_hints:
            candidates.extend(glob.glob(os.path.join(root, "**", hint), recursive=True))

    candidates.extend(
        glob.glob(os.path.join(data_dir, "**", "config.json"), recursive=True)
    )
    candidates.extend(
        glob.glob(os.path.join("/kaggle/input", "**", "config.json"), recursive=True)
    )

    checked = set()
    for c in candidates:
        model_dir = os.path.dirname(c) if c.endswith("config.json") else c
        if model_dir in checked:
            continue
        checked.add(model_dir)

        cfg = os.path.join(model_dir, "config.json")
        if not os.path.exists(cfg):
            continue

        has_weights = (
            len(glob.glob(os.path.join(model_dir, "pytorch_model*.bin"))) > 0
            or len(glob.glob(os.path.join(model_dir, "model*.safetensors"))) > 0
        )
        if has_weights:
            return model_dir

    return None


MODEL_DIR = resolve_local_model_dir(DATA_DIR)
print("Resolved local model dir:", MODEL_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = None
model = None
use_transformer = MODEL_DIR is not None

if use_transformer:
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
        model = AutoModel.from_pretrained(MODEL_DIR, local_files_only=True)
        model.to(device)
    except Exception as e:
        print(
            "Failed to load local transformer model, falling back to hashed BOW embeddings. Error:",
            repr(e),
        )
        tokenizer, model = None, None
        use_transformer = False
else:
    print("No local transformer folder found; falling back to hashed BOW embeddings.")

if use_transformer:
    train_data_emb = compute_embeddings(
        train_data[["text"]], tokenizer, model, batch_size=16, max_length=512
    )
    embedding_features = compute_embeddings(
        data[["text"]], tokenizer, model, batch_size=16, max_length=512
    )
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
else:
    train_data_emb = compute_hashed_bow_embeddings(train_data["text"].values, dim=768)
    embedding_features = compute_hashed_bow_embeddings(data["text"].values, dim=768)

assert train_data_emb.shape[0] == len(train_data)
assert embedding_features.shape[0] == len(data)



## === cell 4
from sklearn.cluster import KMeans

N_TASKS = 8
kmeans = KMeans(n_clusters=N_TASKS, random_state=42, n_init="auto")
train_data["task"] = kmeans.fit_predict(train_data_emb)
data["task"] = kmeans.predict(embedding_features)




## === cell 5
class CustomDataset(Dataset):
    def __init__(self, data, tasks, train_data, train_labels, train_task, n_sample=1):
        self.data = data
        self.tasks = tasks
        self.train_data = train_data
        self.train_labels = (
            train_labels.squeeze() if len(train_labels.shape) > 1 else train_labels
        )
        self.train_task = train_task

        self.n_sample = n_sample

        self.train_num_labels = len(np.unique(self.train_labels))
        self.train_label_indices = {
            i: np.where(self.train_labels == i)[0] for i in range(self.train_num_labels)
        }

        self.train_num_tasks = len(np.unique(self.train_task))
        self.train_task_indices = {
            i: np.where(self.train_task == i)[0] for i in range(self.train_num_tasks)
        }

        self.pairs = self.make_pairs()

    def make_pairs(self):
        pairs = np.empty((0, 2), dtype=int)
        data_len = len(self.data)

        task_indices_sets = {
            task: set(indices) for task, indices in self.train_task_indices.items()
        }

        label_candidates = {
            label: np.array(list(set(indices) & set(range(len(self.train_data)))))
            for label, indices in self.train_label_indices.items()
        }

        print("Making essay pairs...")
        for i in tqdm(range(data_len)):
            e1_task = self.tasks[i]
            task_set = task_indices_sets.get(e1_task, set())

            for label, candidates in label_candidates.items():
                if len(task_set) == 0:
                    continue
                valid_candidates = candidates[np.isin(candidates, list(task_set))]
                if len(valid_candidates) > 0:
                    e2 = np.random.choice(valid_candidates, self.n_sample)
                    for j in e2:
                        pairs = np.vstack((pairs, np.array([i, j], dtype=int)))

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        e1 = self.pairs[idx][0]
        e2 = self.pairs[idx][1]

        x_e1 = self.data[e1]
        x_e2 = self.train_data[e2]
        y_e2 = self.train_labels[e2]

        output = {
            "x_e1": torch.tensor(x_e1, dtype=torch.float32),
            "x_e2": torch.tensor(x_e2, dtype=torch.float32),
            "relative_score": torch.tensor(0.0, dtype=torch.float32),
            "y_e1": torch.tensor(0.0, dtype=torch.float32),
            "y_e2": torch.tensor(float(y_e2), dtype=torch.float32),
        }
        return output


class PairTrainDataset(Dataset):
    def __init__(
        self,
        train_emb: np.ndarray,
        train_labels: np.ndarray,
        train_tasks: np.ndarray,
        n_pairs_per_essay: int = 2,
        max_pairs: int = 120_000,
    ):
        self.train_emb = train_emb
        self.y = train_labels.squeeze() if len(train_labels.shape) > 1 else train_labels
        self.tasks = train_tasks
        self.n_pairs_per_essay = n_pairs_per_essay
        self.max_pairs = max_pairs

        self.task_to_indices = {}
        for t in np.unique(self.tasks):
            self.task_to_indices[int(t)] = np.where(self.tasks == t)[0]

        self.pairs = self._make_pairs()

    def _make_pairs(self):
        pairs = []
        n = len(self.train_emb)
        rng = np.random.default_rng(42)

        for i in range(n):
            t = int(self.tasks[i])
            candidates = self.task_to_indices.get(t, None)
            if candidates is None or len(candidates) < 2:
                continue

            js = rng.choice(candidates, size=self.n_pairs_per_essay, replace=True)
            for j in js:
                if j == i:
                    continue
                pairs.append((i, int(j)))
                if len(pairs) >= self.max_pairs:
                    return np.asarray(pairs, dtype=np.int64)

        return np.asarray(pairs, dtype=np.int64)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        x_e1 = self.train_emb[i]
        x_e2 = self.train_emb[j]
        y_e1 = float(self.y[i])
        y_e2 = float(self.y[j])

        rel = (y_e1 - y_e2 + 5.0) / 10.0
        rel = float(np.clip(rel, 0.0, 1.0))

        return {
            "x_e1": torch.tensor(x_e1, dtype=torch.float32),
            "x_e2": torch.tensor(x_e2, dtype=torch.float32),
            "relative_score": torch.tensor(rel, dtype=torch.float32),
            "y_e1": torch.tensor(y_e1, dtype=torch.float32),
            "y_e2": torch.tensor(y_e2, dtype=torch.float32),
        }




## === cell 6
class Projector(nn.Module):
    def __init__(self, d, dropout=0.2):
        super().__init__()
        self.p = dropout

        self.linear_1 = nn.Linear(d, int(d / 2))
        self.bn_1 = nn.BatchNorm1d(int(d / 2))

        self.linear_2 = nn.Linear(int(d / 2), d)
        self.bn_2 = nn.BatchNorm1d(d)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.bn_1(x)
        x = torch.tanh(x)
        x = F.dropout(x, p=self.p, training=self.training)

        x = self.linear_2(x)
        x = self.bn_2(x)
        x = torch.tanh(x)
        x = F.dropout(x, p=self.p, training=self.training)
        return x




## === cell 7
class AESNet(L.LightningModule):
    def __init__(self, input_dim=768, lr=1e-5):
        super(AESNet, self).__init__()
        self.save_hyperparameters()

        self.lr = lr

        self.feature_extractor = nn.Sequential(
            Projector(input_dim),
            nn.Linear(input_dim, input_dim),
        )

        self.relative_score_head = nn.Linear(input_dim, 1, bias=False)
        self.mse_loss = nn.MSELoss()

    def _scale_back(self, x, min_x=-5, max_x=5):
        output = x * (max_x - min_x) + min_x
        return output

    def forward(self, e1, e2, y_e2):
        x_1 = self.feature_extractor(e1)
        f1 = e1 + x_1
        f1 = F.normalize(f1, dim=-1)

        x_2 = self.feature_extractor(e2)
        f2 = e2 + x_2
        f2 = F.normalize(f2, dim=-1)

        dv = f1 - f2
        relative_score_hat = torch.sigmoid(self.relative_score_head(dv))
        y_e1_hat = relative_score_hat + y_e2
        return relative_score_hat, y_e1_hat

    def training_step(self, batch, batch_idx):
        relative_score_hat, _ = self.forward(
            batch["x_e1"], batch["x_e2"], batch["y_e2"]
        )
        loss = self.mse_loss(relative_score_hat, batch["relative_score"])
        self.log("train loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        relative_score_hat, _ = self.forward(
            batch["x_e1"], batch["x_e2"], batch["y_e2"]
        )
        loss = self.mse_loss(relative_score_hat, batch["relative_score"])

        rs_hat = self._scale_back(relative_score_hat)
        rs = self._scale_back(batch["relative_score"])

        qwk = cohen_kappa_score(
            rs.detach().cpu().numpy().round(0).astype(int),
            rs_hat.detach().cpu().numpy().round(0).astype(int),
            weights="quadratic",
        )
        self.log("validation loss", loss, prog_bar=True)
        self.log("validation qwk", qwk, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        relative_score_hat, _ = self.forward(
            batch["x_e1"], batch["x_e2"], batch["y_e2"]
        )

        diff_hat = self._scale_back(relative_score_hat)  # now in [-5,5]
        y_e1_hat = diff_hat + batch["y_e2"]  # labels are 0..5

        y_e1_hat = torch.clamp(y_e1_hat, 0.0, 5.0)
        return y_e1_hat

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.6,
            total_iters=self.trainer.max_epochs,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }




## === cell 8
input_dim = embedding_features.shape[1]
train_pair_ds = PairTrainDataset(
    train_emb=train_data_emb,
    train_labels=train_data[["labels"]].values,
    train_tasks=train_data["task"].values,
    n_pairs_per_essay=2,
    max_pairs=120_000,
)
NUM_WORKERS = 2
train_pair_dl = DataLoader(
    train_pair_ds,
    batch_size=256,
    shuffle=True,
    num_workers=NUM_WORKERS,
    drop_last=True,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(NUM_WORKERS > 0),
)

tmp_model = AESNet(input_dim=input_dim, lr=2e-4)
trainer_fit = L.Trainer(
    accelerator="auto",
    devices=1,
    max_epochs=2,
    logger=False,
    enable_checkpointing=False,
    enable_model_summary=False,
    enable_progress_bar=True,
    gradient_clip_val=1.0,
)
trainer_fit.fit(tmp_model, train_dataloaders=train_pair_dl)



## === cell 9
test_ds = CustomDataset(
    embedding_features,
    data["task"].values,
    train_data_emb,
    train_data[["labels"]].values,
    train_data["task"].values,
)

test_dl = DataLoader(
    test_ds,
    batch_size=8,
    shuffle=False,
    num_workers=NUM_WORKERS,
    drop_last=False,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(NUM_WORKERS > 0),
)




## === cell 10
def _concat_predict_outputs_to_1d_array(outputs) -> np.ndarray:
    """
    Bugfix: Lightning predict() may yield a list of tensors with varying shapes
    ((B,1), (B,), (B,1,1), etc.). Flatten each batch output to 1D before concat
    so torch.cat only concatenates along a single dimension.
    """
    if isinstance(outputs, torch.Tensor):
        return outputs.detach().float().cpu().reshape(-1).numpy()

    parts_1d = []
    for o in outputs:
        if isinstance(o, torch.Tensor):
            t = o.detach().float().cpu()
        else:
            t = torch.as_tensor(o).detach().float().cpu()
        parts_1d.append(t.reshape(-1))

    if len(parts_1d) == 0:
        return np.array([], dtype=np.float32)

    arr = torch.cat(parts_1d, dim=0).numpy()
    return np.asarray(arr, dtype=np.float32).reshape(-1)


model_paths = glob.glob(f"/kaggle/input/**/models/**/*.ckpt", recursive=True)

test_pred = np.zeros((len(data), 1), dtype=np.float32)
model_count = 0
n_pairs_expected = len(test_ds.pairs)

if len(model_paths) == 0:
    print(
        "No .ckpt models found in /kaggle/input. Using freshly trained AESNet for inference."
    )
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    outputs = trainer.predict(model=tmp_model, dataloaders=test_dl)
    pred_pairs = _concat_predict_outputs_to_1d_array(outputs)

    if pred_pairs.shape[0] != n_pairs_expected:
        pred_pairs = pred_pairs[:n_pairs_expected]

    out_df = pd.DataFrame({"essay_idx": test_ds.pairs[:, 0], "pred_label": pred_pairs})

    out_series = out_df.groupby("essay_idx")["pred_label"].mean()
    out_series = out_series.reindex(np.arange(len(data)))
    if out_series.isna().any():
        out_series = out_series.fillna(out_series.mean())
    test_pred += out_series.values.reshape((-1, 1))
    model_count = 1
else:
    for path in model_paths:
        print("Loading:", path)
        try:
            model = AESNet.load_from_checkpoint(path)
        except Exception as e:
            print(f"Skipping checkpoint (load failed): {path} -> {e}")
            continue

        trainer = L.Trainer(
            accelerator="auto",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
        )
        outputs = trainer.predict(model=model, dataloaders=test_dl)
        pred_pairs = _concat_predict_outputs_to_1d_array(outputs)

        if pred_pairs.shape[0] != n_pairs_expected:
            pred_pairs = pred_pairs[:n_pairs_expected]

        out_df = pd.DataFrame(
            {"essay_idx": test_ds.pairs[:, 0], "pred_label": pred_pairs}
        )
        out_series = out_df.groupby("essay_idx")["pred_label"].mean()
        out_series = out_series.reindex(np.arange(len(data)))
        if out_series.isna().any():
            out_series = out_series.fillna(out_series.mean())

        test_pred += out_series.values.reshape((-1, 1))
        model_count += 1

assert model_count > 0



## === cell 11
test_pred = (test_pred / model_count).round(0).astype(int) + 1
test_pred = np.clip(test_pred, 1, 6)

submission_df = pd.DataFrame(
    {"essay_id": data["essay_id"].values, "score": test_pred.flatten().astype(int)}
)

submission_df = sample_sub[["essay_id"]].merge(submission_df, on="essay_id", how="left")
assert submission_df["score"].isna().sum() == 0
submission_df.to_csv("submission.csv", index=False)

print(submission_df.head())
print("Wrote submission.csv with shape:", submission_df.shape)
