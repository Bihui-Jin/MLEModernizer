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

0.7733314905137582

# 6. Current score

0.66466

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.02994) has done: 'I remove the failing `lightning` dependency and switch to `pytorch_lightning` (which is installed), keeping the same LightningModule/Trainer flow. I also eliminate hard dependencies on missing Kaggle datasets (`aes-deberta-small-embedding-dataset`, `embedding_features.npy`, `kmeans_task_clf.joblib`, and external `.ckpt`s) by computing embeddings locally with an installed SentenceTransformer model and replacing the unavailable kmeans/tasks with a safe single-task fallback. To preserve the core pairwise-regression logic and still produce predictions, I train the same network architecture on synthetic “relative_score=0” pairs derived from the train set (same semantics as your dataset class already encodes) and then run prediction exactly as intended. Finally, I ensure the submission is aligned to `essay_id` and writes a valid `submission.csv` with integer scores clipped to 1–6.'
- What this solution (achieved -0.00305) has done: 'I fix the SentenceTransformer/protobuf crash by avoiding the protobuf-dependent SentenceTransformer encode path and instead computing embeddings locally with a Transformers model (mean-pooled last hidden state), which preserves the same “compute embeddings then train the same network” core logic. Then I fix the prediction aggregation bug causing mismatched lengths by ensuring the number of predictions exactly matches the number of generated test pairs (and slicing defensively if Lightning returns extra batches). Finally, I keep the existing training/pairwise-regression setup intact and only add small stability guards (deterministic seeds, safe pooling, shape checks) so the notebook runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.59407) has done: 'I fix the protobuf-related crash that happens when importing/using `transformers` by avoiding the failing tokenizer/model path and switching embedding computation to a pure-PyTorch TF‑IDF baseline (no protobuf dependency), while keeping the rest of your pipeline intact (pairwise dataset, same LightningModule architecture, same training loop, same submission writing). I also correct a logic bug in `AESNet.forward()` where the predicted absolute score was being computed as `relative_score_hat + y_e2` even though `relative_score_hat` is in [0,1]; this mismatch can severely hurt QWK, so I minimally scale it back to the original diff range via your existing `_scale_back()` and then add `y_e2`. Finally, I keep shapes/dtypes consistent and ensure the script runs end-to-end in the Kaggle environment and writes `submission.csv` with `essay_id,score`.'
- What this solution (achieved 0.60627) has done: 'Your current score (0.59407) is well below the target (0.77333), so we should improve model signal with minimal semantic disruption. The biggest bottleneck is that your training pairs currently learn “relative to a fixed zero” for e1 within the training split (because `y_e1` is only set when `self.data is self.train_data`), which severely weakens supervision; we fix this by explicitly passing the correct `y_e1` vector into the dataset (same pairwise objective, same model). Next, because QWK is an ordinal metric on 1–6, we add a tiny, post-hoc calibration step on the validation set to choose a single scalar shift (and optional scale) for predictions before rounding/clipping; this keeps the model unchanged and typically improves QWK materially. Finally, we keep the same TF‑IDF features, same LightningModule/Trainer flow, and still write a valid `submission.csv`.'
- What this solution (achieved 0.60443) has done: 'Your current score (0.60627) is well below the target (0.77333), so we should increase signal with minimal semantic disruption. The biggest issue is that your final prediction for each essay is just the mean of many noisy pairwise-derived absolute predictions; for QWK on ordinal labels, a tiny additional *monotone* post-processing step often helps a lot without changing the model: we learn optimal bin thresholds on the validation set (ordinal “rounding” via learned cutpoints), and apply the same thresholds to test predictions. This keeps the same TF‑IDF features, the same pairwise dataset, the same Lightning model/training loop, and only changes the last step that converts continuous predictions into integer 1–6 scores. We keep your existing shift/scale calibration too, and then apply thresholds on top (both fitted only on validation), which is typically a safe improvement for QWK.'
- What this solution (achieved 0.62865) has done: 'Your current score (0.60443) is far below the target (0.77333), so we should improve signal while keeping your pairwise/TF‑IDF/Lightning pipeline intact. The biggest win with minimal semantic change is to increase the number of comparison partners per essay (`n_sample`) so each essay’s final score is averaged over more pairwise estimates, reducing variance and typically improving QWK. To keep runtime under control, we also avoid the very expensive “pair against every label for every essay” construction by sampling a small fixed number of labels per essay (still the same pairwise objective/labels, just fewer redundant pairs). Finally, we keep your existing shift/scale + learned thresholds calibration, but fit it on a larger validation split (10%) to make the thresholds more stable and transferable to test.'
- What this solution (achieved 0.66466) has done: 'To move your score up toward the 0.773 target while keeping the exact same TF‑IDF + pairwise Lightning architecture/training semantics, I’m making one primary adjustment: increase per-essay pair averaging signal by sampling more comparison partners (`N_SAMPLE`) and slightly more label bins per essay (`LABELS_PER_ESSAY`) while keeping the total pair count bounded enough to finish under the time limit. This typically improves QWK by reducing variance in the “mean over pairwise-derived absolute predictions” step without changing the model itself. I’m also aligning the test pairing to use the same *training split* reference pool as validation/training (instead of the full train), which avoids a small but real train/val distribution mismatch in calibration and keeps inference consistent with how calibration thresholds were learned. Finally, I add a tiny safety guard to ensure each essay gets at least one pair even if a task/label intersection is empty (prevents rare NaNs that can hurt calibration/prediction stability).'

# 9. Code solution

## === cell 0
import os
import glob
import gc
import random
import math
import warnings
import sys
import importlib

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm

import pytorch_lightning as pl

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)



## === cell 1
BASE = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"
TRAIN_PATH = f"{BASE}/train.csv"
TEST_PATH = f"{BASE}/test.csv"
SAMPLE_SUB_PATH = f"{BASE}/sample_submission.csv"

train_data = pd.read_csv(TRAIN_PATH)
data = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

data = data.rename(columns={"full_text": "text"})
train_data["score"] = train_data["score"] - 1
train_data = train_data.rename(columns={"full_text": "text", "score": "labels"})

assert set(["essay_id", "text"]).issubset(data.columns)
assert set(["essay_id", "text", "labels"]).issubset(train_data.columns)

print(train_data.shape, data.shape, sample_sub.shape)



## === cell 2
from sklearn.feature_extraction.text import TfidfVectorizer


def compute_tfidf_embeddings(
    train_texts, test_texts, max_features=768, ngram_range=(1, 2)
):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        max_features=max_features,
        ngram_range=ngram_range,
        dtype=np.float32,
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train.toarray().astype(np.float32), X_test.toarray().astype(np.float32)


train_texts = train_data["text"].fillna("").tolist()
test_texts = data["text"].fillna("").tolist()

train_data_emb, embedding_features = compute_tfidf_embeddings(
    train_texts, test_texts, max_features=768, ngram_range=(1, 2)
)

print(
    "train embeddings:",
    train_data_emb.shape,
    "test embeddings:",
    embedding_features.shape,
)
gc.collect()



## === cell 3
train_data["task"] = 0
data["task"] = 0

print(
    "Unique tasks (train/test):", train_data["task"].nunique(), data["task"].nunique()
)



## === cell 4
from sklearn.metrics import cohen_kappa_score


def qwk(y_true, y_pred):
    y_true = np.asarray(y_true).round().astype(int)
    y_pred = np.asarray(y_pred).round().astype(int)
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")




## === cell 5
class CustomDataset(Dataset):
    """
    Generates (e1, e2) pairs and a supervised relative_score target derived from labels:
      relative_score = (y_e1 - y_e2) scaled into [0,1] via (diff + 5)/10
    """

    def __init__(
        self,
        data,
        tasks,
        train_data,
        train_labels,
        train_task,
        n_sample=1,
        data_labels=None,
        labels_per_essay=None,
    ):
        self.data = data
        self.tasks = tasks
        self.train_data = train_data
        self.train_labels = np.asarray(train_labels).reshape(-1).astype(np.float32)
        self.train_task = np.asarray(train_task).reshape(-1).astype(int)

        if data_labels is None:
            self.data_labels = None
        else:
            self.data_labels = np.asarray(data_labels).reshape(-1).astype(np.float32)

        self.n_sample = int(n_sample)
        self.labels_per_essay = (
            None if labels_per_essay is None else int(labels_per_essay)
        )

        self.train_num_labels = len(np.unique(self.train_labels.astype(int)))
        self.train_label_indices = {
            i: np.where(self.train_labels.astype(int) == i)[0]
            for i in range(self.train_num_labels)
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
            label: np.array(
                list(set(indices) & set(range(len(self.train_data)))), dtype=int
            )
            for label, indices in self.train_label_indices.items()
        }

        all_labels = np.array(sorted(label_candidates.keys()), dtype=int)

        print("Making essay pairs...")
        for i in tqdm(range(data_len)):
            e1_task = int(self.tasks[i])
            task_set = task_indices_sets.get(e1_task, set(range(len(self.train_data))))
            task_set_list = np.fromiter(task_set, dtype=int)

            if self.labels_per_essay is None or self.labels_per_essay >= len(
                all_labels
            ):
                labels_to_use = all_labels
            else:
                labels_to_use = np.random.choice(
                    all_labels, size=self.labels_per_essay, replace=False
                )

            made_any = False
            for label in labels_to_use:
                candidates = label_candidates.get(int(label), None)
                if candidates is None or len(candidates) == 0:
                    continue

                valid_mask = np.isin(candidates, task_set_list, assume_unique=False)
                valid_candidates = candidates[valid_mask]
                if len(valid_candidates) == 0:
                    continue

                e2 = np.random.choice(valid_candidates, self.n_sample, replace=True)
                pairs = np.vstack(
                    (
                        pairs,
                        np.column_stack(
                            [np.full(len(e2), i, dtype=int), e2.astype(int)]
                        ),
                    )
                )
                made_any = True

            if not made_any:
                e2 = np.random.randint(0, len(self.train_data), size=1, dtype=int)
                pairs = np.vstack((pairs, np.array([[i, int(e2[0])]], dtype=int)))

        return pairs

    @staticmethod
    def _scale_diff_to_unit(diff, min_x=-5.0, max_x=5.0):
        diff = np.clip(diff, min_x, max_x)
        return (diff - min_x) / (max_x - min_x)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        e1 = int(self.pairs[idx][0])
        e2 = int(self.pairs[idx][1])

        x_e1 = self.data[e1]
        x_e2 = self.train_data[e2]
        y_e2 = float(self.train_labels[e2])

        if self.data_labels is None:
            y_e1 = 0.0
        else:
            y_e1 = float(self.data_labels[e1])

        rel = self._scale_diff_to_unit(y_e1 - y_e2)

        output = {
            "x_e1": torch.tensor(x_e1, dtype=torch.float32),
            "x_e2": torch.tensor(x_e2, dtype=torch.float32),
            "relative_score": torch.tensor(rel, dtype=torch.float32),
            "y_e1": torch.tensor(y_e1, dtype=torch.float32),
            "y_e2": torch.tensor(y_e2, dtype=torch.float32),
        }
        return output




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
        x = F.tanh(x)
        x = F.dropout(x, p=self.p, training=self.training)

        x = self.linear_2(x)
        x = self.bn_2(x)
        x = F.tanh(x)
        x = F.dropout(x, p=self.p, training=self.training)

        return x




## === cell 7
class AESNet(pl.LightningModule):
    def __init__(self, input_dim=768, lr=1e-5):
        super(AESNet, self).__init__()
        self.save_hyperparameters()

        self.lr = lr

        self.feature_extractor = nn.Sequential(
            Projector(input_dim), nn.Linear(input_dim, input_dim)
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

        x_1 = self.feature_extractor(e2)
        f2 = e2 + x_1
        f2 = F.normalize(f2, dim=-1)

        dv = f1 - f2
        relative_score_hat = torch.sigmoid(self.relative_score_head(dv))  # in [0,1]

        diff_hat = self._scale_back(relative_score_hat)
        y_e1_hat = diff_hat + y_e2.view_as(diff_hat)
        return relative_score_hat, y_e1_hat

    def training_step(self, batch, batch_idx):
        x_e1 = batch["x_e1"]
        x_e2 = batch["x_e2"]
        y_e2 = batch["y_e2"]
        relative_score = batch["relative_score"]

        relative_score_hat, _ = self.forward(x_e1, x_e2, y_e2)
        loss = self.mse_loss(relative_score_hat.view_as(relative_score), relative_score)

        self.log("train loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_e1 = batch["x_e1"]
        x_e2 = batch["x_e2"]
        y_e2 = batch["y_e2"]
        relative_score = batch["relative_score"]

        relative_score_hat, _ = self.forward(x_e1, x_e2, y_e2)
        loss = self.mse_loss(relative_score_hat.view_as(relative_score), relative_score)

        rs_hat = (
            self._scale_back(relative_score_hat.detach())
            .cpu()
            .numpy()
            .round(0)
            .astype(int)
        )
        rs = (
            self._scale_back(relative_score.detach()).cpu().numpy().round(0).astype(int)
        )
        qwk_val = cohen_kappa_score(rs, rs_hat, weights="quadratic")

        self.log("validation loss", loss, prog_bar=True)
        self.log("validation qwk", qwk_val, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x_e1 = batch["x_e1"]
        x_e2 = batch["x_e2"]
        y_e2 = batch["y_e2"]

        _, y_e1_hat = self.forward(x_e1, x_e2, y_e2)
        return y_e1_hat.view(-1, 1)

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
from sklearn.model_selection import train_test_split

X_train_emb, X_val_emb, y_train, y_val, task_train, task_val = train_test_split(
    train_data_emb,
    train_data["labels"].values.astype(int),
    train_data["task"].values.astype(int),
    test_size=0.10,
    random_state=SEED,
    stratify=train_data["labels"].values.astype(int),
)

N_SAMPLE = 5
LABELS_PER_ESSAY = 5

train_ds = CustomDataset(
    data=X_train_emb,
    tasks=task_train,
    train_data=X_train_emb,
    train_labels=y_train.astype(int),
    train_task=task_train,
    n_sample=N_SAMPLE,
    data_labels=y_train.astype(int),
    labels_per_essay=LABELS_PER_ESSAY,
)
val_ds = CustomDataset(
    data=X_val_emb,
    tasks=task_val,
    train_data=X_train_emb,
    train_labels=y_train.astype(int),
    train_task=task_train,
    n_sample=N_SAMPLE,
    data_labels=y_val.astype(int),
    labels_per_essay=LABELS_PER_ESSAY,
)

train_dl = DataLoader(
    train_ds, batch_size=128, shuffle=True, num_workers=2, drop_last=True
)
val_dl = DataLoader(
    val_ds, batch_size=128, shuffle=False, num_workers=2, drop_last=False
)

print("train pairs:", len(train_ds), "val pairs:", len(val_ds))



## === cell 9
input_dim = train_data_emb.shape[1]
model = AESNet(input_dim=input_dim, lr=1e-5)

trainer = pl.Trainer(
    max_epochs=2,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    logger=False,
    enable_checkpointing=False,
    enable_progress_bar=True,
    deterministic=True,
)

trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)




## === cell 10
def predict_essay_scores_from_pairs(ds, dl):
    out = trainer.predict(model=model, dataloaders=dl)
    pred_arr = torch.cat([o.detach().cpu() for o in out], dim=0).numpy().reshape(-1)
    n_pairs = len(ds.pairs)
    if len(pred_arr) != n_pairs:
        pred_arr = pred_arr[:n_pairs]

    df = pd.DataFrame(
        {"essay_idx": ds.pairs[:, 0].astype(int), "pred_label": pred_arr.astype(float)}
    )
    essay_pred = (
        df.groupby("essay_idx", sort=True)
        .agg({"pred_label": "mean"})["pred_label"]
        .values.reshape(-1)
    )
    return essay_pred


def fit_shift_scale_qwk(y_true_0to5, y_pred_0to5_cont):
    y_true = np.asarray(y_true_0to5).astype(int)
    y_pred = np.asarray(y_pred_0to5_cont).astype(float)

    scales = np.linspace(0.85, 1.15, 13)
    shifts = np.linspace(-0.75, 0.75, 31)

    best = (-1.0, 1.0, 0.0)  # (qwk, scale, shift)
    for a in scales:
        yp_a = a * y_pred
        for b in shifts:
            yp = np.clip(np.rint(yp_a + b), 0, 5).astype(int)
            score = qwk(y_true, yp)
            if score > best[0]:
                best = (score, float(a), float(b))
    return best  # (best_qwk, a, b)


def apply_thresholds(y_cont, thresholds):
    y_cont = np.asarray(y_cont, dtype=float)
    t = np.asarray(thresholds, dtype=float)
    return np.digitize(y_cont, t, right=False).astype(int)


def fit_thresholds_qwk(y_true_0to5, y_cont_0to5, init_thresholds=None):
    y_true = np.asarray(y_true_0to5).astype(int)
    y_cont = np.asarray(y_cont_0to5).astype(float)

    if init_thresholds is None:
        med = []
        for c in range(6):
            vals = y_cont[y_true == c]
            if len(vals) == 0:
                med.append(np.nan)
            else:
                med.append(np.median(vals))
        med = np.asarray(med, dtype=float)

        global_med = np.nanmedian(med)
        med = np.where(np.isnan(med), global_med, med)

        thresholds = []
        for c in range(5):
            thresholds.append(0.5 * (med[c] + med[c + 1]))
        thresholds = np.asarray(thresholds, dtype=float)
    else:
        thresholds = np.asarray(init_thresholds, dtype=float).copy()

    best_thr = thresholds.copy()
    best_score = qwk(y_true, np.clip(apply_thresholds(y_cont, best_thr), 0, 5))

    for _ in range(6):
        for k in range(5):
            lo = -1.0 if k == 0 else best_thr[k - 1] + 1e-6
            hi = 6.0 if k == 4 else best_thr[k + 1] - 1e-6
            center = best_thr[k]
            grid = np.linspace(max(lo, center - 1.0), min(hi, center + 1.0), 41)
            for cand in grid:
                thr_try = best_thr.copy()
                thr_try[k] = cand
                preds = np.clip(apply_thresholds(y_cont, thr_try), 0, 5)
                score = qwk(y_true, preds)
                if score > best_score:
                    best_score = score
                    best_thr = thr_try
    return best_score, best_thr


val_pred_cont = predict_essay_scores_from_pairs(val_ds, val_dl)
val_true = y_val.astype(int)

best_qwk_round, best_a, best_b = fit_shift_scale_qwk(val_true, val_pred_cont)
val_pred_cal = best_a * val_pred_cont + best_b

best_qwk_thr, best_thr = fit_thresholds_qwk(val_true, val_pred_cal)

print(
    f"Calibration on val (rounding): best_qwk={best_qwk_round:.5f}, scale={best_a:.4f}, shift={best_b:.4f}"
)
print(f"Calibration on val (thresholds): best_qwk={best_qwk_thr:.5f}, thr={best_thr}")



## === cell 11
test_ds = CustomDataset(
    data=embedding_features,
    tasks=data["task"].values.astype(int),
    train_data=X_train_emb,
    train_labels=y_train.astype(int),
    train_task=task_train,
    n_sample=N_SAMPLE,
    data_labels=None,
    labels_per_essay=LABELS_PER_ESSAY,
)
test_dl = DataLoader(
    test_ds, batch_size=256, shuffle=False, num_workers=2, drop_last=False
)

test_pred_cont_0to5 = predict_essay_scores_from_pairs(test_ds, test_dl)

test_pred_cal = best_a * test_pred_cont_0to5 + best_b
test_pred_0to5 = np.clip(apply_thresholds(test_pred_cal, best_thr), 0, 5).astype(int)
test_pred_1to6 = (test_pred_0to5 + 1).clip(1, 6)

print(
    "pred shape:",
    test_pred_1to6.shape,
    "min/max:",
    test_pred_1to6.min(),
    test_pred_1to6.max(),
)

submission_df = pd.DataFrame(
    {"essay_id": data["essay_id"].values, "score": test_pred_1to6.astype(int)}
)

if len(submission_df) != len(sample_sub) or not submission_df["essay_id"].equals(
    sample_sub["essay_id"]
):
    submission_df = sample_sub[["essay_id"]].merge(
        submission_df, on="essay_id", how="left"
    )
    mode_score = int(pd.Series(train_data["labels"].values + 1).mode().iloc[0])
    submission_df["score"] = (
        submission_df["score"].fillna(mode_score).astype(int).clip(1, 6)
    )

submission_df.to_csv("submission.csv", index=False)
print(submission_df.head())
print("Wrote submission.csv with shape:", submission_df.shape)
print("Columns:", submission_df.columns.tolist())
print("score value counts:\n", submission_df["score"].value_counts().sort_index())
