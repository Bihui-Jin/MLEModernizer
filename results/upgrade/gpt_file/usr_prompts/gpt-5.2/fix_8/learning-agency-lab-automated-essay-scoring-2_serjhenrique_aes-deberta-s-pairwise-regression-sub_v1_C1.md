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

0.7681975217610972

# 6. Current score

0.51635

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the immediate runtime blocker: the notebook tries to load a Hugging Face transformer offline but no model exists in `/kaggle/input`, so embeddings never get created and everything downstream fails. To preserve the overall pipeline (embed → KMeans task clustering → pairwise AESNet inference → average → round+clip), I add a safe local embedding fallback using scikit-learn TF‑IDF + TruncatedSVD when no local transformers model is found. I also fix a subtle shape/index issue in the prediction aggregation so every `essay_idx` gets a prediction row (even if some indices are missing), ensuring the submission has the correct length and ordering. These changes are purely to make the code run end-to-end and produce a valid `submission.csv` without changing the later model logic.'
- What this solution (achieved 0.01522) has done: 'The runtime error comes from `trainer.predict()` returning a list of batch tensors with inconsistent shapes (some batches yield `(B,)` while others yield `(B,1)`), so `torch.vstack` fails. I make the prediction stacking robust by flattening each batch output to `(batch_size, 1)` before concatenation, without changing any model logic. I also ensure the loaded checkpoint models are put into eval mode and moved to the right device for stable inference. This let the notebook run end-to-end and write a valid `submission.csv` in the required format, moving the score from 0.0 (no valid submission) toward the target.'
- What this solution (achieved 0.01522) has done: 'Your current score is far below the target, and the biggest issue is that the inference pipeline is effectively producing near-random outputs because (a) it rarely finds any valid `.ckpt` files and (b) even when it does, it likely uses a mismatched `input_dim` (your embeddings are 256-D but the model default is 768-D), which can force fallback behavior or break meaningful loading. I make two minimal, core-logic-preserving fixes: enforce that the AESNet `input_dim` always matches the embedding dimension when loading checkpoints (so weights can load correctly when possible), and make the checkpoint discovery more robust (search all `.ckpt` under `/kaggle/input`). These changes should substantially increase QWK toward the target without changing the model architecture, loss, or overall approach (embed → cluster → pairwise inference → average → round/clip). I also ensure Lightning runs in pure inference mode (no accidental training behaviors) for stable predictions.'
- What this solution (achieved 0.4666) has done: 'Your score is far below target because the pipeline is still effectively using an untrained/random AESNet (no compatible `.ckpt` are available), and the current embedding fallback makes that even noisier. To move toward the target with minimal changes, I (1) keep your embed→cluster logic but strengthen the *offline* embedding fallback by using a stronger TF‑IDF representation (char + word) and normalizing it, and (2) replace the random AESNet inference fallback with a deterministic, label-informed nearest-neighbor baseline *using the same embeddings and tasks* (no new data, no leakage from test labels). If compatible checkpoints exist, your original Lightning inference path still be used unchanged; only the “no ckpt / all skipped” path changes to avoid near-random predictions. This should lift QWK substantially toward the target while preserving the overall semantics (embed → task cluster → infer score → round/clip → submission).'
- What this solution (achieved 0.51287) has done: 'Your current score (0.4666) is far below the target (0.7682), so we should improve predictions without changing the overall pipeline. The biggest lift with minimal risk is to keep your embeddings+task clustering, but replace the hard rounding with a tiny, metric-aligned calibration: fit per-task linear scaling (and a global fallback) on out-of-fold train predictions from the same task‑kNN logic, optimizing quadratic weighted kappa via a small discrete search over scale/shift. This preserves core semantics (embed → cluster → predict score → clip) while directly aligning outputs to the QWK metric and typically yields a substantial gain versus raw kNN. If checkpoints exist, your original AESNet inference is still used; we only apply the same lightweight calibration to the final continuous predictions before rounding/clipping.'
- What this solution (achieved 0.51635) has done: 'Your current score (0.51287) is far below the target (0.76820), so we should improve the *existing* kNN+task clustering pipeline without changing the model architecture or training approach. The biggest likely drag is that your “OOF” calibration isn’t actually out-of-fold (it predicts each training point using itself as a neighbor), which inflates the calibration fit and can hurt generalization; I fix it by excluding the query example from its neighbor pool in OOF. Next, your folds are deterministic by index (not stratified by task/label), which can destabilize calibration; I switch to a seeded permutation-based fold assignment (still no leakage, minimal change). Finally, I make the TF‑IDF+SVD fallback deterministic and slightly more stable by using an explicit SVD `n_iter` and consistent sparse stacking when available—this typically improves similarity structure for kNN without changing the overall method.'

# 9. Code solution

## === cell 0
import os
import glob
import gc
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel

import pytorch_lightning as L

from sklearn.cluster import KMeans
from sklearn.metrics import cohen_kappa_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

warnings.filterwarnings("ignore")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)



## === cell 1
DATA_DIR = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"
TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
SAMPLE_SUB_PATH = f"{DATA_DIR}/sample_submission.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

test_df = test_df.rename(columns={"full_text": "text"})
train_df["score"] = train_df["score"] - 1  # keep original logic
train_df = train_df.rename(columns={"full_text": "text", "score": "labels"})

assert "essay_id" in test_df.columns and "text" in test_df.columns
assert (
    "essay_id" in train_df.columns
    and "text" in train_df.columns
    and "labels" in train_df.columns
)




## === cell 2
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


class EmbeddingsDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=1024):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.df.loc[idx, "text"],
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        return enc


def find_local_transformers_model_dir():
    """
    Try to locate a locally-available transformers model directory inside /kaggle/input.
    Minimal heuristic: find folders containing config.json.
    """
    candidates = []
    for base in ["/kaggle/input", "/kaggle/data"]:
        if not os.path.isdir(base):
            continue
        for cfg in glob.glob(os.path.join(base, "**", "config.json"), recursive=True):
            candidates.append(os.path.dirname(cfg))
    for c in candidates:
        lc = c.lower()
        if "deberta" in lc:
            return c
    return candidates[0] if candidates else None


MODEL_DIR = find_local_transformers_model_dir()
FALLBACK_MODEL_NAME = "microsoft/deberta-v3-small"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = None
model = None
use_transformers = False
if MODEL_DIR is not None:
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
        model = AutoModel.from_pretrained(MODEL_DIR, local_files_only=True)
        model.to(device)
        model.eval()
        use_transformers = True
        print(f"Using local transformers model from: {MODEL_DIR}")
    except Exception as e:
        print(f"Local transformers model load failed ({MODEL_DIR}): {repr(e)}")
        tokenizer, model, use_transformers = None, None, False
else:
    print(
        "No local transformers model directory found under /kaggle/input or /kaggle/data."
    )




## === cell 3
@torch.no_grad()
def compute_embeddings_transformers(
    df, tokenizer, model, batch_size=32, max_length=1024
):
    ds = EmbeddingsDataset(df, tokenizer, max_length=max_length)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    embs = []
    for batch in tqdm(dl, total=len(dl), desc="Embedding (transformers)"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        pooled = mean_pooling(outputs.last_hidden_state, batch["attention_mask"])
        embs.append(pooled.detach().cpu().numpy())
    return np.vstack(embs)


def compute_embeddings_tfidf_svd(
    train_texts, test_texts, n_components=256, max_features=250000
):
    def l2norm(a, eps=1e-12):
        n = np.linalg.norm(a, axis=1, keepdims=True)
        return a / (n + eps)

    vec_word = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=2,
        dtype=np.float32,
    )
    Xw_train = vec_word.fit_transform(train_texts)
    Xw_test = vec_word.transform(test_texts)

    vec_char = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=max_features,
        min_df=2,
        dtype=np.float32,
    )
    Xc_train = vec_char.fit_transform(train_texts)
    Xc_test = vec_char.transform(test_texts)

    try:
        from scipy.sparse import hstack  # type: ignore

        X_train = hstack([Xw_train, Xc_train]).tocsr()
        X_test = hstack([Xw_test, Xc_test]).tocsr()
    except Exception:
        X_train, X_test = Xw_train, Xw_test

    n_features = X_train.shape[1]
    n_samples = X_train.shape[0]
    k = int(min(n_components, max(2, n_features - 1), max(2, n_samples - 1)))

    svd = TruncatedSVD(n_components=k, n_iter=10, random_state=42)
    E_train = svd.fit_transform(X_train).astype(np.float32, copy=False)
    E_test = svd.transform(X_test).astype(np.float32, copy=False)

    E_train = l2norm(E_train)
    E_test = l2norm(E_test)
    return E_train, E_test


if use_transformers:
    train_emb = compute_embeddings_transformers(
        train_df[["text"]], tokenizer, model, batch_size=32, max_length=1024
    )
    test_emb = compute_embeddings_transformers(
        test_df[["text"]], tokenizer, model, batch_size=32, max_length=1024
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
else:
    train_emb, test_emb = compute_embeddings_tfidf_svd(
        train_df["text"].astype(str).tolist(),
        test_df["text"].astype(str).tolist(),
        n_components=256,
        max_features=250000,
    )

print("train_emb shape:", train_emb.shape, "test_emb shape:", test_emb.shape)



## === cell 4
n_tasks = 8  # small and stable; acts as a proxy for the original task clustering
kmeans = KMeans(n_clusters=n_tasks, random_state=42, n_init=10)
train_df["task"] = kmeans.fit_predict(train_emb)
test_df["task"] = kmeans.predict(test_emb)




## === cell 5
class CustomDataset(Dataset):
    def __init__(self, data, tasks, train_data, train_labels, train_task, n_sample=1):
        self.data = data
        self.tasks = tasks
        self.train_data = train_data
        self.train_labels = train_labels
        self.train_task = train_task

        self.n_sample = n_sample

        self.train_labels = self.train_labels.astype(int).reshape(-1)

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
        for i in tqdm(range(data_len), total=data_len):
            e1_task = int(self.tasks[i])
            task_set = task_indices_sets.get(e1_task, set(range(len(self.train_data))))

            for label, candidates in label_candidates.items():
                if len(task_set) == 0:
                    continue
                valid_mask = np.isin(
                    candidates, np.fromiter(task_set, dtype=int, count=len(task_set))
                )
                valid_candidates = candidates[valid_mask]
                if len(valid_candidates) > 0:
                    e2 = np.random.choice(valid_candidates, self.n_sample, replace=True)
                    for j in e2:
                        pairs = np.vstack((pairs, np.array([i, j], dtype=int)))

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        e1 = int(self.pairs[idx][0])
        e2 = int(self.pairs[idx][1])

        x_e1 = self.data[e1]
        x_e2 = self.train_data[e2]
        y_e2 = float(self.train_labels[e2])

        output = {
            "x_e1": torch.tensor(x_e1, dtype=torch.float32),
            "x_e2": torch.tensor(x_e2, dtype=torch.float32),
            "relative_score": torch.tensor(0.0, dtype=torch.float32),
            "y_e1": torch.tensor(0.0, dtype=torch.float32),
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
        return x * (max_x - min_x) + min_x

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
        relative_score_hat, y_e1_hat = self.forward(
            batch["x_e1"], batch["x_e2"], batch["y_e2"]
        )
        relative_score_hat = self._scale_back(relative_score_hat)
        y_e1_hat = relative_score_hat + batch["y_e2"]
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




## === cell 7
test_ds = CustomDataset(
    test_emb,
    test_df["task"].values,
    train_emb,
    train_df["labels"].values,
    train_df["task"].values,
)

test_dl = DataLoader(
    test_ds,
    batch_size=2,
    shuffle=False,
    num_workers=2,
    drop_last=False,
    pin_memory=torch.cuda.is_available(),
)

ckpt_paths = glob.glob("/kaggle/input/**/*.ckpt", recursive=True)
ckpt_paths = sorted(set(ckpt_paths))

test_pred = np.zeros((len(test_df), 1), dtype=np.float32)
model_count = 0


def aggregate_preds(output_array, pairs, n_essays):
    out_df = pd.DataFrame(
        {"essay_idx": pairs[:, 0].astype(int), "pred_label": output_array.flatten()}
    )
    out_df = out_df.groupby("essay_idx", sort=False).agg({"pred_label": "mean"})
    out_df = out_df.reindex(np.arange(n_essays))
    out_df["pred_label"] = out_df["pred_label"].fillna(out_df["pred_label"].mean())
    return out_df["pred_label"].values.reshape(-1, 1).astype(np.float32)


def stack_predict_outputs(predict_outputs):
    chunks = []
    for o in predict_outputs:
        if isinstance(o, np.ndarray):
            t = torch.from_numpy(o)
        else:
            t = o
        if not torch.is_tensor(t):
            t = torch.tensor(t)
        t = t.detach().cpu()
        if t.ndim == 0:
            t = t.view(1, 1)
        elif t.ndim == 1:
            t = t.view(-1, 1)
        elif t.ndim >= 2:
            t = t.view(t.shape[0], -1)
            if t.shape[1] != 1:
                t = t[:, :1]
        chunks.append(t)
    return torch.cat(chunks, dim=0).numpy().astype(np.float32, copy=False)


def predict_knn_by_task(
    train_emb, train_labels, train_tasks, test_emb, test_tasks, k=25
):
    train_labels = train_labels.astype(np.float32).reshape(-1)

    def l2norm(a, eps=1e-12):
        n = np.linalg.norm(a, axis=1, keepdims=True)
        return a / (n + eps)

    Tr = l2norm(train_emb.astype(np.float32, copy=False))
    Te = l2norm(test_emb.astype(np.float32, copy=False))

    preds = np.zeros((Te.shape[0],), dtype=np.float32)
    global_mean = float(train_labels.mean())

    task_to_idx = {}
    for t in np.unique(train_tasks):
        task_to_idx[int(t)] = np.where(train_tasks == t)[0]

    for i in tqdm(range(Te.shape[0]), total=Te.shape[0], desc="Predict (task-kNN)"):
        t = int(test_tasks[i])
        idx = task_to_idx.get(t, None)
        if idx is None or len(idx) == 0:
            preds[i] = global_mean
            continue

        v = Te[i]
        A = Tr[idx]  # (n_task, d)
        sims = A @ v  # cosine sims

        kk = int(min(k, len(idx)))
        if kk <= 0:
            preds[i] = global_mean
            continue

        topk_local = np.argpartition(-sims, kk - 1)[:kk]
        top_sims = sims[topk_local]
        top_labels = train_labels[idx[topk_local]]

        w = np.clip(top_sims, -1.0, 1.0)
        w = (w + 1.0) / 2.0  # [0,1]
        wsum = float(w.sum())
        if wsum < 1e-8:
            preds[i] = float(top_labels.mean())
        else:
            preds[i] = float((w * top_labels).sum() / wsum)

    return preds.reshape(-1, 1).astype(np.float32)


def _qwk_from_continuous(y_true_int_0_5, y_cont_0_5):
    y_pred_int = np.clip(np.rint(y_cont_0_5), 0, 5).astype(int)
    return cohen_kappa_score(
        y_true_int_0_5.astype(int), y_pred_int, weights="quadratic"
    )


def fit_taskwise_linear_calibration_oof(train_emb, y_train, tasks, k=25, n_folds=5):
    y_train = y_train.astype(int).reshape(-1)
    tasks = tasks.astype(int).reshape(-1)

    rng = np.random.RandomState(42)
    perm = rng.permutation(len(y_train))
    fold_id = np.empty(len(y_train), dtype=int)
    fold_id[perm] = (np.arange(len(y_train)) % n_folds).astype(int)

    def l2norm(a, eps=1e-12):
        n = np.linalg.norm(a, axis=1, keepdims=True)
        return a / (n + eps)

    Tr_all = l2norm(train_emb.astype(np.float32, copy=False))

    oof = np.zeros((len(y_train),), dtype=np.float32)
    global_mean = float(y_train.mean())

    unique_tasks = np.unique(tasks)
    task_to_indices_all = {int(t): np.where(tasks == t)[0] for t in unique_tasks}

    for f in range(n_folds):
        val_idx = np.where(fold_id == f)[0]
        trn_idx = np.where(fold_id != f)[0]

        for t in unique_tasks:
            t = int(t)
            val_t = val_idx[tasks[val_idx] == t]
            if val_t.size == 0:
                continue

            idx_all_task = task_to_indices_all[t]
            idx_task_tr = idx_all_task[np.isin(idx_all_task, trn_idx)]
            if idx_task_tr.size == 0:
                oof[val_t] = global_mean
                continue

            A = Tr_all[idx_task_tr]
            yA = y_train[idx_task_tr].astype(np.float32)

            for i in val_t:
                if i in idx_task_tr:
                    mask = idx_task_tr != i
                    A_i = A[mask]
                    yA_i = yA[mask]
                else:
                    A_i = A
                    yA_i = yA

                if A_i.shape[0] == 0:
                    oof[i] = global_mean
                    continue

                v = Tr_all[i]
                sims = A_i @ v
                kk = int(min(k, A_i.shape[0]))
                topk_local = np.argpartition(-sims, kk - 1)[:kk]
                top_sims = sims[topk_local]
                top_labels = yA_i[topk_local]

                w = np.clip(top_sims, -1.0, 1.0)
                w = (w + 1.0) / 2.0
                wsum = float(w.sum())
                if wsum < 1e-8:
                    oof[i] = float(top_labels.mean())
                else:
                    oof[i] = float((w * top_labels).sum() / wsum)

    a_grid = np.linspace(0.85, 1.15, 13)
    b_grid = np.linspace(-0.35, 0.35, 15)

    best_global = (1.0, 0.0, -1.0)
    for a in a_grid:
        for b in b_grid:
            q = _qwk_from_continuous(y_train, a * oof + b)
            if q > best_global[2]:
                best_global = (float(a), float(b), float(q))

    task_params = {}
    for t in unique_tasks:
        t = int(t)
        idx = np.where(tasks == t)[0]
        if idx.size < 200:  # stability guard: too few points -> use global
            continue
        yt = y_train[idx]
        yp = oof[idx]
        best_t = (best_global[0], best_global[1], -1.0)
        for a in a_grid:
            for b in b_grid:
                q = _qwk_from_continuous(yt, a * yp + b)
                if q > best_t[2]:
                    best_t = (float(a), float(b), float(q))
        task_params[t] = (best_t[0], best_t[1])

    print(
        f"Calibration fitted. Global QWK (OOF)={best_global[2]:.5f} with a={best_global[0]:.3f}, b={best_global[1]:.3f}. "
        f"Per-task params learned for {len(task_params)}/{len(unique_tasks)} tasks."
    )
    return (best_global[0], best_global[1]), task_params


def apply_taskwise_linear_calibration(pred_cont_0_5, tasks, global_params, task_params):
    a_g, b_g = global_params
    tasks = tasks.astype(int).reshape(-1)
    y = pred_cont_0_5.astype(np.float32).reshape(-1).copy()

    for t in np.unique(tasks):
        t = int(t)
        idx = np.where(tasks == t)[0]
        a, b = task_params.get(t, (a_g, b_g))
        y[idx] = a * y[idx] + b
    return y.reshape(-1, 1).astype(np.float32)


EMB_DIM = int(test_emb.shape[1])

trainer = L.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    logger=False,
    enable_checkpointing=False,
    inference_mode=True,  # stable inference; does not change model logic
)

global_calib, task_calib = fit_taskwise_linear_calibration_oof(
    train_emb=train_emb,
    y_train=train_df["labels"].values,
    tasks=train_df["task"].values,
    k=25,
    n_folds=5,
)

if len(ckpt_paths) == 0:
    print(
        "No .ckpt files found under /kaggle/input/**/*.ckpt; using task-restricted kNN fallback instead of random AESNet."
    )
    test_pred = predict_knn_by_task(
        train_emb=train_emb,
        train_labels=train_df["labels"].values,
        train_tasks=train_df["task"].values,
        test_emb=test_emb,
        test_tasks=test_df["task"].values,
        k=25,
    )
    model_count = 1
else:
    print(f"Found {len(ckpt_paths)} checkpoint(s). Predicting ensemble...")
    for path in ckpt_paths:
        print(path)

        try:
            model = AESNet.load_from_checkpoint(path, input_dim=EMB_DIM)
        except Exception as e:
            print(f"Skipping checkpoint due to load mismatch: {path}\n  {repr(e)}")
            continue

        model.eval()
        output = trainer.predict(model=model, dataloaders=test_dl)
        output = stack_predict_outputs(output)

        preds = aggregate_preds(output, test_ds.pairs, len(test_df))
        test_pred += preds
        model_count += 1

    if model_count == 0:
        print(
            "All checkpoints were skipped due to incompatibility; using task-restricted kNN fallback instead of random AESNet."
        )
        test_pred = predict_knn_by_task(
            train_emb=train_emb,
            train_labels=train_df["labels"].values,
            train_tasks=train_df["task"].values,
            test_emb=test_emb,
            test_tasks=test_df["task"].values,
            k=25,
        )
        model_count = 1
    else:
        test_pred = test_pred / max(model_count, 1)

test_pred = apply_taskwise_linear_calibration(
    pred_cont_0_5=test_pred,
    tasks=test_df["task"].values,
    global_params=global_calib,
    task_params=task_calib,
)



## === cell 8
test_pred = np.round(test_pred, 0).astype(int) + 1
test_pred = np.clip(test_pred, 1, 6)

submission_df = pd.DataFrame(
    {"essay_id": test_df["essay_id"].values, "score": test_pred.flatten()}
)

if len(sample_sub) == len(submission_df) and set(sample_sub["essay_id"]) == set(
    submission_df["essay_id"]
):
    submission_df = sample_sub[["essay_id"]].merge(
        submission_df, on="essay_id", how="left"
    )
else:
    submission_df = submission_df.sort_values("essay_id").reset_index(drop=True)

submission_df.to_csv("submission.csv", index=False)
print(submission_df.head())
print("Wrote submission.csv with shape:", submission_df.shape)
print("score value counts:\n", submission_df["score"].value_counts().sort_index())
