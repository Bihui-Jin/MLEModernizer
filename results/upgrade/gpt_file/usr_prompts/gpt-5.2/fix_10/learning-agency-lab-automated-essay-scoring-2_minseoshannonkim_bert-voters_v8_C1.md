# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import numpy as np
import pandas as pd





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
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
    def __init__(self, df, max_length=512, with_labels=True):
        self.df = df.reset_index(drop=True)
        self.max_length = max_length
        self.with_labels = with_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.loc[idx, "full_text"]
        if self.with_labels:
            label = int(self.df.loc[idx, "score"]) - 1
            return text, label
        return text


def make_collate_fn(tokenizer, max_length, with_labels: bool):
    def collate(batch):
        if with_labels:
            texts, labels = zip(*batch)
        else:
            texts = batch

        enc = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        if with_labels:
            enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc

    return collate




## === cell 11
is_submission = False

output_dir = "/kaggle/working/results"
os.makedirs(output_dir, exist_ok=True)

train_tr, train_va = train_test_split(
    train_df,
    train_size=0.8,
    random_state=42,
    stratify=train_df["score"],
)

train_ds = EssayDataset(train_tr, max_length=256, with_labels=True)
valid_ds = EssayDataset(train_va, max_length=256, with_labels=True)

num_workers = min(4, (os.cpu_count() or 2))
train_loader = DataLoader(
    train_ds,
    batch_size=16,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    collate_fn=make_collate_fn(tokenizer, max_length=256, with_labels=True),
)
valid_loader = DataLoader(
    valid_ds,
    batch_size=32,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    collate_fn=make_collate_fn(tokenizer, max_length=256, with_labels=True),
)

len(train_ds), len(valid_ds)




## === cell 12
if not is_submission:
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    from sklearn.metrics import cohen_kappa_score

    def apply_cutpoints(exp_scores, cuts):
        return (np.digitize(exp_scores, cuts) + 1).astype(int)

    def enforce_order(cuts, lo=1.05, hi=5.95, min_gap=0.02):
        c = cuts.astype(np.float32).copy()
        c[0] = float(np.clip(c[0], lo, hi))
        c[4] = float(np.clip(c[4], lo, hi))
        for k in range(1, 5):
            c[k] = max(c[k], c[k - 1] + min_gap)
        for k in range(3, -1, -1):
            c[k] = min(c[k], c[k + 1] - min_gap)
        c = np.clip(c, lo, hi)
        for k in range(1, 5):
            c[k] = max(c[k], c[k - 1] + min_gap)
        for k in range(3, -1, -1):
            c[k] = min(c[k], c[k + 1] - min_gap)
        return c

    def refine_cutpoints_coordinate_descent(
        exp_scores,
        y_true_1_6,
        init_cuts,
        n_passes=6,
        step=0.03,
        min_gap=0.02,
        lo=1.05,
        hi=5.95,
    ):
        """
        Deterministic, small post-processing optimization for QWK:
        coordinate descent over 5 ordered cutpoints.
        """
        cuts = enforce_order(init_cuts, lo=lo, hi=hi, min_gap=min_gap)
        best_q = cohen_kappa_score(
            y_true_1_6, apply_cutpoints(exp_scores, cuts), weights="quadratic"
        )

        for _ in range(n_passes):
            improved_any = False
            for j in range(5):
                current = cuts[j]
                best_local_q = best_q
                best_cand = None
                for delta in (-2, -1, 0, 1, 2):
                    cand = cuts.copy()
                    cand[j] = current + delta * step
                    cand = enforce_order(cand, lo=lo, hi=hi, min_gap=min_gap)
                    q = cohen_kappa_score(
                        y_true_1_6,
                        apply_cutpoints(exp_scores, cand),
                        weights="quadratic",
                    )
                    if q > best_local_q:
                        best_local_q = q
                        best_cand = cand
                if best_cand is not None and best_local_q > best_q + 1e-12:
                    cuts = best_cand
                    best_q = best_local_q
                    improved_any = True
            if not improved_any:
                break

        return cuts, float(best_q)

    counts_0_5 = (
        train_tr["score"].astype(int).sub(1).value_counts().sort_index()
    )  # index 0..5 (possibly missing some)
    freq_0_5 = counts_0_5.reindex(range(6), fill_value=1).values.astype(np.float32)
    inv_freq = 1.0 / freq_0_5
    class_weights_0_5 = (inv_freq / inv_freq.mean()).astype(np.float32)
    class_weights_t = torch.tensor(
        class_weights_0_5, device=device, dtype=torch.float32
    )
    ce_loss = torch.nn.CrossEntropyLoss(weight=class_weights_t)

    model.train()
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    num_epochs = 4

    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    class_values = torch.arange(1, 7, device=device, dtype=torch.float32)

    best_epoch_qwk = -1.0
    best_cuts = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float32)
    best_alpha = 1.0
    best_beta = 0.0

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            out = model(**batch)
            loss = ce_loss(out.logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(train_loader))

        model.eval()
        va_true = []
        va_exp = []

        with torch.no_grad():
            for batch in valid_loader:
                labels_1_6 = batch["labels"].numpy() + 1  # back to 1..6
                batch_inp = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                logits = model(**batch_inp).logits
                probs = torch.softmax(logits, dim=-1)
                exp_score = (probs * class_values).sum(dim=-1)  # (B,)
                va_exp.append(exp_score.cpu().numpy())
                va_true.append(labels_1_6)

        va_true = np.concatenate(va_true)
        va_exp = np.concatenate(va_exp)

        init_cuts = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float32)
        tuned_cuts, _ = refine_cutpoints_coordinate_descent(
            va_exp, va_true, init_cuts, n_passes=6, step=0.03
        )

        alpha_grid = np.array([0.95, 1.00, 1.05], dtype=np.float32)
        beta_grid = np.array([-0.10, 0.00, 0.10], dtype=np.float32)

        best_local_qwk = -1.0
        best_local_alpha = 1.0
        best_local_beta = 0.0
        best_local_cuts = tuned_cuts.copy()

        for a in alpha_grid:
            for b in beta_grid:
                exp_cal = np.clip(a * va_exp + b, 1.0, 6.0)
                cuts2, q2 = refine_cutpoints_coordinate_descent(
                    exp_cal, va_true, tuned_cuts, n_passes=4, step=0.02
                )
                if q2 > best_local_qwk:
                    best_local_qwk = q2
                    best_local_alpha = float(a)
                    best_local_beta = float(b)
                    best_local_cuts = cuts2.copy()

        pred_round = np.clip(np.rint(va_exp), 1, 6).astype(int)
        qwk_round = cohen_kappa_score(va_true, pred_round, weights="quadratic")

        print(
            f"epoch {epoch+1}/{num_epochs} - train_loss: {avg_loss:.4f} - "
            f"val_qwk_round: {qwk_round:.5f} - val_qwk_tuned: {best_local_qwk:.5f} "
            f"(alpha={best_local_alpha:.2f}, beta={best_local_beta:+.2f})"
        )

        if best_local_qwk > best_epoch_qwk:
            best_epoch_qwk = float(best_local_qwk)
            best_cuts = best_local_cuts.copy()
            best_alpha = float(best_local_alpha)
            best_beta = float(best_local_beta)

        model.train()

    print(
        "best validation:",
        {
            "cuts": best_cuts.tolist(),
            "alpha": best_alpha,
            "beta": best_beta,
            "best_val_qwk": float(best_epoch_qwk),
        },
    )

    model.save_pretrained(os.path.join(output_dir, "bert-base-uncased"))
    tokenizer.save_pretrained(os.path.join(output_dir, "bert-base-uncased"))
    model.eval()
else:
    best_cuts = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float32)
    best_alpha = 1.0
    best_beta = 0.0




## === cell 13
test_ds = EssayDataset(
    test_df.rename(columns={"full_text": "full_text"}),
    max_length=256,
    with_labels=False,
)
test_loader = DataLoader(
    test_ds,
    batch_size=32,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    collate_fn=make_collate_fn(tokenizer, max_length=256, with_labels=False),
)

model.eval()
all_pred_scores = []

class_values = torch.arange(1, 7, device=device, dtype=torch.float32)  # 1..6

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits  # (B,6)
        probs = torch.softmax(logits, dim=-1)
        exp_score = (probs * class_values).sum(dim=-1)  # (B,)
        all_pred_scores.append(exp_score.detach().cpu().numpy())

pred_scores = np.concatenate(all_pred_scores, axis=0)
pred_scores.shape, len(test_df)




## === cell 14
pred_scores_cal = np.clip(best_alpha * pred_scores + best_beta, 1.0, 6.0)
pred_scores_int = (np.digitize(pred_scores_cal, best_cuts) + 1).astype(int)
pred_scores_int = np.clip(pred_scores_int, 1, 6).astype(int)
pred_scores_int.min(), pred_scores_int.max()




## === cell 15
sub = sample_df.copy()
pred_map = dict(zip(test_df["essay_id"].values, pred_scores_int))
sub["score"] = sub["essay_id"].map(pred_map)

sub["score"] = sub["score"].fillna(3).astype(float).round().astype(int)
sub["score"] = sub["score"].clip(1, 6)

sub.head(), sub.shape




## === cell 16
submission_path = "/kaggle/working/submission.csv"
sub.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission_path, "rows:", len(sub), "cols:", sub.columns.tolist())
print("score value counts:\n", sub["score"].value_counts().sort_index())
