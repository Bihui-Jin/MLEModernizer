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

0.6861

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00197) has done: 'I fix the hard failures by (1) removing the invalid local model path usage and consistently loading `bert-base-uncased` from the Hugging Face cache (offline-safe), (2) avoiding the `datasets`-protobuf crash by not using `load_dataset` at all (reading CSVs with pandas instead), and (3) producing predictions in the correct 1–6 integer range with a proper `submission.csv`. To keep the core logic intact, I preserve your BERT sequence-classification approach but ensure `num_labels=6` and map labels 1–6 ↔ 0–5 correctly. I also replace the `pipeline`-based inference (slow and sometimes label-string brittle) with direct batched model inference for speed and stability within the 600s limit. The PCA visualization cells are kept but made non-blocking and safe so they won’t crash the run.'
- What this solution (achieved 0.65066) has done: 'I fix the runtime crash caused by an incompatibility between `transformers` and the installed `protobuf` by forcing the pure-Python protobuf implementation before importing `transformers`. Then I enable the existing training loop (it’s currently disabled with `is_submission=True`), because predicting with an unfine-tuned BERT classifier is what’s driving the extremely low QWK score; this preserves the same model and loss but actually trains it on the provided labels. Finally, I keep the same 1–6 label mapping and submission-writing logic, ensuring the script runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.65687) has done: 'I fix the immediate runtime crash in `transformers` caused by an incompatible protobuf backend by switching to the safe `python` protobuf implementation *and* forcing `transformers`/`protobuf` to be imported in the right order (before any `transformers` submodules are touched). Then I make the training stage actually optimize the metric-aligned target by keeping your 6-class classifier but switching inference from hard argmax to an ordinal-aware expectation over class probabilities (a minimal post-processing change that typically improves QWK for ordered labels). Finally, I ensure submission alignment is correct and fully filled (no NaNs), and that the CSV is written with the required columns and `.csv` suffix.'
- What this solution (achieved 0.65812) has done: 'I fix the immediate crash by removing the explicit `google.protobuf` import and instead forcing the pure-Python protobuf backend *and* pinning a compatible `protobuf` version via `pip` before importing anything from `transformers` (this resolves the `MessageFactory.GetPrototype` error in this Kaggle image). I keep your exact BERT 6-class classification + cross-entropy training loop and the ordinal expectation inference, but make two minimal stability/score-positive fixes: use a proper linear LR scheduler with warmup (common for BERT finetuning and doesn’t change the core approach) and add a simple validation QWK computation to confirm training is working (no early stopping). Finally, I ensure the submission is fully aligned to `sample_submission.csv` and written as a valid `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.68171) has done: 'Your current approach is close to a reasonable baseline but is likely held back by two small, metric-relevant issues: (1) treating an ordinal target (1–6) as plain 6-way classification without class-balance handling, and (2) rounding the expected value without any calibration to QWK. To move your score upward toward the 0.781 target while keeping the same BERT 6-class cross-entropy core logic, I add minimal class-weighting in the loss (via `model.config.problem_type="single_label_classification"` + passing `weight` to `CrossEntropyLoss` externally) and a tiny post-processing calibration step: learn optimal per-class cutpoints on the validation set for converting the expected score to integer labels (this preserves your expectation-based inference but aligns discretization to QWK). I also keep everything deterministic and ensure the submission stays aligned to `sample_submission.csv` with the required columns and range 1–6. These changes are small, don’t alter the model architecture or training loop structure, and are directly aimed at improving QWK.'
- What this solution (achieved 0.69162) has done: 'Your current gap to target is substantial (0.68171 → 0.78114), so the most likely “small but meaningful” win without changing the model/loop is to fix label-discretization to better match QWK. I keep your exact BERT 6-class cross-entropy fine-tuning and expectation inference, but (1) compute validation expected scores every epoch and select the best epoch’s cutpoints by QWK (instead of using only the final epoch), and (2) replace the coarse grid tweak with a tiny, deterministic coordinate-descent refinement of the 5 cutpoints on the validation set (still just post-processing). This keeps architecture/training semantics intact while directly optimizing the only part that converts continuous expectations into the required 1–6 integers for QWK. The script still runs end-to-end and writes `/kaggle/working/submission.csv` with the required schema.'
- What this solution (achieved 0.68328) has done: 'You’re substantially below the target (0.69162 vs 0.78114), so we should nudge performance upward with the smallest changes that directly affect QWK without changing the model architecture or loss. I keep your exact BERT 6-class cross-entropy fine-tuning and expectation-based inference, but fix an important mismatch: your class weights are computed from the train split, yet CrossEntropyLoss expects weights aligned to label indices 0–5 (you currently align them to scores 1–6). I also increase `num_epochs` from 3 to 4 (same training loop/optimizer/scheduler, just one more pass) to reduce underfitting, which is a minimal, metric-relevant change given the large score gap. Everything else (cutpoint tuning, inference, submission formatting, paths) stays the same.'
- What this solution (achieved 0.68271) has done: 'The timeout is dominated by (1) pre-tokenizing 139k essays at `MAX_LEN=256` into dense tensors (very large CPU RAM and time), and (2) running 4 full BERT fine-tuning epochs over that dataset. To preserve the exact model/training logic and evaluation semantics, the main speed win is to avoid the huge up-front tokenization and instead tokenize on-the-fly in the DataLoader (same tokenizer settings), while also enabling the fastest safe kernels (TF32 on GPU) and using `torch.inference_mode()` for eval. These changes are provably equivalent at the algorithm level (same inputs, same loss, same epochs/steps, same decoding/calibration), but remove massive redundant work and memory pressure that causes slowdowns/timeouts. I keep all file paths, splits, epochs, batch sizes, and calibration logic unchanged.'
- What this solution (achieved 0.68353) has done: 'I fix the `MessageFactory.GetPrototype` crash by setting the protobuf implementation env vars *before Python imports anything else* and by forcing a compatible protobuf version at runtime (a common Kaggle image mismatch with recent `transformers`). This is a correctness/stability fix that unblocks model/tokenizer loading and keeps your exact BERT 6-class training/inference logic unchanged. I also make the data loader settings compatible across environments (avoid `prefetch_factor=None` when `num_workers=0`) to prevent occasional runtime errors. No score-degrading changes are introduced; once it runs, the same training + ordinal expectation + cutpoint calibration should continue pushing QWK upward toward the target.'
- What this solution (achieved 0.68213) has done: 'Your current score (0.68353) is well below the target (0.78114), so we should make small, metric-aligned changes that don’t alter the core BERT classification training loop. The biggest likely gain with minimal risk is to stop “fighting” the ordinal nature of the labels at inference by training a tiny, deterministic *QWK-optimized thresholding* on the validation set directly from the model’s expected score (you already do this, but we can make it stronger by also optimizing the *class-value mapping* in the expectation step while keeping 6-way cross-entropy unchanged). Concretely, we keep the exact same model/optimizer/scheduler/epochs, but (1) learn per-class “expected value anchors” (6 monotone scalars) on the validation set for the expectation computation, and (2) then re-run the same cutpoint calibration on top—this usually improves QWK without changing architecture or loss. Finally, we apply the learned anchors + calibration consistently on test and keep submission alignment unchanged.'
- What this solution (achieved 0.67748) has done: 'We keep your exact BERT 6-class fine-tuning + CE-loss core, but make two small, metric-aligned changes that typically improve QWK without changing architecture or loops: (1) tune the cutpoints directly on the validation set to maximize QWK using *the already learned anchors+calibration* (you do this), and additionally (2) apply a simple, deterministic “round-to-nearest-class then cutpoint” ensemble in validation to pick whichever discretization gives higher QWK and then use that same choice for test. This is minimal post-processing only (no new models, no extra data, no early stopping), and it should nudge the score upward toward your 0.781 target. We also ensure submission alignment stays exactly with `sample_submission.csv` and that protobuf/transformers loading remains stable.'
- What this solution (achieved 0.6861) has done: 'Your current score is well below the target (0.67748 vs 0.78114), so we should make the smallest metric-aligned change that can plausibly improve QWK without changing your BERT+CE training core. The biggest low-risk issue here is that you are training and validating on a random stratified split; for this competition, essays come from multiple prompts/domains, and mixing prompts across splits often produces thresholds/anchors that don’t generalize well to the hidden test distribution. I keep the same model, loss, epochs, optimizer, scheduler, inference, and calibration logic, but change the validation split to be prompt-aware using an unsupervised prompt proxy (TF‑IDF + KMeans clusters) and stratify by score *within* those clusters to reduce leakage and improve generalization. This is a data-splitting change only (no architecture/training-loop rewrite) and should move the public score upward toward your target.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import sys
import subprocess

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.3"],
    check=True,
)

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
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer



## === cell 5
MODEL_NAME = "bert-base-uncased"

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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
principal_components = np.zeros((0, 3), dtype=np.float32)
scores = []



## === cell 7
pass



## === cell 8
import gc

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()



## === cell 9
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




## === cell 10
is_submission = False

output_dir = "/kaggle/working/results"
os.makedirs(output_dir, exist_ok=True)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans

MAX_LEN = 256

tfidf = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    strip_accents="unicode",
    lowercase=True,
)
X = tfidf.fit_transform(train_df["full_text"].astype(str).values)

n_clusters = (
    8  # small, deterministic; approximates prompt groups without being too granular
)
kmeans = MiniBatchKMeans(
    n_clusters=n_clusters,
    random_state=42,
    batch_size=4096,
    n_init="auto",
    max_iter=100,
)
clusters = kmeans.fit_predict(X)

tmp = train_df.copy()
tmp["_cluster"] = clusters

tmp["_strat"] = tmp["_cluster"].astype(str) + "_" + tmp["score"].astype(int).astype(str)

vc = tmp["_strat"].value_counts()
rare_keys = set(vc[vc < 2].index.tolist())
if len(rare_keys) > 0:
    tmp.loc[tmp["_strat"].isin(rare_keys), "_strat"] = tmp.loc[
        tmp["_strat"].isin(rare_keys), "_cluster"
    ].astype(str)

train_tr, train_va = train_test_split(
    tmp.drop(columns=["_cluster", "_strat"]),
    train_size=0.8,
    random_state=42,
    stratify=tmp["_strat"],
)

train_ds = EssayDataset(train_tr, max_length=MAX_LEN, with_labels=True)
valid_ds = EssayDataset(train_va, max_length=MAX_LEN, with_labels=True)

num_workers = min(4, (os.cpu_count() or 2))
pin = torch.cuda.is_available()

train_loader_kwargs = dict(
    batch_size=16,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin,
    persistent_workers=(num_workers > 0),
    collate_fn=make_collate_fn(tokenizer, MAX_LEN, with_labels=True),
)
valid_loader_kwargs = dict(
    batch_size=32,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin,
    persistent_workers=(num_workers > 0),
    collate_fn=make_collate_fn(tokenizer, MAX_LEN, with_labels=True),
)

if num_workers > 0:
    train_loader_kwargs["prefetch_factor"] = 4
    valid_loader_kwargs["prefetch_factor"] = 4

train_loader = DataLoader(train_ds, **train_loader_kwargs)
valid_loader = DataLoader(valid_ds, **valid_loader_kwargs)

len(train_ds), len(valid_ds)



## === cell 11
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

    def fit_monotone_anchors_from_probs(
        probs_6, y_true_1_6, init_anchors=None, n_passes=8, step=0.05, min_gap=0.05
    ):
        anchors = (
            np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
            if init_anchors is None
            else init_anchors.astype(np.float32).copy()
        )

        def enforce_anchor_order(a):
            a = a.astype(np.float32).copy()
            a[0] = max(1.0, float(a[0]))
            a[-1] = min(6.0, float(a[-1]))
            for k in range(1, 6):
                a[k] = max(a[k], a[k - 1] + min_gap)
            for k in range(4, -1, -1):
                a[k] = min(a[k], a[k + 1] - min_gap)
            a = np.clip(a, 1.0, 6.0)
            for k in range(1, 6):
                a[k] = max(a[k], a[k - 1] + min_gap)
            for k in range(4, -1, -1):
                a[k] = min(a[k], a[k + 1] - min_gap)
            return a

        anchors = enforce_anchor_order(anchors)
        exp0 = (probs_6 * anchors[None, :]).sum(axis=1)
        cuts, best_q = refine_cutpoints_coordinate_descent(
            exp0, y_true_1_6, np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float32)
        )

        for _ in range(n_passes):
            improved = False
            for j in range(6):
                cur = float(anchors[j])
                best_local_q = best_q
                best_local_a = None
                best_local_c = None

                for delta in (-2, -1, 0, 1, 2):
                    cand_a = anchors.copy()
                    cand_a[j] = cur + delta * step
                    cand_a = enforce_anchor_order(cand_a)

                    exp = (probs_6 * cand_a[None, :]).sum(axis=1)
                    cand_c, q = refine_cutpoints_coordinate_descent(
                        exp, y_true_1_6, cuts, n_passes=4, step=0.02
                    )
                    if q > best_local_q:
                        best_local_q = q
                        best_local_a = cand_a
                        best_local_c = cand_c

                if best_local_a is not None and best_local_q > best_q + 1e-12:
                    anchors = best_local_a
                    cuts = best_local_c
                    best_q = best_local_q
                    improved = True
            if not improved:
                break

        return anchors.astype(np.float32), cuts.astype(np.float32), float(best_q)

    def discretize_with_best_rule(exp_scores_1_6, cuts_5, y_true_1_6):
        pred_cut = apply_cutpoints(exp_scores_1_6, cuts_5)
        q_cut = cohen_kappa_score(y_true_1_6, pred_cut, weights="quadratic")

        pred_round = np.clip(np.rint(exp_scores_1_6), 1, 6).astype(int)
        q_round = cohen_kappa_score(y_true_1_6, pred_round, weights="quadratic")

        if q_cut >= q_round:
            return "cut", pred_cut, float(q_cut), float(q_round)
        else:
            return "round", pred_round, float(q_cut), float(q_round)

    counts_0_5 = train_tr["score"].astype(int).sub(1).value_counts().sort_index()
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
    best_anchors = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    best_rule = "cut"

    non_blocking = bool(torch.cuda.is_available())

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            batch = {
                k: v.to(device, non_blocking=non_blocking) for k, v in batch.items()
            }
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
        va_probs = []
        va_exp_default = []

        with torch.inference_mode():
            for batch in valid_loader:
                labels_1_6 = (batch["labels"].numpy() + 1).astype(np.int64)
                batch_inp = {
                    k: v.to(device, non_blocking=non_blocking)
                    for k, v in batch.items()
                    if k != "labels"
                }
                logits = model(**batch_inp).logits
                probs = torch.softmax(logits, dim=-1)
                exp_score = (probs * class_values).sum(dim=-1)
                va_probs.append(probs.cpu().numpy())
                va_exp_default.append(exp_score.cpu().numpy())
                va_true.append(labels_1_6)

        va_true = np.concatenate(va_true)
        va_probs = np.concatenate(va_probs, axis=0)
        va_exp_default = np.concatenate(va_exp_default)

        anchors_fit, cuts_fit, qwk_anchors = fit_monotone_anchors_from_probs(
            va_probs,
            va_true,
            init_anchors=np.array([1, 2, 3, 4, 5, 6], dtype=np.float32),
        )
        va_exp = (va_probs * anchors_fit[None, :]).sum(axis=1)

        alpha_grid = np.array([0.95, 1.00, 1.05], dtype=np.float32)
        beta_grid = np.array([-0.10, 0.00, 0.10], dtype=np.float32)

        best_local_qwk = -1.0
        best_local_alpha = 1.0
        best_local_beta = 0.0
        best_local_cuts = cuts_fit.copy()
        best_local_anchors = anchors_fit.copy()
        best_local_rule = "cut"

        for a in alpha_grid:
            for b in beta_grid:
                exp_cal = np.clip(a * va_exp + b, 1.0, 6.0)
                cuts2, q2 = refine_cutpoints_coordinate_descent(
                    exp_cal, va_true, cuts_fit, n_passes=4, step=0.02
                )

                rule, _, q_cut, q_round = discretize_with_best_rule(
                    exp_cal, cuts2, va_true
                )
                q_best = max(q_cut, q_round)

                if q_best > best_local_qwk:
                    best_local_qwk = float(q_best)
                    best_local_alpha = float(a)
                    best_local_beta = float(b)
                    best_local_cuts = cuts2.copy()
                    best_local_anchors = anchors_fit.copy()
                    best_local_rule = rule

        pred_round_default = np.clip(np.rint(va_exp_default), 1, 6).astype(int)
        qwk_round_default = cohen_kappa_score(
            va_true, pred_round_default, weights="quadratic"
        )

        print(
            f"epoch {epoch+1}/{num_epochs} - train_loss: {avg_loss:.4f} - "
            f"val_qwk_round_default: {qwk_round_default:.5f} - val_qwk_anchors: {qwk_anchors:.5f} - "
            f"val_qwk_bestdisc: {best_local_qwk:.5f} (rule={best_local_rule}, alpha={best_local_alpha:.2f}, beta={best_local_beta:+.2f})"
        )

        if best_local_qwk > best_epoch_qwk:
            best_epoch_qwk = float(best_local_qwk)
            best_cuts = best_local_cuts.copy()
            best_alpha = float(best_local_alpha)
            best_beta = float(best_local_beta)
            best_anchors = best_local_anchors.copy()
            best_rule = str(best_local_rule)

        model.train()

    print(
        "best validation:",
        {
            "anchors": best_anchors.tolist(),
            "cuts": best_cuts.tolist(),
            "alpha": best_alpha,
            "beta": best_beta,
            "rule": best_rule,
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
    best_anchors = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    best_rule = "cut"



## === cell 12
test_ds = EssayDataset(test_df, max_length=MAX_LEN, with_labels=False)

test_loader_kwargs = dict(
    batch_size=32,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin,
    persistent_workers=(num_workers > 0),
    collate_fn=make_collate_fn(tokenizer, MAX_LEN, with_labels=False),
)
if num_workers > 0:
    test_loader_kwargs["prefetch_factor"] = 4

test_loader = DataLoader(test_ds, **test_loader_kwargs)

model.eval()
all_pred_scores = []

anchors_t = torch.tensor(best_anchors, device=device, dtype=torch.float32)
non_blocking = bool(torch.cuda.is_available())

with torch.inference_mode():
    for batch in test_loader:
        batch = {k: v.to(device, non_blocking=non_blocking) for k, v in batch.items()}
        logits = model(**batch).logits
        probs = torch.softmax(logits, dim=-1)
        exp_score = (probs * anchors_t).sum(dim=-1)
        all_pred_scores.append(exp_score.cpu().numpy())

pred_scores = np.concatenate(all_pred_scores, axis=0)
pred_scores.shape, len(test_df)



## === cell 13
pred_scores_cal = np.clip(best_alpha * pred_scores + best_beta, 1.0, 6.0)

if best_rule == "round":
    pred_scores_int = np.clip(np.rint(pred_scores_cal), 1, 6).astype(int)
else:
    pred_scores_int = (np.digitize(pred_scores_cal, best_cuts) + 1).astype(int)
    pred_scores_int = np.clip(pred_scores_int, 1, 6).astype(int)

pred_scores_int.min(), pred_scores_int.max()



## === cell 14
sub = sample_df.copy()
pred_map = dict(zip(test_df["essay_id"].values, pred_scores_int))
sub["score"] = sub["essay_id"].map(pred_map)

sub["score"] = sub["score"].fillna(3).astype(float).round().astype(int)
sub["score"] = sub["score"].clip(1, 6)

sub.head(), sub.shape



## === cell 15
submission_path = "/kaggle/working/submission.csv"
sub.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission_path, "rows:", len(sub), "cols:", sub.columns.tolist())
print("score value counts:\n", sub["score"].value_counts().sort_index())
print("discretization rule used:", best_rule)
