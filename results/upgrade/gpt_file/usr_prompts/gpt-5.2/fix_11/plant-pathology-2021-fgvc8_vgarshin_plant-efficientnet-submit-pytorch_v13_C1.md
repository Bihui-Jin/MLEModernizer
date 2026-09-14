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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.8034164358264089

# 6. Current score

0.30565

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.23687) has done: 'The crash comes from building `df_sub` by listing the `test_images` directory, which includes nested folders like `test_images/` and produces a bogus “image” entry that can’t be read as a file. I fix this by reading `sample_submission.csv` to get the exact test image list (this also guarantees the submission has the correct number of rows), and I also filter `os.listdir` to only `.jpg` files as a safety fallback. Next, I make the image path join robust with `os.path.join` and ensure label mapping works whether `params.json` stores `LABELS` as `idx->label` or `label->idx`. These are execution/format fixes and should be score-neutral while ensuring a valid `submission.csv` is produced end-to-end.'
- What this solution (achieved 0.30565) has done: 'Your low score is most consistent with a label/threshold/post-processing mismatch rather than a modeling issue: the current `get_labels` forces `"healthy"` whenever `"healthy"` is present, which can suppress true diseases and hurts mean F1 badly. I make the smallest metric-aligned change: only output `"healthy"` when *no* class exceeds the threshold, and otherwise output all predicted classes (including allowing diseased+healthy only if the model truly predicts both). To further move score upward without changing the model/training, I calibrate the decision threshold using a quick out-of-fold (train-set) pass through the same loaded model(s) (no training), selecting the threshold that maximizes sample-wise F1 on a small deterministic subset to stay within the time budget. The rest of the pipeline (model, weights loading, TTA, averaging, submission formatting/paths) remains the same.'
- What this solution (achieved 0.35308) has done: 'Your current gap to the target is large (0.30565 vs 0.8034), and the most likely cause without changing the model is overly strict post-processing that outputs no labels too often or misses the required `healthy` behavior. I keep your model/weights/TTA exactly the same, but replace the single global threshold with a minimal class-wise threshold calibration computed from the same quick train-subset pass you already do (this often gives a big mean-F1 lift for imbalanced multilabel tasks). I also add a tiny, metric-aligned fallback: if nothing exceeds thresholds, output `healthy`; otherwise output predicted labels (unchanged semantics). Finally, I make the calibration deterministic and use the already-loaded `train.csv` labels mapping to avoid any label-order mismatch that can silently kill F1.'
- What this solution (achieved 0.35308) has done: 'Your score is far below the target, so the safest way to move upward without changing the model is to fix post-processing so it matches the competition’s multilabel F1: we should average *logits* across TTA/models and apply `sigmoid` once (your current code averages probabilities, which is typically worse calibrated). Next, keep your per-class threshold calibration but make it consistent with that logit-averaging (calibrate on averaged logits too), and expand the threshold search grid slightly to find better operating points without increasing runtime much. Finally, ensure the submission label list is deterministic and uses descending confidence to help consistency (score-neutral but stability-improving).'
- What this solution (achieved 0.30565) has done: 'Your current score is far below the target, so we should move upward with the smallest changes that improve metric-aligned post-processing without changing the model/training. The biggest likely issue is that per-class threshold calibration is computed on single-view train images (tta=0) but inference uses 4-TTA averaged logits; this mismatch often hurts multilabel F1, so I calibrate thresholds on the same TTA-averaged logits used at test time (still no training, same models). I also avoid an order bias from taking the “first N rows” of train.csv by selecting a deterministic, label-stratified subset (same size) for calibration, which usually improves threshold reliability at essentially the same runtime. Finally, I keep the exact submission formatting and the “healthy if nothing predicted” rule unchanged.'
- What this solution (achieved 0.30565) has done: 'Your current score is far below target, so we should move it upward with minimal, metric-aligned post-processing changes while keeping your model, weights loading, and TTA/logit-averaging intact. The largest remaining score sink is likely label-order mismatch: you build `LABELS_`/`LABEL2IDX` from the model’s params (or a sorted fallback), but the model head order is defined by training; if that order doesn’t exactly match at inference, predictions map to wrong class names and F1 collapses. I (1) force label order to come from `train.csv` consistently (competition canonical set) unless `params.json` explicitly provides an ordered `labels_`, (2) rebuild `LABELS` from that order so `idx->label` always matches the probability vector indices, and (3) ensure calibration uses the same label order as inference. These are minimal changes that preserve the core inference logic and should materially increase mean F1 toward your target if mismatch was the issue.'
- What this solution (achieved 0.30565) has done: 'Your score gap to the target is still large, so the most likely remaining issue is that the model’s output index order still doesn’t match the label names you write into the submission (this can keep mean F1 very low even with a good model). I make a minimal, inference-only fix: detect and use the label order stored in the trained checkpoint (common as `label_names`/`classes`/`labels`) to build `LABELS_`/`LABEL2IDX`, falling back to your current `params.json`/`train.csv` logic only if the checkpoint provides nothing. I also ensure the calibration subset uses the exact same label order and that “healthy” is only emitted when no class passes threshold (already correct), keeping your model/TTA/logit-averaging unchanged. These changes are directly targeted at correcting class-name mapping and should move the score upward toward the target without altering the core model or training.'
- What this solution (achieved 0.30565) has done: 'Your current score is far below the target, so we should move it upward with the smallest inference-only changes that are most likely to fix a silent label-order mismatch. Right now the model head size is set from `LABELS_` *before* loading the checkpoint, so even if the checkpoint contains a different class order (or a different set like excluding `healthy`), you can end up mapping probabilities to the wrong label names and/or calibrating thresholds on the wrong indices, collapsing mean-F1. I (1) infer `out_dim` directly from the checkpoint tensor shapes before instantiating the model head, (2) rebuild `LABELS_` to match that `out_dim` using checkpoint metadata when available, otherwise falling back to a deterministic `train.csv`-based order trimmed/padded to `out_dim`, and (3) keep your same TTA/logit-averaging and per-class threshold calibration, but ensure calibration/inference share the exact same label order and dimension. These are minimal, metric-aligned fixes that preserve your core logic while targeting the most probable cause of the low score.'
- What this solution (achieved 0.30565) has done: 'Your current score is far below the target, so we should move it upward with a minimal, inference-only fix that most commonly causes mean-F1 to collapse: a wrong label order/dimension mapping between the checkpoint head and the labels you write to submission. I (1) infer `out_dim` from the checkpoint and (2) derive the class order by *matching checkpoint head indices to label names using the checkpoint’s own per-class bias statistics* (robust even when the checkpoint doesn’t store label names), preferring a mapping consistent with `train.csv` label frequencies. This keeps your model architecture, weights, TTA, logit-averaging, and threshold calibration intact, but makes the “index -> label” mapping far more likely to be correct. Finally, I keep the “healthy if nothing predicted” rule unchanged and still write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import gc
import sys
import json
import time
import cv2
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data.sampler import SequentialSampler
import torchvision

torch.backends.cudnn.benchmark = True

KAGGLE = os.path.exists("/kaggle/input") or os.path.exists("../input")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)



## === cell 1
pass



## === cell 2
TEST = True
VER = "v5"
if KAGGLE:
    DATA_PATH = "../input/plant-pathology-2021-fgvc8"
    MDLS_PATH = f"../input/plant-models-{VER}"
else:
    DATA_PATH = "./data"
    MDLS_PATH = f"./models_{VER}"

TH = 0.5
TTAS = [0, 1, 2, 3]
FOLDS = [0]
IMGS_PATH = f"{DATA_PATH}/test_images" if TEST else f"{DATA_PATH}/train_images"

start_time = time.time()



## === cell 3
train_csv_path = os.path.join(DATA_PATH, "train.csv")
train_df_all = pd.read_csv(train_csv_path)

canonical_labels = sorted(
    {lbl for s in train_df_all["labels"].astype(str).tolist() for lbl in s.split()}
)

label_counts = {lbl: 0 for lbl in canonical_labels}
for s in train_df_all["labels"].astype(str).tolist():
    for lbl in s.split():
        if lbl in label_counts:
            label_counts[lbl] += 1
label_prevalence = np.array(
    [label_counts[lbl] for lbl in canonical_labels], dtype=np.float32
)

params_path = os.path.join(MDLS_PATH, "params.json")
params = {}
if os.path.exists(params_path):
    with open(params_path) as file:
        params = json.load(file)
    WORKERS = 2 if KAGGLE else int(params.get("workers", 2))
else:
    params = {
        "backbone": "efficientnet_b0",
        "dropout": 0.2,
        "img_size": 512,
        "batch_size": 16,
        "workers": 2,
    }
    WORKERS = 2 if KAGGLE else 2

LABELS_ = params.get("labels_", None)
if LABELS_ is None or not isinstance(LABELS_, (list, tuple)) or len(LABELS_) == 0:
    LABELS_ = canonical_labels
else:
    LABELS_ = [str(x) for x in LABELS_]

LABELS = {str(i): lbl for i, lbl in enumerate(LABELS_)}
LABEL2IDX = {lbl: i for i, lbl in enumerate(LABELS_)}

print("loaded params:", params)
print("initial num classes (pre-ckpt):", len(LABELS_))
print("initial LABELS_ (pre-ckpt, first 20):", LABELS_[:20])
print("DEVICE:", DEVICE)



## === cell 4
sub_path = os.path.join(DATA_PATH, "sample_submission.csv")
if os.path.exists(sub_path):
    df_sub = pd.read_csv(sub_path)
    df_sub["labels"] = "healthy"
else:
    img_files = sorted([f for f in os.listdir(IMGS_PATH) if f.lower().endswith(".jpg")])
    df_sub = pd.DataFrame({"image": img_files})
    df_sub["labels"] = "healthy"

print(df_sub.head())
print("df_sub shape:", df_sub.shape)




## === cell 5
def flip(img, axis=0):
    if axis == 1:
        return img[::-1, :, :]
    elif axis == 2:
        return img[:, ::-1, :]
    elif axis == 3:
        return img[::-1, ::-1, :]
    else:
        return img


class PlantDataset(data.Dataset):
    def __init__(self, df, size, labels, transform=None, tta=0):
        self.df = df.reset_index(drop=True)
        self.size = int(size)
        self.labels = labels
        self.transform = transform
        self.tta = int(tta)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_name = row.image
        img_path = os.path.join(IMGS_PATH, img_name)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        img = img.astype(np.float32) / 255.0

        if self.labels:
            img = img.transpose(2, 0, 1)
            label = np.zeros(len(self.labels)).astype(np.float32)
            for lbl in str(row.labels).split():
                if lbl in self.labels:
                    label[self.labels[lbl]] = 1
            return torch.tensor(img), torch.tensor(label)
        else:
            img = flip(img, axis=self.tta)
            img = img.transpose(2, 0, 1)
            return torch.tensor(img.copy())




## === cell 6
class EffNet(nn.Module):
    def __init__(self, params, out_dim):
        super().__init__()
        backbone = params.get("backbone", "efficientnet_b0")

        if not hasattr(torchvision.models, backbone):
            raise ValueError(f"Unknown torchvision backbone: {backbone}")

        self.enet = getattr(torchvision.models, backbone)(weights=None)

        if hasattr(self.enet, "classifier") and isinstance(
            self.enet.classifier, nn.Sequential
        ):
            nc = self.enet.classifier[-1].in_features
            self.enet.classifier = nn.Identity()
        else:
            raise RuntimeError(
                "Unexpected EfficientNet classifier structure in torchvision."
            )

        self.myfc = nn.Sequential(
            nn.Dropout(float(params.get("dropout", 0.2))),
            nn.Linear(nc, int(nc / 4)),
            nn.Dropout(float(params.get("dropout", 0.2))),
            nn.Linear(int(nc / 4), out_dim),
        )

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x




## === cell 7
def _extract_label_order_from_checkpoint(state_obj):
    """
    Change (score-directed): if the checkpoint stores the class names/order used in training,
    we should use it to map output indices -> labels. Otherwise F1 can collapse due to permuted labels.
    """
    if not isinstance(state_obj, dict):
        return None

    candidate_keys = [
        "label_names",
        "labels",
        "classes",
        "class_names",
        "target_names",
        "idx2label",
        "idx_to_label",
    ]
    for k in candidate_keys:
        if k in state_obj:
            v = state_obj[k]
            if isinstance(v, dict):
                try:
                    items = sorted(v.items(), key=lambda x: int(x[0]))
                    return [str(lbl) for _, lbl in items]
                except Exception:
                    pass
            if isinstance(v, (list, tuple)) and len(v) > 0:
                return [str(x) for x in v]

    for outer_k in ["meta", "metadata", "cfg", "config", "params", "hparams"]:
        if outer_k in state_obj and isinstance(state_obj[outer_k], dict):
            v = _extract_label_order_from_checkpoint(state_obj[outer_k])
            if v is not None:
                return v

    return None


def _infer_out_dim_from_state_dict(state_dict):
    """
    Change (score-directed): infer model head output dimension from checkpoint tensors so
    we never instantiate a wrong-sized head (which would force strict=False loads and can
    silently desync label mapping).
    """
    if not isinstance(state_dict, dict):
        return None
    for k in ["myfc.3.weight", "myfc.3.bias"]:
        if k in state_dict and hasattr(state_dict[k], "shape"):
            t = state_dict[k]
            if k.endswith(".weight") and len(t.shape) == 2:
                return int(t.shape[0])
            if k.endswith(".bias") and len(t.shape) == 1:
                return int(t.shape[0])
    return None


def _infer_label_order_from_bias(
    bias_vec,
    canonical_labels,
    label_prevalence,
    ensure_healthy_present=True,
):
    """
    Change (score-directed): If checkpoint doesn't store label names/order, infer a plausible
    mapping from head index -> label using bias statistics and train.csv prevalence.
    This aims to correct catastrophic label permutation with minimal inference-only logic.

    Heuristic:
    - In multilabel sigmoid heads, more prevalent classes tend to get higher bias (less negative).
    - So map highest bias indices to most prevalent labels.
    """
    bias_vec = np.asarray(bias_vec, dtype=np.float32).reshape(-1)
    out_dim = int(bias_vec.shape[0])

    labels = list(canonical_labels)

    if len(labels) != out_dim:
        if "healthy" in labels and len(labels) - 1 == out_dim:
            labels = [x for x in labels if x != "healthy"]
        else:
            if len(labels) > out_dim:
                labels = labels[:out_dim]
            else:
                labels = labels + [f"extra_{i}" for i in range(out_dim - len(labels))]

    prev_map = {
        lbl: float(label_prevalence[canonical_labels.index(lbl)])
        for lbl in canonical_labels
    }
    prev = np.array([prev_map.get(lbl, 0.0) for lbl in labels], dtype=np.float32)

    idx_by_bias = np.argsort(-bias_vec)  # high -> low
    lbl_by_prev = np.argsort(-prev)  # high -> low prevalence

    inferred = [None] * out_dim
    for rank in range(out_dim):
        inferred[int(idx_by_bias[rank])] = str(labels[int(lbl_by_prev[rank])])

    if (
        ensure_healthy_present
        and "healthy" in canonical_labels
        and "healthy" not in inferred
    ):
        inferred[int(idx_by_bias[-1])] = "healthy"

    used = set()
    for i in range(out_dim):
        if inferred[i] in used:
            inferred[i] = f"extra_fix_{i}"
        used.add(inferred[i])

    return inferred


models = []
ckpt_label_order = None
inferred_out_dim = None
ckpt_bias = None

for n_fold in FOLDS:
    path = f"{MDLS_PATH}/model_best_{n_fold}.pth"
    state_obj = None
    state_dict = None

    if os.path.exists(path):
        state_obj = torch.load(path, map_location="cpu")
        ckpt_label_order = ckpt_label_order or _extract_label_order_from_checkpoint(
            state_obj
        )

        state_dict = (
            state_obj["state_dict"]
            if isinstance(state_obj, dict) and "state_dict" in state_obj
            else state_obj
        )
        inferred_out_dim = inferred_out_dim or _infer_out_dim_from_state_dict(
            state_dict
        )

        if (
            ckpt_bias is None
            and isinstance(state_dict, dict)
            and "myfc.3.bias" in state_dict
        ):
            try:
                ckpt_bias = (
                    state_dict["myfc.3.bias"]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                    .reshape(-1)
                )
            except Exception:
                ckpt_bias = None

        print("loaded checkpoint object:", path)
    else:
        print("WARNING: model weights not found, using untrained model:", path)

    out_dim = int(inferred_out_dim) if inferred_out_dim is not None else len(LABELS_)
    model = EffNet(params, out_dim=out_dim)

    if state_dict is not None:
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            new_state = {}
            for k, v in state_dict.items():
                nk = (
                    k.replace("module.", "")
                    if isinstance(k, str) and k.startswith("module.")
                    else k
                )
                new_state[nk] = v
            model.load_state_dict(new_state, strict=False)

    model.float().eval().to(DEVICE)
    models.append(model)

    del state_obj, state_dict
    gc.collect()

final_out_dim = models[0].myfc[-1].out_features if len(models) > 0 else len(LABELS_)

if (
    ckpt_label_order is not None
    and isinstance(ckpt_label_order, list)
    and len(ckpt_label_order) == final_out_dim
):
    LABELS_ = [str(x) for x in ckpt_label_order]
    print("Using label order from checkpoint metadata:", LABELS_)
elif ckpt_bias is not None and int(len(ckpt_bias)) == int(final_out_dim):
    LABELS_ = _infer_label_order_from_bias(
        bias_vec=ckpt_bias,
        canonical_labels=canonical_labels,
        label_prevalence=label_prevalence,
        ensure_healthy_present=True,
    )
    print(
        "Using label order inferred from checkpoint bias + train prevalence (first 20):",
        LABELS_[:20],
    )
else:
    base = list(canonical_labels)
    if len(base) != final_out_dim:
        if "healthy" in base and len(base) - 1 == final_out_dim:
            base = [x for x in base if x != "healthy"]
        else:
            if len(base) > final_out_dim:
                base = base[:final_out_dim]
            else:
                base = base + [f"extra_{i}" for i in range(final_out_dim - len(base))]
    LABELS_ = [str(x) for x in base]
    if ckpt_label_order is not None:
        print(
            "Checkpoint label order found but incompatible; using train.csv-derived order. "
            f"(found={len(ckpt_label_order)}, expected={final_out_dim})"
        )
    print("Final LABELS_ (aligned to out_dim):", LABELS_)

LABELS = {str(i): lbl for i, lbl in enumerate(LABELS_)}
LABEL2IDX = {lbl: i for i, lbl in enumerate(LABELS_)}

print("final num classes:", len(LABELS_))
print("final_out_dim:", final_out_dim)



## === cell 8
datasets, loaders = [], []
for tta in TTAS:
    dataset = PlantDataset(
        df=df_sub, size=params["img_size"], labels=None, transform=None, tta=tta
    )
    datasets.append(dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(params["batch_size"]),
        sampler=SequentialSampler(dataset),
        num_workers=int(WORKERS),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    loaders.append(loader)




## === cell 9
def get_labels(row_prob, labels, th, class_order=None):
    if np.isscalar(th):
        idxs = np.where(row_prob > float(th))[0]
    else:
        th_vec = np.asarray(th, dtype=np.float32)
        if th_vec.shape[0] != row_prob.shape[0]:
            idxs = np.where(row_prob > float(TH if np.isscalar(TH) else 0.5))[0]
        else:
            idxs = np.where(row_prob > th_vec)[0]

    if idxs.size == 0:
        return "healthy"

    idxs = idxs[np.argsort(-row_prob[idxs])]

    labs = [labels.get(str(int(i)), None) for i in idxs]
    labs = [x for x in labs if x is not None]
    if len(labs) == 0:
        return "healthy"
    return " ".join(labs)


def _f1_samples(y_true, y_pred, eps=1e-12):
    tp = (y_true * y_pred).sum(axis=1)
    fp = ((1 - y_true) * y_pred).sum(axis=1)
    fn = (y_true * (1 - y_pred)).sum(axis=1)
    f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return float(f1.mean())


def _build_calib_subset(train_df, labels_list, label2idx, max_items, seed=42):
    rng = np.random.RandomState(int(seed))
    C = len(labels_list)
    y = np.zeros((len(train_df), C), dtype=np.uint8)
    for i, s in enumerate(train_df["labels"].astype(str).tolist()):
        for lbl in s.split():
            j = label2idx.get(lbl, None)
            if j is not None:
                y[i, j] = 1

    pos_counts = y.sum(axis=0).astype(np.int64)
    cls_order = np.argsort(np.where(pos_counts > 0, pos_counts, 10**9))

    chosen = set()
    per_class_quota = max(1, int(max_items) // max(1, min(C, 8)))
    for c in cls_order:
        if pos_counts[c] <= 0:
            continue
        idx = np.where(y[:, c] == 1)[0]
        if idx.size == 0:
            continue
        rng.shuffle(idx)
        take = min(per_class_quota, idx.size)
        for k in idx[:take]:
            chosen.add(int(k))
            if len(chosen) >= max_items:
                break
        if len(chosen) >= max_items:
            break

    if len(chosen) < max_items:
        remaining = np.setdiff1d(
            np.arange(len(train_df)),
            np.fromiter(chosen, dtype=np.int64),
            assume_unique=False,
        )
        rng.shuffle(remaining)
        need = max_items - len(chosen)
        for k in remaining[:need]:
            chosen.add(int(k))

    chosen = sorted(chosen)
    return train_df.iloc[chosen].copy()


def _predict_logits_with_tta(models, df, params, ttalist, imgs_path, workers, device):
    global IMGS_PATH
    old_imgs_path = IMGS_PATH
    IMGS_PATH = imgs_path

    local_loaders = []
    for tta in ttalist:
        ds = PlantDataset(
            df=df, size=params["img_size"], labels=None, transform=None, tta=int(tta)
        )
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=int(params["batch_size"]),
            sampler=SequentialSampler(ds),
            num_workers=int(workers),
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        local_loaders.append(dl)

    all_model_tta_logits = []
    with torch.no_grad():
        for model in models:
            tta_logits = []
            for loader in local_loaders:
                batch_logits = []
                for xb in loader:
                    xb = xb.to(device, non_blocking=True).float()
                    lg = model(xb).detach().cpu().numpy()
                    batch_logits.append(lg)
                tta_logits.append(np.concatenate(batch_logits, axis=0))
            all_model_tta_logits.append(np.stack(tta_logits, axis=0))  # (T,N,C)

    all_model_tta_logits = np.stack(all_model_tta_logits, axis=0)  # (M,T,N,C)
    logits_mean = all_model_tta_logits.mean(axis=(0, 1))  # (N,C)

    IMGS_PATH = old_imgs_path
    return logits_mean


def calibrate_thresholds_on_train(
    models,
    params,
    labels_list,
    label2idx,
    data_path,
    max_items=768,
    ttalist=(0, 1, 2, 3),
):
    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    n = min(len(train_df), int(max_items))

    train_df = _build_calib_subset(
        train_df, labels_list, label2idx, max_items=n, seed=SEED
    )

    y_true = np.zeros((len(train_df), len(labels_list)), dtype=np.float32)
    for i, s in enumerate(train_df["labels"].astype(str).tolist()):
        for lbl in s.split():
            if lbl in label2idx:
                y_true[i, label2idx[lbl]] = 1.0

    train_logits_mean = _predict_logits_with_tta(
        models=models,
        df=train_df,
        params=params,
        ttalist=ttalist,
        imgs_path=os.path.join(data_path, "train_images"),
        workers=WORKERS,
        device=DEVICE,
    )
    y_prob = 1.0 / (1.0 + np.exp(-train_logits_mean))

    grid = np.array(
        [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
        dtype=np.float32,
    )
    C = y_true.shape[1]
    ths = np.full((C,), float(TH if np.isscalar(TH) else 0.5), dtype=np.float32)

    y_pred = (y_prob > ths[None, :]).astype(np.float32)
    base_f1 = _f1_samples(y_true, y_pred)

    for c in range(C):
        best_th_c = ths[c]
        best_f1 = base_f1

        if y_true[:, c].sum() <= 0:
            continue

        for th_c in grid:
            y_pred_c = y_pred.copy()
            y_pred_c[:, c] = (y_prob[:, c] > th_c).astype(np.float32)
            f1 = _f1_samples(y_true, y_pred_c)
            if f1 > best_f1:
                best_f1 = f1
                best_th_c = float(th_c)

        ths[c] = best_th_c
        y_pred[:, c] = (y_prob[:, c] > ths[c]).astype(np.float32)
        base_f1 = best_f1

    return ths, base_f1


try:
    TH_vec, f1_cal = calibrate_thresholds_on_train(
        models=models,
        params=params,
        labels_list=LABELS_,
        label2idx=LABEL2IDX,
        data_path=DATA_PATH,
        max_items=768,
        ttalist=tuple(TTAS),
    )
    TH = TH_vec
    print(
        f"calibrated per-class thresholds on train subset (mean sample-F1≈{f1_cal:.4f})"
    )
    print("thresholds (first 10):", np.array(TH[:10]).round(3).tolist())
except Exception as e:
    print(
        "WARNING: per-class threshold calibration failed, using default scalar TH. Error:",
        repr(e),
    )

all_model_tta_logits = []
with torch.no_grad():
    for i, model in enumerate(models):
        tta_logits = []
        for j, loader in enumerate(loaders):
            batch_logits = []
            for img_data in loader:
                img_data = img_data.to(DEVICE, non_blocking=True).float()
                lg = model(img_data).detach().cpu().numpy()  # (B, C) logits
                batch_logits.append(lg)
            logits_loader = np.concatenate(batch_logits, axis=0)  # (N, C)
            tta_logits.append(logits_loader)
            print(f"model {i} | loader {j} -> done")
        tta_logits = np.stack(tta_logits, axis=0)  # (T, N, C)
        all_model_tta_logits.append(tta_logits)

all_model_tta_logits = np.stack(all_model_tta_logits, axis=0)  # (M, T, N, C)
logits_mean = all_model_tta_logits.mean(axis=(0, 1))  # (N, C)
probs = 1.0 / (1.0 + np.exp(-logits_mean))

df_sub["labels"] = [get_labels(x, LABELS, TH) for x in probs]

elapsed_time = time.time() - start_time
print(f"time elapsed: {elapsed_time // 60:.0f} min {elapsed_time % 60:.0f} sec")



## === cell 10
print("value counts:")
print(df_sub.labels.value_counts().head(20))
print(df_sub.head())



## === cell 11
df_sub = df_sub[["image", "labels"]]
df_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_sub.shape)
print(df_sub.head())
print("submission.csv exists:", os.path.exists("submission.csv"))
print("num rows:", len(df_sub))
