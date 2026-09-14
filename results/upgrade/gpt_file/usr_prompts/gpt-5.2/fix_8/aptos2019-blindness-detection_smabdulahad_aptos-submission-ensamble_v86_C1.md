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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.12

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
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
timm==1.0.19
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.8900593895168976

# 6. Current score

0.70296

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.03301) has done: 'I fix the pipeline so it runs end-to-end without relying on missing external model weight files (the current `/kaggle/input/aptos_ensamble-models/...` paths don’t exist, causing the empty-model-list cascade). The minimal safe fallback is to use a timm pretrained ImageNet backbone with the same 5-class head and run inference; this preserves the overall approach (timm model(s) + softmax + argmax) while ensuring a valid `submission.csv` is always produced. I also make model loading robust (skip missing files, map_location, handle checkpoints with `state_dict`) and fix the inference loop so it can’t `torch.cat()` an empty list. Finally, I ensure the submission rows align exactly with `test.csv` and that the output column names match Kaggle’s required format.'
- What this solution (achieved -0.04199) has done: 'Your current score is far below the target, and the main reason is that the fallback path uses an ImageNet-pretrained model with a fresh random 5-class head, which effectively produces near-random class predictions and destroys QWK. To move toward the target while preserving your inference-only ensemble core logic, I load an ImageNet-pretrained backbone and keep its pretrained head by using `num_classes=1000`, then map those 1000 logits to 5 DR classes via a fixed, deterministic binning of the expected severity (computed from the softmax over 1000 classes). I also switch preprocessing to the model’s own `timm` data config (still Resize/Normalize, but consistent with the chosen backbone) to stabilize predictions. These are minimal changes that keep the approach “timm model(s) + softmax + weighted averaging + final discrete label,” but should massively improve score versus random.'
- What this solution (achieved -0.07962) has done: 'Your current score is far below the target, so we should make the smallest change that turns the fallback from “near-random” into “reasonable” while preserving your inference-only timm approach. The main issue is that when ensemble checkpoints are missing, you keep the 1000-way ImageNet head and then use a heuristic mapping that doesn’t reflect DR severity; instead, we keep the backbone pretrained but switch to a proper 5-class head and use a deterministic, training-set-prior bias so predictions aren’t random. Concretely: use `pretrained=True, num_classes=5` (timm load pretrained backbone and randomly init head), then calibrate logits by adding `log(train_class_prior)` before softmax to stabilize class distribution and improve QWK without changing architecture/training loops. We also make the transform match the actual fallback model (not hardcoded efficientnet_b0) to avoid mismatch.'
- What this solution (achieved 0.06559) has done: 'Your score is far below the target, and the main reason is that when the external ensemble checkpoints are missing you fall back to an ImageNet-pretrained backbone with a randomly initialized 5-class head, which yields near-random predictions and very poor QWK. To move sharply toward the target while keeping the same core “timm model(s) + softmax + weighted averaging + argmax” inference logic, I keep the fallback model’s pretrained 1000-class head and convert its output to a 5-class distribution via a simple, deterministic intensity-based mapping (computed from the input image itself), then ensemble it exactly as before. This avoids any training or architecture changes and keeps runtime within limits, while making predictions strongly correlated with DR severity instead of random. I also make sure the mapping is stable (no randomness) and the submission stays aligned to `test.csv`.'
- What this solution (achieved 0.78147) has done: 'Your current score (0.06559) is far below the target (0.89006), and the main cause is that the fallback path never uses a DR-trained head: it ignores the ImageNet logits and predicts from a crude intensity heuristic, which won’t correlate well with DR severity. I keep your core inference-only “timm model(s) + softmax + weighted averaging + argmax” logic, but make the fallback produce a real 5-class prediction by (1) using the timm pretrained backbone with a 5-class head and (2) fitting only that head on the provided `train.csv`/images for 1 short epoch (backbone frozen) to get non-random DR logits. This is a minimal change in approach (still same architecture family and same inference semantics) but should move QWK sharply toward the target while keeping runtime under the 600s constraint. I also keep your class-prior logit bias (it’s aligned with QWK stability) and ensure the submission remains aligned to `test.csv`.'
- What this solution (achieved 0.75613) has done: 'Your current score (0.78147) is below the target (0.89006), so we should make a small, safe improvement that keeps your exact “timm model(s) + softmax + weighted average + argmax” inference semantics. The biggest gain with minimal disruption is to make the fallback head-training less noisy and more DR-aligned by (1) using the model’s *training* transforms during the 1-epoch head fit (but keeping your current *inference* transforms unchanged), and (2) applying standard class-balanced weights in the CrossEntropy loss so the head doesn’t collapse to the dominant class. These changes do not alter your architecture, do not add extra epochs, and should move QWK upward toward the target by improving ordinal class separation and distribution. Everything still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.70296) has done: 'We keep your exact ensemble/inference semantics and the 1‑epoch “train only the head” fallback, but make two minimal changes that typically move QWK up: (1) use a stratified train/val split and select the best head checkpoint by validation QWK (same 1 epoch total, no extra epochs), preventing head overfit/collapse; and (2) replace the fixed log-prior bias (which can over-push to majority class) with a small tunable prior strength chosen on the validation split, then applied at test-time. These are calibration-level tweaks (not architecture/loss/loop rewrites) and are aimed at nudging your 0.756 score upward toward 0.890 while staying stable and within time. The pipeline still runs end-to-end and writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import warnings
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tqdm import tqdm

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




## === cell 1
class BlindnessDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, test=False):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_id = self.annotations.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, img_id + ".png")
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.test:
            return image
        else:
            label = int(self.annotations.iloc[idx, 1])
            return image, label




## === cell 2
test_csv_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_root_dir = "/kaggle/input/aptos2019-blindness-detection/test_images"
train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
train_root_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"

test_df = pd.read_csv(test_csv_file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = min(4, (os.cpu_count() or 2))



## === cell 3
model_paths = {
    "resnet18": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/resnet18(WD_1e-3)_aptos.pth",
    "efficientnet_b1": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/efficentNet_b1.pth",
    "efficientnet_b2": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/efficentNet_b2.pth",
    "efficientnet_b3": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/efficentNet_b3.pth",
    "inception_resnet_v2": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/inception_resnet_v2.pth",
    "inception_v4": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/inception_v4.pth",
    "seresnext50_32x4d": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/seresnext50_32x4d.pth",
    "seresnext101_32x4d": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/seresnext101_32x4d.pth",
}

model_names = {
    "resnet18": "resnet18",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_b1": "efficientnet_b1",
    "efficientnet_b2": "efficientnet_b2",
    "efficientnet_b3": "efficientnet_b3",
    "efficientnet_b4": "efficientnet_b4",
    "efficientnet_b5": "efficientnet_b5",
    "inception_resnet_v2": "inception_resnet_v2",
    "inception_v4": "inception_v4",
    "seresnext50_32x4d": "seresnext50_32x4d",
    "seresnext101_32x4d": "seresnext101_32x4d",
}

validation_scores = {
    "resnet18": 0.887,
    "efficientnet_b0": 0.8922,
    "efficientnet_b1": 0.894,
    "efficientnet_b2": 0.898,
    "efficientnet_b3": 0.9127,
    "efficientnet_b4": 0.893,
    "efficientnet_b5": 0.870,
    "inception_resnet_v2": 0.896,
    "inception_v4": 0.8875,
    "seresnext50_32x4d": 0.8652,
    "seresnext101_32x4d": 0.9083,
}

total_score = sum(validation_scores.values())
weights = {k: v / total_score for k, v in validation_scores.items()}


def _load_checkpoint_to_model(model: nn.Module, ckpt_path: str) -> bool:
    if not os.path.exists(ckpt_path):
        return False
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if (
        isinstance(ckpt, dict)
        and "state_dict" in ckpt
        and isinstance(ckpt["state_dict"], dict)
    ):
        state = ckpt["state_dict"]
        cleaned = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            if nk.startswith("model."):
                nk = nk[len("model.") :]
            cleaned[nk] = v
        state = cleaned
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        return False
    model.load_state_dict(state, strict=False)
    return True


def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    assert y_true.shape == y_pred.shape
    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < n_classes and 0 <= b < n_classes:
            O[a, b] += 1.0
    act_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E * (O.sum() / E.sum())

    W = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return 0.0
    return 1.0 - num / den


def stratified_split_indices(labels, val_frac=0.12, seed=SEED):
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.RandomState(seed)
    idx_all = np.arange(len(labels))
    train_idx, val_idx = [], []
    for c in range(5):
        idx_c = idx_all[labels == c]
        rng.shuffle(idx_c)
        n_val = max(1, int(round(len(idx_c) * val_frac)))
        val_idx.append(idx_c[:n_val])
        train_idx.append(idx_c[n_val:])
    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


models_list = []
models_keys = []
models_output_dim = None

for model_key, path in model_paths.items():
    model_name = model_names.get(model_key, model_key)
    model = timm.create_model(model_name, pretrained=False, num_classes=5)
    ok = _load_checkpoint_to_model(model, path)
    if not ok:
        continue
    model.to(device)
    model.eval()
    models_list.append(model)
    models_keys.append(model_key)
    models_output_dim = 5

fallback_trained_head = False
prior_strength = 1.0  # will be tuned on a small val split if we have to use fallback
if len(models_list) == 0:
    fallback_key = "efficientnet_b0"
    fallback_name = model_names[fallback_key]

    model = timm.create_model(fallback_name, pretrained=True, num_classes=5)

    for p in model.parameters():
        p.requires_grad = False
    head_params = []
    for name, p in model.named_parameters():
        if any(x in name for x in ["classifier", "fc", "head"]):
            p.requires_grad = True
            head_params.append(p)

    model.to(device)

    _cfg = resolve_data_config({}, model=model)

    transform_train = create_transform(**_cfg, is_training=True)
    transform_infer = create_transform(**_cfg, is_training=False)

    train_df_for_split = pd.read_csv(train_csv_file)
    labels_all = train_df_for_split["diagnosis"].values.astype(np.int64)
    tr_idx, va_idx = stratified_split_indices(labels_all, val_frac=0.12, seed=SEED)

    train_dataset_full = BlindnessDataset(
        train_csv_file, train_root_dir, transform=transform_train, test=False
    )
    val_dataset_full = BlindnessDataset(
        train_csv_file, train_root_dir, transform=transform_infer, test=False
    )

    train_subset = torch.utils.data.Subset(train_dataset_full, tr_idx.tolist())
    val_subset = torch.utils.data.Subset(val_dataset_full, va_idx.tolist())

    train_loader = DataLoader(
        train_subset,
        batch_size=16,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=32,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    if len(head_params) == 0:
        for name, p in model.named_parameters():
            if name.split(".")[0] in {"classifier", "fc", "head"}:
                p.requires_grad = True
                head_params.append(p)

    optimizer = torch.optim.AdamW(head_params, lr=2e-3, weight_decay=1e-4)

    counts = (
        train_df_for_split["diagnosis"]
        .value_counts()
        .reindex([0, 1, 2, 3, 4], fill_value=0)
        .values.astype(np.float32)
    )
    counts = np.clip(counts, 1.0, None)
    inv_freq = 1.0 / counts
    class_w = inv_freq / inv_freq.mean()
    class_w_t = torch.tensor(class_w, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_w_t)

    counts_tr = (
        pd.Series(labels_all[tr_idx])
        .value_counts()
        .reindex([0, 1, 2, 3, 4], fill_value=0)
        .values.astype(np.float32)
    )
    prior_tr = counts_tr / max(counts_tr.sum(), 1.0)
    prior_tr = np.clip(prior_tr, 1e-6, 1.0)
    log_prior_tr = torch.log(torch.tensor(prior_tr, dtype=torch.float32, device=device))

    model.train()
    best_qwk = -1e9
    best_state = None

    for images, labels in tqdm(
        train_loader, total=len(train_loader), desc="Train head (1 epoch)"
    ):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        do_eval = False
        if not hasattr(model, "_iter_ct"):
            model._iter_ct = 0
        model._iter_ct += 1
        if model._iter_ct % max(
            1, len(train_loader) // 5
        ) == 0 or model._iter_ct == len(train_loader):
            do_eval = True

        if do_eval:
            model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for vimg, vlab in val_loader:
                    vimg = vimg.to(device, non_blocking=True)
                    vlab = vlab.numpy().astype(np.int64)
                    vlogits = model(vimg)
                    vpred = (
                        torch.argmax(vlogits, dim=1)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.int64)
                    )
                    y_true.append(vlab)
                    y_pred.append(vpred)
            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)
            qwk = quadratic_weighted_kappa(y_true, y_pred, n_classes=5)
            if qwk > best_qwk:
                best_qwk = qwk
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            model.train()

    model.eval()
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    strengths = [0.0, 0.25, 0.5, 0.75, 1.0]
    best_s = 1.0
    best_qwk_s = -1e9
    y_true_all = labels_all[va_idx].astype(np.int64)

    val_logits_list = []
    with torch.no_grad():
        for vimg, _ in val_loader:
            vimg = vimg.to(device, non_blocking=True)
            val_logits_list.append(model(vimg).detach().cpu())
    val_logits = torch.cat(val_logits_list, dim=0)  # (Nval,5)

    for s in strengths:
        adj = val_logits + (s * log_prior_tr.detach().cpu()).unsqueeze(0)
        pred = torch.argmax(adj, dim=1).numpy().astype(np.int64)
        qwk = quadratic_weighted_kappa(y_true_all, pred, n_classes=5)
        if qwk > best_qwk_s:
            best_qwk_s = qwk
            best_s = s
    prior_strength = float(best_s)

    models_list = [model]
    models_keys = [fallback_key]
    models_output_dim = 5
    fallback_trained_head = True

    transform = transform_infer
else:
    _transform_model_for_config = models_list[0]
    _cfg = resolve_data_config({}, model=_transform_model_for_config)
    transform = create_transform(**_cfg, is_training=False)

wsum = sum(weights.get(k, 1.0) for k in models_keys)
weights_in_use = {k: (weights.get(k, 1.0) / wsum) for k in models_keys}

train_df = pd.read_csv(train_csv_file)
prior_counts = (
    train_df["diagnosis"]
    .value_counts()
    .reindex([0, 1, 2, 3, 4], fill_value=0)
    .values.astype(np.float32)
)
prior = prior_counts / max(prior_counts.sum(), 1.0)
prior = np.clip(prior, 1e-6, 1.0)
log_prior = torch.log(torch.tensor(prior, dtype=torch.float32, device=device))  # (5,)

test_dataset = BlindnessDataset(
    test_csv_file, test_root_dir, transform=transform, test=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)



## === cell 4
all_outputs = []

with torch.no_grad():
    for images in tqdm(test_loader, total=len(test_loader), desc="Inference"):
        images = images.to(device, non_blocking=True)

        per_model = []
        for model_key, model in zip(models_keys, models_list):
            logits = model(images)

            s = prior_strength if fallback_trained_head else 1.0
            logits = logits + (s * log_prior).unsqueeze(0)

            probs = nn.functional.softmax(logits, dim=1)
            per_model.append(weights_in_use[model_key] * probs)

        weighted_outputs = torch.stack(per_model, dim=0).sum(dim=0)  # (B,5)
        all_outputs.append(weighted_outputs.detach().cpu().numpy())

all_outputs = np.concatenate(all_outputs, axis=0)
final_predictions = np.argmax(all_outputs, axis=1).astype(int)

assert len(final_predictions) == len(
    test_df
), "Prediction count must match test.csv rows."



## === cell 5
submission_df = pd.DataFrame(
    {
        "id_code": test_df["id_code"].values,
        "diagnosis": final_predictions,
    }
)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(submission_df.head())
print(f"Saved: {submission_path}  shape={submission_df.shape}")
print(
    f"Used models: {models_keys}  fallback_trained_head={fallback_trained_head}  prior_strength={prior_strength:.2f}"
)
