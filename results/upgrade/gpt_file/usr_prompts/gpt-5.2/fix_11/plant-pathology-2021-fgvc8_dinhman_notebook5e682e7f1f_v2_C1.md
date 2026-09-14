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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.7357710064635279

# 6. Current score

0.35262

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.26731) has done: 'The timeout is dominated by per-image CPU preprocessing: `skimage.transform.resize` is slow, `RandomCrop` introduces extra work, and logits are accumulated with repeated `np.vstack` (quadratic growth). I replace resizing/cropping with equivalent OpenCV operations (same geometry semantics), make the crop deterministic (center crop) to avoid random overhead while keeping the same transform structure, and preallocate the logits array to avoid repeated concatenations. I also enable faster DataLoader settings (more workers, persistent workers, prefetching) and reduce per-batch CPU/GPU sync by writing into the preallocated array directly—all of which preserve the model, checkpoint loading, inference loop semantics, and output formatting.'
- What this solution (achieved 0.3087) has done: 'The crash comes from `Normalize` receiving a Torch tensor because `ToTensor()` runs before it; changing `Normalize` to handle both NumPy arrays and Torch tensors fixes the runtime error without changing the model or inference logic. I also keep the ImageNet normalization (important for ResNet calibration) but apply it in a type-safe way and ensure the tensor dtype is float32. Finally, I keep the existing prediction→threshold→fallback-to-argmax logic and ensure a valid `submission.csv` with the exact required columns is written.'
- What this solution (achieved 0.30565) has done: 'Your current gap to the target is large (0.3087 → 0.7358), so the smallest score-relevant change is to fix the prediction post-processing rather than touching the model or inference loop. The competition is multi-label with mean F1, and a fixed global threshold of 0.50 is typically far from optimal; we can instead choose a threshold using the provided `train.csv` by running inference on the training images with the same pretrained checkpoint and selecting the threshold that maximizes mean F1. This preserves your model, weights, transforms, and inference semantics; it only calibrates the final label decision rule to better match the evaluation metric. The chosen threshold is then applied unchanged to test predictions and we still write a valid `submission.csv` in the required space-delimited format.'
- What this solution (achieved 0.30565) has done: 'Your current score is far below target, and the largest score-limiting issue here is that you’re tuning a single global threshold on the *training set itself*, which overfits and doesn’t transfer to the hidden test set, depressing mean F1. I keep your exact model, weights, transforms, and inference loop, but change threshold calibration to use an out-of-fold (OOF) procedure so the chosen threshold generalizes better while still being a single global threshold (same post-processing semantics). Concretely: run inference on train once, compute OOF probabilities with a fast K-fold split, select the best threshold by OOF mean F1, then apply it unchanged to test predictions and write the same `submission.csv`. This is a minimal change focused only on post-processing calibration and should move the score upward toward your target.'
- What this solution (achieved 0.30565) has done: 'Your score gap to the target is large, so the most “minimal but score-relevant” improvement is still in post-processing rather than changing the model. The current OOF threshold tuning is effectively random because folds are assigned without stratification for multi-label prevalence, which can pick a threshold that doesn’t generalize and hurts hidden-test F1. I keep the same single global-threshold decision rule, but compute folds using iterative multilabel stratification (implemented locally with NumPy) so each fold has similar class frequencies, then re-select the threshold on those OOF folds. This is a small change (only how folds are formed) that typically improves threshold calibration stability and should move the score upward toward your target while preserving your architecture/inference semantics and still writing a valid `submission.csv`.'
- What this solution (achieved 0.35038) has done: 'Your current score is far below the target, so we should only adjust post-processing (decision rule) rather than touching the model or inference. The largest issue is using a single global threshold; mean-F1 in this competition typically benefits from per-class thresholds because class prevalences differ a lot, while keeping the same overall “threshold then argmax-fallback” semantics. I keep your model/checkpoint/transforms and inference loops identical, but change threshold calibration to pick one threshold per class using out-of-fold (iterative multilabel-stratified) predictions on train. Then I apply those per-class thresholds to test probabilities and keep your required “at least one label via argmax” fallback, writing the same `submission.csv` format.'
- What this solution (achieved 0.34742) has done: 'I keep your model, weights, transforms, and inference loops intact, and only adjust the post-processing calibration to better match mean F1 without changing evaluation semantics. Your current per-class threshold grid is quite coarse (0.05 steps), which can easily miss good operating points; I switch to a finer but still fast grid (0.01 steps) and select thresholds by maximizing the true OOF mean-F1 of the *full multilabel prediction with argmax fallback* (rather than averaging per-class F1 independently). This preserves the same decision rule (thresholding + “at least one label via argmax”), but tunes it in a way that matches the competition metric more directly and should move your score upward toward the target. The output submission format and paths remain unchanged, and the notebook still writes `submission.csv`.'
- What this solution (achieved 0.34742) has done: 'Your current score is far below the target, so we should improve it without changing the model/training core logic. The biggest score limiter in your post-processing is optimizing thresholds on OOF folds but using probabilities inferred on the *full training set model* (no fold-specific models), which makes “OOF” effectively just a partitioned in-sample evaluation and can select thresholds that don’t transfer. I keep your exact model, checkpoint, transforms, and inference loops, but change threshold selection to a more stable, metric-aligned procedure: (1) compute per-class thresholds by maximizing per-class F1 on a held-out fold *with the same argmax-fallback semantics applied globally afterward*, and (2) optionally do a tiny global scaling step to avoid overly aggressive multi-label predictions. These are minimal post-processing-only changes aimed at increasing hidden-test mean F1 toward your target while still producing the same `submission.csv` format.'
- What this solution (achieved 0.35262) has done: 'We keep your model, checkpoint loading, transforms, and inference loops unchanged, and only adjust the post-processing calibration because your current score (0.34742) is far below the target (0.73577). The biggest issue is that the “OOF” threshold tuning is not actually out-of-fold because you only have one model (trained on all train), so thresholds can become unstable; the smallest robust fix is to tune thresholds on a small held-out validation split (stratified by label cardinality) and then apply them to test. We also tune thresholds directly against the competition mean-F1 with the same “threshold + argmax fallback” semantics (not per-class independent F1), using a coordinate-descent style sweep that’s fast and stable. This should move the score upward toward the target without changing the core logic or adding new training.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd

import cv2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import torchvision.models as models
from torchvision.transforms import transforms

from tqdm import tqdm

DATA_DIR = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_DIR, "sample_submission.csv")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

test_fram = pd.read_csv(SAMPLE_SUB_CSV)
print(test_fram.head())
print("Num test rows:", len(test_fram))
print("Example image id:", test_fram.values[1][0])




## === cell 1
class LeafDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform):
        df = pd.read_csv(csv_file)
        self.image_names = df.iloc[:, 0].to_numpy()
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.image_names[idx])

        image = cv2.imread(img_name, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_name}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        labels = [1]
        sample = {"image": image, "labels": labels}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    def __call__(self, sample):
        image = sample["image"]
        labels = sample["labels"]
        image = np.ascontiguousarray(image.transpose((2, 0, 1)), dtype=np.float32)
        return {"image": torch.from_numpy(image), "labels": labels}


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, labels = sample["image"], sample["labels"]
        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return {"image": img, "labels": labels}


class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = (
            (output_size, output_size) if isinstance(output_size, int) else output_size
        )
        assert len(self.output_size) == 2

    def __call__(self, sample):
        image, labels = sample["image"], sample["labels"]
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        if h < new_h or w < new_w:
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            return {"image": image, "labels": labels}

        top = (h - new_h) // 2
        left = (w - new_w) // 2
        image = image[top : top + new_h, left : left + new_w]
        return {"image": image, "labels": labels}


class Normalize(object):
    """
    Type-safe normalization: supports both NumPy (HWC) and Torch tensor (CHW).
    Keeps ImageNet normalization (needed for ResNet calibration).
    """

    def __init__(self, mean, std):
        self.mean_np = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std_np = np.array(std, dtype=np.float32).reshape(3, 1, 1)
        self.mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std_t = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, sample):
        x = sample["image"]

        if isinstance(x, torch.Tensor):
            x = x.to(dtype=torch.float32)
            x = x * (1.0 / 255.0)
            mean = self.mean_t.to(device=x.device)
            std = self.std_t.to(device=x.device)
            x = (x - mean) / std
            sample["image"] = x
            return sample

        x = x.astype(np.float32) * (1.0 / 255.0)
        x = (x - self.mean_np) / self.std_np
        sample["image"] = x
        return sample


infer_transform = transforms.Compose(
    [
        Rescale(256),
        RandomCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

leafDatasets = LeafDataset(
    SAMPLE_SUB_CSV,
    TEST_IMG_DIR,
    transform=infer_transform,
)

print("Dataset item keys:", leafDatasets[0].keys())
print("Image tensor shape:", leafDatasets[0]["image"].shape)
print("Image dtype:", leafDatasets[0]["image"].dtype)



## === cell 2
batch_size = 64  # Keep as-is.

num_workers = min(8, os.cpu_count() or 2)
pin = torch.cuda.is_available()
test_loader = DataLoader(
    leafDatasets,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    pin_memory=pin,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)



## === cell 3
try:
    resnet = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
except Exception:
    resnet = models.resnet152(weights=None)

num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 6)

print(resnet.fc)



## === cell 4
load_path = "../input/modelres/resnet.pkl"
if os.path.exists(load_path):
    state = torch.load(load_path, map_location="cpu")
    resnet.load_state_dict(state)
    print("Loaded checkpoint:", load_path)
else:
    print("Checkpoint not found; using torchvision initialization:", load_path)



## === cell 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet.to(device)
resnet.eval()

n_test = len(leafDatasets)
y_pred_logits = np.empty((n_test, 6), dtype=np.float32)

with torch.no_grad():
    stream = tqdm(
        test_loader,
        desc=f"Infer test ({device})",
        total=(n_test + batch_size - 1) // batch_size,
    )
    offset = 0
    for sample in stream:
        X = sample["image"].to(device, non_blocking=True)
        X = X.float()
        pred = resnet(X)  # logits
        bs = pred.shape[0]
        y_pred_logits[offset : offset + bs] = pred.detach().cpu().numpy()
        offset += bs

print("Pred logits shape:", y_pred_logits.shape)



## === cell 6
LABELS = ["complex", "frog_eye_leaf_spot", "healthy", "powdery_mildew", "rust", "scab"]
label_to_idx = {l: i for i, l in enumerate(LABELS)}


def encode_multilabel_space_delimited(s: str, n_classes: int = 6) -> np.ndarray:
    y = np.zeros((n_classes,), dtype=np.int64)
    if isinstance(s, str) and len(s.strip()) > 0:
        for token in s.split():
            if token in label_to_idx:
                y[label_to_idx[token]] = 1
    return y


def mean_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-12
    f1s = []
    for c in range(y_true.shape[1]):
        yt = y_true[:, c].astype(np.int64)
        yp = y_pred[:, c].astype(np.int64)
        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        f1 = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
        f1s.append(f1)
    return float(np.mean(f1s))


def apply_thresholds_with_argmax_fallback(
    probs: np.ndarray, thr_vec: np.ndarray
) -> np.ndarray:
    pred_bin = (probs >= thr_vec.reshape(1, -1)).astype(np.int64)
    empty = pred_bin.sum(axis=1) == 0
    if empty.any():
        am = np.argmax(probs, axis=1)
        pred_bin[empty, :] = 0
        pred_bin[empty, am[empty]] = 1
    return pred_bin


train_df = pd.read_csv(TRAIN_CSV)
y_train_true = np.stack(
    [encode_multilabel_space_delimited(x, 6) for x in train_df["labels"].tolist()],
    axis=0,
)

train_dataset_for_thr = LeafDataset(
    TRAIN_CSV,
    TRAIN_IMG_DIR,
    transform=infer_transform,
)

train_loader_for_thr = DataLoader(
    train_dataset_for_thr,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    pin_memory=pin,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)

n_train = len(train_dataset_for_thr)
y_train_logits = np.empty((n_train, 6), dtype=np.float32)

with torch.no_grad():
    stream = tqdm(
        train_loader_for_thr,
        desc=f"Infer train (thr-tune) ({device})",
        total=(n_train + batch_size - 1) // batch_size,
    )
    offset = 0
    for sample in stream:
        X = sample["image"].to(device, non_blocking=True).float()
        pred = resnet(X)
        bs = pred.shape[0]
        y_train_logits[offset : offset + bs] = pred.detach().cpu().numpy()
        offset += bs

train_probs = 1.0 / (1.0 + np.exp(-y_train_logits))

card = y_train_true.sum(axis=1)
bins = np.clip(card, 0, 4)  # 0..4+ as stratification bins
rng = np.random.RandomState(SEED)
idx_all = np.arange(n_train)
val_mask = np.zeros(n_train, dtype=bool)
for b in np.unique(bins):
    ids = idx_all[bins == b]
    rng.shuffle(ids)
    n_val = max(1, int(round(0.20 * len(ids))))
    val_mask[ids[:n_val]] = True
val_idx = np.where(val_mask)[0]
trn_idx = np.where(~val_mask)[0]
print("Held-out split sizes:", len(trn_idx), "train-like,", len(val_idx), "val")


def val_mean_f1_for_thresholds(thr_vec: np.ndarray) -> float:
    pred_val = apply_thresholds_with_argmax_fallback(train_probs[val_idx], thr_vec)
    return mean_f1_score(y_train_true[val_idx], pred_val)


thr_grid = np.round(np.arange(0.05, 0.951, 0.01), 2).astype(np.float32)
thr_vec = np.full((6,), 0.50, dtype=np.float32)

base = val_mean_f1_for_thresholds(thr_vec)
print(f"Val mean-F1 @ all 0.50: {base:.6f}")

for c in range(6):
    best_t = float(thr_vec[c])
    best_sc = base
    thr_try = thr_vec.copy()
    for t in thr_grid:
        thr_try[c] = float(t)
        sc = val_mean_f1_for_thresholds(thr_try)
        if sc > best_sc + 1e-12:
            best_sc = sc
            best_t = float(t)
    thr_vec[c] = best_t
    base = best_sc  # carry forward improvement

delta_grid = np.round(np.arange(-0.10, 0.101, 0.01), 2).astype(np.float32)
best_delta = 0.0
best_sc = val_mean_f1_for_thresholds(thr_vec)
for d in delta_grid:
    thr_try = np.clip(thr_vec + float(d), 0.01, 0.99).astype(np.float32)
    sc = val_mean_f1_for_thresholds(thr_try)
    if sc > best_sc + 1e-12:
        best_sc = sc
        best_delta = float(d)

best_thr_per_class = np.clip(thr_vec + best_delta, 0.01, 0.99).astype(np.float32)
final_val = val_mean_f1_for_thresholds(best_thr_per_class)

print("Selected per-class thresholds (val-tuned) after global delta adjustment:")
for c, lab in enumerate(LABELS):
    print(f"  {lab:>20s}: thr={best_thr_per_class[c]:.2f}")
print(f"Chosen global delta: {best_delta:+.2f}")
print(f"Final held-out val mean-F1 with tuned thresholds: {final_val:.6f}")



## === cell 7
probs = 1.0 / (1.0 + np.exp(-y_pred_logits))

mask = probs >= best_thr_per_class.reshape(1, -1)

indices = []
argmax_idx = np.argmax(probs, axis=1)
for i in range(probs.shape[0]):
    idxs = np.flatnonzero(mask[i]).tolist()
    if len(idxs) == 0:
        idxs = [int(argmax_idx[i])]
    indices.append(idxs)

print("Example predicted indices:", indices[0])



## === cell 8
labels = ["complex", "frog_eye_leaf_spot", "healthy", "powdery_mildew", "rust", "scab"]
testlabels = [" ".join(labels[i] for i in idxs) for idxs in indices]

print(testlabels[:5], " ... total:", len(testlabels))



## === cell 9
sub = pd.read_csv(SAMPLE_SUB_CSV)
assert len(sub) == len(
    testlabels
), f"Predictions ({len(testlabels)}) != submission rows ({len(sub)})"
sub["labels"] = testlabels
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
