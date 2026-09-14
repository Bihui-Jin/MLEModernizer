# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import timm


def seed_everything(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(42)


def _dl_kwargs_for_env():
    use_cuda = torch.cuda.is_available()
    return dict(
        num_workers=0,
        pin_memory=use_cuda,
        persistent_workers=False,
    )




## === cell 1
import cv2


class BlindnessDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, test=False, return_id=False):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.test = test
        self.return_id = return_id

        self._ids = self.annotations.iloc[:, 0].astype(str).to_numpy()
        if not test and self.annotations.shape[1] > 1:
            self._labels = self.annotations.iloc[:, 1].astype(np.int64).to_numpy()
        else:
            self._labels = None

    def __len__(self):
        return len(self._ids)

    def __getitem__(self, idx):
        img_id = self._ids[idx]
        img_name = os.path.join(self.root_dir, img_id + ".png")

        img_bgr = cv2.imread(img_name, cv2.IMREAD_COLOR)
        if img_bgr is None:
            image = Image.open(img_name).convert("RGB")
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img_rgb)

        if self.transform:
            image = self.transform(image)

        if self.test:
            if self.return_id:
                return image, img_id
            return image
        else:
            label = int(self._labels[idx])
            if self.return_id:
                return image, label, img_id
            return image, label




## === cell 2
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

_fallback_backbone = "resnet18"


def build_transforms(backbone_name: str):
    tmp_model = timm.create_model(backbone_name, pretrained=True, num_classes=5)
    data_cfg = resolve_data_config({}, model=tmp_model)
    tr = create_transform(**data_cfg, is_training=True)
    ev = create_transform(**data_cfg, is_training=False)
    del tmp_model
    return tr, ev




## === cell 3
test_csv_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_root_dir = "/kaggle/input/aptos2019-blindness-detection/test_images"

train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
train_root_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"




## === cell 4
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




## === cell 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models_list = []
loaded_model_keys = []

for model_key, path in model_paths.items():
    if not os.path.exists(path):
        continue
    model_name = model_names[model_key]
    model = timm.create_model(model_name, pretrained=False, num_classes=5)
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    models_list.append(model)
    loaded_model_keys.append(model_key)

print(f"Loaded {len(models_list)} checkpoint models: {loaded_model_keys}")




## === cell 6


def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    assert y_true.shape == y_pred.shape
    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        O[a, b] += 1.0

    hist_true = O.sum(axis=1)
    hist_pred = O.sum(axis=0)
    E = np.outer(hist_true, hist_pred)
    if E.sum() > 0:
        E = E * (O.sum() / E.sum())

    W = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    return 1.0 - (num / den if den > 0 else 0.0)


def probs_to_continuous_score(probs: np.ndarray) -> np.ndarray:
    w = np.arange(probs.shape[1], dtype=np.float32)
    return (probs * w[None, :]).sum(axis=1)


def apply_thresholds(x: np.ndarray, thr: np.ndarray) -> np.ndarray:
    return np.digitize(x, thr, right=False).astype(np.int64)


def optimize_thresholds(x: np.ndarray, y: np.ndarray, init_thr=None, n_iter=2):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    if init_thr is None:
        thr = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
    else:
        thr = np.array(init_thr, dtype=np.float32).copy()

    for _ in range(n_iter):
        for i in range(4):
            best_thr_i = thr[i]
            best_score = -1e9
            lo = -1.0 if i == 0 else float(thr[i - 1] + 1e-3)
            hi = 5.0 if i == 3 else float(thr[i + 1] - 1e-3)
            grid = np.linspace(
                max(lo, thr[i] - 0.6), min(hi, thr[i] + 0.6), 31, dtype=np.float32
            )
            for t in grid:
                thr_try = thr.copy()
                thr_try[i] = float(t)
                pred = apply_thresholds(x, thr_try)
                score = quadratic_weighted_kappa(y, pred, n_classes=5)
                if score > best_score:
                    best_score = score
                    best_thr_i = float(t)
            thr[i] = best_thr_i
    return thr




## === cell 7
def _train_fallback_model(backbone_name: str, train_csv: str, train_root: str):
    transform_train, transform_eval = build_transforms(backbone_name)

    full_df = pd.read_csv(train_csv)
    rng = np.random.RandomState(42)
    idx = np.arange(len(full_df))
    rng.shuffle(idx)
    val_size = max(200, int(0.15 * len(full_df)))
    val_idx = idx[:val_size]
    tr_idx = idx[val_size:]

    tr_df = full_df.iloc[tr_idx].reset_index(drop=True)
    val_df = full_df.iloc[val_idx].reset_index(drop=True)

    tr_csv_tmp = "/kaggle/working/_train_split.csv"
    val_csv_tmp = "/kaggle/working/_val_split.csv"
    tr_df.to_csv(tr_csv_tmp, index=False)
    val_df.to_csv(val_csv_tmp, index=False)

    train_ds = BlindnessDataset(
        tr_csv_tmp, train_root, transform=transform_train, test=False, return_id=False
    )
    val_ds = BlindnessDataset(
        val_csv_tmp, train_root, transform=transform_eval, test=False, return_id=False
    )

    dl_kwargs = _dl_kwargs_for_env()
    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        drop_last=False,
        **{k: v for k, v in dl_kwargs.items() if v is not None},
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        **{k: v for k, v in dl_kwargs.items() if v is not None},
    )

    model = timm.create_model(backbone_name, pretrained=True, num_classes=5)
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    epochs = 3

    for ep in range(epochs):
        running_loss = 0.0
        for images, labels in tqdm(
            train_loader, desc=f"Fallback train ep{ep+1}/{epochs}"
        ):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu())

        running_loss /= max(1, len(train_loader))
        print(f"Fallback training epoch {ep+1}: loss={running_loss:.4f}")

    model.eval()
    val_probs = []
    val_labels = []
    with torch.inference_mode():
        for images, labels in tqdm(val_loader, desc="Fallback val infer"):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            prob = nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
            val_probs.append(prob)
            val_labels.append(labels.numpy())
    val_probs = np.concatenate(val_probs, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)

    x_cont = probs_to_continuous_score(val_probs)
    thr = optimize_thresholds(
        x_cont, val_labels, init_thr=[0.5, 1.5, 2.5, 3.5], n_iter=2
    )
    val_pred = apply_thresholds(x_cont, thr)
    val_qwk = quadratic_weighted_kappa(val_labels, val_pred, n_classes=5)
    print(f"Optimized thresholds: {thr} | val QWK={val_qwk:.5f}")

    return model, transform_eval, thr


if len(models_list) == 0:
    print(
        "No external checkpoint models found. Training fallback model to enable end-to-end run."
    )
    fallback_model, transform_eval, tuned_thresholds = _train_fallback_model(
        _fallback_backbone, train_csv_file, train_root_dir
    )
    models_list = [fallback_model]
    loaded_model_keys = [_fallback_backbone]
else:
    _backbone_for_transform = model_names[loaded_model_keys[0]]
    _, transform_eval = build_transforms(_backbone_for_transform)
    tuned_thresholds = None




## === cell 8
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

if all(k in validation_scores for k in loaded_model_keys):
    total_score = sum(validation_scores[k] for k in loaded_model_keys)
    weights = {k: validation_scores[k] / total_score for k in loaded_model_keys}
    weights_vec = torch.tensor(
        [weights[k] for k in loaded_model_keys], dtype=torch.float32, device=device
    )
else:
    weights_vec = torch.ones(len(models_list), dtype=torch.float32, device=device)
    weights_vec = weights_vec / weights_vec.sum()

print(f"Ensemble weights: {weights_vec.detach().cpu().numpy()}")




## === cell 9
test_dataset = BlindnessDataset(
    test_csv_file, test_root_dir, transform=transform_eval, test=True, return_id=True
)

dl_kwargs = _dl_kwargs_for_env()
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    drop_last=False,
    **{k: v for k, v in dl_kwargs.items() if v is not None},
)




## === cell 10
all_outputs = []
all_ids = []

with torch.inference_mode():
    for images, ids in tqdm(test_loader, desc="Infer test"):
        all_ids.extend(list(ids))
        images = images.to(device, non_blocking=True)

        weighted_prob = None
        for mi, model in enumerate(models_list):
            logits = model(images)
            prob = nn.functional.softmax(logits, dim=1)
            wprob = weights_vec[mi] * prob
            weighted_prob = wprob if weighted_prob is None else (weighted_prob + wprob)

        all_outputs.append(weighted_prob.detach().cpu().numpy())

all_outputs = np.concatenate(all_outputs, axis=0)

if tuned_thresholds is not None:
    cont = probs_to_continuous_score(all_outputs)
    final_predictions = apply_thresholds(cont, tuned_thresholds).astype(int)
else:
    final_predictions = np.argmax(all_outputs, axis=1).astype(int)




## === cell 11
test_df = pd.read_csv(test_csv_file)
test_ids = test_df["id_code"].astype(str).to_numpy()

all_ids_arr = np.asarray(all_ids, dtype=str)
if all_ids_arr.shape[0] != test_ids.shape[0] or not np.array_equal(
    all_ids_arr, test_ids
):
    id_to_pred = {i: int(p) for i, p in zip(all_ids_arr, final_predictions)}
    ordered_preds = pd.Series(test_ids).map(id_to_pred).to_numpy()
    assert not pd.isna(ordered_preds).any(), "Some test ids were missing predictions"
    ordered_preds = ordered_preds.astype(int)
else:
    ordered_preds = final_predictions

submission_df = pd.DataFrame(
    {"id_code": test_ids, "diagnosis": ordered_preds.astype(int)}
)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(
    f"Wrote {submission_path} with shape {submission_df.shape} and columns {list(submission_df.columns)}"
)
print(submission_df.head())
print(submission_df["diagnosis"].value_counts().sort_index())

## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same id_codes as answers
