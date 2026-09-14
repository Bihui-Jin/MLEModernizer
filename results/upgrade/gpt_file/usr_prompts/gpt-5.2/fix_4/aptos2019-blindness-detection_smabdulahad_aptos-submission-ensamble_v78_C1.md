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

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.75593) has done: 'The main blocker is missing external model checkpoint files, which makes `models_list` empty and breaks inference and submission creation. I add a minimal, self-contained fallback that keeps the same ensemble inference semantics when checkpoints exist, but automatically trains a small timm model on the provided `train_images/train.csv` when they don’t. I also fix robustness issues: ensure RGB loading, set `num_workers=0` for Kaggle notebook stability, load weights with proper `map_location`, and guarantee `submission.csv` is always written with the required columns and row order.'
- What this solution (achieved 0.7625) has done: 'Your current score (0.75593) is well below the target (0.89006), so we should make a small, legitimate change that improves predictions without changing the model architecture or training loop style. The biggest gain with minimal risk is to align the image preprocessing with what timm pretrained backbones expect: use timm’s model-specific `resolve_data_config` + `create_transform` instead of a generic 224-resize transform. This preserves your core approach (single-model fallback training for 1 epoch; softmax + argmax; ensemble when checkpoints exist) but typically yields a sizeable kappa lift because the normalization/interpolation/crop behavior matches the backbone. I also ensure the same transform is used consistently for both training and test in the fallback path, keeping semantics intact and avoiding train/test preprocessing mismatch.'

# 9. Code solution

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
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

_default_backbone_for_transform = "resnet18"
_tmp_model = timm.create_model(
    _default_backbone_for_transform, pretrained=True, num_classes=5
)
_data_cfg = resolve_data_config({}, model=_tmp_model)

transform_train = create_transform(**_data_cfg, is_training=True)
transform_eval = create_transform(**_data_cfg, is_training=False)

del _tmp_model



## === cell 3
test_csv_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_root_dir = "/kaggle/input/aptos2019-blindness-detection/test_images"

train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
train_root_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"

test_dataset = BlindnessDataset(
    test_csv_file, test_root_dir, transform=transform_eval, test=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)



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

if len(loaded_model_keys) > 0:
    total_score = sum(validation_scores[k] for k in loaded_model_keys)
    weights = {k: validation_scores[k] / total_score for k in loaded_model_keys}
else:
    weights = {}




## === cell 7
def _quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    assert y_true.shape == y_pred.shape
    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < n_classes and 0 <= b < n_classes:
            O[a, b] += 1.0

    act_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    E = np.outer(act_hist, pred_hist)
    E = E / E.sum() * O.sum()

    W = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return 0.0
    return 1.0 - num / den


def _apply_thresholds(x, thr):
    thr = np.asarray(thr, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    return np.digitize(x, thr).astype(int)


def _fit_thresholds_bruteforce(
    x, y, init_thr=(0.5, 1.5, 2.5, 3.5), step=0.05, radius=0.6
):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=int)

    best_thr = np.array(init_thr, dtype=np.float64)
    best_score = -1e9

    grids = []
    for t0 in init_thr:
        lo = t0 - radius
        hi = t0 + radius
        grids.append(np.arange(lo, hi + 1e-12, step, dtype=np.float64))

    for a in grids[0]:
        for b in grids[1]:
            if b <= a:
                continue
            for c in grids[2]:
                if c <= b:
                    continue
                for d in grids[3]:
                    if d <= c:
                        continue
                    thr = (a, b, c, d)
                    pred = _apply_thresholds(x, thr)
                    score = _quadratic_weighted_kappa(y, pred, n_classes=5)
                    if score > best_score:
                        best_score = score
                        best_thr = np.array(thr, dtype=np.float64)
    return best_thr, float(best_score)




## === cell 8
def train_fallback_model():
    full_df = pd.read_csv(train_csv_file)

    idx = np.arange(len(full_df))
    rng = np.random.RandomState(42)
    rng.shuffle(idx)
    split = int(0.9 * len(idx))
    tr_idx = idx[:split]
    va_idx = idx[split:]

    tr_csv = "/kaggle/working/_train_split.csv"
    va_csv = "/kaggle/working/_val_split.csv"
    full_df.iloc[tr_idx].to_csv(tr_csv, index=False)
    full_df.iloc[va_idx].to_csv(va_csv, index=False)

    train_dataset = BlindnessDataset(
        tr_csv, train_root_dir, transform=transform_train, test=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = timm.create_model("resnet18", pretrained=True, num_classes=5)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

    model.train()
    for images, labels in tqdm(train_loader, desc="Training fallback (1 epoch)"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

    val_dataset = BlindnessDataset(
        va_csv, train_root_dir, transform=transform_eval, test=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model.eval()
    y_true = []
    x_pred = []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Calibrating thresholds"):
            images = images.to(device, non_blocking=True)
            prob = nn.functional.softmax(model(images), dim=1)
            exp = (prob * torch.arange(5, device=prob.device, dtype=prob.dtype)).sum(
                dim=1
            )
            x_pred.append(exp.detach().cpu().numpy())
            y_true.append(labels.numpy())

    y_true = np.concatenate(y_true, axis=0)
    x_pred = np.concatenate(x_pred, axis=0)

    thr, qwk = _fit_thresholds_bruteforce(
        x_pred, y_true, init_thr=(0.5, 1.5, 2.5, 3.5), step=0.05, radius=0.6
    )
    print(f"Fitted thresholds: {thr.tolist()} | val QWK={qwk:.5f}")

    return model, thr


fallback_model = None
fallback_thresholds = None
if len(models_list) == 0:
    print(
        "No external checkpoints found. Training a fallback model from provided train set..."
    )
    fallback_model, fallback_thresholds = train_fallback_model()



## === cell 9
all_outputs = []

with torch.no_grad():
    for images in tqdm(test_loader, desc="Infer test"):
        images = images.to(device, non_blocking=True)

        if len(models_list) > 0:
            outs = []
            for model_key, model in zip(loaded_model_keys, models_list):
                prob = nn.functional.softmax(model(images), dim=1)
                outs.append(weights[model_key] * prob)
            weighted_outputs = torch.stack(outs, dim=0).sum(dim=0)
        else:
            prob = nn.functional.softmax(fallback_model(images), dim=1)
            weighted_outputs = prob

        all_outputs.append(weighted_outputs.cpu().numpy())

all_outputs = np.concatenate(all_outputs, axis=0)

if len(models_list) == 0 and fallback_thresholds is not None:
    exp = (all_outputs * np.arange(5, dtype=np.float64)[None, :]).sum(axis=1)
    final_predictions = _apply_thresholds(exp, fallback_thresholds).astype(int)
else:
    final_predictions = np.argmax(all_outputs, axis=1).astype(int)



## === cell 10
test_ids = pd.read_csv(test_csv_file)["id_code"].values
assert len(test_ids) == len(final_predictions), "Prediction length mismatch vs test.csv"

submission_df = pd.DataFrame({"id_code": test_ids, "diagnosis": final_predictions})

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(
    f"Wrote {submission_path} with shape {submission_df.shape} and columns {list(submission_df.columns)}"
)
