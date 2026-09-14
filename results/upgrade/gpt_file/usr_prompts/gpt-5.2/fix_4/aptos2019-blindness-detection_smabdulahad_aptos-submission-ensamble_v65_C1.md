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

0.7783556176172796

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.10291) has done: 'I fix the missing model-weights path issue by making the script robust: it try to load the provided checkpoints if they exist, and otherwise fall back to a timm ImageNet-pretrained model with the same architecture so inference can still run end-to-end. I also fix the ensemble loop so it never calls `torch.cat` on an empty list, and ensure we always produce `final_predictions` for the submission. To keep the core logic (softmax + weighted averaging + argmax) intact, I only adjust the weights dictionary to match whatever models are actually loaded, and keep the submission columns/ordering identical to `test.csv`. This produce a valid `submission.csv` and should yield a non-trivial score (better than a crash), moving you toward the target.'
- What this solution (achieved 0.12533) has done: 'Your current score is far below the target, and the biggest likely reason (with minimal code changes) is that the model you *think* you’re using (a trained checkpoint ensemble) is not actually available, so you’re effectively submitting an ImageNet-pretrained classifier on a very different domain. To move the score sharply upward without changing the core architecture/training logic, I (1) robustly discover and load any `.pth` checkpoints that exist in the Kaggle input tree, (2) ensure the classifier head matches (5 classes) and handle common checkpoint formats (`state_dict`, `model`, `module.` prefixes), and (3) fix inference-time preprocessing to a standard “center crop after resize” (still 224) which typically improves stability on fundus images without changing the model itself. The ensemble logic (softmax → weighted average → argmax) and submission format remain identical.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm




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
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0] + ".png")
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.test:
            return image
        else:
            label = int(self.annotations.iloc[idx, 1])
            return image, label




## === cell 2
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)



## === cell 3
test_csv_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_root_dir = "/kaggle/input/aptos2019-blindness-detection/test_images"

test_dataset = BlindnessDataset(
    test_csv_file, test_root_dir, transform=transform, test=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)




## === cell 4
def find_checkpoints(root="/kaggle/input", exts=(".pth", ".pt")):
    found = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(exts):
                found.append(os.path.join(dirpath, fn))
    return sorted(found)


available_ckpts = find_checkpoints("/kaggle/input")
print(f"Found {len(available_ckpts)} checkpoint files under /kaggle/input")
for p in available_ckpts[:20]:
    print(" -", p)



## === cell 5
model_paths = {
    "efficientnet_b4": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/1/efficentNet_b4.pth",
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



## === cell 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def best_match_checkpoint(model_key: str, ckpt_paths):
    key = model_key.lower()
    candidates = []
    for p in ckpt_paths:
        pl = p.lower()
        if key in pl or key.replace("_", "") in pl:
            candidates.append(p)
    candidates = sorted(candidates, key=lambda x: (len(x), x))
    return candidates[0] if candidates else None


def extract_state_dict(state):
    if isinstance(state, dict):
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            return state["state_dict"]
        if "model" in state and isinstance(state["model"], dict):
            return state["model"]
    return state


def strip_module_prefix(sd):
    if not isinstance(sd, dict):
        return sd
    if any(k.startswith("module.") for k in sd.keys()):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


models_list = []
loaded_model_keys = []

for model_key, path in model_paths.items():
    model_name = model_names[model_key]

    ckpt_to_use = path
    if not os.path.exists(ckpt_to_use):
        matched = best_match_checkpoint(model_key, available_ckpts)
        if matched is not None:
            ckpt_to_use = matched

    if ckpt_to_use is not None and os.path.exists(ckpt_to_use):
        model = timm.create_model(model_name, pretrained=False, num_classes=5)
        state = torch.load(ckpt_to_use, map_location="cpu")
        sd = strip_module_prefix(extract_state_dict(state))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Loaded checkpoint for {model_key} from: {ckpt_to_use}")
        if missing:
            print(f"  - missing keys (first 5): {missing[:5]}")
        if unexpected:
            print(f"  - unexpected keys (first 5): {unexpected[:5]}")
    else:
        model = timm.create_model(model_name, pretrained=True, num_classes=5)
        print(
            f"WARNING: No checkpoint found for {model_key}; using ImageNet pretrained weights."
        )

    model.to(device)
    model.eval()

    models_list.append(model)
    loaded_model_keys.append(model_key)

if len(models_list) == 0:
    raise RuntimeError(
        "No models available for inference (no checkpoints found and no fallback created)."
    )

print("Loaded models:", loaded_model_keys)



## === cell 7
validation_scores = {
    "resnet18": 0.879,
    "efficientnet_b0": 0.8922,
    "efficientnet_b1": 0.894,
    "efficientnet_b2": 0.898,
    "efficientnet_b3": 0.897,
    "efficientnet_b4": 0.893,
    "efficientnet_b5": 0.870,
    "inception_resnet_v2": 0.896,
    "inception_v4": 0.8875,
    "seresnext50_32x4d": 0.8652,
    "seresnext101_32x4d": 0.9083,
}

used_scores = {k: validation_scores.get(k, 1.0) for k in loaded_model_keys}
total_score = sum(used_scores.values())
weights = {k: v / total_score for k, v in used_scores.items()}
print("Ensemble weights:", weights)




## === cell 8
def quadratic_weighted_kappa(y_true, y_pred, num_classes=5):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    assert y_true.shape == y_pred.shape

    O = np.zeros((num_classes, num_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < num_classes and 0 <= b < num_classes:
            O[a, b] += 1.0

    act_hist = np.bincount(y_true, minlength=num_classes).astype(np.float64)
    pred_hist = np.bincount(y_pred, minlength=num_classes).astype(np.float64)

    E = np.outer(act_hist, pred_hist)
    E = E / E.sum() * O.sum() if E.sum() > 0 else E

    W = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i in range(num_classes):
        for j in range(num_classes):
            W[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    return 1.0 - num / den if den > 0 else 0.0


def probs_to_expected_grade(probs_5):
    classes = np.arange(5, dtype=np.float32)
    return (probs_5 * classes[None, :]).sum(axis=1)


def apply_thresholds(x, th):
    x = np.asarray(x)
    th = np.asarray(th)
    return np.digitize(x, bins=th, right=False).astype(int)


def infer_probs(loader):
    all_out = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Infer", leave=False):
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            images = images.to(device, non_blocking=True)

            outputs_per_model = []
            for model_key, model in zip(loaded_model_keys, models_list):
                probs = nn.functional.softmax(model(images), dim=1)  # [B,5]
                outputs_per_model.append(weights[model_key] * probs)

            weighted_outputs = torch.stack(outputs_per_model, dim=0).sum(dim=0)  # [B,5]
            all_out.append(weighted_outputs.cpu().numpy())
    return np.concatenate(all_out, axis=0)


def fit_thresholds(y_true, x, init_th=(0.5, 1.5, 2.5, 3.5)):
    y_true = np.asarray(y_true, dtype=int)
    x = np.asarray(x, dtype=np.float32)

    th = np.array(init_th, dtype=np.float32)

    lo, hi = float(np.percentile(x, 1)), float(np.percentile(x, 99))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = float(x.min()), float(x.max())
    lo, hi = float(lo), float(hi)

    best = quadratic_weighted_kappa(y_true, apply_thresholds(x, th))
    for _ in range(6):
        improved = False
        for i in range(4):
            left = lo if i == 0 else float(th[i - 1] + 1e-3)
            right = hi if i == 3 else float(th[i + 1] - 1e-3)
            if left >= right:
                continue

            grid = np.linspace(left, right, 25, dtype=np.float32)
            local_best_th = th[i]
            local_best = best
            for cand in grid:
                th_try = th.copy()
                th_try[i] = cand
                score = quadratic_weighted_kappa(y_true, apply_thresholds(x, th_try))
                if score > local_best:
                    local_best = score
                    local_best_th = cand
            if local_best > best:
                th[i] = local_best_th
                best = local_best
                improved = True
        if not improved:
            break
    return th, best




## === cell 9
train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
train_root_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"
train_df = pd.read_csv(train_csv_file)

rng = np.random.RandomState(42)
idx = np.arange(len(train_df))
rng.shuffle(idx)
val_size = int(0.15 * len(idx))
val_idx = idx[:val_size]

val_csv_path = "/kaggle/working/_val_split.csv"
train_df.iloc[val_idx].to_csv(val_csv_path, index=False)

val_dataset = BlindnessDataset(
    val_csv_path, train_root_dir, transform=transform, test=False
)
val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

val_probs = infer_probs(val_loader)
val_expected = probs_to_expected_grade(val_probs)
val_y = train_df.iloc[val_idx]["diagnosis"].values.astype(int)

best_th, best_qwk = fit_thresholds(val_y, val_expected, init_th=(0.5, 1.5, 2.5, 3.5))
print("Fitted thresholds:", best_th)
print("Validation QWK (on split):", float(best_qwk))



## === cell 10
test_probs = infer_probs(test_loader)
test_expected = probs_to_expected_grade(test_probs)

final_predictions = apply_thresholds(test_expected, best_th).astype(int)



## === cell 11
test_ids = pd.read_csv(test_csv_file)["id_code"].values
if len(test_ids) != len(final_predictions):
    raise RuntimeError(
        f"Prediction length mismatch: ids={len(test_ids)} preds={len(final_predictions)}"
    )

submission_df = pd.DataFrame({"id_code": test_ids, "diagnosis": final_predictions})
submission_df.to_csv("submission.csv", index=False)

print(submission_df.head())
print("Wrote submission.csv with shape:", submission_df.shape)
