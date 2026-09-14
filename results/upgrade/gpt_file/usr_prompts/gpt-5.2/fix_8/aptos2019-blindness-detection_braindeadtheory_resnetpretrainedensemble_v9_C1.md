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

3.8

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
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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

0.8791714649727451

# 6. Current score

0.71893

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.08307) has done: 'I fix the crash by removing the hardcoded dependency on a missing `../input/resnet152/FinalResnet152_0.pt` file and instead load torchvision’s built-in ResNet-152 ImageNet weights (available in the Kaggle environment). I also make the tqdm import compatible with scripts (the notebook-specific tqdm module can break) and ensure images are always converted to RGB to avoid occasional channel-related errors. Finally, I keep your inference logic intact (argmax over 5 logits) and guarantee a valid `submission.csv` with the required `id_code,diagnosis` columns is written.'
- What this solution (achieved 0.01829) has done: 'Your score is strongly negative because the model is doing inference with a randomly initialized final classification layer (and no fine-tuning), so predictions are essentially random. To move the score up toward the 0.879 target while keeping your core inference logic (argmax over 5 logits) unchanged, the minimal legitimate fix is to load a trained checkpoint for the modified 5-class head (and full model) if it exists in the dataset working directory. I add a small, robust checkpoint search/load step that looks for common `.pt/.pth` files under `../input/aptos2019-blindness-detection/` (and falls back to current behavior if none are found), plus deterministic settings to avoid run-to-run drift. The output submission format/path remains identical (`submission.csv` with `id_code,diagnosis`).'
- What this solution (achieved 0.01829) has done: 'Your current score is low because the checkpoint search is accidentally picking up large non-model `.pth/.pt` files (e.g., the 7.7GB `train_images.zip` placeholder or other irrelevant binaries) and then “loading” them with `strict=False`, which leaves the 5-class head effectively random. I make the checkpoint finder strict to only consider plausible PyTorch state_dict files by (1) excluding known non-model paths/files and huge archive-like sizes, and (2) verifying the loaded object looks like a model/state_dict with at least the `fc.weight` tensor of shape `[5, in_features]` before applying it. If no valid checkpoint exists, the code fall back exactly as before, still producing `submission.csv`. These are minimal changes focused only on making sure a real trained checkpoint (if present) is loaded correctly, which should move the kappa score upward toward your target.'
- What this solution (achieved 0.01829) has done: 'Your score is still near-random because the model is almost certainly running with an untrained 5-class head (either no checkpoint exists in `../input`, or the loader is rejecting/ignoring a usable checkpoint). To move the score upward toward the 0.879 target while preserving your exact model and argmax inference logic, I (1) enable “fast but safe” checkpoint validation by also accepting common alternative head names (`classifier.*`, `head.*`, `last_linear.*`) and then remapping them into `fc.*` when shapes match, and (2) ensure inference uses `torch.inference_mode()` plus an explicit `softmax`-free argmax (same semantics) with correct device placement. These changes keep architecture/training untouched, only improving the odds that a real trained checkpoint in the dataset is correctly loaded (which is the main determinant of kappa here). The script still writes a valid `submission.csv` with `id_code,diagnosis` aligned to `test.csv`.'
- What this solution (achieved 0.01829) has done: 'Your current score indicates the 5-class head is effectively untrained; the most likely cause is that your checkpoint loader is rejecting/ignoring real checkpoints because it requires an exact `fc.weight` shape match, which won’t hold if the checkpoint is from a different backbone or stores weights under different keys. To move the score up toward the 0.879 target without changing your architecture or inference semantics (still ResNet152 + Linear(5) and argmax), I make the loader also accept a checkpoint that at least contains a plausible 5-class classifier head (any `[5, *]` weight + `[5]` bias), then transplant that head into your current model when possible. Additionally, I ensure the model is put in `eval()` before prediction inside `compute_predictions` to avoid any accidental train-mode behavior if the function is reused. The submission writing remains identical (`submission.csv` with `id_code,diagnosis` aligned to `test.csv`).'
- What this solution (achieved 0.01829) has done: 'Your current score (0.01829) is far below the target (0.87917), and the most likely reason is still that you’re effectively using an ImageNet backbone with a randomly initialized 5-class head (i.e., no real DR-trained weights are being loaded). The smallest legitimate way to move kappa upward without changing your model/inference logic is to (1) make checkpoint loading actually succeed when a suitable state_dict exists by also handling common wrappers (e.g., `model_state`, `ema_state_dict`) and key prefixes (e.g., `model.`), and (2) stop “best-guess” selecting by file size and instead prioritize checkpoints whose tensors clearly include a 5-class head (and ideally a full ResNet state). These changes keep the exact same architecture (ResNet152 + Linear(5)) and the same argmax inference, but materially increase the chance you load meaningful trained weights if they’re present, which is the main driver toward your target score. The script still runs end-to-end and always writes a valid `submission.csv` with `id_code,diagnosis` aligned to `test.csv`.'
- What this solution (achieved 0.71893) has done: 'Your current score is near-random because the model is effectively an ImageNet backbone plus a randomly initialized 5-class head, and the checkpoint search is (correctly) not finding any usable DR-trained weights in the provided dataset. To move the score materially upward toward the 0.879 target while keeping your core architecture (ResNet152 + Linear(5)), loss/metric semantics (argmax class 0–4), and inference loop intact, the minimal legitimate change is to add a short fine-tuning step on `train.csv` using the same preprocessing and then run the same argmax inference on test. To keep changes minimal and stable within the 600s limit, we freeze the backbone and train only `model.fc` (the 5-class head) for a small, fixed number of epochs with standard CrossEntropyLoss. This preserves your overall pipeline and produces a valid `submission.csv` aligned to `test.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

from tqdm import tqdm



## === cell 1
transform = transforms.Compose(
    [
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.460, 0.247, 0.080], [0.249, 0.138, 0.081]),
    ]
)


class APTOSDataset(Dataset):
    """Eye images dataset."""

    def __init__(self, csv_file, filetype, transform=None):
        self.eye_frame = pd.read_csv(csv_file)
        self.filetype = filetype
        self.transform = transform

    def __len__(self):
        return len(self.eye_frame)

    def __getitem__(self, idx):
        if self.filetype == "train":
            img_name = os.path.join(
                "../input/aptos2019-blindness-detection/train_images",
                self.eye_frame.loc[idx, "id_code"] + ".png",
            )
            image = Image.open(img_name).convert("RGB")
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
            return image, int(self.eye_frame.loc[idx, "diagnosis"])
        else:
            img_name = os.path.join(
                "../input/aptos2019-blindness-detection/test_images",
                self.eye_frame.loc[idx, "id_code"] + ".png",
            )
            image = Image.open(img_name).convert("RGB")
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
            return image, self.eye_frame.loc[idx, "id_code"]




## === cell 2
test_dataset = APTOSDataset(
    csv_file="../input/aptos2019-blindness-detection/test.csv",
    filetype="test",
    transform=transform,
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=24,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

weights = torchvision.models.ResNet152_Weights.DEFAULT
model = torchvision.models.resnet152(weights=weights)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)
model = model.to(device)


def _is_plausible_model_file(path, sz):
    fn = os.path.basename(path).lower()
    bad_name_tokens = [
        "train_images",
        "test_images",
        "sample_submission",
        "train.csv",
        "test.csv",
        ".zip",
        ".csv",
        ".md",
        ".txt",
        ".json",
    ]
    if any(tok in fn for tok in bad_name_tokens):
        return False
    if sz >= 2_000_000_000:  # 2GB
        return False
    if sz < 200_000:
        return False
    return True


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
            "ema_state_dict",
            "ema",
            "model_state",
            "model_ema",
        ]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        if any(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    return None


def _normalize_state_dict_keys(state):
    if state is None:
        return None

    prefixes = ("module.", "model.", "net.", "backbone.", "encoder.")
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if any(k.startswith(p) for k in state.keys()):
                state = {
                    k[len(p) :] if k.startswith(p) else k: v for k, v in state.items()
                }
                changed = True

    return state


def _find_any_5class_head(state):
    if state is None:
        return None, None

    if "fc.weight" in state:
        w = state.get("fc.weight")
        b = state.get("fc.bias", None)
        if isinstance(w, torch.Tensor) and w.ndim == 2 and w.shape[0] == 5:
            if (b is None) or (isinstance(b, torch.Tensor) and b.shape == (5,)):
                return w, b

    alt_prefixes = [
        "classifier.",
        "head.",
        "last_linear.",
        "final.",
        "logits.",
        "model.fc.",
        "net.fc.",
    ]
    for p in alt_prefixes:
        w_key = p + "weight"
        b_key = p + "bias"
        if w_key in state:
            w = state.get(w_key)
            b = state.get(b_key, None)
            if isinstance(w, torch.Tensor) and w.ndim == 2 and w.shape[0] == 5:
                if (b is None) or (isinstance(b, torch.Tensor) and b.shape == (5,)):
                    return w, b

    for k, w in state.items():
        if not (isinstance(w, torch.Tensor) and k.endswith(".weight")):
            continue
        if w.ndim == 2 and w.shape[0] == 5:
            b_key = k[: -len(".weight")] + ".bias"
            b = state.get(b_key, None)
            if (b is None) or (isinstance(b, torch.Tensor) and b.shape == (5,)):
                return w, b

    return None, None


def _state_dict_matches_5class_resnet_head_exact(state, in_features):
    w = state.get("fc.weight", None)
    b = state.get("fc.bias", None)
    if not isinstance(w, torch.Tensor):
        return False
    if list(w.shape) != [5, in_features]:
        return False
    if isinstance(b, torch.Tensor) and list(b.shape) != [5]:
        return False
    return True


def _score_checkpoint_candidate(state, in_features):
    if state is None or not isinstance(state, dict):
        return -10_000

    score = 0
    head_w, head_b = _find_any_5class_head(state)
    if isinstance(head_w, torch.Tensor) and head_w.ndim == 2 and head_w.shape[0] == 5:
        score += 10_000
        if head_w.shape[1] == in_features:
            score += 5_000
    if _state_dict_matches_5class_resnet_head_exact(state, in_features):
        score += 20_000

    score += min(len(state), 5000)
    return score


def _find_and_load_best_checkpoint(model, search_roots, device):
    exts = (".pt", ".pth", ".bin")
    candidates = []
    for root in search_roots:
        if not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                lfn = fn.lower()
                if not lfn.endswith(exts):
                    continue
                full = os.path.join(dirpath, fn)
                try:
                    sz = os.path.getsize(full)
                except OSError:
                    continue
                if not _is_plausible_model_file(full, sz):
                    continue
                candidates.append(full)

    scored = []
    for path in candidates:
        try:
            ckpt = torch.load(path, map_location="cpu")
        except Exception:
            continue
        state = _extract_state_dict(ckpt)
        state = _normalize_state_dict_keys(state)
        if state is None:
            continue
        s = _score_checkpoint_candidate(state, model.fc.in_features)
        if s > 0:
            scored.append((s, path))

    scored.sort(reverse=True, key=lambda x: x[0])

    for _, path in scored:
        try:
            ckpt = torch.load(path, map_location=device)
        except Exception:
            continue

        state = _extract_state_dict(ckpt)
        state = _normalize_state_dict_keys(state)
        if state is None:
            continue

        if _state_dict_matches_5class_resnet_head_exact(state, model.fc.in_features):
            missing, unexpected = model.load_state_dict(state, strict=False)
            return path, missing, unexpected

        head_w, head_b = _find_any_5class_head(state)
        if isinstance(head_w, torch.Tensor):
            if head_w.shape[1] == model.fc.in_features:
                with torch.no_grad():
                    model.fc.weight.copy_(
                        head_w.to(device=device, dtype=model.fc.weight.dtype)
                    )
                    if head_b is not None:
                        model.fc.bias.copy_(
                            head_b.to(device=device, dtype=model.fc.bias.dtype)
                        )
                return (
                    path,
                    ["(partial) loaded classifier head only"],
                    ["(partial) ignored non-matching backbone"],
                )

    return None, None, None


search_roots = [
    "../input/aptos2019-blindness-detection",
    "../input",
]
ckpt_path, missing, unexpected = _find_and_load_best_checkpoint(
    model, search_roots, device
)
if ckpt_path is not None:
    print(f"Loaded checkpoint (validated/partial): {ckpt_path}")
    print(f"Missing keys/info: {missing}; Unexpected keys/info: {unexpected}")
else:
    print(
        "No valid trained checkpoint found in ../input; using ImageNet backbone + randomly initialized 5-class head (likely low score)."
    )



## === cell 3
train_dataset = APTOSDataset(
    csv_file="../input/aptos2019-blindness-detection/train.csv",
    filetype="train",
    transform=transform,
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=24,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

for p in model.parameters():
    p.requires_grad = False
for p in model.fc.parameters():
    p.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=3e-4)

model.train()
epochs = 2  # small fixed training; deterministic seeds set above; avoids timeouts while improving score vs random
for epoch in range(epochs):
    running_loss = 0.0
    n = 0
    for inputs, labels in tqdm(
        train_loader, desc=f"Train head epoch {epoch+1}/{epochs}"
    ):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        running_loss += loss.item() * bs
        n += bs

    print(f"Epoch {epoch+1}/{epochs} - head train loss: {running_loss / max(n, 1):.5f}")

model.eval()




## === cell 4
def compute_predictions(model, model_type, data_loader, device):
    model.eval()

    if model_type == "train":
        predictions = []
        correct_pred, num_examples = 0, 0
        for inputs, labels in tqdm(data_loader, desc="Predict (train)"):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.append(preds)
            num_examples += labels.size(0)
            correct_pred += (preds == labels).sum()
        return predictions, correct_pred.item() / num_examples * 100

    else:
        predictions = []
        img_ids = []
        for inputs, img_id in tqdm(data_loader, desc="Predict (test)"):
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.detach().cpu().tolist())
            img_ids.extend(list(img_id))

        final_predictions = pd.DataFrame({"id_code": img_ids, "diagnosis": predictions})
        final_predictions.to_csv("submission.csv", index=False)
        return final_predictions




## === cell 5
with torch.inference_mode():
    model.eval()
    print("Computing Test Predictions")
    test_predictions = compute_predictions(model, "test", test_loader, device)
    print(test_predictions.head())
    print("Wrote submission.csv")
