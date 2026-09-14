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

3.7

# 3. Installed packages

fastai==2.8.5
geopandas==0.14.4
numpy==1.26.4
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

0.8694537518105276

# 6. Current score

0.17356

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.03655) has done: 'I fix the empty-predictions runtime error by ensuring the FastAI test dataloader is created correctly (with a no-label DataBlock and `num_workers=0` for Kaggle stability) and by adding a safe fallback that runs direct PyTorch inference over the image paths if the dataloader yields zero batches. This keeps your core model (ResNet18 + 5-way head) and prediction logic (argmax over logits) unchanged while making the pipeline robust end-to-end. I also make the data-root detection work with Kaggle’s `/kaggle/input/...` layout in addition to `../input`, since your environment paths show `/kaggle/input` and `/kaggle/data`. Finally, I ensure `submission.csv` is always written with the required columns and row alignment.'
- What this solution (achieved -0.03655) has done: 'Your current score is strongly negative because the model used at inference is effectively an untrained 5-class head on top of ImageNet ResNet18 (your `model_aptos.pth` is not found, so predictions are close to random), which destroys QWK. The smallest change that should move the score substantially toward your target is to ensure we actually load a trained checkpoint if it exists in the dataset and, if not, fall back to a competition-proven pretrained DR model available directly in `fastai` (without changing the architecture or prediction rule). To keep core logic identical, I keep ResNet18 + 5-way linear head, the same preprocessing (Resize(224)+ImageNet normalize), and the same argmax label decision; I only (1) broaden checkpoint discovery to common filenames/locations and (2) add a safe fastai-learner load fallback for a `.pkl` export if present. This should move performance upward toward the target band while remaining minimal and Kaggle-stable.'
- What this solution (achieved 0.0) has done: 'Your current negative QWK is most consistent with a label/metric mismatch: you are training/using a 5-way classifier but your inference turns logits into hard classes with plain `argmax`, which is typically badly calibrated for QWK and can yield very poor kappa even when accuracy is not terrible. To keep the core model and inference pipeline intact, I only add a small, deterministic post-processing step that maps the model’s continuous “severity score” (expected value of class probabilities) into 0–4 using fixed thresholds tuned to the competition’s common class imbalance, while leaving the model weights, preprocessing, and dataloader logic unchanged. This is a minimal semantic change (still outputs integer 0–4) but aligns predictions better with QWK’s ordinal nature and should move the score substantially upward toward your target band. I also ensure the submission remains correctly aligned and always writes `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with a submission that’s mostly predicting a single class (or near-random) because no trained checkpoint is actually being loaded, so the ImageNet-init ResNet18 head produces poorly ordered outputs for QWK. To move the score toward your target with minimal change and identical model/inference semantics, I (1) make checkpoint discovery more robust by also checking common Kaggle working/output locations (in addition to `/kaggle/input` and `/kaggle/data`), and (2) verify that any loaded checkpoint really changed the weights (otherwise warn clearly). This keeps your architecture (ResNet18 + 5-way head), transforms (Resize(224)+ImageNet normalize), and prediction mapping (expected value + fixed thresholds) unchanged, but increases the chance you’re using trained weights rather than random. The rest of the pipeline (dataloader + direct inference fallback + submission writing) is kept intact.'
- What this solution (achieved 0.17356) has done: 'Your current 0.0 score is most consistent with the model producing near-constant or poorly ordered class outputs (often because no truly trained DR checkpoint is loaded), and the current fixed 0.5/1.5/2.5/3.5 bucketing of an expected-value score is usually not well-calibrated for QWK on this competition. To move the score upward toward your target with minimal semantic change, I keep your exact model and inference pipeline but add a tiny, deterministic calibration step: compute thresholds from the model’s own predicted severity scores so the final class distribution matches the competition’s typical/known train distribution proportions. This preserves “logits → softmax → expected severity → thresholds → int 0..4”, but chooses thresholds more appropriately than the fixed ones, which should improve ordinal agreement and thus QWK when the model has any signal. I also keep your robust checkpoint discovery/loading and submission writing unchanged, only adding this post-processing and a safe fallback to fixed thresholds if something goes wrong.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import torch
from torch import nn
from torchvision import models

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

CANDIDATE_INPUT_ROOTS = ["../input", "/kaggle/input", "/kaggle/data", "/kaggle/working"]
INPUT_ROOT = None
for r in CANDIDATE_INPUT_ROOTS:
    if os.path.isdir(r):
        INPUT_ROOT = r
        break
if INPUT_ROOT is None:
    raise FileNotFoundError(
        "Could not find an input root (../input, /kaggle/input, /kaggle/data, /kaggle/working)."
    )

print("Using INPUT_ROOT:", INPUT_ROOT)
print("Listing INPUT_ROOT:", os.listdir(INPUT_ROOT)[:50])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def find_file(root, filename):
    """Find filename under root recursively and return the first match."""
    for dirpath, dirnames, filenames in os.walk(root):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None


def find_first_existing(root, filenames):
    """Return first file found under root from a list of candidate filenames."""
    for fn in filenames:
        p = find_file(root, fn)
        if p is not None:
            return p
    return None


def _tensor_checksum(state_dict: dict, n_first: int = 12) -> float:
    """Small deterministic checksum to detect whether loading actually changed weights (helps catch wrong keys)."""
    s = 0.0
    c = 0
    for k in sorted(state_dict.keys()):
        v = state_dict[k]
        if torch.is_tensor(v) and v.numel() > 0:
            s += float(v.reshape(-1)[:32].float().sum().item())
            c += 1
            if c >= n_first:
                break
    return s


torch_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
torch_model.fc = nn.Linear(torch_model.fc.in_features, 5)

CKPT_CANDIDATES = [
    "model_aptos.pth",
    "aptos_resnet18.pth",
    "resnet18_aptos.pth",
    "best.pth",
    "model.pth",
    "stage-2.pth",
    "resnet18.pth",
    "model-best.pth",
]
PKL_CANDIDATES = [
    "export.pkl",
    "model.pkl",
    "learner.pkl",
]

SEARCH_ROOTS = []
for r in [INPUT_ROOT, "/kaggle/input", "/kaggle/data", "/kaggle/working"]:
    if r and os.path.isdir(r) and r not in SEARCH_ROOTS:
        SEARCH_ROOTS.append(r)

model_path = None
pkl_path = None
for r in SEARCH_ROOTS:
    if model_path is None:
        model_path = find_first_existing(r, CKPT_CANDIDATES)
    if pkl_path is None:
        pkl_path = find_first_existing(r, PKL_CANDIDATES)

loaded = False

if model_path is not None:
    print("Found PyTorch model checkpoint at:", model_path)
    try:
        before_sd = torch_model.state_dict()
        before_ck = _tensor_checksum(before_sd)

        ckpt = torch.load(model_path, map_location="cpu")
        if isinstance(ckpt, dict):
            state = ckpt.get("state_dict", ckpt)
            cleaned = {}
            for k, v in state.items():
                nk = k
                if nk.startswith("module."):
                    nk = nk[len("module.") :]
                if nk.startswith("model."):
                    nk = nk[len("model.") :]
                cleaned[nk] = v
            missing, unexpected = torch_model.load_state_dict(cleaned, strict=False)
            after_ck = _tensor_checksum(torch_model.state_dict())

            print(
                "Loaded state_dict. Missing keys:",
                len(missing),
                "Unexpected keys:",
                len(unexpected),
            )
            if abs(after_ck - before_ck) < 1e-6:
                print(
                    "Warning: checkpoint load did not change model weights checksum; checkpoint may be incompatible."
                )
            loaded = True
        elif isinstance(ckpt, nn.Module):
            torch_model = ckpt
            loaded = True
        else:
            raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")
    except Exception as e:
        print(
            "Warning: failed to load .pth checkpoint, will try .pkl if available. Error:",
            repr(e),
        )

if (not loaded) and (pkl_path is not None):
    print("Found fastai exported learner at:", pkl_path)
    try:
        from fastai.vision.all import load_learner

        learn = load_learner(pkl_path, cpu=True)
        fa_model = learn.model
        torch_model = fa_model
        loaded = True
        print("Loaded model from fastai learner export.")
    except Exception as e:
        print(
            "Warning: failed to load fastai learner export; using torchvision init weights only. Error:",
            repr(e),
        )

if not loaded:
    print(
        "Warning: no trained checkpoint found/loaded under search roots; using torchvision init weights only (likely low score)."
    )

torch_model = torch_model.to(device)
torch_model.eval()
print("Model ready:", type(torch_model).__name__)


def logits_to_expected_score(logits: torch.Tensor) -> np.ndarray:
    """Expected severity score in [0,4] from logits [N,5]."""
    probs = torch.softmax(logits, dim=1)
    classes = torch.arange(5, device=probs.device, dtype=probs.dtype).view(1, -1)
    score = (probs * classes).sum(dim=1)
    return score.detach().cpu().numpy()


def expected_score_to_labels_by_thresholds(
    score: np.ndarray, thr: np.ndarray
) -> np.ndarray:
    """Map continuous score to integer labels using 4 thresholds."""
    thr = np.asarray(thr, dtype=np.float32).reshape(-1)
    if thr.shape[0] != 4:
        raise ValueError("thr must have 4 elements")
    return np.digitize(score.astype(np.float32), bins=thr).astype(np.int64)


def infer_thresholds_match_train_prior(
    score: np.ndarray, train_csv_path: str
) -> np.ndarray:
    """
    Minimal, deterministic calibration for QWK:
    Choose thresholds so predicted label proportions match train label proportions.
    This keeps the same semantic pipeline (expected value + thresholds) but sets thresholds
    more appropriately than fixed [0.5,1.5,2.5,3.5].
    """
    fixed = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
    try:
        tr = pd.read_csv(train_csv_path)
        if "diagnosis" not in tr.columns:
            return fixed
        counts = (
            tr["diagnosis"]
            .value_counts()
            .reindex([0, 1, 2, 3, 4], fill_value=0)
            .to_numpy()
        )
        if counts.sum() <= 0:
            return fixed
        props = counts / counts.sum()

        cum = np.cumsum(props)[:-1]  # length 4
        q = np.clip(cum, 1e-6, 1 - 1e-6)
        thr = np.quantile(score.astype(np.float64), q).astype(np.float32)

        eps = 1e-4
        for i in range(1, len(thr)):
            if thr[i] <= thr[i - 1]:
                thr[i] = thr[i - 1] + eps
        return thr
    except Exception:
        return fixed




## === cell 1
from fastai.vision.all import *
from PIL import Image

CANDIDATE_ROOTS = [
    os.path.join(INPUT_ROOT, "aptos2019-blindness-detection"),
    os.path.join(INPUT_ROOT, "kaggle", "data", "aptos2019-blindness-detection"),
    os.path.join(INPUT_ROOT, "kaggle", "input", "aptos2019-blindness-detection"),
    INPUT_ROOT,  # fallback to flat layout
]

DATA_ROOT = None
for r in CANDIDATE_ROOTS:
    if os.path.exists(os.path.join(r, "test.csv")):
        if os.path.isdir(os.path.join(r, "test_images")) or os.path.isdir(
            os.path.join(r, "test_images", "test_images")
        ):
            DATA_ROOT = r
            break

if DATA_ROOT is None:
    found_test = find_file(INPUT_ROOT, "test.csv")
    if found_test:
        DATA_ROOT = os.path.dirname(found_test)

if DATA_ROOT is None:
    raise FileNotFoundError(
        f"Could not locate aptos2019-blindness-detection data root under {INPUT_ROOT}"
    )

test_csv = os.path.join(DATA_ROOT, "test.csv")
train_csv = os.path.join(DATA_ROOT, "train.csv")
test_folder = os.path.join(DATA_ROOT, "test_images")
if not os.path.isdir(test_folder) and os.path.isdir(
    os.path.join(test_folder, "test_images")
):
    test_folder = os.path.join(test_folder, "test_images")

print("Using DATA_ROOT:", DATA_ROOT)
print("Using test.csv:", test_csv)
print("Using train.csv:", train_csv)
print("Using test_images folder:", test_folder)

dff = pd.read_csv(test_csv)
if "id_code" not in dff.columns:
    raise ValueError("test.csv must contain 'id_code' column")

dff["image_path"] = (
    dff["id_code"].astype(str).map(lambda x: os.path.join(test_folder, f"{x}.png"))
)

missing = [p for p in dff["image_path"].tolist() if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(
        f"Missing {len(missing)} test images. Example missing file: {missing[0]}"
    )

dff = dff.reset_index(drop=True)

imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

dblock = DataBlock(
    blocks=(ImageBlock,),
    get_x=ColReader("image_path"),
    splitter=IndexSplitter([]),  # everything goes to "valid"
    item_tfms=Resize(224),
    batch_tfms=Normalize.from_stats(*imagenet_stats),
)

dls = dblock.dataloaders(dff, bs=32, shuffle=False, num_workers=0, path=DATA_ROOT)
test_dl = dls.valid

all_logits = []
torch_model.eval()
with torch.no_grad():
    for batch in test_dl:
        if isinstance(batch, (tuple, list)):
            xb = batch[0]
        else:
            xb = batch
        xb = xb.to(device)
        logits = torch_model(xb)
        all_logits.append(logits.detach().cpu())

if len(all_logits) == 0:
    print(
        "Warning: test dataloader produced 0 batches; falling back to direct torch inference."
    )
    from torchvision import transforms

    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_stats[0], std=imagenet_stats[1]),
        ]
    )

    direct_logits = []
    with torch.no_grad():
        for p in dff["image_path"].tolist():
            img = Image.open(p).convert("RGB")
            xb = tfm(img).unsqueeze(0).to(device)
            direct_logits.append(torch_model(xb).detach().cpu())
    logits_all = torch.cat(direct_logits, dim=0)
else:
    logits_all = torch.cat(all_logits, dim=0)

score = logits_to_expected_score(logits_all)
thr = infer_thresholds_match_train_prior(score, train_csv)
labels = expected_score_to_labels_by_thresholds(score, thr).astype(int).tolist()

assert len(labels) == len(
    dff
), f"Predictions length {len(labels)} != test rows {len(dff)}"
print("Using thresholds:", thr.tolist())
print(
    "Predictions ready. Label distribution:",
    pd.Series(labels).value_counts().sort_index().to_dict(),
)



## === cell 2
ids = dff["id_code"].astype(str).tolist()
submit = pd.DataFrame({"id_code": ids, "diagnosis": labels})
submit["diagnosis"] = submit["diagnosis"].astype(int)
submit = submit[["id_code", "diagnosis"]]

out_path = "./submission.csv"
submit.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(submit.head())
print("Rows:", len(submit), "Columns:", list(submit.columns))
