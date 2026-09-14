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

0.5505869260275978

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.23101) has done: 'The failure comes from trying to load pretrained ensemble weights from a Kaggle input dataset that is not present in your environment, leaving `models_list` empty and causing downstream `torch.cat()` and `final_predictions` errors. I keep your inference-only pipeline intact, but add a safe fallback: if no weight files are found, we create the same model architecture with ImageNet pretrained weights so predictions can still be generated. I also make image loading robust (`RGB` conversion) and ensure `torch.load(..., map_location=...)` works on CPU/GPU consistently. Finally, I always write a valid `submission.csv` with the required columns and correct row alignment.'
- What this solution (achieved 0.1353) has done: 'Your current fallback uses ImageNet-pretrained backbones with a 5-class head that has random weights, so predictions are essentially noise and QWK stays low; we can move toward the target by making the fallback produce *sensible ordinal labels* from pretrained features without changing your overall inference-only flow. I keep the same dataloader, transforms, and `timm.create_model(...)` usage, but in the “no finetuned weights found” case I switch the model to output pretrained logits (1000 classes) and then map those logits to 5 ordered bins using the distribution from `train.csv`. This preserves the core idea (single-pass inference + arg-like discretization) while making outputs non-random and typically much closer to a reasonable kappa baseline. The submission writing stays identical and still produces a valid `submission.csv`.'
- What this solution (achieved 0.73661) has done: 'Your current fallback turns ImageNet probabilities into labels using only the max-softmax “confidence”, which is a weak signal for an ordinal medical severity task and tends to produce near-random ordering. To move the score upward toward the 0.55 target with minimal change and without altering the inference-only structure, I keep the same model/dataloader flow but switch the fallback to extract a stable embedding (`num_classes=0`) and use simple cosine similarity to 5 class prototypes computed from the training set. This preserves the same overall pipeline (single forward pass + deterministic postprocess) while making predictions align with DR class structure much better than confidence binning. I also keep the original finetuned-weight path intact and only activate the prototype fallback when finetuned weights are missing.'
- What this solution (achieved 0.0) has done: 'I fix the deterministic-algorithms crash by setting the required `CUBLAS_WORKSPACE_CONFIG` environment variable *before* importing torch, and by falling back to deterministic-safe cosine similarity via `F.linear` (normalized dot product) if needed. I also make sure the prototype tensor is on the correct device/dtype and guard against any remaining edge cases so `final_predictions` is always created. Finally, I keep your inference/prototype fallback logic intact and ensure a valid `submission.csv` is always written with the correct columns and row alignment.'
- What this solution (achieved 0.0) has done: 'Your current score is 0.0 because the script crashes in the non-fallback branch: when finetuned weights are found it call the models (which output 5 logits), but `models_list` currently contains a `num_classes=0` backbone in the missing-weights case and then later tries to `argmax` on `all_outputs` only if `fallback_prototypes` is False. To move the score upward toward the 0.5506 target with minimal change, I (1) make the code robust so it never ends up in an inconsistent “no prototypes but also no logits” state, and (2) ensure the finetuned-weight path and fallback-prototype path are mutually consistent and always produce `final_predictions`. I also expand `model_paths` to include more potential local weight filenames if present, but keep the same ensemble/inference logic; if none exist, we stay with the prototype fallback. Finally, I keep submission formatting identical and add a couple of guards so the CSV is always written.'
- What this solution (achieved 0.0) has done: 'I make the script reliably produce a submission by fixing the only remaining execution hazard: `infer_with_prototypes()` currently references `device` and `models_list` as outer-scope globals, which can break in Kaggle cell execution/order and lead to “Not yielded” when calibration runs. I pass `device` and `models_list` explicitly into `infer_with_prototypes()` (and use those inside), keeping the exact same prototype inference logic and calibration loop. This is a minimal, score-relevant change because it prevents calibration/inference from crashing and allows the prototype fallback (your best-performing path when finetuned weights are missing) to run end-to-end. No model architecture, transforms, losses, or training loops are changed.'
- What this solution (achieved 0.0) has done: 'Your “Not yielded” comes from an execution-risky prototype-building/calibration path: `persistent_workers=True` can crash in Kaggle notebook contexts, and the prototype-building uses `seresnext101_32x4d` which is heavy enough to time out/oom when extracting embeddings for 1200 images. To get a valid submission reliably and move the score upward toward the 0.5506 target, I keep the exact same prototype fallback logic (feature extractor → class prototypes → cosine similarity → expected score → thresholding), but (1) make DataLoader worker settings robust, (2) build prototypes on a slightly larger yet still safe stratified subset with a lighter batch size to reduce memory spikes, and (3) expand the threshold search very slightly (still tiny) so QWK calibration is less brittle without changing evaluation semantics. These are minimal, score-relevant changes that preserve your overall approach and should produce a submission end-to-end within the time limit.'
- What this solution (achieved 0.0) has done: 'Your current pipeline likely reports “Not yielded” because it can time out while building prototypes (heavy `seresnext101_32x4d` over a large subset) or crash on GPU determinism edge-cases during the prototype stage, preventing the final `submission.csv` write. I keep your exact inference-only logic (prototype fallback vs finetuned logits) but make two minimal, score-relevant stabilizations: (1) reduce prototype build compute by capping per-class sampling more tightly while keeping stratification, and (2) make the DataLoader settings safer (workers=0 on Kaggle to avoid multiprocessing stalls) while preserving deterministic behavior. I also add a final safety guard to always produce a submission (even if prototype building fails unexpectedly) by falling back to the calibrated “expected score + thresholds” defaults using random-but-deterministic class priors from train.csv—this is only used if an exception occurs, and it ensures a valid CSV is always written. These changes are aimed at reliably generating a submission and moving score upward from “no submission” toward your target band without changing the core model/metric semantics.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm

ImageFile.LOAD_TRUNCATED_IMAGES = True

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass




## === cell 1
class BlindnessDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, test=False):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.test = test

        self._fallback_image = Image.new("RGB", (224, 224), (0, 0, 0))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_id = self.annotations.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, img_id + ".png")

        try:
            image = Image.open(img_name).convert("RGB")
        except Exception:
            image = self._fallback_image

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
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)



## === cell 3
test_csv_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_root_dir = "/kaggle/input/aptos2019-blindness-detection/test_images"

test_df = pd.read_csv(test_csv_file)

_test_ordered_path = "/kaggle/working/_test_ordered.csv"
test_df.to_csv(_test_ordered_path, index=False)

test_dataset = BlindnessDataset(
    _test_ordered_path, test_root_dir, transform=transform, test=True
)

_cuda = torch.cuda.is_available()

_NUM_WORKERS = 0

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=_NUM_WORKERS,
    pin_memory=_cuda,
    persistent_workers=False,
)



## === cell 4
"""
Guarantee consistent inference mode:
- If any finetuned 5-class weights are found and loaded, do logits averaging + argmax.
- Otherwise build class prototypes from a pretrained feature extractor and predict via cosine similarity.

Keep prototype-fallback path. Minimal stability/correctness changes only.
"""

candidate_weight_paths = [
    "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/1/seresnext101_32x4d.pth",
    "/kaggle/input/aptos_ensamble-models/seresnext101_32x4d.pth",
    "/kaggle/input/aptos2019-blindness-detection/seresnext101_32x4d.pth",
]

model_name = "seresnext101_32x4d"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

found_weight_path = None
for p in candidate_weight_paths:
    if os.path.exists(p):
        found_weight_path = p
        break

models_list = []
loaded_any = False
fallback_prototypes = False

if found_weight_path is not None:
    model = timm.create_model(model_name, pretrained=False, num_classes=5)
    state = torch.load(found_weight_path, map_location="cpu")
    model.load_state_dict(state)
    loaded_any = True
    fallback_prototypes = False
    models_list.append(model)
else:
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    loaded_any = False
    fallback_prototypes = True
    models_list.append(model)

for m in models_list:
    m.to(device)
    m.eval()

print(
    f"Using {len(models_list)} model(s). Loaded finetuned weights: {loaded_any}. Fallback prototypes: {fallback_prototypes}"
)

train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
train_root_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"

prototypes = None
counts = None

prototype_build_ok = True

if fallback_prototypes:
    try:
        train_df_full = pd.read_csv(train_csv_file)
        rng = np.random.RandomState(SEED)

        PROTO_MAX = 900

        per_class_caps = []
        for c in range(5):
            idx_c = np.where(train_df_full["diagnosis"].astype(int).values == c)[0]
            rng.shuffle(idx_c)
            per_class_caps.append(
                min(
                    len(idx_c),
                    max(50, int(round(PROTO_MAX * len(idx_c) / len(train_df_full)))),
                )
            )

        total = sum(per_class_caps)
        if total > PROTO_MAX:
            order = np.argsort(per_class_caps)[::-1]
            i = 0
            while total > PROTO_MAX and i < 10000:
                c = int(order[i % 5])
                if per_class_caps[c] > 50:
                    per_class_caps[c] -= 1
                    total -= 1
                i += 1

        selected_indices = []
        for c in range(5):
            idx_c = np.where(train_df_full["diagnosis"].astype(int).values == c)[0]
            rng.shuffle(idx_c)
            selected_indices.append(idx_c[: per_class_caps[c]])
        selected_indices = np.unique(np.concatenate(selected_indices))
        proto_df = train_df_full.iloc[selected_indices].reset_index(drop=True)

        proto_csv = "/kaggle/working/_train_for_prototypes.csv"
        proto_df.to_csv(proto_csv, index=False)

        train_dataset = BlindnessDataset(
            proto_csv, train_root_dir, transform=transform, test=False
        )

        _cuda = torch.cuda.is_available()

        _NUM_WORKERS = 0

        train_loader = DataLoader(
            train_dataset,
            batch_size=12 if _cuda else 8,
            shuffle=False,
            num_workers=_NUM_WORKERS,
            pin_memory=_cuda,
            persistent_workers=False,
        )

        sums = None
        counts = torch.zeros(5, dtype=torch.long)

        with torch.no_grad():
            for images, labels in tqdm(
                train_loader, desc="Building class prototypes (capped subset)"
            ):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                feats = [
                    m(images).unsqueeze(0) for m in models_list
                ]  # [n_models, B, D]
                feats = torch.cat(feats, dim=0)
                feats = torch.mean(feats, dim=0)  # [B, D]

                if sums is None:
                    sums = torch.zeros(
                        (5, feats.shape[1]), device=device, dtype=feats.dtype
                    )

                for c in range(5):
                    mask = labels == c
                    if mask.any():
                        sums[c] += feats[mask].sum(dim=0)
                        counts[c] += int(mask.sum().item())

        counts_safe = counts.clamp(min=1).to(device)
        prototypes = sums / counts_safe.unsqueeze(1)
        prototypes = torch.nn.functional.normalize(prototypes, p=2, dim=1).to(device)

        print("Prototype subset size:", len(proto_df))
        print("Prototype counts:", counts.cpu().tolist())
    except Exception as e:
        prototype_build_ok = False
        prototypes = None
        print(
            "WARNING: Prototype building failed; will use backup predictions to ensure submission:",
            repr(e),
        )



## === cell 5
import torch.nn.functional as F


def quadratic_weighted_kappa(y_true, y_pred, num_classes=5):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_true = np.clip(y_true, 0, num_classes - 1)
    y_pred = np.clip(y_pred, 0, num_classes - 1)

    O = np.zeros((num_classes, num_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        O[a, b] += 1.0

    act_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E * (O.sum() / E.sum())

    W = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i in range(num_classes):
        for j in range(num_classes):
            W[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)

    denom = (W * E).sum()
    if denom == 0:
        return 0.0
    return 1.0 - (W * O).sum() / denom


def infer_expected_scores(loader, prototypes_tensor, temp, models, device):
    """Return continuous expected score in [0,4] for each sample (prototype fallback path)."""
    expected_list = []
    prot = F.normalize(prototypes_tensor.to(device=device), p=2, dim=1)

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (tuple, list)) and len(batch) >= 1:
                images = batch[0]
            else:
                images = batch

            images = images.to(device, non_blocking=True)

            feats = [m(images).unsqueeze(0) for m in models]
            feats = torch.cat(feats, dim=0).mean(dim=0)
            feats = F.normalize(feats, p=2, dim=1)

            sims = F.linear(feats, prot.to(dtype=feats.dtype))  # [B, 5]
            probs = torch.softmax(sims / float(temp), dim=1)
            expected = (probs * torch.arange(5, device=device, dtype=probs.dtype)).sum(
                dim=1
            )
            expected_list.append(expected.detach().cpu().numpy())

    if len(expected_list) == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(expected_list, axis=0).astype(np.float32)


def apply_thresholds(expected_scores, thresholds):
    """
    Map expected_scores to ordinal labels 0..4 using 4 thresholds:
      pred = number of thresholds exceeded.
    """
    e = np.asarray(expected_scores, dtype=np.float32).reshape(-1)
    th = np.asarray(thresholds, dtype=np.float32).reshape(-1)
    if th.shape[0] != 4:
        raise ValueError("thresholds must have length 4.")
    th = np.sort(th)
    pred = (e[:, None] > th[None, :]).sum(axis=1).astype(int)
    return np.clip(pred, 0, 4)


FALLBACK_TEMP = 2.5
FALLBACK_THRESHOLDS = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)

if fallback_prototypes and prototype_build_ok:
    train_df = pd.read_csv(train_csv_file)

    n = len(train_df)
    val_size = max(1, int(0.10 * n))

    rng = np.random.RandomState(SEED)
    val_indices = []
    for c in range(5):
        idx_c = np.where(train_df["diagnosis"].astype(int).values == c)[0]
        rng.shuffle(idx_c)
        take = max(1, int(round(0.10 * len(idx_c))))
        val_indices.append(idx_c[:take])
    val_indices = np.unique(np.concatenate(val_indices))
    if len(val_indices) < val_size:
        remaining = np.setdiff1d(np.arange(n), val_indices, assume_unique=False)
        rng.shuffle(remaining)
        val_indices = np.concatenate(
            [val_indices, remaining[: (val_size - len(val_indices))]]
        )
    elif len(val_indices) > val_size:
        rng.shuffle(val_indices)
        val_indices = val_indices[:val_size]

    val_mask = np.zeros(n, dtype=bool)
    val_mask[val_indices] = True
    va_df = train_df.loc[val_mask].reset_index(drop=True)

    va_csv = "/kaggle/working/_train_va.csv"
    va_df.to_csv(va_csv, index=False)

    va_dataset = BlindnessDataset(
        va_csv, train_root_dir, transform=transform, test=False
    )
    _cuda = torch.cuda.is_available()

    _NUM_WORKERS = 0

    va_loader = DataLoader(
        va_dataset,
        batch_size=24 if _cuda else 16,
        shuffle=False,
        num_workers=_NUM_WORKERS,
        pin_memory=_cuda,
        persistent_workers=False,
    )

    temps = [1.25, 1.5, 2.0, 2.5, 3.0, 3.5]
    best_temp = FALLBACK_TEMP
    best_kappa = -1e9

    y_true = va_df["diagnosis"].astype(int).values

    for t in temps:
        exp_scores = infer_expected_scores(
            va_loader, prototypes, temp=float(t), models=models_list, device=device
        )
        y_pred = apply_thresholds(exp_scores, FALLBACK_THRESHOLDS)
        k = quadratic_weighted_kappa(y_true, y_pred, num_classes=5)
        if k > best_kappa:
            best_kappa = k
            best_temp = float(t)

    FALLBACK_TEMP = best_temp
    print(f"Calibrated FALLBACK_TEMP={FALLBACK_TEMP} using val QWK={best_kappa:.5f}")

    exp_scores = infer_expected_scores(
        va_loader,
        prototypes,
        temp=float(FALLBACK_TEMP),
        models=models_list,
        device=device,
    )

    base = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
    deltas = np.array([-0.35, -0.20, -0.10, 0.0, 0.10, 0.20, 0.35], dtype=np.float32)

    best_th = base.copy()
    best_kappa_th = -1e9

    cand0 = base[0] + deltas
    cand1 = base[1] + deltas
    cand2 = base[2] + deltas
    cand3 = base[3] + deltas

    for t0 in cand0:
        for t1 in cand1:
            if t1 <= t0:
                continue
            for t2 in cand2:
                if t2 <= t1:
                    continue
                for t3 in cand3:
                    if t3 <= t2:
                        continue
                    th = np.array([t0, t1, t2, t3], dtype=np.float32)
                    y_pred = apply_thresholds(exp_scores, th)
                    k = quadratic_weighted_kappa(y_true, y_pred, num_classes=5)
                    if k > best_kappa_th:
                        best_kappa_th = k
                        best_th = th

    FALLBACK_THRESHOLDS = best_th
    print(
        "Calibrated FALLBACK_THRESHOLDS="
        f"{FALLBACK_THRESHOLDS.tolist()} using val QWK={best_kappa_th:.5f}"
    )



## === cell 6
all_outputs = []
all_preds = []

_prot_for_test = None
if fallback_prototypes and prototype_build_ok:
    if prototypes is None:
        prototype_build_ok = False
    else:
        _prot_for_test = F.normalize(prototypes.to(device), p=2, dim=1)

with torch.no_grad():
    for images in tqdm(test_loader, desc="Infer test"):
        images = images.to(device, non_blocking=True)

        if not fallback_prototypes:
            outputs = [model(images).unsqueeze(0) for model in models_list]
            outputs = torch.cat(outputs, dim=0)  # [n_models, batch, 5]
            averaged_outputs = torch.mean(outputs, dim=0)  # [batch, 5]
            all_outputs.append(averaged_outputs.detach().cpu().numpy())
        else:
            if prototype_build_ok and _prot_for_test is not None:
                feats = [
                    m(images).unsqueeze(0) for m in models_list
                ]  # [n_models, B, D]
                feats = torch.cat(feats, dim=0)
                feats = torch.mean(feats, dim=0)  # [B, D]
                feats = torch.nn.functional.normalize(feats, p=2, dim=1)  # [B, D]

                sims = F.linear(feats, _prot_for_test.to(dtype=feats.dtype))  # [B, 5]
                probs = torch.softmax(sims / float(FALLBACK_TEMP), dim=1)  # [B, 5]
                expected = (
                    probs * torch.arange(5, device=device, dtype=probs.dtype)
                ).sum(dim=1)
                expected_np = expected.detach().cpu().numpy().astype(np.float32)
                preds = apply_thresholds(expected_np, FALLBACK_THRESHOLDS)
            else:
                train_df = pd.read_csv(
                    "/kaggle/input/aptos2019-blindness-detection/train.csv"
                )
                pri = (
                    train_df["diagnosis"]
                    .astype(int)
                    .value_counts()
                    .reindex(range(5), fill_value=1)
                    .values
                )
                pri = pri / pri.sum()
                rng = np.random.RandomState(SEED)
                preds = rng.choice(np.arange(5), size=images.shape[0], p=pri).astype(
                    int
                )

            all_preds.append(preds)

if not fallback_prototypes:
    if len(all_outputs) == 0:
        raise RuntimeError("No logits collected for test inference.")
    all_outputs = np.concatenate(all_outputs, axis=0)
    final_predictions = np.argmax(all_outputs, axis=1).astype(int)
    final_predictions = np.clip(final_predictions, 0, 4)
else:
    if len(all_preds) == 0:
        raise RuntimeError("No predictions collected for test inference.")
    final_predictions = np.concatenate(all_preds, axis=0).astype(int)
    final_predictions = np.clip(final_predictions, 0, 4)

print("Pred shape:", final_predictions.shape)
print("Pred label counts:", np.bincount(final_predictions, minlength=5).tolist())



## === cell 7
test_ids = pd.read_csv(_test_ordered_path)["id_code"].values

assert len(test_ids) == 367, f"Unexpected test size: {len(test_ids)}"
assert len(test_ids) == len(
    final_predictions
), "Mismatch between test ids and predictions length."

submission_df = pd.DataFrame({"id_code": test_ids, "diagnosis": final_predictions})
submission_df.to_csv("submission.csv", index=False)

print(submission_df.head())
print("Wrote submission.csv with shape:", submission_df.shape)
