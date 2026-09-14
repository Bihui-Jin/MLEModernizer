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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

albumentations==2.0.8
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
scipy==1.15.3
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
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8931701420368692

# 6. Current score

0.39873

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.08408) has done: 'I fix the Albumentations v2 API break that prevents `sub_aug` from being created (the root cause of the cascading `NameError`s), by switching `RandomResizedCrop(height, width, ...)` to the new `RandomResizedCrop(size=(h,w), ...)` signature. I also keep the rest of the inference/ensemble logic identical, only adding a safe CPU fallback for the augmentation if Albumentations ever fails at runtime so the notebook always produces a valid `submission.csv`. Finally, I make the file/path resolution a bit more robust while keeping the same weights-loading behavior, so it runs end-to-end in this Kaggle environment and writes the correct submission format.'
- What this solution (achieved 0.27354) has done: 'Your current score (0.08408) is consistent with the models effectively being near-random on 5 classes, which typically happens when the custom `.pth` weights are not being found/loaded (your logs likely show the warnings). The smallest score-improving change that preserves your core inference/ensemble logic is to (1) resolve the weight paths robustly by searching `/kaggle/input` for the specific filenames you reference, and (2) load state dicts with a safer key-cleaning that also handles `model`/`net` wrappers so the heads actually get trained weights. I keep your exact model architectures, TTA loop, and 50/50 logit ensembling unchanged; the only goal is to ensure the intended weights are truly loaded. This should move accuracy sharply upward toward your target band because the solution becomes a real pretrained ensemble instead of random heads.'
- What this solution (achieved 0.08744) has done: 'Your current score (0.27354) is far below the target (0.89317), which strongly suggests the ensemble is still effectively untrained because the `.pth` weights aren’t being properly matched to the model keys (so most layers stay at ImageNet/random head). I keep your exact model architectures, TTA loop, and 50/50 logit ensembling, but make the state-dict loading stricter in a safe way: detect common checkpoint wrappers and automatically remap keys (including `backbone.`/`encoder.`/`classifier.`/`fc.` patterns) so the weights actually land on the intended layers. I also instantiate a fresh EfficientNet per fold (same architecture) so fold loading can’t be polluted by partially-mismatched keys from a previous fold. These are minimal, directly score-relevant changes that should move accuracy sharply upward toward your target band without changing inference semantics.'
- What this solution (achieved 0.39013) has done: 'Your score is far below the target, and the current code strongly suggests the custom weights are either not being found or (more likely) not being loaded into the right layers (so you’re effectively predicting with an ImageNet backbone + random 5-class head). I make the smallest score-relevant change: improve checkpoint key cleaning/remapping so common training wrappers (e.g., `model.`, `module.`, `backbone.`, `state_dict`) and common head names (`fc`, `classifier`, `head`) map correctly to your torchvision ResNeXt/EfficientNet modules. I also add a “best-candidate” selection that tries a few remap variants and chooses the one with the highest overlap before loading (still `strict=False`, so core inference logic is unchanged). This should move the accuracy sharply upward toward your target without changing your architectures, TTA loops, or ensembling math.'
- What this solution (achieved 0.07436) has done: 'Your current score is far below the target, so we should only make changes that increase accuracy without changing the modeling/ensemble logic. The biggest likely remaining issue is preprocessing mismatch: EfficientNet-B4 and ResNeXt50 expect different normalization stats in torchvision, but your pipeline forces ImageNet mean/std for both, which can severely hurt if your checkpoints were trained with the model-specific default transforms. I keep the same TTA loop, same architectures, and same 50/50 logit ensembling, but apply the correct per-model normalization inside `predict_model_tta` (ResNeXt uses ImageNet mean/std; EfficientNet-B4 uses the weights’ default mean/std). This is a minimal, metric-aligned change that typically yields a large accuracy jump when weights are correctly loaded.'
- What this solution (achieved 0.39873) has done: 'Your score is far below the target, so we should only make changes that plausibly increase accuracy without altering the ensemble math, architectures, loss, or training approach (inference-only here). The most likely remaining issue is a preprocessing mismatch with the checkpoints: your TTA augmentation is always producing 512×512 crops, but these public Cassava checkpoints are commonly trained/inferred at 380 (EffNet-B4) and 512 (ResNeXt), and mixing sizes can tank accuracy. I keep the same TTA loop and augmentations, but make the crop size model-specific (380 for EffNet-B4, 512 for ResNeXt) while keeping normalization per model as you already do. I also make the resize deterministic by ensuring each model always receives its expected input size before normalization, which typically yields a large jump when weights are actually loaded.'

# 9. Code solution

## === cell 0
import os
import glob
import re

import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import softmax

import torch
import torch.nn as nn
from torchvision import models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)



## === cell 1
resnet_model_path = "../input/rn-wc-tta-calr-clahe-cutmix/model(24).pth"
effnet_model_path = "../input/en-b4-tta-calr-clahe-v2-8/model(15).pth"
effnet_model_folds_path = [
    "../input/en-b4-tta-calr-clahe-v3-5-folds/EFFICIENT_NET_B4_0_11.pth",
    "../input/en-b4-tta-calr-clahe-v3-5-folds/EFFICIENT_NET_B4_1_10.pth",
    "../input/en-b4-tta-calr-clahe-v3-5-folds/EFFICIENT_NET_B4_2_13.pth",
    "../input/en-b4-tta-calr-clahe-v3-5-folds/EFFICIENT_NET_B4_3_12.pth",
    "../input/en-b4-tta-calr-clahe-v3-5-folds/EFFICIENT_NET_B4_4_11.pth",
]

sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"




## === cell 2
def _find_pth_by_basename(basename: str, root: str = "/kaggle/input"):
    matches = glob.glob(os.path.join(root, "**", basename), recursive=True)
    matches = [m for m in matches if os.path.isfile(m)]
    return sorted(matches)[0] if matches else None


def _find_pth_by_regex(pattern: str, root: str = "/kaggle/input"):
    cand = []
    for p in glob.glob(os.path.join(root, "**", "*.pth"), recursive=True):
        if re.search(pattern, os.path.basename(p), flags=re.IGNORECASE):
            cand.append(p)
    return sorted(cand)[0] if cand else None


def _maybe_resolve_path(path: str, fallback_pattern: str):
    if path and os.path.exists(path):
        return path

    if path:
        bn = os.path.basename(path)
        found = _find_pth_by_basename(bn)
        if found is not None:
            return found

    found = _find_pth_by_regex(fallback_pattern)
    return found  # may be None


resnet_model_path = _maybe_resolve_path(
    resnet_model_path, r"(resn|resnext|rn|next50|32x4d).*\.pth|model.*\.pth"
)

resolved_folds = []
for p in effnet_model_folds_path:
    if os.path.exists(p):
        resolved_folds.append(p)
    else:
        bn = os.path.basename(p)
        found = _find_pth_by_basename(bn)
        if found is not None:
            resolved_folds.append(found)

if len(resolved_folds) == 0:
    b4_pths = sorted(glob.glob("/kaggle/input/**/*.pth", recursive=True))
    b4_pths = [
        p
        for p in b4_pths
        if re.search(r"(b4|efficient.*b4|eff.*b4)", os.path.basename(p), re.IGNORECASE)
    ]
    resolved_folds = b4_pths[:5] if len(b4_pths) else []

effnet_model_folds_path = resolved_folds

print("Resolved resnet weights:", resnet_model_path)
print("Resolved effnet fold weights:", effnet_model_folds_path)




## === cell 3
def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
            "ema_state_dict",
            "ema",
        ]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


def _clean_state_dict(sd):
    sd = _extract_state_dict(sd)
    if not isinstance(sd, dict):
        return sd

    new_sd = {}
    for k, v in sd.items():
        nk = k
        for pref in ["module.", "model.", "net.", "backbone.", "encoder."]:
            if nk.startswith(pref):
                nk = nk[len(pref) :]
        new_sd[nk] = v
    return new_sd


def _try_variant_remaps(sd: dict):
    variants = []

    variants.append(dict(sd))

    for pref in ["module.", "model.", "net."]:
        if any(k.startswith(pref) for k in sd.keys()):
            variants.append(
                {k[len(pref) :]: v for k, v in sd.items() if k.startswith(pref)}
            )

    for pref in ["module.", "model.", "net."]:
        variants.append({pref + k: v for k, v in sd.items()})

    def _swap_prefix(d, a, b):
        out = {}
        for k, v in d.items():
            out[k.replace(a, b, 1) if k.startswith(a) else k] = v
        return out

    for base in list(variants):
        variants.append(_swap_prefix(base, "head.", "classifier."))
        variants.append(_swap_prefix(base, "classifier.", "fc."))
        variants.append(_swap_prefix(base, "fc.", "classifier."))

        tmp = {}
        for k, v in base.items():
            nk = k
            if "classifier.1" in nk:
                nk = nk.replace("classifier.1", "classifier.0")
            tmp[nk] = v
        variants.append(tmp)

        tmp2 = {}
        for k, v in base.items():
            nk = k
            if "classifier.0" in nk:
                nk = nk.replace("classifier.0", "classifier.1")
            tmp2[nk] = v
        variants.append(tmp2)

    uniq = []
    seen = set()
    for v in variants:
        sig = (len(v), hash(tuple(sorted(v.keys()))))
        if sig not in seen:
            seen.add(sig)
            uniq.append(v)
    return uniq


def _pick_best_state_dict_for_model(sd: dict, model: nn.Module) -> dict:
    if not isinstance(sd, dict):
        return sd

    model_keys = set(model.state_dict().keys())
    if not model_keys:
        return sd

    best = sd
    best_overlap = len(model_keys.intersection(sd.keys()))

    for cand in _try_variant_remaps(sd):
        overlap = len(model_keys.intersection(cand.keys()))
        if overlap > best_overlap:
            best_overlap = overlap
            best = cand

    return best


def _load_weights_best_effort(model: nn.Module, weight_path: str, desc: str):
    ckpt = torch.load(weight_path, map_location="cpu")
    sd = _clean_state_dict(ckpt)
    sd = _pick_best_state_dict_for_model(sd, model)

    missing, unexpected = model.load_state_dict(sd, strict=False)

    model_sd = model.state_dict()
    overlap = len(set(sd.keys()).intersection(model_sd.keys()))
    overlap_ratio = overlap / max(1, len(model_sd))

    print(f"Loaded {desc}: {os.path.basename(weight_path)}")
    print(
        f"{desc} missing keys: {len(missing)} | unexpected keys: {len(unexpected)} | key-overlap ratio: {overlap_ratio:.3f}"
    )
    return missing, unexpected




## === cell 4
resnet_model = models.resnext50_32x4d(
    weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2
)
resnet_model.fc = nn.Linear(2048, 5)
resnet_model.to(DEVICE).eval()

if resnet_model_path is not None and os.path.exists(resnet_model_path):
    _load_weights_best_effort(resnet_model, resnet_model_path, desc="ResNeXt")
else:
    print(
        "WARNING: ResNeXt custom weights not found; using ImageNet backbone + random 5-class head."
    )


def _build_effnet_b4():
    m = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    if isinstance(m.classifier, nn.Sequential):
        in_features = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_features, 5)
    else:
        in_features = m.classifier.in_features
        m.classifier = nn.Linear(in_features, 5)
    return m


effnet_model = _build_effnet_b4().to(DEVICE).eval()




## === cell 5
def load_image_np(path: str) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        return np.array(im)




## === cell 6
def _get_norm_for_model(model_name: str):
    if model_name.lower().startswith("effnet"):
        w = models.EfficientNet_B4_Weights.IMAGENET1K_V1
    else:
        w = models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2
    mean = list(w.transforms().mean)
    std = list(w.transforms().std)
    return A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0)


def _get_sub_aug_no_norm(model_name: str) -> A.Compose:
    size = 380 if model_name.lower().startswith("effnet") else 512
    return A.Compose(
        [
            A.RandomResizedCrop(size=(size, size), scale=(0.5, 1.0)),
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.8),
        ],
        p=1.0,
    )


def _img_to_tensor_chw(img_hwc: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(img_hwc).permute(2, 0, 1).contiguous().float()
    return x




## === cell 7
def predict_model_tta(
    model: torch.nn.Module,
    image_paths: list,
    tta_count: int,
    batch_size: int = 16,
    model_name: str = "resnet",
) -> np.ndarray:
    n = len(image_paths)
    preds = np.zeros((n, 5), dtype=np.float32)

    norm = _get_norm_for_model(model_name)
    sub_aug_no_norm = _get_sub_aug_no_norm(model_name)

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        batch_paths = image_paths[start:end]
        bs = len(batch_paths)

        imgs0 = [load_image_np(p) for p in batch_paths]

        logits_sum = torch.zeros((bs, 5), device=DEVICE, dtype=torch.float32)
        for _ in range(tta_count):
            batch_imgs = []
            for img0 in imgs0:
                try:
                    img = sub_aug_no_norm(image=img0)["image"]
                except Exception:
                    img = img0

                img = norm(image=img)["image"]
                batch_imgs.append(_img_to_tensor_chw(img))

            x = torch.stack(batch_imgs, dim=0).to(DEVICE, non_blocking=True)
            out = model(x).detach().float()
            logits_sum += out

        logits_avg = (logits_sum / float(tta_count)).cpu().numpy()
        preds[start:end] = logits_avg

    return preds




## === cell 8
sample_sub = pd.read_csv(sample_sub_path)
image_paths = [
    os.path.join(test_images_path, iid) for iid in sample_sub["image_id"].tolist()
]

tta_count = 5
resnet_predictions = predict_model_tta(
    resnet_model, image_paths, tta_count=tta_count, batch_size=16, model_name="resnet"
)
print("ResNeXt preds shape:", resnet_predictions.shape)



## === cell 9
tta_count = 1
effnet_fold_preds = []

if len(effnet_model_folds_path) > 0:
    for model_path in effnet_model_folds_path:
        if not os.path.exists(model_path):
            continue

        fold_model = _build_effnet_b4().to(DEVICE).eval()
        _load_weights_best_effort(fold_model, model_path, desc="EffNetB4 fold")

        fold_preds = predict_model_tta(
            fold_model,
            image_paths,
            tta_count=tta_count,
            batch_size=16,
            model_name="effnet",
        )
        effnet_fold_preds.append(fold_preds)

if len(effnet_fold_preds) == 0:
    print(
        "WARNING: EfficientNet fold weights not found; using ImageNet backbone + random 5-class head."
    )
    effnet_predictions = predict_model_tta(
        effnet_model,
        image_paths,
        tta_count=tta_count,
        batch_size=16,
        model_name="effnet",
    )
else:
    effnet_predictions = np.mean(np.stack(effnet_fold_preds, axis=0), axis=0)

print("EffNet preds shape:", effnet_predictions.shape)



## === cell 10
combine_logits = (effnet_predictions * 0.5) + (resnet_predictions * 0.5)
combine_preds = softmax(combine_logits, axis=1).argmax(axis=1).astype(int)

sub_df = pd.DataFrame(
    {"image_id": sample_sub["image_id"].values, "label": combine_preds}
)
sub_df.to_csv("submission.csv", index=False)

print(sub_df.head())
print("Saved submission.csv with shape:", sub_df.shape)
print(
    "submission.csv exists:",
    os.path.exists("submission.csv"),
    "size:",
    os.path.getsize("submission.csv") if os.path.exists("submission.csv") else None,
)
