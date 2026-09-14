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

0.8926602111463993

# 6. Current score

0.06608

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.07285) has done: 'I fix the runtime failures by removing the hard dependency on missing external model weight files and making the ensemble robust when no weights are found. To preserve the core “multi-model softmax ensemble” logic, the code try to load each checkpoint if it exists, otherwise it skip that model and proceed with the remaining ones. If none are available (as in your environment), it fall back to using timm pretrained ImageNet weights for the same model architectures so inference can still run end-to-end and produce a valid `submission.csv`. I also fix image loading to ensure RGB inputs (some PNGs can be RGBA/grayscale) and make `state_dict` loading compatible with common checkpoint formats.'
- What this solution (achieved 0.00653) has done: 'Your current score is low because the fallback uses ImageNet-pretrained classifiers with randomly initialized 5-class heads, so predictions are essentially noise. To move the score toward your target while preserving the same “multi-model softmax ensemble” inference logic, the smallest legitimate fix is to actually use diabetic-retinopathy-trained weights; in this environment those are typically available via the competition’s dataset or your Kaggle “input” datasets. I make the code automatically search `/kaggle/input/**` for matching checkpoint filenames (e.g., `resnet18.pth`, `efficentNet_b3.pth`, etc.) and load them when found, otherwise fall back exactly as before. I also ensure that the ensemble weights are renormalized over only the successfully loaded models so the weighted sum stays correctly scaled.'
- What this solution (achieved -0.04809) has done: 'Your score is near-random because none of the DR-trained checkpoints are being found, so the code falls back to ImageNet-pretrained backbones with a random 5-class head. The smallest change to move the score toward your target is to (1) point `model_paths` at the correct dataset name (`aptos-ensamble-models` not `aptos_ensamble-models`) and (2) make checkpoint loading robust to common “head name” differences (e.g., `classifier.*` vs `fc.*`) while keeping the same architectures and weighted-softmax ensemble logic. This should enable actually loading the intended DR-trained weights when present, drastically improving QWK without changing your inference semantics. I also keep the weight renormalization over only the successfully loaded models (already correct) and leave transforms/model list intact.'
- What this solution (achieved 0.01011) has done: 'Your current score is far below the target because the intended DR-trained checkpoints still aren’t being loaded, so the ensemble falls back to ImageNet-pretrained backbones with effectively random 5-class heads. The smallest score-relevant change is to make checkpoint discovery robust to (a) slightly different dataset folder names and (b) the common “efficentNet_*.pth” misspelling by searching for multiple filename variants per model. I also keep your exact ensemble/softmax logic, but tighten checkpoint loading to prefer strict loading after a smarter key-remap (and only then fall back to non-strict), which increases the chance we load the trained classification head correctly. These changes should move QWK substantially upward toward your target without changing the model list, transforms, or prediction semantics.'
- What this solution (achieved 0.06608) has done: 'Your score is far below the target because the code is still not loading DR-trained checkpoints (so predictions are effectively random), so the smallest relevant change is to make checkpoint discovery actually find the common variants in `/kaggle/input` (including `.pt/.bin`, different casing, and common “_foldX”/“best” naming). Then, to increase the chance the trained 5-class head loads correctly without changing model logic, we add a minimal key-normalization step that also handles `model_state_dict`/Lightning-style keys and strips common prefixes (`model.`, `net.`). Finally, we keep your weighted-softmax ensemble exactly the same, but ensure we don’t silently accept a “loaded” checkpoint that has near-zero overlap (which would behave like random weights), falling back to ImageNet only in that case.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd
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
        transforms.Resize((224, 224)),
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
    test_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True
)



## === cell 4
model_paths = {
    "resnet18": "/kaggle/input/aptos-ensamble-models/pytorch/ensamble_v2/2/resnet18.pth",
    "efficientnet_b0": "/kaggle/input/aptos-ensamble-models/pytorch/ensamble_v2/2/efficentNet_b0.pth",
    "efficientnet_b1": "/kaggle/input/aptos-ensamble-models/pytorch/ensamble_v2/2/efficentNet_b1.pth",
    "efficientnet_b2": "/kaggle/input/aptos-ensamble-models/pytorch/ensamble_v2/2/efficentNet_b2.pth",
    "efficientnet_b3": "/kaggle/input/aptos-ensamble-models/pytorch/ensamble_v2/2/efficentNet_b3.pth",
    "efficientnet_b4": "/kaggle/input/aptos-ensamble-models/pytorch/ensamble_v2/2/efficentNet_b4.pth",
    "efficientnet_b5": "/kaggle/input/aptos-ensamble-models/pytorch/ensamble_v2/2/efficentNet_b5.pth",
    "inception_resnet_v2": "/kaggle/input/aptos-ensamble-models/pytorch/ensamble_v2/2/inception_resnet_v2.pth",
    "inception_v4": "/kaggle/input/aptos-ensamble-models/pytorch/ensamble_v2/2/inception_v4.pth",
    "seresnext50_32x4d": "/kaggle/input/aptos-ensamble-models/pytorch/ensamble_v2/2/seresnext50_32x4d.pth",
    "seresnext101_32x4d": "/kaggle/input/aptos-ensamble-models/pytorch/ensamble_v2/2/seresnext101_32x4d.pth",
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
validation_scores = {
    "resnet18": 0.879,
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



## === cell 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "model_state_dict", "net", "weights"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def _normalize_state_dict_keys(sd: dict) -> dict:
    if not isinstance(sd, dict):
        return sd
    out = {}
    for k, v in sd.items():
        nk = k
        for prefix in ("module.", "model.", "net."):
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]
        out[nk] = v
    return out


def _candidate_checkpoint_filenames(model_key: str, original_path: str) -> list[str]:
    """
    Score-relevant: broaden filename variants to actually find DR-trained weights.
    This preserves the same ensemble inference; it only improves checkpoint discovery.
    """
    base = os.path.basename(original_path)
    stem, ext = os.path.splitext(base)
    exts = [ext] if ext else [".pth"]
    exts = list(dict.fromkeys(exts + [".pth", ".pt", ".bin"]))

    cands = set()
    for e in exts:
        cands.add(stem + e)

    suffixes = ["", "_best", "-best", "_final", "-final", "_last", "-last"]
    fold_suffixes = [
        "",
        "_fold0",
        "_fold1",
        "_fold2",
        "_fold3",
        "_fold4",
        "-fold0",
        "-fold1",
        "-fold2",
        "-fold3",
        "-fold4",
    ]
    for e in exts:
        for s in suffixes:
            for f in fold_suffixes:
                cands.add(f"{stem}{s}{f}{e}")

    if model_key.startswith("efficientnet_b"):
        b = model_key.split("_")[-1]  # b0..b5
        bases = [
            f"efficientnet_{b}",
            f"efficientnet-{b}",
            f"efficentNet_{b}",
            f"efficentnet_{b}",
            f"EfficientNet_{b}",
            f"EfficientNet-{b}",
        ]
        for bstem in bases:
            for e in exts:
                cands.add(bstem + e)
                for s in suffixes:
                    for f in fold_suffixes:
                        cands.add(f"{bstem}{s}{f}{e}")

    return list(cands)


def _resolve_checkpoint_path(original_path: str, model_key: str) -> str | None:
    """
    Score-relevant: robustly locate checkpoints inside /kaggle/input even if the dataset
    folder name differs or the filename varies slightly.
    """
    if os.path.exists(original_path):
        return original_path

    orig_dir = os.path.dirname(original_path)
    for fname in _candidate_checkpoint_filenames(model_key, original_path):
        trial = os.path.join(orig_dir, fname)
        if os.path.exists(trial):
            return trial

    for fname in _candidate_checkpoint_filenames(model_key, original_path):
        hits = glob.glob(os.path.join("/kaggle/input", "**", fname), recursive=True)
        if len(hits) > 0:
            return hits[0]

    stem = os.path.splitext(os.path.basename(original_path))[0]
    patterns = [
        os.path.join("/kaggle/input", "**", stem + ".pth"),
        os.path.join("/kaggle/input", "**", stem + ".pt"),
        os.path.join("/kaggle/input", "**", stem + ".bin"),
    ]
    for pat in patterns:
        hits = glob.glob(pat, recursive=True)
        if len(hits) > 0:
            return hits[0]

    return None


def _remap_head_keys_if_needed(sd: dict, model: nn.Module) -> dict:
    """
    Score-relevant: improve loading compatibility so we actually load the trained 5-class head
    (not a randomly initialized head), while keeping the same architectures.
    """
    if not isinstance(sd, dict):
        return sd

    model_sd = model.state_dict()
    model_keys = set(model_sd.keys())
    sd_keys = set(sd.keys())

    if len(sd_keys & model_keys) / max(1, len(model_keys)) > 0.90:
        return sd

    remap_candidates = [
        ("classifier.", "fc."),
        ("fc.", "classifier."),
        ("head.", "fc."),
        ("fc.", "head."),
        ("head.fc.", "fc."),
    ]

    best_sd = sd
    best_overlap = len(sd_keys & model_keys)

    for src, dst in remap_candidates:
        trial = {}
        for k, v in sd.items():
            if k.startswith(src):
                trial[dst + k[len(src) :]] = v
            else:
                trial[k] = v
        overlap = len(set(trial.keys()) & model_keys)
        if overlap > best_overlap:
            best_overlap = overlap
            best_sd = trial

    return best_sd


models_list = []
active_model_keys = []
loaded_paths = {}

for model_key, path in model_paths.items():
    model_name = model_names[model_key]

    resolved = _resolve_checkpoint_path(path, model_key)
    if resolved is None:
        continue

    model = timm.create_model(model_name, pretrained=False, num_classes=5)
    ckpt = torch.load(resolved, map_location="cpu")
    sd = _extract_state_dict(ckpt)

    if isinstance(sd, dict):
        sd = _normalize_state_dict_keys(sd)
        sd = _remap_head_keys_if_needed(sd, model)

    model_keys = set(model.state_dict().keys())
    sd_keys = set(sd.keys()) if isinstance(sd, dict) else set()
    overlap_ratio = (
        (len(model_keys & sd_keys) / max(1, len(model_keys)))
        if isinstance(sd, dict)
        else 0.0
    )
    if overlap_ratio < 0.20:
        continue

    try:
        model.load_state_dict(sd, strict=True)
    except Exception:
        model.load_state_dict(sd, strict=False)

    model.to(device)
    model.eval()

    models_list.append(model)
    active_model_keys.append(model_key)
    loaded_paths[model_key] = resolved

if len(models_list) == 0:
    for model_key in model_paths.keys():
        model_name = model_names[model_key]
        model = timm.create_model(model_name, pretrained=True, num_classes=5)
        model.to(device)
        model.eval()
        models_list.append(model)
        active_model_keys.append(model_key)

active_total = sum(weights[k] for k in active_model_keys)
active_weights = {k: (weights[k] / active_total) for k in active_model_keys}

print("Active models:", active_model_keys)
print("Loaded checkpoints:", {k: loaded_paths.get(k, None) for k in active_model_keys})
print(
    "Num models:",
    len(models_list),
    "| weight sum:",
    float(sum(active_weights.values())),
)



## === cell 7
all_outputs = []

with torch.no_grad():
    for images in tqdm(test_loader):
        images = images.to(device, non_blocking=True)

        per_model = []
        for model_key, model in zip(active_model_keys, models_list):
            probs = nn.functional.softmax(model(images), dim=1)
            per_model.append(active_weights[model_key] * probs)

        weighted_outputs = torch.stack(per_model, dim=0).sum(dim=0)
        all_outputs.append(weighted_outputs.cpu().numpy())

all_outputs = np.concatenate(all_outputs, axis=0)
final_predictions = np.argmax(all_outputs, axis=1).astype(int)



## === cell 8
submission_df = pd.DataFrame(
    {
        "id_code": pd.read_csv(test_csv_file)["id_code"].values,
        "diagnosis": final_predictions,
    }
)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission_df.head())
print("rows:", len(submission_df), "cols:", list(submission_df.columns))
