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

0.889861199791967

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0666) has done: 'Your code fails because it references a Kaggle dataset of pretrained ensemble weights (`/kaggle/input/aptos_ensamble-models/...`) that is not present in your environment, so no models load and inference crashes. To keep the core inference/ensemble logic intact while making it run end-to-end, I add a minimal fallback: if no weight files exist, build the same model architectures with ImageNet pretrained weights and run the same weighted-softmax ensemble. I also fix the weight/model-key alignment bug by iterating over `(model_key, model)` pairs directly (instead of zipping two separate sequences that can desync) and make image loading robust (`convert("RGB")`). Finally, the script always writes a valid `submission.csv` with the required columns.'
- What this solution (achieved -0.01856) has done: 'Your current score is far below the target, and the biggest issue is that you’re effectively running an untrained 5-class head for every “fallback pretrained=True, num_classes=5” model (the classifier layer is randomly initialized), so predictions are near-random. To move the score toward the target without changing the ensemble/inference core logic, I (1) robustly locate and load the provided `.pth` weights if they exist anywhere under `/kaggle/input`, (2) correctly handle common checkpoint formats (`state_dict`, `model`, `module.*`) and load with `strict=False` only when needed, and (3) if a weight file truly doesn’t exist, build the model with an ImageNet backbone but keep a sane head initialization and warn (rather than silently producing random outputs). This keeps the same architecture/ensemble approach and should materially increase QWK toward your target because you actually be using the intended trained weights. The script still runs end-to-end and always writes a valid `submission.csv` with the required columns.'
- What this solution (achieved -0.01471) has done: 'Your score is extremely far below the target, so the most likely cause is still that the intended trained ensemble weights are not actually being loaded (or are being loaded into mismatched architectures), yielding near-random predictions. I make a minimal, score-relevant change: robustly resolve checkpoint paths and, when a file is missing/unloadable, *exclude that model from the ensemble weights* (instead of silently mixing in a random 5-class head that destroys QWK). I also fix the `efficientnet_b3` inconsistency (it has a validation score but no weight path, so it was never used correctly) and make the “loaded_from_files” counter accurate so you can verify what’s really being ensembled. The model architectures, inference approach (weighted softmax averaging), transforms, and output semantics remain the same.'
- What this solution (achieved -0.01489) has done: 'Your current score is far below the target, and with this inference-only script the main score lever (without changing model/ensemble core logic) is making the final discrete labels better aligned with QWK. I keep the same models, same weighted-softmax averaging, and same image preprocessing, but add a minimal post-processing step that learns 4 optimal thresholds on a small held-out split of the provided training set using the *same ensemble predictions* and maximizes quadratic weighted kappa. Then I apply those thresholds to the test-set ensemble expected value to produce integer labels 0–4, which typically improves QWK substantially versus plain argmax while preserving evaluation semantics. I also ensure deterministic splitting and robustly handle the case where some models are missing by calibrating only on the actually-ensembled models.'
- What this solution (achieved 0.11338) has done: 'Your current run didn’t yield a Kaggle score mainly because the data paths point to `/kaggle/data/...`, but in this environment the competition files are under `/kaggle/input/aptos2019-blindness-detection/...`; so the script fail before it can write `submission.csv`. I make the smallest change that fixes this by adding a tiny path resolver and using it for `train.csv`, `test.csv`, `train_images/`, and `test_images/`. I also add a defensive fallback to set `num_workers=0` if multiprocessing data loading fails in the runtime, which helps ensure the notebook completes and produces a valid submission without changing the model/ensemble logic. Everything else (models, weighted-softmax ensembling, and threshold fitting for QWK) stays the same.'

# 9. Code solution

## === cell 0
import os
import random
import re
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
        img_name = os.path.join(
            self.root_dir, str(self.annotations.iloc[idx, 0]) + ".png"
        )
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
def resolve_first_existing(*candidates: str) -> str:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(
        "None of the candidate paths exist:\n" + "\n".join(str(c) for c in candidates)
    )


test_csv_file = resolve_first_existing(
    "/kaggle/input/aptos2019-blindness-detection/test.csv",
    "/kaggle/data/aptos2019-blindness-detection/test.csv",
    "/kaggle/input/test.csv",
    "/kaggle/data/test.csv",
)
test_root_dir = resolve_first_existing(
    "/kaggle/input/aptos2019-blindness-detection/test_images",
    "/kaggle/data/aptos2019-blindness-detection/test_images",
    "/kaggle/input/test_images",
    "/kaggle/data/test_images",
)

test_dataset = BlindnessDataset(
    test_csv_file, test_root_dir, transform=transform, test=True
)

_num_workers = 2
try:
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=_num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    _ = next(iter(test_loader))
except Exception as e:
    print(
        f"WARNING: DataLoader with num_workers={_num_workers} failed ({type(e).__name__}); retrying with num_workers=0."
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
    "resnet18": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/1/resnet18.pth",
    "efficientnet_b0": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/1/efficentNet_b0.pth",
    "efficientnet_b1": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/1/efficentNet_b1.pth",
    "efficientnet_b2": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/1/efficentNet_b2.pth",
    "efficientnet_b3": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/1/efficentNet_b3.pth",
    "efficientnet_b4": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/1/efficentNet_b4.pth",
    "efficientnet_b5": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/1/efficentNet_b5.pth",
    "inception_resnet_v2": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/1/inception_resnet_v2.pth",
    "inception_v4": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/1/inception_v4.pth",
    "seresnext50_32x4d": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/1/seresnext50_32x4d.pth",
    "seresnext101_32x4d": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/1/seresnext101_32x4d.pth",
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
def _iter_ckpt_files(search_root: str):
    exts = (".pth", ".pt", ".bin")
    for root, _, files in os.walk(search_root):
        for f in files:
            if f.lower().endswith(exts):
                yield os.path.join(root, f)


def _normalize_token(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _candidate_name_tokens(model_key: str) -> list[str]:
    toks = {model_key, model_key.replace("efficientnet", "efficentnet")}
    toks.add(model_names.get(model_key, model_key))
    toks.add(
        model_names.get(model_key, model_key).replace("efficientnet", "efficentnet")
    )
    toks.add(model_key.replace("_", ""))
    toks.add(model_names.get(model_key, model_key).replace("_", ""))
    toks.add(model_key.replace("-", ""))
    toks.add(model_names.get(model_key, model_key).replace("-", ""))
    return sorted({_normalize_token(t) for t in toks})


def resolve_model_path(path: str, model_key: str) -> str | None:
    if os.path.exists(path):
        return path

    b = os.path.basename(path)
    for root, _, files in os.walk("/kaggle/input"):
        if b in files:
            return os.path.join(root, b)

    want_tokens = _candidate_name_tokens(model_key)

    best = None
    best_score = -1
    for p in _iter_ckpt_files("/kaggle/input"):
        fn = _normalize_token(os.path.basename(p))
        dn = _normalize_token(os.path.dirname(p))

        score = 0
        for t in want_tokens:
            if t and t in fn:
                score += 3 * len(t)
            if t and t in dn:
                score += 1 * len(t)

        if "ensamble" in dn or "ensemble" in dn:
            score += 20
        if "aptos" in dn:
            score += 6

        if score > best_score:
            best_score = score
            best = p

    if best_score >= 20:
        return best
    return None


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in (
            "state_dict",
            "model",
            "net",
            "model_state_dict",
            "weights",
            "ema",
            "model_ema",
        ):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def strip_prefix(state_dict: dict, prefix: str):
    out = {}
    for k, v in state_dict.items():
        if isinstance(k, str) and k.startswith(prefix):
            out[k[len(prefix) :]] = v
        else:
            out[k] = v
    return out


def _key_overlap_ratio(model: nn.Module, state: dict) -> float:
    model_keys = set(model.state_dict().keys())
    state_keys = set(state.keys())
    if not model_keys:
        return 0.0
    return len(model_keys & state_keys) / float(len(model_keys))


def _find_classifier_weight_shapes(state: dict):
    cand_keys = []
    for k in state.keys():
        lk = k.lower()
        if lk.endswith(
            ("classifier.weight", "fc.weight", "head.weight", "last_linear.weight")
        ):
            cand_keys.append(k)
    shapes = []
    for k in cand_keys:
        v = state.get(k, None)
        if torch.is_tensor(v) and v.ndim == 2:
            shapes.append((k, tuple(v.shape)))
    return shapes


def _make_model_with_optional_head(model_name: str, num_classes: int = 5):
    return timm.create_model(model_name, pretrained=False, num_classes=num_classes)


def _attach_linear_head_if_needed(model: nn.Module, num_classes: int = 5) -> bool:
    if hasattr(model, "get_classifier") and hasattr(model, "reset_classifier"):
        try:
            model.reset_classifier(num_classes=num_classes)
            return True
        except Exception:
            pass

    in_features = None
    if hasattr(model, "num_features"):
        in_features = int(model.num_features)
    elif hasattr(model, "get_classifier"):
        clf = model.get_classifier()
        if hasattr(clf, "in_features"):
            in_features = int(clf.in_features)

    if in_features is None:
        return False

    if hasattr(model, "classifier") and isinstance(
        getattr(model, "classifier"), nn.Module
    ):
        model.classifier = nn.Linear(in_features, num_classes)
        return True
    if hasattr(model, "fc") and isinstance(getattr(model, "fc"), nn.Module):
        model.fc = nn.Linear(in_features, num_classes)
        return True
    if hasattr(model, "head") and isinstance(getattr(model, "head"), nn.Module):
        model.head = nn.Linear(in_features, num_classes)
        return True
    return False


def _try_load_state_dict_into_model(
    model: nn.Module, state: dict, min_overlap: float = 0.70
):
    try:
        model.load_state_dict(state, strict=True)
        return True, None
    except Exception as e1:
        overlap = _key_overlap_ratio(model, state)
        if overlap < min_overlap:
            return (
                False,
                f"strict load failed ({type(e1).__name__}); overlap={overlap:.3f} < {min_overlap:.2f}",
            )
        try:
            model.load_state_dict(state, strict=False)
            return (
                True,
                f"strict load failed ({type(e1).__name__}); used strict=False (overlap={overlap:.3f})",
            )
        except Exception as e2:
            return False, f"failed to load ({type(e2).__name__})"


def _verify_logits5(model: nn.Module, device: torch.device) -> bool:
    model.eval()
    with torch.no_grad():
        x = torch.zeros(1, 3, 224, 224, device=device)
        y = model(x)
        return torch.is_tensor(y) and y.ndim == 2 and y.shape[1] == 5


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models_dict = {}
model_output_dims = {}
loaded_from_files = 0
excluded = []
load_warnings = []

for model_key, original_path in model_paths.items():
    model_name = model_names[model_key]
    resolved = resolve_model_path(original_path, model_key=model_key)

    if resolved is None:
        excluded.append((model_key, original_path))
        continue

    model = _make_model_with_optional_head(model_name, num_classes=5)

    try:
        ckpt = torch.load(resolved, map_location="cpu")
    except Exception as e:
        excluded.append((model_key, resolved))
        load_warnings.append(
            f"{model_key}: torch.load failed ({type(e).__name__}); excluded."
        )
        continue

    state = extract_state_dict(ckpt)
    if not isinstance(state, dict):
        excluded.append((model_key, resolved))
        load_warnings.append(f"{model_key}: checkpoint not a state-dict; excluded.")
        continue

    state = strip_prefix(state, "module.")
    state = strip_prefix(state, "model.")
    state = strip_prefix(state, "net.")

    head_shapes = _find_classifier_weight_shapes(state)
    head_out = None
    for _, shp in head_shapes:
        head_out = int(shp[0])
        break

    if head_out is not None and head_out != 5:
        model2 = _make_model_with_optional_head(model_name, num_classes=0)
        ok_head = _attach_linear_head_if_needed(model2, num_classes=5)
        if ok_head:
            model = model2
            load_warnings.append(
                f"{model_key}: checkpoint head_out={head_out} != 5; rebuilt {model_name} with num_classes=0 and attached 5-class head to improve load compatibility."
            )

    loaded_ok, warn = _try_load_state_dict_into_model(model, state, min_overlap=0.70)
    if not loaded_ok:
        excluded.append((model_key, resolved))
        load_warnings.append(f"{model_key}: {warn}; excluded.")
        continue

    model.to(device).eval()

    if not _verify_logits5(model, device=device):
        excluded.append((model_key, resolved))
        load_warnings.append(
            f"{model_key}: model forward not [B,5] logits after load; excluded to avoid invalid softmax."
        )
        continue

    if warn is not None:
        load_warnings.append(f"{model_key}: {warn} from {resolved}")

    loaded_from_files += 1
    models_dict[model_key] = model
    model_output_dims[model_key] = 5

if loaded_from_files == 0:
    fallback_key = "efficientnet_b0"
    fallback_name = model_names[fallback_key]
    print(
        "WARNING: No trained checkpoints found under /kaggle/input. "
        "Falling back to a single ImageNet-pretrained model (efficientnet_b0) so the notebook runs end-to-end."
    )

    backbone = timm.create_model(fallback_name, pretrained=True, num_classes=0)
    ok = _attach_linear_head_if_needed(backbone, num_classes=5)
    backbone = backbone.to(device).eval()  # ensure head + backbone are on same device

    if (not ok) or (not _verify_logits5(backbone, device=device)):
        backbone = (
            timm.create_model(fallback_name, pretrained=True, num_classes=5)
            .to(device)
            .eval()
        )

    models_dict = {fallback_key: backbone}
    model_output_dims = {fallback_key: 5}
else:
    print(
        f"Models with weights loaded from checkpoint files: {loaded_from_files}/{len(model_paths)}"
    )
    print(f"Models included in ensemble: {len(models_dict)}")
    if excluded:
        print(
            "Models excluded due to missing/unusable weights (to avoid degrading QWK):"
        )
        for k, p in excluded[:20]:
            print(f"  - {k}: {p}")
        if len(excluded) > 20:
            print(f"  ... ({len(excluded)-20} more)")
    if load_warnings:
        print("Load warnings:")
        for w in load_warnings[:60]:
            print(f"  - {w}")
        if len(load_warnings) > 60:
            print(f"  ... ({len(load_warnings)-60} more)")



## === cell 6
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



## === cell 7
ensemble_keys = list(models_dict.keys())
if len(ensemble_keys) == 0:
    raise RuntimeError("No models available; cannot produce submission.")

scores = {k: float(validation_scores.get(k, 1.0)) for k in ensemble_keys}
total_score = float(sum(scores.values()))
if not np.isfinite(total_score) or total_score <= 0:
    total_score = float(len(scores))
weights = {k: float(v) / total_score for k, v in scores.items()}

print(
    f"Ensembling {len(ensemble_keys)} models. Sum(weights)={sum(weights.values()):.6f}"
)




## === cell 8
def quadratic_weighted_kappa(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 5
) -> float:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    assert y_true.shape == y_pred.shape

    O = np.zeros((num_classes, num_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < num_classes and 0 <= b < num_classes:
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
    num = (W * O).sum()
    return float(1.0 - num / denom)


def probs_to_expected_value(probs_5: np.ndarray) -> np.ndarray:
    idx = np.arange(5, dtype=np.float32)
    return (probs_5 * idx[None, :]).sum(axis=1)


def apply_thresholds(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    return np.digitize(x, th, right=True).astype(int)


def fit_thresholds_coordinate_descent(
    x: np.ndarray, y: np.ndarray, init_th=None, n_iter: int = 3
) -> np.ndarray:
    x = x.astype(np.float32)
    y = y.astype(int)

    if init_th is None:
        means = []
        for c in range(5):
            xc = x[y == c]
            means.append(float(np.mean(xc)) if len(xc) else float(c))
        means = np.array(means, dtype=np.float32)
        init_th = np.array(
            [(means[i] + means[i + 1]) / 2.0 for i in range(4)], dtype=np.float32
        )

    th = init_th.astype(np.float32).copy()
    th.sort()

    cand = np.unique(np.quantile(x, np.linspace(0.02, 0.98, 97)).astype(np.float32))
    best = quadratic_weighted_kappa(y, apply_thresholds(x, th))

    for _ in range(n_iter):
        for k in range(4):
            low = -np.inf if k == 0 else th[k - 1] + 1e-4
            high = np.inf if k == 3 else th[k + 1] - 1e-4
            cands_k = cand[(cand > low) & (cand < high)]
            if cands_k.size == 0:
                continue

            best_k = th[k]
            best_score = best
            for v in cands_k:
                th_try = th.copy()
                th_try[k] = v
                score = quadratic_weighted_kappa(y, apply_thresholds(x, th_try))
                if score > best_score:
                    best_score = score
                    best_k = v
            th[k] = best_k
            best = best_score

    th.sort()
    return th


def fit_1d_scale_for_expected_value(
    x: np.ndarray, y: np.ndarray
) -> tuple[float, float]:
    x = x.astype(np.float32)
    y = y.astype(int)

    a_grid = np.linspace(0.85, 1.15, 13, dtype=np.float32)
    b_grid = np.linspace(-0.30, 0.30, 13, dtype=np.float32)

    best = -1.0
    best_a, best_b = 1.0, 0.0
    for a in a_grid:
        for b in b_grid:
            x2 = np.clip(a * x + b, 0.0, 4.0)
            th2 = fit_thresholds_coordinate_descent(x2, y, n_iter=2)
            pred2 = apply_thresholds(x2, th2)
            score = quadratic_weighted_kappa(y, pred2)
            if score > best:
                best = score
                best_a, best_b = float(a), float(b)
    return best_a, best_b




## === cell 9
train_csv_file = resolve_first_existing(
    "/kaggle/input/aptos2019-blindness-detection/train.csv",
    "/kaggle/data/aptos2019-blindness-detection/train.csv",
    "/kaggle/input/train.csv",
    "/kaggle/data/train.csv",
)
train_root_dir = resolve_first_existing(
    "/kaggle/input/aptos2019-blindness-detection/train_images",
    "/kaggle/data/aptos2019-blindness-detection/train_images",
    "/kaggle/input/train_images",
    "/kaggle/data/train_images",
)

train_df = pd.read_csv(train_csv_file)

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

rng = np.random.default_rng(seed)
idx = np.arange(len(train_df))
rng.shuffle(idx)

calib_size = min(800, len(idx) // 3)
calib_idx = idx[:calib_size]

calib_df = train_df.iloc[calib_idx].reset_index(drop=True)
calib_csv_path = "/kaggle/working/_calib.csv"
calib_df.to_csv(calib_csv_path, index=False)

calib_dataset = BlindnessDataset(
    calib_csv_path, train_root_dir, transform=transform, test=False
)

_num_workers = 2
try:
    calib_loader = DataLoader(
        calib_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=_num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    _ = next(iter(calib_loader))
except Exception as e:
    print(
        f"WARNING: Calib DataLoader with num_workers={_num_workers} failed ({type(e).__name__}); retrying with num_workers=0."
    )
    calib_loader = DataLoader(
        calib_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def logits_to_probs5(logits: torch.Tensor, out_dim: int) -> torch.Tensor:
    if out_dim == 5:
        return nn.functional.softmax(logits, dim=1)
    return nn.functional.softmax(logits[:, :5], dim=1)


def predict_probs(loader: DataLoader) -> tuple[np.ndarray, np.ndarray | None]:
    probs_all = []
    y_all = []
    is_labeled = False

    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, labels = batch
                is_labeled = True
                y_all.append(labels.numpy().astype(int))
            else:
                images = batch

            images = images.to(device, non_blocking=True)

            weighted_probs = None
            for model_key in ensemble_keys:
                model = models_dict[model_key]
                out_dim = model_output_dims.get(model_key, 5)
                logits = model(images)
                probs5 = logits_to_probs5(logits, out_dim=out_dim)

                w = float(weights.get(model_key, 0.0))
                if weighted_probs is None:
                    weighted_probs = probs5.mul(w)
                else:
                    weighted_probs = weighted_probs + probs5.mul(w)

            if weighted_probs is None:
                raise RuntimeError(
                    "Ensemble produced no predictions; check model loading."
                )

            probs_all.append(weighted_probs.detach().cpu().numpy())

    probs_all = np.concatenate(probs_all, axis=0)
    if is_labeled:
        y_all = np.concatenate(y_all, axis=0)
        return probs_all, y_all
    return probs_all, None


calib_probs, calib_y = predict_probs(calib_loader)
calib_x = probs_to_expected_value(calib_probs)

a_ev, b_ev = fit_1d_scale_for_expected_value(calib_x, calib_y)
calib_x_cal = np.clip(a_ev * calib_x + b_ev, 0.0, 4.0)

th = fit_thresholds_coordinate_descent(calib_x_cal, calib_y, n_iter=3)
calib_pred = apply_thresholds(calib_x_cal, th)
calib_kappa = quadratic_weighted_kappa(calib_y, calib_pred)
print(f"Expected-value calibration: a={a_ev:.4f}, b={b_ev:.4f}")
print("Fitted thresholds:", th)
print(
    f"Calibration QWK (sanity check, not used in submission score directly): {calib_kappa:.5f}"
)



## === cell 10
test_probs, _ = predict_probs(test_loader)
test_x = probs_to_expected_value(test_probs)

test_x_cal = np.clip(a_ev * test_x + b_ev, 0.0, 4.0)

final_predictions = apply_thresholds(test_x_cal, th)
final_predictions = np.clip(final_predictions, 0, 4).astype(int)

test_ids = pd.read_csv(test_csv_file)["id_code"].astype(str).values
if len(test_ids) != len(final_predictions):
    raise RuntimeError(
        f"Prediction length mismatch: ids={len(test_ids)} preds={len(final_predictions)}"
    )

submission_df = pd.DataFrame({"id_code": test_ids, "diagnosis": final_predictions})
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Wrote submission to: {submission_path}")
print(submission_df.head())
print(
    "Prediction distribution:",
    submission_df["diagnosis"].value_counts().sort_index().to_dict(),
)
