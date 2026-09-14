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

-0.01471

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0666) has done: 'Your code fails because it references a Kaggle dataset of pretrained ensemble weights (`/kaggle/input/aptos_ensamble-models/...`) that is not present in your environment, so no models load and inference crashes. To keep the core inference/ensemble logic intact while making it run end-to-end, I add a minimal fallback: if no weight files exist, build the same model architectures with ImageNet pretrained weights and run the same weighted-softmax ensemble. I also fix the weight/model-key alignment bug by iterating over `(model_key, model)` pairs directly (instead of zipping two separate sequences that can desync) and make image loading robust (`convert("RGB")`). Finally, the script always writes a valid `submission.csv` with the required columns.'
- What this solution (achieved -0.01856) has done: 'Your current score is far below the target, and the biggest issue is that you’re effectively running an untrained 5-class head for every “fallback pretrained=True, num_classes=5” model (the classifier layer is randomly initialized), so predictions are near-random. To move the score toward the target without changing the ensemble/inference core logic, I (1) robustly locate and load the provided `.pth` weights if they exist anywhere under `/kaggle/input`, (2) correctly handle common checkpoint formats (`state_dict`, `model`, `module.*`) and load with `strict=False` only when needed, and (3) if a weight file truly doesn’t exist, build the model with an ImageNet backbone but keep a sane head initialization and warn (rather than silently producing random outputs). This keeps the same architecture/ensemble approach and should materially increase QWK toward your target because you actually be using the intended trained weights. The script still runs end-to-end and always writes a valid `submission.csv` with the required columns.'
- What this solution (achieved -0.01471) has done: 'Your score is extremely far below the target, so the most likely cause is still that the intended trained ensemble weights are not actually being loaded (or are being loaded into mismatched architectures), yielding near-random predictions. I make a minimal, score-relevant change: robustly resolve checkpoint paths and, when a file is missing/unloadable, *exclude that model from the ensemble weights* (instead of silently mixing in a random 5-class head that destroys QWK). I also fix the `efficientnet_b3` inconsistency (it has a validation score but no weight path, so it was never used correctly) and make the “loaded_from_files” counter accurate so you can verify what’s really being ensembled. The model architectures, inference approach (weighted softmax averaging), transforms, and output semantics remain the same.'

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
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2,
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
def _find_by_basename(search_root: str, basename: str) -> str | None:
    for root, _, files in os.walk(search_root):
        if basename in files:
            return os.path.join(root, basename)
    return None


def resolve_model_path(path: str) -> str | None:
    if os.path.exists(path):
        return path
    b = os.path.basename(path)
    return _find_by_basename("/kaggle/input", b)


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "net", "model_state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def strip_prefix(state_dict: dict, prefix: str):
    out = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
        else:
            out[k] = v
    return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models_dict = {}
loaded_from_files = 0
missing = []
load_warnings = []

for model_key, original_path in model_paths.items():
    model_name = model_names[model_key]
    resolved = resolve_model_path(original_path)

    if resolved is None:
        missing.append((model_key, original_path))
        model = timm.create_model(model_name, pretrained=True, num_classes=5)
        model.to(device).eval()
        models_dict[model_key] = model
        continue

    model = timm.create_model(model_name, pretrained=False, num_classes=5)
    ckpt = torch.load(resolved, map_location="cpu")
    state = extract_state_dict(ckpt)

    if not isinstance(state, dict):
        load_warnings.append(
            f"{model_key}: checkpoint at {resolved} not a dict; using fallback pretrained backbone."
        )
        model = timm.create_model(model_name, pretrained=True, num_classes=5)
        model.to(device).eval()
        models_dict[model_key] = model
        continue

    state = strip_prefix(state, "module.")

    loaded_ok = False
    try:
        model.load_state_dict(state, strict=True)
        loaded_ok = True
    except Exception as e:
        try:
            model.load_state_dict(state, strict=False)
            loaded_ok = True
            load_warnings.append(
                f"{model_key}: strict load failed ({type(e).__name__}); loaded with strict=False from {resolved}."
            )
        except Exception as e2:
            load_warnings.append(
                f"{model_key}: failed to load weights from {resolved} ({type(e2).__name__}); using fallback pretrained backbone."
            )
            model = timm.create_model(model_name, pretrained=True, num_classes=5)

    if loaded_ok:
        loaded_from_files += 1

    model.to(device).eval()
    models_dict[model_key] = model

print(
    f"Models with weights loaded from .pth files: {loaded_from_files}/{len(model_paths)}"
)
if missing:
    print(
        "Missing model weight files (these models will be excluded from the weighted ensemble):"
    )
    for k, p in missing:
        print(f"  - {k}: {p}")
if load_warnings:
    print("Load warnings:")
    for w in load_warnings:
        print(f"  - {w}")



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
missing_keys = {k for k, _ in missing}

available_keys = list(models_dict.keys())
ensemble_keys = [k for k in available_keys if k not in missing_keys]

if len(ensemble_keys) == 0:
    ensemble_keys = available_keys

scores = {k: validation_scores[k] for k in ensemble_keys if k in validation_scores}

if len(scores) != len(ensemble_keys):
    for k in ensemble_keys:
        if k not in scores:
            scores[k] = 1.0

total_score = float(sum(scores.values()))
weights = {k: float(v) / total_score for k, v in scores.items()}

print(
    f"Ensembling {len(ensemble_keys)}/{len(available_keys)} models. Sum(weights)={sum(weights.values()):.6f}"
)



## === cell 8
all_outputs = []

with torch.no_grad():
    for images in tqdm(test_loader):
        images = images.to(device, non_blocking=True)
        weighted_outputs = None

        for model_key in ensemble_keys:
            model = models_dict[model_key]
            probs = nn.functional.softmax(model(images), dim=1)
            w = weights.get(model_key, 0.0)
            if weighted_outputs is None:
                weighted_outputs = w * probs
            else:
                weighted_outputs = weighted_outputs + w * probs

        all_outputs.append(weighted_outputs.detach().cpu().numpy())

all_outputs = np.concatenate(all_outputs, axis=0)
final_predictions = np.argmax(all_outputs, axis=1).astype(int)



## === cell 9
submission_df = pd.DataFrame(
    {
        "id_code": pd.read_csv(test_csv_file)["id_code"].values,
        "diagnosis": final_predictions,
    }
)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Wrote submission to: {submission_path}")
print(submission_df.head())
print(
    "Prediction distribution:",
    submission_df["diagnosis"].value_counts().sort_index().to_dict(),
)
