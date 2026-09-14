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

0.0761537016384402

# 6. Current score

0.30447

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.08137) has done: 'The run fails because the notebook tries to load pretrained ensemble weights from a Kaggle dataset path that doesn’t exist in your environment, so `loaded_models` is never created and submission generation crashes. I make model loading robust by checking for the existence of each `.pth` and skipping missing ones; if none are found, the code fall back to a single timm model with `pretrained=True` (same inference pipeline) so a valid submission is always produced. I also fix the inference transform to match the training-time normalization (score-positive and consistent) and make AMP conditional on CUDA to avoid CPU autocast issues. Finally, I keep all paths and the submission format unchanged and ensure `/kaggle/working/submission.csv` is always written.'
- What this solution (achieved 0.0) has done: 'Your current submission is produced by a single pretrained ResNet18 fallback (since the intended ensemble weights path doesn’t exist), and it uses plain argmax over softmax probabilities; that combination often yields very poor QWK because class thresholds aren’t calibrated. To move the score upward toward your target with minimal change and without touching the model, I keep the same inference pipeline but add a tiny, validation-based “threshold optimization” step: learn 4 cutpoints on a small held-out split of `train.csv` by minimizing negative quadratic kappa using the model’s expected class value. Then I apply those learned cutpoints to the test expected values to generate ordinal predictions (0–4), which typically improves QWK substantially while preserving your core logic and producing the same `submission.csv` format.'
- What this solution (achieved 0.0) has done: 'Your current score (0.0) is below the target (0.07615), so we want a small, low-risk improvement. I keep your exact model/inference approach, but make the threshold learning more stable for QWK by (1) optimizing thresholds using out-of-fold predictions (5-fold stratified CV) instead of a single split, and (2) doing a tiny final refinement pass starting from the CV thresholds. This avoids overfitting to one validation split and usually lifts QWK without changing architecture, loss, or training. I also ensure the thresholds are always strictly increasing by parameterizing them as cumulative positive gaps.'
- What this solution (achieved 0.41733) has done: 'I make the code reliably produce a submission in this environment and nudge QWK upward toward your target by fixing two small but high-impact issues without changing your model logic. First, the fallback currently uses `pretrained=True` but still sets `num_classes=5`, which discards pretrained weights for the classifier head and often yields near-random predictions; I keep the same fallback model family but load it as a true ImageNet model and map its 1000-class output to a stable 5-class ordinal prediction via expected-value thresholds. Second, the threshold learning currently evaluates the model on train images but uses random ImageNet weights (fallback), so it overfits noise; I instead learn thresholds on the same model outputs (expected values) but with a small amount of smoothing/clipping to avoid extreme thresholds when predictions are uninformative. This keeps the architecture/training approach unchanged (still pure inference + thresholding) while making the fallback meaningful enough to reach a small positive QWK and produce a valid `submission.csv`.'
- What this solution (achieved 0.35523) has done: 'Your current score (0.41733) is far above the target (0.07615), and since higher is better we should *reduce* performance slightly toward the target with the smallest, safest change. The lowest-risk lever here is the post-processing that optimizes QWK: we disable threshold optimization and instead use fixed, conservative thresholds, which typically lower QWK while preserving the same model/inference core. To keep things stable and deterministic, we still compute expected values the same way and generate the same submission schema, but we won’t tune thresholds on any validation split. This should move your score downward toward the target without touching model architecture, training, or feature extraction.'
- What this solution (achieved 0.0) has done: 'Your current score (0.35523) is far above the target (0.07615), so we should intentionally move performance downward with the smallest low-risk change. The safest lever (without touching model architecture or inference core) is post-processing: make predictions less aligned to the ordinal structure by replacing expected-value thresholding with plain `argmax` on the averaged class probabilities. This typically reduces QWK while keeping the same ensemble averaging and submission format. I keep everything else identical and still guarantee `/kaggle/working/submission.csv` is written correctly.'
- What this solution (achieved 0.30447) has done: 'Your current 0.0 score is far below the target 0.07615, and the main low-risk lever (without changing your model/inference core) is ordinal post-processing suited to QWK. I keep the exact same ensemble probability inference, but replace plain `argmax` with an expected-value + fixed-threshold mapping (a standard ordinal calibration that often improves QWK compared to argmax). Because your fallback can be an ImageNet 1000-class model, I also add a minimal, deterministic 1000→5 “binning” step so expected values and thresholds are well-defined in both ensemble and fallback cases. This is intentionally conservative (fixed thresholds; no training/threshold optimization) to nudge score upward toward the target while minimizing change risk.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm

from sklearn.model_selection import train_test_split, StratifiedKFold
from torchvision import transforms, models
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import cohen_kappa_score

import copy
import timm

torch.manual_seed(42)
np.random.seed(42)




## === cell 1
class BlindnessDataset(Dataset):
    def __init__(
        self,
        csv_file,
        root_dir,
        transform=None,
        augmentations=None,
        max_count=None,
        test=False,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.max_count = max_count
        self.test = test

        if not test:
            self.class_counts = (
                self.annotations["diagnosis"].value_counts().sort_index()
            )
        else:
            self.class_counts = None

        if max_count:
            self.oversample(max_count)

    def oversample(self, max_count):  # Over sampling classes to balance
        samples = []
        for diagnosis in self.class_counts.index:
            class_samples = self.annotations[self.annotations["diagnosis"] == diagnosis]
            oversampled_class = class_samples.sample(max_count, replace=True)
            samples.append(oversampled_class)
        self.annotations = pd.concat(samples).reset_index(drop=True)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0] + ".png")
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.test:
            return image

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
def select_model(model_name, input_size):
    if model_name == "efficientnet_b5":
        return timm.create_model(
            "efficientnet_b5", pretrained=False, num_classes=5, in_chans=3
        )
    elif model_name == "inception_resnet_v2":
        return timm.create_model(
            "inception_resnet_v2", pretrained=False, num_classes=5, in_chans=3
        )
    elif model_name == "inception_v4":
        return timm.create_model(
            "inception_v4", pretrained=False, num_classes=5, in_chans=3
        )
    elif model_name == "seresnext50_32x4d":
        return timm.create_model(
            "seresnext50_32x4d", pretrained=False, num_classes=5, in_chans=3
        )
    elif model_name == "seresnext101_32x4d":
        return timm.create_model(
            "seresnext101_32x4d", pretrained=False, num_classes=5, in_chans=3
        )
    elif model_name == "resnet18(WD_1e-3)_aptos":
        return timm.create_model(
            "resnet18", pretrained=False, num_classes=5, in_chans=3
        )
    else:
        raise ValueError(f"Unknown model name {model_name}")




## === cell 4
def load_model(model_name, input_size, model_path, device):
    model = select_model(model_name, input_size).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model




## === cell 5
def load_all_models(model_paths, device):
    loaded = []
    missing = []
    for model_name, input_size, model_path in model_paths:
        if not os.path.exists(model_path):
            missing.append(model_path)
            continue
        try:
            model = load_model(model_name, input_size, model_path, device)
            loaded.append((model_name, model))
        except Exception as e:
            missing.append(f"{model_path} (load error: {repr(e)})")
            continue

    if missing:
        print("Some model weights were not found / could not be loaded. Skipping:")
        for p in missing:
            print(" -", p)
    return loaded




## === cell 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_dir = "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/1"



## === cell 7
"""
model_paths = [
    ('efficientnet_b5', 512, os.path.join(save_dir, 'efficientnet_b5.pth')),
    ('resnet18(WD_1e-3)_aptos', 512, os.path.join(save_dir, 'resnet18(WD_1e-3)_aptos.pth')),
    ('inception_resnet_v2', 512, os.path.join(save_dir, 'inception_resnet_v2.pth')),
    ('inception_v4', 512, os.path.join(save_dir, 'inception_v4.pth')),
    ('seresnext50_32x4d', 512, os.path.join(save_dir, 'seresnext50_32x4d.pth')),
    ('seresnext101_32x4d', 384, os.path.join(save_dir, 'seresnext101_32x4d.pth'))
]
"""

model_paths = [
    (
        "resnet18(WD_1e-3)_aptos",
        512,
        os.path.join(save_dir, "resnet18(WD_1e-3)_aptos.pth"),
    ),
    ("inception_resnet_v2", 512, os.path.join(save_dir, "inception_resnet_v2.pth")),
    ("inception_v4", 512, os.path.join(save_dir, "inception_v4.pth")),
    ("seresnext50_32x4d", 512, os.path.join(save_dir, "seresnext50_32x4d.pth")),
]



## === cell 8
loaded_models = load_all_models(model_paths, device)

if len(loaded_models) == 0:
    print(
        "No ensemble weights loaded; falling back to a single pretrained ImageNet model for inference."
    )
    fallback_model = timm.create_model(
        "resnet18", pretrained=True, num_classes=1000, in_chans=3
    ).to(device)
    fallback_model.eval()
    loaded_models = [("fallback_resnet18_imagenet", fallback_model)]




## === cell 9
def _predict_proba_ensemble(models, loader, device):
    use_amp = torch.cuda.is_available()
    all_model_probs = []
    for model_name, model in models:
        probs_list = []
        model.eval()
        with torch.inference_mode():
            for batch in loader:
                if isinstance(batch, (tuple, list)):
                    images = batch[0]
                else:
                    images = batch
                images = images.to(device, non_blocking=True)
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        probs = torch.softmax(outputs, dim=1)
                else:
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)
                probs_list.append(probs.detach().cpu().numpy())
        all_model_probs.append(np.concatenate(probs_list, axis=0))
    return np.mean(np.stack(all_model_probs, axis=0), axis=0)


def _expected_value_from_proba(p):
    classes = np.arange(p.shape[1], dtype=np.float32)
    return (p * classes[None, :]).sum(axis=1)


def _apply_thresholds(x, thr):
    return np.digitize(x, bins=np.array(thr, dtype=np.float32), right=False).astype(int)


def _clip_and_sort_thr(thr):
    thr = np.asarray(thr, dtype=np.float32).copy()
    thr = np.clip(thr, 0.0, 4.0)
    thr.sort()
    for i in range(1, 4):
        if thr[i] <= thr[i - 1]:
            thr[i] = min(4.0, thr[i - 1] + 1e-3)
    return thr


def _compress_proba_to_5bins(p):
    if p.shape[1] == 5:
        return p
    if p.shape[1] != 1000:
        p5 = p[:, :5]
        p5 = p5 / np.clip(p5.sum(axis=1, keepdims=True), 1e-12, None)
        return p5

    p = p.astype(np.float32, copy=False)
    p5 = p.reshape(p.shape[0], 5, 200).sum(axis=2)
    p5 = p5 / np.clip(p5.sum(axis=1, keepdims=True), 1e-12, None)
    return p5




## === cell 10
def predict_ensemble(models, test_csv_file, test_root_dir, submission_file, device):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_dataset = BlindnessDataset(
        test_csv_file, test_root_dir, transform=transform, test=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    test_probs = _predict_proba_ensemble(models, test_loader, device)

    test_probs_5 = _compress_proba_to_5bins(test_probs)
    test_exp = _expected_value_from_proba(test_probs_5)

    fixed_thr = _clip_and_sort_thr([0.5, 1.5, 2.5, 3.5])
    final_predictions = _apply_thresholds(test_exp, fixed_thr)
    final_predictions = np.clip(final_predictions, 0, 4).astype(int)

    test_df = pd.read_csv(test_csv_file)
    submission_df = pd.DataFrame(
        {"id_code": test_df["id_code"].values, "diagnosis": final_predictions}
    )
    submission_df.to_csv(submission_file, index=False)
    print(
        f"Wrote submission: {submission_file} with shape {submission_df.shape} and columns {list(submission_df.columns)}"
    )




## === cell 11
test_csv_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_root_dir = "/kaggle/input/aptos2019-blindness-detection/test_images"
submission_file = "/kaggle/working/submission.csv"
predict_ensemble(loaded_models, test_csv_file, test_root_dir, submission_file, device)

sub = pd.read_csv(submission_file)
assert list(sub.columns) == ["id_code", "diagnosis"]
assert len(sub) == pd.read_csv(test_csv_file).shape[0]
assert sub["diagnosis"].between(0, 4).all()
print(sub.head())
