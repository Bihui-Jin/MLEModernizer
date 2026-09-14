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

0.749463163992592

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.29138) has done: 'The run is failing because it expects external pretrained weight files (`/kaggle/input/aptos_ensamble-models/...`) that are not present in your environment, which leaves the ensemble empty and breaks inference. I add a minimal fallback that uses a timm ImageNet-pretrained EfficientNet when those files are missing, preserving the same inference semantics (softmax → weighted sum → argmax) and producing a valid `submission.csv`. I also make model loading robust to common checkpoint formats and ensure images are consistently converted to RGB to avoid occasional PIL mode issues. These changes are primarily to unblock end-to-end execution and yield a valid submission; score likely be lower than the target without the intended competition-trained weights.'
- What this solution (achieved -0.06455) has done: 'Your current score is far below the target, so the smallest legitimate way to move it upward is to keep your exact inference pipeline but make the fallback model and preprocessing consistent with the intended EfficientNet inputs. I (1) switch the fallback to use timm’s correct pretrained head (load pretrained features then replace classifier to 5 classes), (2) use timm’s model-specific normalization and input size via `resolve_data_config/create_transform`, and (3) add lightweight test-time augmentation (horizontal flip) averaged in probability space, which keeps the same “softmax → weighted sum → argmax” semantics. These are minimal changes that typically yield a large kappa gain over the current “random-ish” ImageNet-head mismatch, while still producing the same valid `submission.csv` format.'
- What this solution (achieved -0.09867) has done: 'Your current score is far below the target, so we should increase performance with the smallest changes that keep your inference/ensemble semantics intact. The main issue is that your fallback model replaces the classifier with random weights, making predictions nearly random; instead we keep the same EfficientNet-B0 backbone but use it as a fixed feature extractor and add a tiny, deterministic “severity head” that maps ImageNet features to DR grades without training. We also align preprocessing with the actually-used fallback model (not a temporary one) and keep your existing softmax→weighted sum→argmax plus horizontal-flip TTA unchanged. These changes are minimal (no training loop, no architecture overhaul for the intended checkpoints case) and should move QWK upward substantially vs the current random-head fallback.'
- What this solution (achieved 0.0) has done: 'Your current score is far below the target, so we should increase performance with the smallest change that fixes the main issue: the fallback model is not trained for DR, so its predictions are essentially uncorrelated with the label distribution. To move QWK upward without changing your core inference semantics (softmax → weighted sum → argmax, with hflip TTA), I keep your exact ensemble structure but replace the deterministic “severity head” with a calibrated, non-training fallback that uses the training label prior (class histogram) to produce stable predictions closer to the expected test distribution. This is a minimal, legitimate adjustment (no leakage: it uses only `train.csv` labels, not test labels) and typically improves kappa substantially versus random-like outputs. I also ensure the fallback weight key matches `validation_scores` so weighting is well-defined.'

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
import cv2

from timm.data import resolve_data_config, create_transform




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
_fallback_model_name_for_transform = "efficientnet_b0"
_tmp_model = timm.create_model(_fallback_model_name_for_transform, pretrained=True)
_data_cfg = resolve_data_config({}, model=_tmp_model)
transform = create_transform(**_data_cfg, is_training=False)
del _tmp_model



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
"""
model_paths = {
    'resnet18': "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/resnet18(WD_1e-3)_aptos.pth",
    'efficientnet_b5': "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/efficientnet_b5.pth",
    'inception_resnet_v2': "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/inception_resnet_v2.pth",
    'inception_v4': "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/inception_v4.pth",
    'seresnext50_32x4d': "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/seresnext50_32x4d.pth",
    'seresnext101_32x4d': "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/seresnext101_32x4d.pth"
}
"""
model_paths = {
    "efficientnet_b0": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/1/efficentNet_b0.pth",
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


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model", "model_state_dict", "net", "weights"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def _replace_classifier(model, num_classes=5):
    if hasattr(model, "reset_classifier"):
        model.reset_classifier(num_classes=num_classes)
        return model
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
        in_features = getattr(model.classifier, "in_features", None)
        if in_features is not None:
            model.classifier = nn.Linear(in_features, num_classes)
            return model
    if hasattr(model, "fc") and isinstance(model.fc, nn.Module):
        in_features = getattr(model.fc, "in_features", None)
        if in_features is not None:
            model.fc = nn.Linear(in_features, num_classes)
            return model
    return model


class PriorLogitsModel(nn.Module):
    def __init__(self, prior_probs: np.ndarray):
        super().__init__()
        prior_probs = np.asarray(prior_probs, dtype=np.float64)
        prior_probs = np.clip(prior_probs, 1e-6, 1.0)
        prior_probs = prior_probs / prior_probs.sum()
        logits = np.log(prior_probs).astype(np.float32)
        self.register_buffer("logits", torch.from_numpy(logits).view(1, -1))

    def forward(self, x):
        return self.logits.expand(x.shape[0], -1)


models_list = []
loaded_model_keys = []

for model_key, path in model_paths.items():
    model_name = model_names[model_key]
    model = timm.create_model(model_name, pretrained=False, num_classes=5)
    if os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu")
        state = _extract_state_dict(ckpt)
        if isinstance(state, dict) and any(
            k.startswith("module.") for k in state.keys()
        ):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        models_list.append(model.to(device).eval())
        loaded_model_keys.append(model_key)
    else:
        print(f"WARNING: missing checkpoint for {model_key} at {path}. Skipping.")

if len(models_list) == 0:
    train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
    train_df = pd.read_csv(train_csv_file)
    counts = (
        train_df["diagnosis"]
        .value_counts()
        .reindex([0, 1, 2, 3, 4], fill_value=0)
        .values
    )
    prior = counts / max(counts.sum(), 1)
    print(
        "WARNING: no checkpoints found; using label-prior fallback model for inference."
    )
    model = PriorLogitsModel(prior_probs=prior)
    models_list = [model.to(device).eval()]
    loaded_model_keys = ["efficientnet_b0"]



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
    "inception_v4": 0.8875,  # 0.888,
    "seresnext50_32x4d": 0.8652,  # 0.709,
    "seresnext101_32x4d": 0.9083,  # 0.951
}



## === cell 7
total_score = sum(validation_scores[k] for k in loaded_model_keys)
weights = {k: validation_scores[k] / total_score for k in loaded_model_keys}



## === cell 8
hflip = transforms.RandomHorizontalFlip(p=1.0)

all_outputs = []

with torch.no_grad():
    for images in tqdm(test_loader):
        images = images.to(device, non_blocking=True)

        images_flip = torch.flip(images, dims=[3])

        outputs = []
        for model_key, model in zip(loaded_model_keys, models_list):
            logits = model(images)
            logits_flip = model(images_flip)

            probs = nn.functional.softmax(logits, dim=1)
            probs_flip = nn.functional.softmax(logits_flip, dim=1)

            probs_avg = 0.5 * (probs + probs_flip)
            outputs.append(weights[model_key] * probs_avg.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        weighted_outputs = torch.sum(outputs, dim=0)
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

submission_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_df.shape)
print(submission_df.head())
