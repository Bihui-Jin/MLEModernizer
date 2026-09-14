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

0.8900593895168976

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I make the script robust by handling missing model checkpoint files: it fall back to ImageNet‑pretrained weights, skip any model that cannot be loaded, recompute the ensemble weights accordingly, and guard against empty output lists. This fixes the runtime errors and guarantees a valid `submission.csv` is written, while preserving the original ensemble logic as much as possible.'
- What this solution (achieved 0.0) has done: 'I prevent the script from trying to download model weights by disabling the `pretrained=True` flag when creating models with timm, and add a small safeguard for the ensemble‑weight calculation in case no models are loaded. This removes the network‑related crash, guarantees a `submission.csv` is written, and lets the pipeline produce a non‑zero score, moving it toward the target.'
- What this solution (achieved 0.0) has done: 'I add a deterministic seed for reproducibility and replace the arg‑max class selection with a weighted‑average (expected value) followed by rounding and clipping to the 0‑4 range. This simple post‑processing usually aligns the predictions more closely with the continuous nature of the quadratic weighted kappa metric, moving the score toward the target without altering the core model or training logic.'
- What this solution (achieved 0.0) has done: 'I make the ensemble only use models whose checkpoint files actually exist and switch them to ImageNet‑pretrained weights (instead of random initialization). This removes noisy random models from the prediction, keeping the original architecture and weighting logic untouched while giving each retained model a sensible starting point, which should raise the validation‑style score toward the target.'
- What this solution (achieved 0.0) has done: 'I make the ensemble always include every listed architecture, even when a checkpoint file is missing. Missing checkpoints now fall back to the ImageNet‑pretrained weights already loaded by `timm.create_model`. This gives the model a sensible initialization instead of skipping it, so the ensemble predictions become informative rather than uniform (all‑2), moving the validation‑style score toward the target. No other logic is altered.'
- What this solution (achieved -0.0161) has done: 'I adjust the prediction step to use the class with the highest ensembled probability (arg‑max) instead of rounding the expected value. This small change aligns the output directly with the discrete labels, which typically improves the quadratic weighted kappa score while keeping the model architecture, ensembling, and data handling unchanged.'
- What this solution (achieved 0.0) has done: 'I replace the arg‑max class selection with a probability‑weighted expected value followed by rounding and clipping to the valid 0‑4 range. This post‑processing aligns better with the quadratic weighted kappa metric while keeping the model architecture and ensembling unchanged, and should move the score upward toward the target.'
- What this solution (achieved -0.0161) has done: 'I replace the expected‑value rounding with a direct arg‑max selection for the final class prediction. Using the class with highest ensembled probability (arg‑max) aligns the output with the discrete label space required by the quadratic weighted kappa metric and is expected to move the score closer to the target without altering the core model or training logic.'
- What this solution (achieved 0.0) has done: 'I replace the arg‑max based prediction with a probability‑weighted expected value (rounded to the nearest integer) because the quadratic weighted kappa metric better rewards predictions that reflect the underlying probability distribution. This small post‑processing change keeps the model architecture, ensembling, and data handling untouched while moving the score toward the target.'

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
import warnings

torch.manual_seed(42)
np.random.seed(42)




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
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_paths = {
    "resnet18": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/resnet18(WD_1e-3)_aptos.pth",
    "efficientnet_b1": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/efficentNet_b1.pth",
    "efficientnet_b2": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/efficentNet_b2.pth",
    "efficientnet_b3": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/efficentNet_b3.pth",
    "inception_resnet_v2": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/inception_resnet_v2.pth",
    "inception_v4": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/inception_v4.pth",
    "seresnext50_32x4d": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/seresnext50_32x4d.pth",
    "seresnext101_32x4d": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/seresnext101_32x4d.pth",
}
model_names = {
    "resnet18": "resnet18",
    "efficientnet_b1": "efficientnet_b1",
    "efficientnet_b2": "efficientnet_b2",
    "efficientnet_b3": "efficientnet_b3",
    "inception_resnet_v2": "inception_resnet_v2",
    "inception_v4": "inception_v4",
    "seresnext50_32x4d": "seresnext50_32x4d",
    "seresnext101_32x4d": "seresnext101_32x4d",
}
validation_scores = {
    "resnet18": 0.887,
    "efficientnet_b1": 0.894,
    "efficientnet_b2": 0.898,
    "efficientnet_b3": 0.9127,
    "inception_resnet_v2": 0.896,
    "inception_v4": 0.8875,
    "seresnext50_32x4d": 0.8652,
    "seresnext101_32x4d": 0.9083,
}
models_list = []
used_keys = []

for model_key, path in model_paths.items():
    if not os.path.exists(path):
        warnings.warn(
            f"Checkpoint not found for {model_key}. Using ImageNet‑pretrained weights."
        )
    model_name = model_names.get(model_key)
    if model_name is None:
        continue

    model = timm.create_model(model_name, pretrained=True, num_classes=5)

    if os.path.exists(path):
        try:
            state_dict = torch.load(path, map_location=device)
            model.load_state_dict(state_dict)
        except Exception as e:
            warnings.warn(
                f"Error loading {model_key} weights from {path}: {e}. "
                "Falling back to ImageNet‑pretrained weights."
            )

    model.to(device)
    model.eval()
    models_list.append(model)
    used_keys.append(model_key)

if used_keys:
    total_score = sum(validation_scores[k] for k in used_keys)
    weights = {k: validation_scores[k] / total_score for k in used_keys}
else:
    warnings.warn("No models loaded; using uniform predictions.")
    weights = {}




## === cell 4
all_outputs = []

with torch.no_grad():
    for images in tqdm(test_loader, desc="Inference"):
        images = images.to(device)
        model_outputs = []
        for model_key, model in zip(used_keys, models_list):
            logits = model(images)
            probs = nn.functional.softmax(logits, dim=1)  # (B,5)
            weighted = weights.get(model_key, 0) * probs
            model_outputs.append(weighted.unsqueeze(0))  # (1,B,5)

        if not model_outputs:
            uniform = torch.full((images.size(0), 5), 1.0 / 5, device=device)
            all_outputs.extend(uniform.cpu().numpy())
            continue

        stacked = torch.cat(model_outputs, dim=0)  # (M,B,5)
        ensemble = torch.sum(stacked, dim=0)  # (B,5)
        all_outputs.extend(ensemble.cpu().numpy())

all_outputs = np.array(all_outputs)

expected_vals = np.sum(all_outputs * np.arange(5), axis=1)
final_predictions = np.rint(expected_vals).astype(int)
final_predictions = np.clip(final_predictions, 0, 4)




## === cell 5
submission_df = pd.DataFrame(
    {
        "id_code": pd.read_csv(test_csv_file)["id_code"],
        "diagnosis": final_predictions,
    }
)
submission_df.to_csv("submission.csv", index=False)
