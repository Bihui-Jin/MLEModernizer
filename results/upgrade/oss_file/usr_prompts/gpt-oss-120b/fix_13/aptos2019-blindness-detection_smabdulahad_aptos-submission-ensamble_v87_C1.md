# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 2 other files
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
        input/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 2 other files
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
            test_images/
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
            train_images/
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 2 other files
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> input/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> working/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.8949546090873821

# 6. Current score

-0.00378

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -0.05249) has done: 'The fix adds robust model loading: if a checkpoint file is missing, the code now falls back to an ImageNet‑pretrained version of the same architecture, ensuring the ensemble is populated and inference runs without errors. This prevents the empty‑list issue when concatenating model outputs and restores the creation of `final_predictions`, allowing a valid `submission.csv` to be written.'
- What this solution (achieved 0.00343) has done: 'I keep the overall data loading, model loading and weighted‑ensemble inference unchanged, but replace the naïve arg‑max conversion with a calibrated expected‑value rounding step. Using the soft‑max probabilities to compute an expected class and then rounding (and clipping) usually aligns better with the quadratic weighted kappa metric, moving the score upward toward the target while preserving the core logic.'
- What this solution (achieved -0.00596) has done: 'I corrected the checkpoint file names in the model paths dictionary (the original paths contained a typo “efficentNet”). With the proper filenames the script now loads the fine‑tuned checkpoints instead of falling back to generic ImageNet‑pretrained models, which substantially raises the quadratic weighted kappa score toward the target while keeping the original architecture and inference logic unchanged.'
- What this solution (achieved -0.00989) has done: 'Implemented a focused fix to the weighting scheme: the ensemble weights are now normalized **only over the models actually used** (those listed in `model_paths`). This prevents the probability mass from being unintentionally down‑scaled, which was biasing predictions toward lower severity classes and causing the very low QWK score. The rest of the pipeline remains unchanged, preserving the core logic and model architecture.'
- What this solution (achieved -0.14362) has done: 'I replace the expected‑value rounding step with a simple argmax class selection, which aligns better with the Quadratic Weighted Kappa metric for this problem. The change is limited to the prediction‑generation part of cell 8, preserving the rest of the pipeline, model ensemble, and weighting logic.'
- What this solution (achieved -0.03213) has done: 'I replace the simple arg‑max conversion with a calibrated expected‑value rounding step.  
The weighted soft‑max outputs are first normalised to form a proper probability distribution, then the expected class value is computed, rounded to the nearest integer and clipped to the valid label range [0, 4]. This small change aligns the predictions better with the quadratic weighted kappa metric and moves the score toward the target while preserving the existing model and ensemble logic.'
- What this solution (achieved -0.05193) has done: 'The update makes model loading robust by checking several possible directories for the checkpoint files; if a checkpoint is found it is loaded, otherwise a pretrained ImageNet version is used. This ensures the ensemble actually uses the fine‑tuned weights when they exist, which should raise the quadratic weighted kappa toward the target while keeping the original architecture and inference pipeline unchanged.'
- What this solution (achieved -0.08507) has done: 'I adjust the prediction step to use the class with the highest probability (argmax) instead of rounding the expected value. This simple change aligns better with the Quadratic Weighted Kappa metric for this problem and is expected to raise the score toward the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.13639) has done: 'I replace the simple arg‑max conversion with an expected‑value rounding step, which better aligns the predicted class with the quadratic weighted kappa metric and should raise the score toward the target while keeping the model ensemble and loading logic unchanged.'
- What this solution (achieved -0.0294) has done: 'The score is low because the script often cannot find the fine‑tuned checkpoint files and falls back to generic ImageNet‑pretrained models. I added a more robust search that looks for any “.pth” file containing the model key in each candidate directory, so the proper fine‑tuned weights are loaded when present. This change keeps the original architecture and ensemble logic unchanged while greatly improving the relevance of the predictions, moving the QWK score much closer to the target.'
- What this solution (achieved -0.03614) has done: 'I make the checkpoint‑search robust by walking through all sub‑folders of the candidate directories so fine‑tuned *.pth* files are actually found and loaded. This ensures the ensemble uses the intended pretrained models rather than falling back to generic ImageNet weights, which should raise the quadratic weighted kappa score toward the target.'
- What this solution (achieved -0.00378) has done: 'I expand the ensemble by automatically adding any models listed in `validation_scores` that were not already loaded from checkpoints. This keeps the original loading logic but ensures the ensemble contains all high‑scoring architectures (using ImageNet‑pretrained weights when fine‑tuned checkpoints are missing), which should raise the quadratic weighted kappa toward the target. No other core logic is altered.'

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
from multiprocessing import Pool
import cv2




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
        image = Image.open(img_name)

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




## === cell 4
candidate_base_dirs = [
    "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2",
    "/kaggle/input/aptos2019-blindness-detection/ensamble_v2/2",
    "/kaggle/input/aptos2019-blindness-detection",
    "/kaggle/working/aptos2019-blindness-detection",
]

model_paths = {
    "efficientnet_b1": "efficientnet_b1.pth",
    "efficientnet_b2": "efficientnet_b2.pth",
    "efficientnet_b3": "efficientnet_b3.pth",
    "seresnext101_32x4d": "seresnext101_32x4d.pth",
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
models_list = []
loaded_keys = []

for model_key, filename in model_paths.items():
    model_name = model_names[model_key]
    checkpoint_path = None

    for base in candidate_base_dirs:
        if not os.path.isdir(base):
            continue
        for root, _, files in os.walk(base):
            for f in files:
                if f.endswith(".pth") and model_key in f:
                    candidate = os.path.join(root, f)
                    if os.path.isfile(candidate):
                        checkpoint_path = candidate
                        break
            if checkpoint_path:
                break
        if checkpoint_path:
            break

    try:
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint.get("state_dict", checkpoint)
            model = timm.create_model(model_name, pretrained=False, num_classes=5)
            model.load_state_dict(state_dict, strict=False)
        else:
            model = timm.create_model(model_name, pretrained=True, num_classes=5)
    except Exception:
        model = timm.create_model(model_name, pretrained=True, num_classes=5)

    model.to(device)
    model.eval()
    models_list.append(model)
    loaded_keys.append(model_key)

for extra_key in validation_scores.keys():
    if extra_key in loaded_keys:
        continue
    if extra_key not in model_names:
        continue  # safety: skip if we lack a name mapping
    model_name = model_names[extra_key]
    try:
        extra_model = timm.create_model(model_name, pretrained=True, num_classes=5)
    except Exception:
        continue
    extra_model.to(device)
    extra_model.eval()
    models_list.append(extra_model)
    loaded_keys.append(extra_key)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/65638730.py in <cell line: 0>()
     45 # ensemble, ensuring the weighting scheme can use all available scores.
     46 # ----------------------------------------------------------------------
---> 47 for extra_key in validation_scores.keys():
     48     if extra_key in loaded_keys:
     49         continue

NameError: name 'validation_scores' is not defined

## === cell 6
validation_scores = {
    "resnet18": 0.887,
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




## === cell 7
used_keys = [k for k in loaded_keys if k in validation_scores]
total_score = sum(validation_scores[k] for k in used_keys) or 1e-12  # avoid div‑0
weights = {k: validation_scores[k] / total_score for k in used_keys}




## === cell 8
all_outputs = []

with torch.no_grad():
    for images in tqdm(test_loader, desc="Inference"):
        images = images.to(device)
        outputs = [
            weights[model_key]
            * nn.functional.softmax(model(images), dim=1).unsqueeze(0)
            for model_key, model in zip(loaded_keys, models_list)
            if model_key in weights
        ]
        if not outputs:
            continue
        outputs = torch.cat(outputs)  # (num_models, batch, 5)
        weighted_outputs = torch.sum(outputs, dim=0)  # (batch, 5)

        probs = weighted_outputs
        all_outputs.extend(probs.cpu().numpy())

all_outputs = np.array(all_outputs)  # (num_samples, 5)

expected_vals = np.dot(all_outputs, np.arange(5))
final_predictions = np.clip(np.rint(expected_vals), 0, 4).astype(int)




## === cell 9
submission_df = pd.DataFrame(
    {"id_code": pd.read_csv(test_csv_file)["id_code"], "diagnosis": final_predictions}
)

submission_df.to_csv("submission.csv", index=False)
