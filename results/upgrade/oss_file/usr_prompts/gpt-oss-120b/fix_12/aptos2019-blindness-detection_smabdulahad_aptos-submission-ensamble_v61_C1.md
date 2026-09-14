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

- What this solution (achieved 0.05303) has done: 'I add defensive loading for the model checkpoint: if the file is missing, the code fall back to a pretrained EfficientNet‑b0 model so a model is always available. The weight dictionary be rebuilt only for the models that were actually loaded, preventing empty‑list errors when concatenating outputs. These changes fix the FileNotFoundError, the empty‑tensor list, and the undefined final predictions, allowing the script to run end‑to‑end and write a valid `submission.csv`.'
- What this solution (achieved -0.01538) has done: 'I add a cheap test‑time augmentation (horizontal flip) and average the soft‑max outputs, then compute the expected class value instead of a plain argmax. This keeps the original model ensemble logic unchanged while often raising the quadratic weighted kappa, moving the score from the very low 0.053 toward the target 0.749.'
- What this solution (achieved 0.21116) has done: 'I fix the checkpoint path typo and add a safe fallback that automatically finds any EfficientNet‑B0 `.pth` file in the same directory. This ensures the pretrained model is loaded instead of falling back to generic ImageNet weights, which dramatically improves prediction quality and moves the quadratic weighted kappa much closer to the target score. No core logic or architecture changes are made.'
- What this solution (achieved 0.10602) has done: 'I add a fallback that automatically searches the whole `/kaggle/input` tree for a checkpoint matching each model name, so the fine‑tuned weights are used when they exist. I also switch the final decision from rounding the expected class to a plain `argmax` which aligns better with the quadratic weighted‑kappa metric. These small changes keep the original ensemble logic intact while moving the score toward the target.'
- What this solution (achieved 0.0) has done: 'I replace the argmax post‑processing with a probability‑weighted expected‑value calculation followed by rounding and clipping. This keeps the ensemble and augmentation logic unchanged while producing predictions that better reflect the model confidence, which typically raises the quadratic weighted kappa and moves the score toward the target.'
- What this solution (achieved -0.00327) has done: 'The script failed because no model checkpoints were found, leaving the model list empty and causing a `torch.cat` error and undefined predictions. I added a safe fallback that loads each architecture with ImageNet‑pretrained weights when a fine‑tuned checkpoint is missing, guarantees at least one model is available, and keeps the original weighting logic. These tweaks fix the runtime errors while preserving the original ensemble‑averaging and expected‑value post‑processing, moving the solution toward the target score.'
- What this solution (achieved 0.03846) has done: 'I replace the expected‑value rounding with a simple argmax of the weighted class probabilities. Argmax usually aligns better with the quadratic weighted‑kappa metric when the probability estimates are noisy, and this change is minimal, keeps the ensemble logic unchanged, and moves the score toward the target. The rest of the pipeline—including model loading, weighting, and test‑time augmentation—remains the same.'
- What this solution (achieved 0.0) has done: 'I replace the arg‑max post‑processing with a probability‑weighted expected‑value calculation (rounding to the nearest integer) and add a fallback that, when none of the ensemble models were fine‑tuned, predicts using the class‑frequency prior from the training set. This change keeps the original loading, augmentation and weighting logic untouched while providing a more sensible prediction strategy that moves the quadratic weighted‑kappa score toward the target.'
- What this solution (achieved 0.02866) has done: 'I keep the existing data loading, model creation and weighting logic, but change the post‑processing so that the ensemble’s raw class probabilities are used directly instead of being replaced by a class‑frequency prior when no fine‑tuned checkpoints are found. I also switch from the expected‑value → rounding approach to a simple argmax over the probability vector, which aligns better with the quadratic weighted kappa metric and prevents all predictions from collapsing to a single class. These minimal edits keep the core architecture unchanged while moving the score toward the target.'
- What this solution (achieved 0.0) has done: 'I keep the overall architecture and ensemble logic unchanged but improve the post‑processing and weighting so that the predictions are less noisy and slightly corrected toward the true class distribution.  
1. Compute the class‑frequency prior from the training labels and blend it with the ensemble probabilities (α = 0.1).  
2. When a model was not loaded from a fine‑tuned checkpoint, reduce its weight (multiply by 0.5) so that poorly‑trained ImageNet‑only models influence the final prediction less.  
3. Convert the blended probabilities to an expected‑value prediction, round to the nearest integer and clip to the valid range [0, 4].  
These minimal changes keep the core model loading, augmentation, and ensemble steps intact while moving the quadratic weighted‑kappa score toward the target.'

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



## === cell 4
model_dir = "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/1"
default_ckpt_name = "efficientnet_b0.pth"  # corrected spelling
default_path = os.path.join(model_dir, default_ckpt_name)

if not os.path.isfile(default_path):
    import glob

    candidates = glob.glob(os.path.join(model_dir, "*efficientnet*b0*.pth"))
    if candidates:
        default_path = candidates[0]
        print(f"Found alternative checkpoint: {default_path}")
    else:
        print(
            f"Warning: checkpoint not found at {default_path}. Will use ImageNet pre‑trained weights."
        )

model_paths = {
    "efficientnet_b0": default_path,
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_list = []
loaded_keys = []
fine_tuned_flags = (
    []
)  # track whether each model was loaded from a fine‑tuned checkpoint

import glob  # needed for automatic checkpoint search

for model_key, model_name in model_names.items():
    path = model_paths.get(model_key, None)

    if not path or not os.path.isfile(path):
        candidates = glob.glob(f"/kaggle/input/**/*.pth", recursive=True)
        matches = [
            c for c in candidates if model_key.lower() in os.path.basename(c).lower()
        ]
        if matches:
            path = matches[0]
            print(f"Auto‑found checkpoint for {model_key}: {path}")

    fine_tuned_loaded = False
    model = None

    if path and os.path.isfile(path):
        try:
            state_dict = torch.load(path, map_location=device)
            model = timm.create_model(model_name, pretrained=False, num_classes=5)
            model.load_state_dict(state_dict)
            fine_tuned_loaded = True
        except Exception as e:
            print(
                f"Warning: could not load checkpoint for {model_key} ({path}). Details: {e}"
            )

    if not fine_tuned_loaded:
        try:
            model = timm.create_model(model_name, pretrained=True, num_classes=5)
            print(f"Using ImageNet‑pretrained {model_name} for {model_key}.")
        except Exception as e:
            print(f"Error creating pretrained model {model_name}: {e}")
            continue

    model.to(device)
    model.eval()
    models_list.append(model)
    loaded_keys.append(model_key)
    fine_tuned_flags.append(fine_tuned_loaded)

if not models_list:
    raise RuntimeError("No models could be loaded or created. Abort.")



## === cell 5
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

fine_tuned_map = dict(zip(loaded_keys, fine_tuned_flags))

total_score = sum(
    validation_scores[k] * (1.0 if fine_tuned_map.get(k, False) else 0.5)
    for k in loaded_keys
    if k in validation_scores
)

if total_score == 0:
    weights = {k: 1.0 / len(loaded_keys) for k in loaded_keys}
else:
    weights = {
        k: (validation_scores[k] * (1.0 if fine_tuned_map.get(k, False) else 0.5))
        / total_score
        for k in loaded_keys
        if k in validation_scores
    }

train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
train_labels = pd.read_csv(train_csv_file)["diagnosis"]
class_counts = train_labels.value_counts().sort_index()
class_prior = (class_counts / class_counts.sum()).values  # shape (5,)



## === cell 6
all_outputs = []
alpha = 0.1  # blending factor for class prior

with torch.no_grad():
    for images in tqdm(test_loader, desc="Predicting"):
        images = images.to(device)

        probs_orig = []
        for model_key, model in zip(loaded_keys, models_list):
            prob = nn.functional.softmax(model(images), dim=1)  # (B,5)
            weighted = weights[model_key] * prob
            probs_orig.append(weighted.unsqueeze(0))  # (1,B,5)

        images_flipped = torch.flip(images, dims=[3])
        probs_flip = []
        for model_key, model in zip(loaded_keys, models_list):
            prob = nn.functional.softmax(model(images_flipped), dim=1)
            weighted = weights[model_key] * prob
            probs_flip.append(weighted.unsqueeze(0))

        outputs_tensor = torch.cat(probs_orig + probs_flip, dim=0)  # (2*models,B,5)
        weighted_outputs = torch.mean(outputs_tensor, dim=0)  # (B,5)

        blended = (1 - alpha) * weighted_outputs + alpha * torch.tensor(
            class_prior, device=device
        )

        all_outputs.extend(blended.cpu().numpy())

all_outputs = np.array(all_outputs)  # (num_samples,5)

expected_vals = (all_outputs * np.arange(5)).sum(axis=1)
final_predictions = np.rint(expected_vals).astype(int)
final_predictions = np.clip(final_predictions, 0, 4)



## === cell 7
submission_df = pd.DataFrame(
    {"id_code": pd.read_csv(test_csv_file)["id_code"], "diagnosis": final_predictions}
)
submission_df.to_csv("submission.csv", index=False)
