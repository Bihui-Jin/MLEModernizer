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

3.13

# 3. Installed packages

albumentations==2.0.8
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
tqdm==4.67.1

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

0.8747355696585071

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11136) has done: 'I fix the script so it always creates a valid `submission.csv`. The changes load the sample‑submission file using an absolute path, wrap model weight loading in a try/except (falling back to ImageNet‑pretrained weights if the custom checkpoints are missing), and write the output to the standard Kaggle working directory. These minimal adjustments keep the original architecture and ensemble logic while ensuring the pipeline runs end‑to‑end and produces a submission file.'
- What this solution (achieved 0.13303) has done: 'I correct the checkpoint paths (fix the typo “casava‑aug” → “cassava‑aug” and add a fallback to a generic “cassava‑leaf‑disease‑classification” folder) so the fine‑tuned weights can be loaded, and then quickly evaluate both models on a small validation split of the training data. Using the validation accuracies I recompute the ensemble weights (higher‑performing model gets a larger weight). This keeps the original architecture and inference pipeline but should raise the accuracy toward the target without changing any core logic.'
- What this solution (achieved 0.14013) has done: 'I keep the overall pipeline unchanged but modify the ensemble weighting logic so that the model with the higher validation accuracy receives all the weight (the other gets zero). This simple change aligns the predictions with the better‑performing model and is expected to move the Kaggle score closer to the target while preserving the core architecture and inference code.'
- What this solution (achieved 0.59043) has done: 'I expanded the checkpoint search to automatically locate any *.pth* files under /kaggle/input so the fine‑tuned weights are more likely to be loaded, and I changed the ensemble weighting to use the validation accuracies proportionally (instead of giving all weight to the single best model). These tweaks keep the original models and data pipeline unchanged while giving a realistic boost toward the target accuracy.'
- What this solution (achieved 0.20291) has done: 'I keep the overall pipeline unchanged but modify the ensemble‐weight calculation so that the model with the higher validation accuracy receives all the weight (the other gets zero). This simple binary weighting often improves the final Kaggle accuracy when one model consistently outperforms the other, moving the score closer to the target while preserving the core logic.'
- What this solution (achieved 0.11136) has done: 'I improve the ensemble weighting by using proportional weights based on each model’s validation accuracy rather than assigning all weight to the single best model. I also increase the validation subset size to obtain a more reliable accuracy estimate, which should move the competition score closer to the target while keeping the core modeling logic unchanged.'
- What this solution (achieved 0.05755) has done: 'I fix the model loading logic so that any discovered checkpoint *.pth* files are actually applied to the ResNet‑50 and EfficientNet‑V2‑S architectures. The original code only loaded the checkpoint into a temporary variable without calling `load_state_dict`, so the models remained un‑trained (or only ImageNet‑pretrained), yielding a very low validation accuracy. By correctly loading the weights the ensemble predictions become much stronger, moving the leaderboard score toward the target while preserving the existing architecture and inference pipeline.'
- What this solution (achieved 0.12294) has done: 'I added a small flexible checkpoint‑loading routine that checks whether a loaded `.pth` file contains the raw state‑dict or is wrapped inside a dictionary (e.g., under `'model'` or `'state_dict'`). This ensures the pretrained weights are actually applied to both ResNet‑50 and EfficientNet‑V2‑S models instead of falling back to ImageNet defaults, which markedly raises validation accuracy and moves the leaderboard score toward the target while keeping the original architecture and inference logic unchanged. The rest of the pipeline (data handling, ensemble weighting, and CSV submission) remains the same.'
- What this solution (achieved 0.45142) has done: 'I simplify the image preprocessing by removing the CLAHE augmentation, keeping only resizing, normalization and tensor conversion. This cleaner preprocessing is better aligned with ImageNet‑pretrained weights and typically yields higher classification accuracy, moving the validation and leaderboard scores closer to the target while leaving the model architecture and overall pipeline unchanged.'
- What this solution (achieved 0.58296) has done: 'I keep the overall data handling, model definitions and inference pipeline unchanged, but replace the simple proportional ensemble weighting with a small validation‑based weight search. By computing each model’s soft‑max probabilities on the validation split and trying a range of weights (0 → 1), we select the combination that yields the highest validation accuracy and use those weights for the final test predictions. This modest change is expected to raise the validation and Kaggle accuracy toward the target while preserving the core logic.'
- What this solution (achieved 0.30979) has done: 'I slightly refine the ensemble weighting by searching on a finer 0.01 grid (instead of 0.05) to capture a better mix of the two models, and I add a simple test‑time horizontal‑flip augmentation during inference: each image is run through the models twice (original and horizontally‑flipped) and the resulting probabilities are averaged before applying the learned ensemble weights. These minimal changes keep the original architectures and training logic intact while expected to raise validation and Kaggle accuracy toward the target.'

# 9. Code solution

## === cell 0
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.models import EfficientNet_V2_S_Weights
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm
import copy
import cv2
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob  # added to enable recursive search for .pth/.pt/.ckpt files

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 1
def _load_checkpoint_into_model(model, possible_paths, model_name):
    """
    Load a checkpoint into ``model`` from a list of candidate files.
    Handles raw state_dicts as well as dicts wrapped under keys like
    'model' or 'state_dict', and tolerates DataParallel prefixes.
    """
    state_dict = None
    for p in possible_paths:
        if os.path_exists(p):
            try:
                loaded = torch.load(p, map_location=device)
                if isinstance(loaded, dict):
                    if "model" in loaded:
                        loaded = loaded["model"]
                    elif "state_dict" in loaded:
                        loaded = loaded["state_dict"]
                state_dict = loaded
                print(f"Loaded {model_name} checkpoint from {p}")
                break
            except Exception as e:
                print(f"Failed loading {model_name} from {p}: {e}")

    if state_dict is not None:
        try:
            new_state = {}
            for k, v in state_dict.items():
                new_key = k.replace("module.", "")
                new_state[new_key] = v
            model.load_state_dict(new_state, strict=False)
            print(f"{model_name} weights loaded into model.")
        except Exception as e:
            print(f"Error loading state_dict into {model_name} model: {e}")
    else:
        print(
            f"Could not find custom {model_name} weights, using ImageNet pretrained model."
        )
    return model


resnet_model = models.resnet50(pretrained=False)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 5)

possible_resnet_paths = [
    "/kaggle/input/cassava-aug/pytorch/default/1/cassava_leaf_best_model_fine_aug.pth",
    "/kaggle/input/cassava-leaf-disease-classification/cassava_leaf_best_model_fine_aug.pth",
]
possible_resnet_paths.extend(
    glob.glob("/kaggle/input/**/*.pth", recursive=True)
    + glob.glob("/kaggle/input/**/*.pt", recursive=True)
    + glob.glob("/kaggle/input/**/*.ckpt", recursive=True)
)

resnet_model = _load_checkpoint_into_model(
    resnet_model, possible_resnet_paths, "ResNet"
)
if not any(os.path.exists(p) for p in possible_resnet_paths):
    resnet_model = models.resnet50(pretrained=True)
    num_ftrs = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(num_ftrs, 5)

resnet_model = resnet_model.to(device)
resnet_model.eval()


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/294132913.py in <cell line: 0>()
     52 )
     53 
---> 54 resnet_model = _load_checkpoint_into_model(
     55     resnet_model, possible_resnet_paths, "ResNet"
     56 )

/tmp/ipykernel_55/294132913.py in _load_checkpoint_into_model(model, possible_paths, model_name)
      7     state_dict = None
      8     for p in possible_paths:
----> 9         if os.path_exists(p):
     10             try:
     11                 loaded = torch.load(p, map_location=device)

AttributeError: module 'os' has no attribute 'path_exists'

## === cell 2
efficientnet_model = models.efficientnet_v2_s(weights=None)
num_features_efficientnet = efficientnet_model.classifier[1].in_features
efficientnet_model.classifier[1] = nn.Linear(num_features_efficientnet, 5)

possible_eff_paths = [
    "/kaggle/input/eff-t/pytorch/default/1/Eff.pth",
    "/kaggle/input/cassava-leaf-disease-classification/Eff.pth",
]
possible_eff_paths.extend(
    glob.glob("/kaggle/input/**/*.pth", recursive=True)
    + glob.glob("/kaggle/input/**/*.pt", recursive=True)
    + glob.glob("/kaggle/input/**/*.ckpt", recursive=True)
)

efficientnet_model = _load_checkpoint_into_model(
    efficientnet_model, possible_eff_paths, "EfficientNet"
)
if not any(os.path.exists(p) for p in possible_eff_paths):
    print("Using ImageNet pretrained EfficientNet weights.")
    efficientnet_model = models.efficientnet_v2_s(
        weights=EfficientNet_V2_S_Weights.DEFAULT
    )
    num_features_efficientnet = efficientnet_model.classifier[1].in_features
    efficientnet_model.classifier[1] = nn.Linear(num_features_efficientnet, 5)

efficientnet_model = efficientnet_model.to(device)
efficientnet_model.eval()

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/1413862384.py in <cell line: 0>()
     13 )
     14 
---> 15 efficientnet_model = _load_checkpoint_into_model(
     16     efficientnet_model, possible_eff_paths, "EfficientNet"
     17 )

/tmp/ipykernel_55/294132913.py in _load_checkpoint_into_model(model, possible_paths, model_name)
      7     state_dict = None
      8     for p in possible_paths:
----> 9         if os.path_exists(p):
     10             try:
     11                 loaded = torch.load(p, map_location=device)

AttributeError: module 'os' has no attribute 'path_exists'
