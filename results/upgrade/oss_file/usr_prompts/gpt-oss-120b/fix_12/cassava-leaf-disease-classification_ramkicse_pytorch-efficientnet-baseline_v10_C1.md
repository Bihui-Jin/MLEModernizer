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
seaborn==0.12.2
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

0.8705046841946207

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.22384) has done: 'I fixed the runtime error caused by missing weight files by removing the per‑batch weight loading and instead using a single EfficientNet‑B7 model pretrained on ImageNet. The model is now always built with ImageNet weights, and predictions are generated directly from this model. This eliminates the FileNotFoundError and ensures a valid `submission.csv` is written.'
- What this solution (achieved 0.22347) has done: 'I adjust the dataset class so that test mode does not expect a `label` column, and I simplify the prediction step: if none of the external weight files are found, the script still run inference using the ImageNet‑pretrained EfficientNet‑B7 model (the base model that is already built). This prevents the runtime error and guarantees a valid `submission.csv` is written.'
- What this solution (achieved 0.61099) has done: 'I keep the existing EfficientNet‑B7 model and inference flow, but improve the predictions by (1) applying a simple test‑time augmentation that averages the model output for the original and horizontally‑flipped images, and (2) re‑weighting the averaged class probabilities with the empirical class‑frequency priors from the training set before taking the arg‑max. These lightweight changes preserve the core architecture and inference logic while are expected to raise the accuracy toward the target score.'
- What this solution (achieved 0.23692) has done: 'I add a lightweight test‑time augmentation that also includes vertical flips (and the combination of both flips) and remove the class‑frequency prior re‑weighting, because the prior scaling can bias predictions away from the true classes. These changes keep the same EfficientNet‑B7 model and inference flow while giving the model more diverse views of each image, which should raise the accuracy toward the target score.'
- What this solution (achieved 0.61099) has done: 'I re‑introduce the class‑frequency prior scaling that was removed in the last version because it helps push the predictions toward the more common classes and historically raised the validation accuracy. The training‑set label distribution is computed once, turned into a probability vector, and multiplied element‑wise with the averaged TTA probabilities before taking the argmax. This tiny change keeps the exact model architecture and inference flow while moving the score upward toward the target.'
- What this solution (achieved 0.61099) has done: 'Implemented a fix for the probability accumulation bug by correctly gathering batch predictions before weighting and adding them to the overall score matrix. Added class‑frequency prior scaling using the training label distribution to improve accuracy toward the target. Updated the fallback logic when no external weights are found to follow the same aggregation pattern.'

# 9. Code solution

## === cell 0
try:
    get_ipython().system(
        "pip install ../input/efficientnetpytorch-install/dist/efficientnet_pytorch-0.7.0.tar"
    )
except Exception:
    pass



## === cell 1
TRAINING = False
WEIGHT_BASE_PATH = "../input/ramki-cassava-weights/"

WEIGHT_FILES = [
    WEIGHT_BASE_PATH + "fold-0-weight-at-epoch-42-acc-0.87126.pth",
    WEIGHT_BASE_PATH + "fold-1-loss-weight-at-epoch-33-loss-0.48675.pth",
    WEIGHT_BASE_PATH + "fold-2-weight-at-epoch-47-acc-0.86212.pth",
]



## === cell 2
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

SummaryWriter = None



## === cell 3
try:
    from efficientnet_pytorch import EfficientNet
except ImportError:
    from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights

    class EfficientNet:
        @staticmethod
        def from_name(name, num_classes):
            model = efficientnet_b7(weights=None)
            model.classifier[1] = nn.Linear(
                model.classifier[1].in_features, num_classes
            )
            return model

        @staticmethod
        def from_pretrained(name, num_classes):
            model = efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
            model.classifier[1] = nn.Linear(
                model.classifier[1].in_features, num_classes
            )
            return model




## === cell 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")



## === cell 5
SEED = 42
N_FOLDS = 5
N_EPOCHS = 50
BATCH_SIZE = 16
IMG_SIZE = 224
LR = 0.001
NUM_CLASSES = 5


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


seed_everything(SEED)



## === cell 6
base_path = "../input/cassava-leaf-disease-classification/"
train_path = base_path + "train_images/"
test_path = base_path + "test_images/"

train_csv = pd.read_csv(base_path + "train.csv")
sample = pd.read_csv(base_path + "sample_submission.csv")




## === cell 7
class CasavaDataset(Dataset):
    def __init__(self, dataframe, transforms=None, test=False):
        self.df = dataframe
        self.transforms = transforms
        self.test = test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx].image_id
        img_path = (test_path if self.test else train_path) + img_id
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]
        label = -1 if self.test else self.df.iloc[idx].label
        return image, label




## === cell 8
import albumentations as A
from albumentations.pytorch import ToTensorV2

transforms_train = A.Compose(
    [
        A.Resize(height=IMG_SIZE, width=IMG_SIZE),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ],
    p=1.0,
)

transforms_valid = A.Compose(
    [
        A.Resize(height=IMG_SIZE, width=IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)




## === cell 9
def build_model():
    model_name = "efficientnet-b7"
    model = EfficientNet.from_pretrained(model_name, num_classes=NUM_CLASSES)
    print("Model created with ImageNet pretrained weights")
    return model.to(device)




## === cell 10
model = build_model()



## === cell 11
testset = CasavaDataset(sample, transforms=transforms_valid, test=True)
test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)



## === cell 12
import re


def _extract_weight(path):
    """Extract a numeric score from the weight filename.
    Prefer validation accuracy (acc-...), otherwise use 1/loss if loss-... is present.
    Fallback to weight=1.0."""
    m = re.search(r"acc-([0-9.]+)", path)
    if m:
        return float(m.group(1))
    m = re.search(r"loss-([0-9.]+)", path)
    if m:
        loss_val = float(m.group(1))
        return 1.0 / (loss_val + 1e-6)
    return 1.0


def tta_predict_batch(model, images):
    """Return averaged class probabilities for multiple flip augmentations."""
    probs = []
    probs.append(model(images).softmax(dim=1))
    probs.append(model(torch.flip(images, dims=[3])).softmax(dim=1))  # horiz
    probs.append(model(torch.flip(images, dims=[2])).softmax(dim=1))  # vert
    probs.append(
        model(torch.flip(torch.flip(images, dims=[3]), dims=[2])).softmax(dim=1)
    )  # horiz+vert
    return torch.stack(probs).mean(0)


model.eval()
all_probs_weighted = np.zeros((len(sample), NUM_CLASSES), dtype=np.float32)
total_weight = 0.0
valid_folds = 0

for w_path in WEIGHT_FILES:
    if not os.path.isfile(w_path):
        print(f"Weight file not found: {w_path} – skipping")
        continue
    try:
        state = torch.load(w_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded weights from {w_path}")
    except Exception as e:
        print(f"Failed to load {w_path}: {e}")
        continue

    fold_weight = _extract_weight(w_path)
    batch_probs = []
    with torch.no_grad():
        for images, _ in tqdm(
            test_loader, desc=f"Predicting with {os.path.basename(w_path)}"
        ):
            images = images.to(device)
            prob = tta_predict_batch(model, images)  # (B, NUM_CLASSES)
            batch_probs.append(prob.cpu().numpy())
    if not batch_probs:
        continue
    fold_probs = np.concatenate(batch_probs, axis=0)  # (N_test, NUM_CLASSES)

    all_probs_weighted += fold_probs * fold_weight
    total_weight += fold_weight
    valid_folds += 1

if valid_folds == 0:
    print(
        "No external weights loaded – using ImageNet‑pretrained model for prediction."
    )
    batch_probs = []
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc="Predicting with base model"):
            images = images.to(device)
            prob = tta_predict_batch(model, images)
            batch_probs.append(prob.cpu().numpy())
    all_probs_weighted = np.concatenate(batch_probs, axis=0)
    total_weight = 1.0
    valid_folds = 1

avg_probs = all_probs_weighted / total_weight  # (N_test, NUM_CLASSES)

class_counts = train_csv["label"].value_counts().sort_index()
class_prior = (class_counts / class_counts.sum()).values.astype(
    np.float32
)  # (NUM_CLASSES,)
scaled_probs = avg_probs * class_prior  # broadcast multiplication

test_pred = scaled_probs.argmax(axis=1).tolist()

sample["label"] = test_pred
submission_path = "submission.csv"
sample.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")



## === cell 13
print(sample.head())
