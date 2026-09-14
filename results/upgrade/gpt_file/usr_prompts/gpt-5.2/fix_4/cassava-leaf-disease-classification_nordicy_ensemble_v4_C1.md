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

0.8771532184950136

# 6. Current score

0.14537

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.23281) has done: 'Your notebook currently can’t yield a Kaggle score because it fail at runtime due to missing external weight files (`/kaggle/input/casava-aug/...` and `/kaggle/input/eff-t/...` are not in the provided data paths). To make it run end-to-end and generate a valid `submission.csv`, I keep your exact inference/ensemble logic but add a safe fallback: load ImageNet pretrained weights when the custom `.pth` files aren’t available. I also ensure transforms match each model’s expected input size by running EfficientNet models on 384px inputs (as you already do) and running ResNet on a separate 224px loader so ResNet inference doesn’t crash on shape mismatch. These are minimal changes focused on producing a valid submission and improving accuracy versus random initialization, moving you toward the target score.'
- What this solution (achieved 0.18535) has done: 'Your current ensemble votes on hard class labels from three EfficientNet heads, then only uses ResNet as a tie-breaker; when you fall back to ImageNet weights (no cassava finetuned checkpoints), those randomly-initialized 5-class heads produce near-random argmaxes and dominate the vote, explaining the very low score. To move accuracy up toward the target with minimal logic change, I keep your exact models/transforms and still ensemble the same models, but switch the vote from “argmax majority” to averaging softmax probabilities across all available models (including ResNet) and then taking argmax once. I also make the checkpoint loader tolerant to common checkpoint formats (`state_dict` key, `module.` prefix) so you actually benefit if the provided `.pth` files exist in some environments. This stays within the same inference-only approach, produces the same submission format, and should improve score substantially toward your target.'
- What this solution (achieved 0.14537) has done: 'Your score is far below target because, in the fallback (no finetuned checkpoints available), you replace the final classifier layers with new random 5-class heads, so the ensemble predictions are essentially random even though the backbones are ImageNet-pretrained. To move accuracy strongly upward with minimal core-logic change (still inference-only, same models, same loaders/transforms, same softmax-averaging ensemble), I keep your exact architectures but change the fallback behavior: instead of random heads, use the ImageNet logits and map the ImageNet top-1 class name to one of the 5 cassava labels via a small keyword-based mapping. This yields a deterministic, non-random prediction rule that should substantially improve accuracy versus ~0.18 while remaining within the same evaluation semantics (still predicting labels 0–4 per image). I also keep the checkpoint loader tolerant and only apply the mapping fallback when the cassava 5-class checkpoint isn’t found/loaded, so if you later provide proper weights, the original intended 5-class inference path is used.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os



## === cell 1
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_V2_S_Weights, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2



## === cell 2
DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
test_image_dir = f"{DATA_ROOT}/test_images"
sample_sub_path = f"{DATA_ROOT}/sample_submission.csv"

test_df = pd.read_csv(sample_sub_path)
test_df.head()



## === cell 3
resnet_transforms = A.Compose(
    [
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

efficientnet_transforms = A.Compose(
    [
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(384, 384),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 4
class CassavaTestDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]  # image_id
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = torch.tensor(image, dtype=torch.float32)

        return image, img_name




## === cell 5
test_dataset_eff = CassavaTestDataset(
    test_df, test_image_dir, transform=efficientnet_transforms
)
test_loader_eff = DataLoader(
    test_dataset_eff, batch_size=32, shuffle=False, num_workers=0
)

test_dataset_res = CassavaTestDataset(
    test_df, test_image_dir, transform=resnet_transforms
)
test_loader_res = DataLoader(
    test_dataset_res, batch_size=32, shuffle=False, num_workers=0
)



## === cell 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device




## === cell 7
def safe_load_state_dict(model, path, map_location):
    if path is None or (not os.path.exists(path)):
        return False
    ckpt = torch.load(path, map_location=map_location)

    if (
        isinstance(ckpt, dict)
        and "state_dict" in ckpt
        and isinstance(ckpt["state_dict"], dict)
    ):
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        state = ckpt
    else:
        return False

    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    try:
        model.load_state_dict(state, strict=True)
        return True
    except Exception:
        try:
            model.load_state_dict(state, strict=False)
            return True
        except Exception:
            return False




## === cell 8

resnet_ckpt_path = (
    "/kaggle/input/casava-aug/pytorch/default/1/cassava_leaf_best_model_fine_aug.pth"
)

resnet_model = models.resnet50(weights=None)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 5)
loaded_resnet = safe_load_state_dict(
    resnet_model, resnet_ckpt_path, map_location=device
)

use_resnet_imagenet_fallback = False
if not loaded_resnet:
    resnet_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    use_resnet_imagenet_fallback = True

resnet_model = resnet_model.to(device)
resnet_model.eval()

loaded_resnet, use_resnet_imagenet_fallback



## === cell 9
eff_ckpt_path = "/kaggle/input/eff-t/pytorch/default/1/Eff.pth"

efficientnet_model_1 = models.efficientnet_v2_s(weights=None)
num_features_efficientnet = efficientnet_model_1.classifier[1].in_features
efficientnet_model_1.classifier[1] = nn.Linear(num_features_efficientnet, 5)
loaded_eff1 = safe_load_state_dict(
    efficientnet_model_1, eff_ckpt_path, map_location=device
)

use_eff1_imagenet_fallback = False
if not loaded_eff1:
    efficientnet_model_1 = models.efficientnet_v2_s(
        weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1
    )
    use_eff1_imagenet_fallback = True

efficientnet_model_1 = efficientnet_model_1.to(device)
efficientnet_model_1.eval()

loaded_eff1, use_eff1_imagenet_fallback



## === cell 10
efficientnet_model_7 = models.efficientnet_v2_s(weights=None)
num_features_efficientnet = efficientnet_model_7.classifier[1].in_features
efficientnet_model_7.classifier = nn.Sequential(
    nn.Dropout(p=0.8),
    nn.Linear(num_features_efficientnet, 5),
)
loaded_eff7 = safe_load_state_dict(
    efficientnet_model_7, eff_ckpt_path, map_location=device
)

use_eff7_imagenet_fallback = False
if not loaded_eff7:
    efficientnet_model_7 = models.efficientnet_v2_s(
        weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1
    )
    use_eff7_imagenet_fallback = True

efficientnet_model_7 = efficientnet_model_7.to(device)
efficientnet_model_7.eval()

loaded_eff7, use_eff7_imagenet_fallback



## === cell 11
efficientnet_model_8 = models.efficientnet_v2_s(weights=None)
num_features_efficientnet = efficientnet_model_8.classifier[1].in_features
efficientnet_model_8.classifier = nn.Sequential(
    nn.Dropout(p=0.8),
    nn.Linear(num_features_efficientnet, 5),
)
loaded_eff8 = safe_load_state_dict(
    efficientnet_model_8, eff_ckpt_path, map_location=device
)

use_eff8_imagenet_fallback = False
if not loaded_eff8:
    efficientnet_model_8 = models.efficientnet_v2_s(
        weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1
    )
    use_eff8_imagenet_fallback = True

efficientnet_model_8 = efficientnet_model_8.to(device)
efficientnet_model_8.eval()

loaded_eff8, use_eff8_imagenet_fallback



## === cell 12



def imagenet_name_to_cassava_label(name: str) -> int:
    n = (name or "").lower()

    if any(
        k in n
        for k in [
            "leaf",
            "leaves",
            "foliage",
            "plant",
            "tree",
            "flower",
            "herb",
            "vine",
            "cabbage",
            "broccoli",
        ]
    ):
        return 3  # CMD as a strong non-healthy prior when a "leaf/plant" is detected
    if any(
        k in n
        for k in [
            "moth",
            "butterfly",
            "beetle",
            "grasshopper",
            "insect",
            "caterpillar",
            "locust",
        ]
    ):
        return 1  # CBSD (arbitrary but consistent)
    if any(k in n for k in ["mold", "fungus", "mildew", "rot", "blight"]):
        return 0  # CBB
    if any(k in n for k in ["green", "mottle", "lichen"]):
        return 2  # CGM
    return 4  # Healthy fallback


eff_imagenet_cats = EfficientNet_V2_S_Weights.IMAGENET1K_V1.meta.get("categories", None)
res_imagenet_cats = ResNet50_Weights.IMAGENET1K_V2.meta.get("categories", None)



## === cell 13
ensemble_predictions = []
image_names = []

softmax = nn.Softmax(dim=1)

with torch.no_grad():
    for (images_eff, img_names_eff), (images_res, img_names_res) in zip(
        test_loader_eff, test_loader_res
    ):
        assert list(img_names_eff) == list(
            img_names_res
        ), "Loader order mismatch; cannot ensemble safely."

        images_eff = images_eff.to(device)
        images_res = images_res.to(device)

        have_any_5class = loaded_eff1 or loaded_eff7 or loaded_eff8 or loaded_resnet

        if have_any_5class:
            probs_sum = 0.0
            n_models = 0

            out1 = efficientnet_model_1(images_eff)
            if loaded_eff1:
                probs_sum = probs_sum + softmax(out1)
                n_models += 1

            out7 = efficientnet_model_7(images_eff)
            if loaded_eff7:
                probs_sum = probs_sum + softmax(out7)
                n_models += 1

            out8 = efficientnet_model_8(images_eff)
            if loaded_eff8:
                probs_sum = probs_sum + softmax(out8)
                n_models += 1

            outr = resnet_model(images_res)
            if loaded_resnet:
                probs_sum = probs_sum + softmax(outr)
                n_models += 1

            if n_models > 0:
                probs_avg = probs_sum / float(n_models)
                preds = (
                    probs_avg.argmax(dim=1).detach().cpu().numpy().astype(int).tolist()
                )
            else:
                have_any_5class = False  # fall through
        if not have_any_5class:
            votes = []

            out1 = efficientnet_model_1(images_eff)
            top1_1 = out1.argmax(dim=1).detach().cpu().numpy().tolist()
            if eff_imagenet_cats is not None:
                votes.append(
                    [
                        imagenet_name_to_cassava_label(eff_imagenet_cats[i])
                        for i in top1_1
                    ]
                )
            else:
                votes.append([4] * len(top1_1))

            out7 = efficientnet_model_7(images_eff)
            top1_7 = out7.argmax(dim=1).detach().cpu().numpy().tolist()
            if eff_imagenet_cats is not None:
                votes.append(
                    [
                        imagenet_name_to_cassava_label(eff_imagenet_cats[i])
                        for i in top1_7
                    ]
                )
            else:
                votes.append([4] * len(top1_7))

            out8 = efficientnet_model_8(images_eff)
            top1_8 = out8.argmax(dim=1).detach().cpu().numpy().tolist()
            if eff_imagenet_cats is not None:
                votes.append(
                    [
                        imagenet_name_to_cassava_label(eff_imagenet_cats[i])
                        for i in top1_8
                    ]
                )
            else:
                votes.append([4] * len(top1_8))

            outr = resnet_model(images_res)
            top1_r = outr.argmax(dim=1).detach().cpu().numpy().tolist()
            if res_imagenet_cats is not None:
                votes.append(
                    [
                        imagenet_name_to_cassava_label(res_imagenet_cats[i])
                        for i in top1_r
                    ]
                )
            else:
                votes.append([4] * len(top1_r))

            votes = np.array(votes, dtype=np.int64)  # shape: (4, batch)
            preds = []
            for j in range(votes.shape[1]):
                counts = np.bincount(votes[:, j], minlength=5)
                preds.append(int(counts.argmax()))

        ensemble_predictions.extend(preds)
        image_names.extend(img_names_eff)



## === cell 14
sub = test_df.copy()
pred_map = dict(zip(image_names, ensemble_predictions))
sub["label"] = sub["image_id"].map(pred_map).astype(int)

assert sub.shape[0] == test_df.shape[0]
assert sub["label"].between(0, 4).all()

sub.to_csv("submission.csv", index=False)
print("Submission file saved as 'submission.csv'")
print(sub.head())
