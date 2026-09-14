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

0.66526

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.75593) has done: 'The main blocker is missing external model checkpoint files, which makes `models_list` empty and breaks inference and submission creation. I add a minimal, self-contained fallback that keeps the same ensemble inference semantics when checkpoints exist, but automatically trains a small timm model on the provided `train_images/train.csv` when they don’t. I also fix robustness issues: ensure RGB loading, set `num_workers=0` for Kaggle notebook stability, load weights with proper `map_location`, and guarantee `submission.csv` is always written with the required columns and row order.'
- What this solution (achieved 0.7625) has done: 'Your current score (0.75593) is well below the target (0.89006), so we should make a small, legitimate change that improves predictions without changing the model architecture or training loop style. The biggest gain with minimal risk is to align the image preprocessing with what timm pretrained backbones expect: use timm’s model-specific `resolve_data_config` + `create_transform` instead of a generic 224-resize transform. This preserves your core approach (single-model fallback training for 1 epoch; softmax + argmax; ensemble when checkpoints exist) but typically yields a sizeable kappa lift because the normalization/interpolation/crop behavior matches the backbone. I also ensure the same transform is used consistently for both training and test in the fallback path, keeping semantics intact and avoiding train/test preprocessing mismatch.'
- What this solution (achieved 0.66526) has done: 'I remove the hard failure when external checkpoint models are missing and re-enable a self-contained fallback path that trains the same kind of timm classifier (num_classes=5) on the provided train images, so the notebook always runs end-to-end and writes `submission.csv`. I keep the ensemble inference semantics when checkpoints exist, but when they don’t, we train a single fallback model for a small fixed amount of time and then run the exact same inference pipeline to produce predictions. I also fix DataLoader stability in this Kaggle environment by using `num_workers=0` (avoids multiprocessing/persistent worker issues) and ensure transforms are always defined consistently via timm’s `resolve_data_config` + `create_transform`. Finally, the script always write a valid submission with correct columns and id ordering.'

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
import timm


def seed_everything(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(42)


def _dl_kwargs_for_env():
    use_cuda = torch.cuda.is_available()
    return dict(
        num_workers=0,
        pin_memory=use_cuda,
        persistent_workers=False,
    )




## === cell 1
import cv2


class BlindnessDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, test=False, return_id=False):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.test = test
        self.return_id = return_id

        self._ids = self.annotations.iloc[:, 0].astype(str).to_numpy()
        if not test and self.annotations.shape[1] > 1:
            self._labels = self.annotations.iloc[:, 1].astype(np.int64).to_numpy()
        else:
            self._labels = None

    def __len__(self):
        return len(self._ids)

    def __getitem__(self, idx):
        img_id = self._ids[idx]
        img_name = os.path.join(self.root_dir, img_id + ".png")

        img_bgr = cv2.imread(img_name, cv2.IMREAD_COLOR)
        if img_bgr is None:
            image = Image.open(img_name).convert("RGB")
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img_rgb)

        if self.transform:
            image = self.transform(image)

        if self.test:
            if self.return_id:
                return image, img_id
            return image
        else:
            label = int(self._labels[idx])
            if self.return_id:
                return image, label, img_id
            return image, label




## === cell 2
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

_fallback_backbone = "resnet18"


def build_transforms(backbone_name: str):
    tmp_model = timm.create_model(backbone_name, pretrained=True, num_classes=5)
    data_cfg = resolve_data_config({}, model=tmp_model)
    tr = create_transform(**data_cfg, is_training=True)
    ev = create_transform(**data_cfg, is_training=False)
    del tmp_model
    return tr, ev




## === cell 3
test_csv_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_root_dir = "/kaggle/input/aptos2019-blindness-detection/test_images"

train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
train_root_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"



## === cell 4
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
loaded_model_keys = []

for model_key, path in model_paths.items():
    if not os.path.exists(path):
        continue
    model_name = model_names[model_key]
    model = timm.create_model(model_name, pretrained=False, num_classes=5)
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    models_list.append(model)
    loaded_model_keys.append(model_key)

print(f"Loaded {len(models_list)} checkpoint models: {loaded_model_keys}")




## === cell 6
def _train_fallback_model(backbone_name: str, train_csv: str, train_root: str):
    transform_train, transform_eval = build_transforms(backbone_name)

    train_ds = BlindnessDataset(
        train_csv, train_root, transform=transform_train, test=False, return_id=False
    )

    dl_kwargs = _dl_kwargs_for_env()
    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        drop_last=False,
        **{k: v for k, v in dl_kwargs.items() if v is not None},
    )

    model = timm.create_model(backbone_name, pretrained=True, num_classes=5)
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    epochs = 1

    for ep in range(epochs):
        running_loss = 0.0
        for images, labels in tqdm(
            train_loader, desc=f"Fallback train ep{ep+1}/{epochs}"
        ):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach().cpu())

        running_loss /= max(1, len(train_loader))
        print(f"Fallback training epoch {ep+1}: loss={running_loss:.4f}")

    model.eval()
    return model, transform_eval


if len(models_list) == 0:
    print(
        "No external checkpoint models found. Training fallback model to enable end-to-end run."
    )
    fallback_model, transform_eval = _train_fallback_model(
        _fallback_backbone, train_csv_file, train_root_dir
    )
    models_list = [fallback_model]
    loaded_model_keys = [_fallback_backbone]
else:
    _backbone_for_transform = model_names[loaded_model_keys[0]]
    _, transform_eval = build_transforms(_backbone_for_transform)



## === cell 7
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

if all(k in validation_scores for k in loaded_model_keys):
    total_score = sum(validation_scores[k] for k in loaded_model_keys)
    weights = {k: validation_scores[k] / total_score for k in loaded_model_keys}
    weights_vec = torch.tensor(
        [weights[k] for k in loaded_model_keys], dtype=torch.float32, device=device
    )
else:
    weights_vec = torch.ones(len(models_list), dtype=torch.float32, device=device)
    weights_vec = weights_vec / weights_vec.sum()

print(f"Ensemble weights: {weights_vec.detach().cpu().numpy()}")



## === cell 8
test_dataset = BlindnessDataset(
    test_csv_file, test_root_dir, transform=transform_eval, test=True, return_id=True
)

dl_kwargs = _dl_kwargs_for_env()
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    drop_last=False,
    **{k: v for k, v in dl_kwargs.items() if v is not None},
)



## === cell 9
all_outputs = []
all_ids = []

with torch.inference_mode():
    for images, ids in tqdm(test_loader, desc="Infer test"):
        all_ids.extend(list(ids))
        images = images.to(device, non_blocking=True)

        weighted_prob = None
        for mi, model in enumerate(models_list):
            logits = model(images)
            prob = nn.functional.softmax(logits, dim=1)
            wprob = weights_vec[mi] * prob
            weighted_prob = wprob if weighted_prob is None else (weighted_prob + wprob)

        all_outputs.append(weighted_prob.detach().cpu().numpy())

all_outputs = np.concatenate(all_outputs, axis=0)
final_predictions = np.argmax(all_outputs, axis=1).astype(int)



## === cell 10
test_df = pd.read_csv(test_csv_file)
test_ids = test_df["id_code"].astype(str).to_numpy()

all_ids_arr = np.asarray(all_ids, dtype=str)
if all_ids_arr.shape[0] != test_ids.shape[0] or not np.array_equal(
    all_ids_arr, test_ids
):
    id_to_pred = {i: int(p) for i, p in zip(all_ids_arr, final_predictions)}
    ordered_preds = pd.Series(test_ids).map(id_to_pred).to_numpy()
    assert not pd.isna(ordered_preds).any(), "Some test ids were missing predictions"
    ordered_preds = ordered_preds.astype(int)
else:
    ordered_preds = final_predictions

submission_df = pd.DataFrame(
    {"id_code": test_ids, "diagnosis": ordered_preds.astype(int)}
)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(
    f"Wrote {submission_path} with shape {submission_df.shape} and columns {list(submission_df.columns)}"
)
print(submission_df.head())
print(submission_df["diagnosis"].value_counts().sort_index())
