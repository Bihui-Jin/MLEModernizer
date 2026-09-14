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

3.8

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

0.8948393061819762

# 6. Current score

-0.06777

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.07031) has done: 'I fix the immediate runtime blocker: the script tries to load model weight files from `../input/resnet*` paths that don’t exist in your environment, which prevents any submission from being written. To preserve the ensemble/core inference logic while making it run end-to-end, I add a safe fallback that uses ImageNet-pretrained ResNet weights when the competition-specific `.pt` files are missing, and I make `torch.load` device-safe via `map_location`. I also replace the notebook-only `tqdm_notebook` import with a standard `tqdm` that works in Kaggle script runs, and ensure the submission `diagnosis` column is an integer and written to `submission.csv`. These changes are execution/stability focused (score is currently “Not yielded”), and keep the same model architectures and ensemble averaging logic.'
- What this solution (achieved -0.06777) has done: 'I remove the hard failure on missing external `.pt` weight files and replace it with a safe fallback that still instantiates the exact same ResNet101/152 architectures and runs inference end-to-end. The fallback uses torchvision ImageNet weights (when available) or random init if weights cannot be fetched, so the pipeline always produces `submission.csv` in the required format. I also make the DataLoader settings robust in Kaggle by disabling `persistent_workers/prefetch_factor` when `num_workers=0`, avoiding runtime crashes in restricted environments. Finally, I keep the ensemble averaging logic identical and ensure outputs are aligned to `test.csv` order and written as integer `diagnosis`.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

Image.MAX_IMAGE_PIXELS = None
try:
    from PIL import ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
except Exception:
    pass



## === cell 1
transform = transforms.Compose(
    [
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.460, 0.247, 0.080], [0.249, 0.138, 0.081]),
    ]
)


class APTOSDataset(Dataset):
    """Eye images dataset."""

    def __init__(self, csv_file, filetype, transform=None):
        self.eye_frame = pd.read_csv(csv_file)
        self.filetype = filetype
        self.transform = transform

        self._ids = self.eye_frame["id_code"].to_numpy()
        if self.filetype == "train":
            self._labels = self.eye_frame["diagnosis"].to_numpy(dtype=np.int64)
        else:
            self._labels = None

    def __len__(self):
        return len(self.eye_frame)

    def __getitem__(self, idx):
        if self.filetype == "train":
            img_name = os.path.join(
                "../input/aptos2019-blindness-detection/train_images",
                self._ids[idx] + ".png",
            )
            image = Image.open(img_name).convert("RGB")
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
            return image, int(self._labels[idx])
        else:
            img_name = os.path.join(
                "../input/aptos2019-blindness-detection/test_images",
                self._ids[idx] + ".png",
            )
            image = Image.open(img_name).convert("RGB")
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
            return image, self._ids[idx]




## === cell 2
train_dataset = APTOSDataset(
    csv_file="../input/aptos2019-blindness-detection/train.csv",
    filetype="train",
    transform=transform,
)

test_dataset = APTOSDataset(
    csv_file="../input/aptos2019-blindness-detection/test.csv",
    filetype="test",
    transform=transform,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

_num_workers = 4
_common_loader_kwargs = dict(
    num_workers=_num_workers,
    pin_memory=torch.cuda.is_available(),
)
if _num_workers > 0:
    _common_loader_kwargs.update(dict(persistent_workers=True, prefetch_factor=4))

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=12,
    shuffle=True,
    **_common_loader_kwargs,
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=24,
    shuffle=False,
    **_common_loader_kwargs,
)




## === cell 3
def _get_imagenet_weights(model_name: str):
    try:
        if model_name == "resnet152":
            return torchvision.models.ResNet152_Weights.DEFAULT
        if model_name == "resnet101":
            return torchvision.models.ResNet101_Weights.DEFAULT
    except Exception:
        return None
    return None


def build_resnet(model_name: str, weights_path: str, num_classes: int = 5):
    """
    Bugfix: original code hard-failed when external competition weight files were missing.
    Minimal change: keep the same architectures and FC head, but fall back to ImageNet weights
    (or random init) so inference can run end-to-end and write submission.csv.
    """
    imagenet_weights = _get_imagenet_weights(model_name)

    if model_name == "resnet152":
        model = torchvision.models.resnet152(weights=imagenet_weights)
    elif model_name == "resnet101":
        model = torchvision.models.resnet101(weights=imagenet_weights)
    else:
        raise ValueError(f"Unsupported model_name={model_name}")

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    if weights_path is not None and os.path.exists(weights_path):
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state)
        print(f"Loaded competition weights: {weights_path}")
    else:
        print(
            f"WARNING: Weights not found at '{weights_path}'. "
            f"Falling back to torchvision ImageNet weights ({imagenet_weights is not None}) "
            "with a fresh 5-class head."
        )

    model = model.to(device)
    return model


model0 = build_resnet("resnet152", "../input/resnet/FinalResnet152_0.pt")
model1 = build_resnet("resnet101", "../input/resnet/FinalResnet02.pt")
model2 = build_resnet("resnet101", "../input/resnet/FinalResnet01.pt")
model3 = build_resnet("resnet101", "../input/resnet/FinalResnet00.pt")
model4 = build_resnet("resnet152", "../input/resnet0/pretrainedResnet151_0.pt")
model5 = build_resnet("resnet101", "../input/resnet0/pretrainedResnet1.pt")



## === cell 4
from tqdm.auto import tqdm


def compute_predictions(model, model_type, data_loader, device):
    use_amp = device.type == "cuda"
    if model_type == "train":
        predictions = []
        correct_pred, num_examples = 0, 0
        model.eval()
        with torch.inference_mode():
            for inputs, labels in tqdm(data_loader, desc="Predict(train)"):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                predictions.append(preds)
                num_examples += labels.size(0)
                correct_pred += (preds == labels).sum()
        return predictions, correct_pred.item() / num_examples * 100
    else:
        predictions = []
        img_ids = []
        out = []
        model.eval()
        with torch.inference_mode():
            for inputs, img_id in tqdm(data_loader, desc="Predict(test)"):
                inputs = inputs.to(device, non_blocking=True)
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                predictions.extend(preds.detach().cpu())
                img_ids.extend(list(img_id))
                out.extend(outputs.detach().cpu())
        predictions = [int(pred.item()) for pred in predictions]
        final_predictions = pd.DataFrame({"id_code": img_ids, "diagnosis": predictions})
        return final_predictions, out, img_ids




## === cell 5
def fine_tune_if_needed(model, local_ckpt_path: str, epochs: int = 1, lr: float = 1e-4):
    if os.path.exists(local_ckpt_path):
        state = torch.load(local_ckpt_path, map_location="cpu")
        model.load_state_dict(state)
        print(f"Loaded local fine-tuned weights: {local_ckpt_path}")
        return model

    print(
        f"Skipping fine-tuning because '{local_ckpt_path}' does not exist. "
        "Proceeding with current weights for inference."
    )
    model.eval()
    return model


model0 = fine_tune_if_needed(model0, "ft_resnet152_m0.pt", epochs=1, lr=1e-4)
model1 = fine_tune_if_needed(model1, "ft_resnet101_m1.pt", epochs=1, lr=1e-4)
model2 = fine_tune_if_needed(model2, "ft_resnet101_m2.pt", epochs=1, lr=1e-4)
model3 = fine_tune_if_needed(model3, "ft_resnet101_m3.pt", epochs=1, lr=1e-4)
model4 = fine_tune_if_needed(model4, "ft_resnet152_m4.pt", epochs=1, lr=1e-4)
model5 = fine_tune_if_needed(model5, "ft_resnet101_m5.pt", epochs=1, lr=1e-4)



## === cell 6
print("Computing Test Predictions")
test_predictions0, out0, ids0 = compute_predictions(model0, "test", test_loader, device)
test_predictions1, out1, ids1 = compute_predictions(model1, "test", test_loader, device)
test_predictions2, out2, ids2 = compute_predictions(model2, "test", test_loader, device)
test_predictions3, out3, ids3 = compute_predictions(model3, "test", test_loader, device)
test_predictions4, out4, ids4 = compute_predictions(model4, "test", test_loader, device)
test_predictions5, out5, ids5 = compute_predictions(model5, "test", test_loader, device)

out = (
    torch.stack(out0)
    + torch.stack(out1)
    + torch.stack(out2)
    + torch.stack(out3)
    + torch.stack(out4)
    + torch.stack(out5)
) / 6.0

img_ids = np.array(ids0)
_, predictions = torch.max(out, 1)
predictions = predictions.numpy().astype(int)

final_predictions = pd.DataFrame(
    {"id_code": img_ids, "diagnosis": predictions.astype(int)}
)

test_ids = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")[
    "id_code"
].astype(str)
final_predictions["id_code"] = final_predictions["id_code"].astype(str)
final_predictions = test_ids.to_frame().merge(
    final_predictions, on="id_code", how="left"
)
final_predictions["diagnosis"] = final_predictions["diagnosis"].fillna(0).astype(int)

final_predictions.to_csv("submission.csv", index=False)

print(final_predictions.head())
print("Wrote submission.csv with shape:", final_predictions.shape)
print(
    "diagnosis value counts:\n",
    final_predictions["diagnosis"].value_counts().sort_index(),
)
