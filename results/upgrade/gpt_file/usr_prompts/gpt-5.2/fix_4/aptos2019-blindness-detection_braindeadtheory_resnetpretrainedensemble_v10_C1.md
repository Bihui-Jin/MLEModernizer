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

0.8883768355086908

# 6. Current score

0.02443

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.02982) has done: 'I fix the model loading crash by preventing ImageNet weights from overwriting the final `fc` layer when the checkpoint is missing; this preserves your architecture and lets inference run. I also make the dataloader more robust in Kaggle by using a safe `num_workers` default and ensure `out` tensors are concatenated correctly (the current `stack(list_of_batches)` pattern can mis-shape). Finally, I keep the same argmax-over-ensemble-logits prediction logic but make sure ID ordering is validated and the submission is written with the required columns to `submission.csv`.'
- What this solution (achieved 0.02443) has done: 'Your current score is low because the “fallback” path is being triggered (the `/kaggle/input/resnet/*.pt` checkpoints aren’t present), so you are ensembling essentially untrained random classification heads on ImageNet backbones. The smallest change that moves you toward the target is to ensure the exact same architecture can actually load weights by (1) auto-discovering the checkpoint files if they exist anywhere under `/kaggle/input`, and (2) making the load tolerant to common checkpoint formats (`state_dict`, `model_state_dict`, `module.` prefixes). This preserves your model architecture and argmax-over-ensemble-logits semantics, but makes it much more likely you’re using real trained weights instead of random heads. If no checkpoints exist, the code still produces a valid `submission.csv` exactly as before.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))



## === cell 1
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm



## === cell 2
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

    def __len__(self):
        return len(self.eye_frame)

    def __getitem__(self, idx):
        row = self.eye_frame.iloc[idx]
        img_id = row["id_code"]

        if self.filetype == "train":
            img_name = os.path.join(
                "../input/aptos2019-blindness-detection/train_images",
                img_id + ".png",
            )
            image = Image.open(img_name).convert("RGB")
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
            return image, int(row["diagnosis"])
        else:
            img_name = os.path.join(
                "../input/aptos2019-blindness-detection/test_images",
                img_id + ".png",
            )
            image = Image.open(img_name).convert("RGB")
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
            return image, img_id




## === cell 3
test_csv_path = "../input/aptos2019-blindness-detection/test.csv"
test_dataset = APTOSDataset(
    csv_file=test_csv_path, filetype="test", transform=transform
)

num_workers = 4
try:
    cpu_cnt = os.cpu_count() or 2
    num_workers = min(4, max(0, cpu_cnt - 1))
except Exception:
    num_workers = 0

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=24,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device, "| num_workers:", num_workers)




## === cell 4
def _find_checkpoint_path(preferred_path: str, basename: str):
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path

    for root, _, files in os.walk("/kaggle/input"):
        if basename in files:
            return os.path.join(root, basename)
    return preferred_path  # keep original for warning messages


def _extract_state_dict(loaded_obj):
    if isinstance(loaded_obj, dict):
        if "state_dict" in loaded_obj and isinstance(loaded_obj["state_dict"], dict):
            return loaded_obj["state_dict"]
        if "model_state_dict" in loaded_obj and isinstance(
            loaded_obj["model_state_dict"], dict
        ):
            return loaded_obj["model_state_dict"]
    return loaded_obj


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _load_resnet_with_optional_ckpt(arch: str, ckpt_path: str, num_classes: int = 5):
    if arch == "resnet152":
        model = torchvision.models.resnet152(weights=None)
        base = torchvision.models.resnet152(
            weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V1
        )
    elif arch == "resnet101":
        model = torchvision.models.resnet101(weights=None)
        base = torchvision.models.resnet101(
            weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V1
        )
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    ckpt_path_resolved = ckpt_path
    if ckpt_path is not None:
        ckpt_path_resolved = _find_checkpoint_path(
            ckpt_path, os.path.basename(ckpt_path)
        )

    if ckpt_path_resolved and os.path.exists(ckpt_path_resolved):
        state = torch.load(ckpt_path_resolved, map_location="cpu")
        state = _extract_state_dict(state)
        state = _strip_module_prefix(state)

        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {ckpt_path_resolved}")
        if len(missing) or len(unexpected):
            print(
                f"Checkpoint load (non-strict) | missing: {len(missing)} unexpected: {len(unexpected)}"
            )
    else:
        base_state = base.state_dict()
        base_state.pop("fc.weight", None)
        base_state.pop("fc.bias", None)
        missing, unexpected = model.load_state_dict(base_state, strict=False)
        print(
            f"WARNING: Checkpoint not found: {ckpt_path} (resolved: {ckpt_path_resolved})"
        )
        print(
            f"Falling back to ImageNet weights for {arch} (fc head randomly initialized). "
            f"Missing keys: {len(missing)}, unexpected: {len(unexpected)}"
        )

    return model.to(device)


model0 = _load_resnet_with_optional_ckpt(
    "resnet152", "../input/resnet/FinalResnet152_0.pt"
)
model1 = _load_resnet_with_optional_ckpt(
    "resnet101", "../input/resnet/FinalResnet02.pt"
)
model2 = _load_resnet_with_optional_ckpt(
    "resnet101", "../input/resnet/FinalResnet01.pt"
)




## === cell 5
def compute_predictions(model, model_type, data_loader, device):
    if model_type == "train":
        predictions = []
        correct_pred, num_examples = 0, 0
        for inputs, labels in tqdm(data_loader, desc="Predict(train)"):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.append(preds.detach().cpu())
            num_examples += labels.size(0)
            correct_pred += (preds == labels).sum()
        return predictions, correct_pred.item() / num_examples * 100.0
    else:
        predictions = []
        img_ids = []
        out_batches = []
        for inputs, img_id in tqdm(data_loader, desc="Predict(test)"):
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.detach().cpu().tolist())
            img_ids.extend(list(img_id))
            out_batches.append(outputs.detach().cpu())
        final_predictions_df = pd.DataFrame(
            {"id_code": img_ids, "diagnosis": predictions}
        )
        return final_predictions_df, out_batches, img_ids




## === cell 6
with torch.no_grad():
    model0.eval()
    model1.eval()
    model2.eval()

    print("Computing Test Predictions")
    test_predictions0, out0_batches, ids0 = compute_predictions(
        model0, "test", test_loader, device
    )
    test_predictions1, out1_batches, ids1 = compute_predictions(
        model1, "test", test_loader, device
    )
    test_predictions2, out2_batches, ids2 = compute_predictions(
        model2, "test", test_loader, device
    )



## === cell 7
if not (ids0 == ids1 == ids2):
    raise RuntimeError(
        "Model prediction ID orders do not match; cannot ensemble safely."
    )

out0 = torch.cat(out0_batches, dim=0)
out1 = torch.cat(out1_batches, dim=0)
out2 = torch.cat(out2_batches, dim=0)

out = (out0 + out1 + out2) / 3.0
img_ids = np.array(ids0)
_, predictions = torch.max(out, 1)
predictions = predictions.numpy().astype(int)

final_predictions = pd.DataFrame({"id_code": img_ids, "diagnosis": predictions})

sample_path = "../input/aptos2019-blindness-detection/sample_submission.csv"
sample = pd.read_csv(sample_path)

final_predictions = sample[["id_code"]].merge(
    final_predictions, on="id_code", how="left"
)
final_predictions["diagnosis"] = final_predictions["diagnosis"].fillna(0).astype(int)

final_predictions.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", final_predictions.shape)
print(final_predictions.head())
print(
    "diagnosis value counts:\n",
    final_predictions["diagnosis"].value_counts().sort_index(),
)
