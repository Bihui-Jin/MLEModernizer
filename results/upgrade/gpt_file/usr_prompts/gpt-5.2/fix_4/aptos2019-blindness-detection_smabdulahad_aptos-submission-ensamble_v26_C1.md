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

0.5442382715616201

# 6. Current score

0.39661

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.03012) has done: 'The runtime failure is caused by hardcoded model checkpoint paths that don’t exist in this Kaggle environment, which leaves the ensemble empty and cascades into later errors. I add a small, safe fallback: if those checkpoints aren’t found, load standard ImageNet-pretrained timm models instead (same architectures, still 5-class head) so inference runs end-to-end and produces a valid `submission.csv`. I also fix image loading to always convert to RGB, add `map_location` for checkpoint loading, and make the ensemble loop robust to missing models (skip missing ones and renormalize weights). These changes preserve the core ensemble/inference logic while ensuring the notebook always writes a valid submission.'
- What this solution (achieved -0.10038) has done: 'Your current pipeline is doing plain argmax over 5-class softmax, which is usually poorly aligned with Quadratic Weighted Kappa (ordinal metric) and can easily yield negative kappa if the class distribution is badly skewed. I keep the same ensemble+inference core logic, but change only the prediction post-processing to an ordinal “expected value” regression-style score (sum of class prob * class index) followed by simple, fixed thresholds to map back to classes 0–4. To keep changes minimal and stable, the thresholds be derived from the training label distribution quantiles (no extra training), which typically moves predictions away from “all one class” collapse and improves kappa substantially. The code still run end-to-end and write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.39661) has done: 'Your current negative kappa is most likely driven by a mismatch between the model’s probability calibration and the fixed post-processing: you’re setting thresholds using **test-set severity quantiles**, which is unstable and can easily collapse predictions in an ordinal metric. I keep the exact same ensemble/inference logic, but change only the thresholding step to use **train-based severity cutpoints** computed by running the same ensemble over the training set (no extra training, same model forward pass). This aligns the ordinal mapping with the training label distribution without “peeking” at test predictions for calibration, which should move the score upward toward your target. I also make DataLoader settings deterministic/safe (no behavior change to model) and ensure submission row alignment stays correct.'

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

test_loader = DataLoader(
    test_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True
)



## === cell 4
model_paths = {
    "resnet18": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/1/resnet18(WD_1e-3)_aptos.pth",
    "efficientnet_b5": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/1/efficientnet_b5.pth",
    "inception_resnet_v2": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/1/inception_resnet_v2.pth",
    "inception_v4": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/1/inception_v4.pth",
    "seresnext50_32x4d": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/1/seresnext50_32x4d.pth",
    "seresnext101_32x4d": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/1/seresnext101_32x4d.pth",
}

model_names = {
    "resnet18": "resnet18",
    "efficientnet_b5": "efficientnet_b5",
    "inception_resnet_v2": "inception_resnet_v2",
    "inception_v4": "inception_v4",
    "seresnext50_32x4d": "seresnext50_32x4d",
    "seresnext101_32x4d": "seresnext101_32x4d",
}



## === cell 5
validation_scores = {
    "resnet18": 0.887,
    "efficientnet_b5": 0.952,
    "inception_resnet_v2": 0.822,
    "inception_v4": 0.888,
    "seresnext50_32x4d": 0.709,
    "seresnext101_32x4d": 0.951,
}



## === cell 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models_list = []
active_model_keys = []

for model_key, ckpt_path in model_paths.items():
    model_name = model_names[model_key]
    has_ckpt = os.path.exists(ckpt_path)

    model = timm.create_model(model_name, pretrained=not has_ckpt, num_classes=5)

    if has_ckpt:
        state = torch.load(ckpt_path, map_location="cpu")
        if (
            isinstance(state, dict)
            and "state_dict" in state
            and isinstance(state["state_dict"], dict)
        ):
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)

    model.to(device)
    model.eval()

    models_list.append(model)
    active_model_keys.append(model_key)

if len(models_list) == 0:
    raise RuntimeError(
        "No models could be created/loaded. Check dataset availability and timm model names."
    )



## === cell 7
active_scores = {k: validation_scores[k] for k in active_model_keys}
total_score = sum(active_scores.values())
weights = {k: (v / total_score) for k, v in active_scores.items()}




## === cell 8
def predict_expected_severity(loader: DataLoader) -> np.ndarray:
    all_outputs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Infer"):
            if isinstance(batch, (tuple, list)):
                images = batch[0]
            else:
                images = batch

            images = images.to(device, non_blocking=True)

            outputs = []
            for model_key, model in zip(active_model_keys, models_list):
                logits = model(images)
                probs = nn.functional.softmax(logits, dim=1)
                outputs.append(weights[model_key] * probs)

            weighted_outputs = torch.stack(outputs, dim=0).sum(dim=0)
            all_outputs.append(weighted_outputs.cpu().numpy())

    all_outputs = np.concatenate(all_outputs, axis=0)
    class_values = np.arange(5, dtype=np.float32)
    severity = (all_outputs * class_values[None, :]).sum(axis=1)  # in [0,4]
    return severity




## === cell 9
train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
train_root_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"

train_df = pd.read_csv(train_csv_file)
y_train = train_df["diagnosis"].astype(int).values

train_dataset = BlindnessDataset(
    train_csv_file, train_root_dir, transform=transform, test=False
)
train_loader = DataLoader(
    train_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True
)

train_severity = predict_expected_severity(train_loader)

counts = np.bincount(y_train, minlength=5).astype(np.float64)
probs = counts / counts.sum()
cum = np.cumsum(probs)  # cum[0]=P(y<=0), ..., cum[3]=P(y<=3)

q = np.clip(cum[:4], 1e-6, 1 - 1e-6)
thresholds = np.quantile(train_severity, q)
thresholds = np.maximum.accumulate(thresholds)  # enforce monotonicity



## === cell 10
test_severity = predict_expected_severity(test_loader)

final_predictions = np.digitize(test_severity, thresholds, right=False).astype(int)
final_predictions = np.clip(final_predictions, 0, 4)



## === cell 11
test_ids = pd.read_csv(test_csv_file)["id_code"].values
if len(test_ids) != len(final_predictions):
    raise RuntimeError(
        f"Prediction length mismatch: {len(final_predictions)} vs {len(test_ids)}"
    )

submission_df = pd.DataFrame({"id_code": test_ids, "diagnosis": final_predictions})
submission_df.to_csv("submission.csv", index=False)

print(submission_df.head())
print("Thresholds used:", thresholds)
print("Pred distribution:", np.bincount(final_predictions, minlength=5))
print("Wrote submission.csv with shape:", submission_df.shape)
