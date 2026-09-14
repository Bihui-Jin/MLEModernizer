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

0.8486945958951383

# 6. Current score

0.29885

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.12364) has done: 'I fix the immediate runtime blocker by removing the dependency on missing external pretrained weights and instead load each model with `pretrained=True` from `timm`, keeping the same ensemble + softmax-weighted averaging logic. I also make the dataset image loading robust (force RGB and close files) to avoid PIL mode issues, and fix the empty-model-list case that caused `torch.cat()` to fail. Finally, I ensure the submission is created with the required columns and correct length, and written to `submission.csv` in the working directory.'
- What this solution (achieved 0.17746) has done: 'Your score is far below the target, and the main issue is that you’re using ImageNet-pretrained classifiers with random (untrained) 5-class heads (`num_classes=5`), so predictions are essentially noise for this task. To move the score toward the target without changing the core “ensemble + softmax weighted averaging” inference logic, the minimal effective fix is to use each model’s pretrained ImageNet head (keep `num_classes` default) and map the 1000-class probabilities down to 5 DR classes via a fixed, deterministic binning of the expected severity (computed from class index). This keeps the same inference semantics (softmax → weighted average → argmax), but makes the base probabilities non-random and usually much more structured than an untrained head. I also keep the current weights logic, but make it robust if a key is missing, and ensure the submission remains aligned and valid.'
- What this solution (achieved 0.18972) has done: 'Your current low score is mainly driven by the ImageNet→5-class “hard binning” (one-hot) mapping, which destroys most ranking information that quadratic weighted kappa benefits from. I keep your exact ensemble + softmax-weighted averaging core logic, but change the mapping to a deterministic *soft* 5-bin aggregation over the 1000 ImageNet probabilities (summing probability mass into 5 contiguous bins) so the output remains a true probability distribution. Then I convert the final 5-class probabilities into an ordinal prediction using the expected severity (and rounding), which is a minimal post-processing aligned with the ordinal nature of the metric. These are small, inference-only changes that should move the score up toward your target without changing the overall approach.'
- What this solution (achieved 0.18605) has done: 'Your current approach is bottlenecked by an arbitrary ImageNet→5-class binning that doesn’t correspond to DR severity, so even with soft bins the model outputs remain largely misaligned with the ordinal target, keeping QWK low. To move the score upward toward 0.848 with minimal change and without altering your ensemble/inference structure, I keep the same “softmax → per-model weighting → average” logic but calibrate the final discrete prediction using tuned ordinal thresholds (a standard, metric-aligned post-processing for QWK). I also (safely) compute thresholds from your provided per-model validation scores to avoid hardcoding random values, and keep a fallback default if anything is missing. This keeps everything inference-only, fast, and produces the same required `submission.csv`.'
- What this solution (achieved 0.29885) has done: 'Your current gap to the target is large, and the biggest lever that doesn’t change your core “pretrained timm ensemble + softmax-weighted averaging” logic is to replace the arbitrary ImageNet→5 mapping with a deterministic ordinal mapping based on each ImageNet class index. Concretely, instead of summing contiguous 200-class bins, we compute an expected ImageNet class index (0–999) from the 1000-way probabilities and then map that scalar to DR labels using fixed quantile thresholds; this preserves more rank information, which QWK rewards. I keep your per-model weighting and your thresholding-to-label mechanism, but compute the thresholds from the 1000-class expectation quantiles (with a safe fallback), and ensure the submission remains aligned and valid. Changes are inference-only, fast, and keep the same overall semantics (softmax → weighted average → ordinal discretization).'

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
        img_id = self.annotations.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, f"{img_id}.png")

        with Image.open(img_name) as im:
            image = im.convert("RGB")

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
model_paths = {"resnet18": None, "seresnext50_32x4d": None, "seresnext101_32x4d": None}

model_names = {
    "resnet18": "resnet18",
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

for model_key in model_paths.keys():
    model_name = model_names[model_key]

    model = timm.create_model(model_name, pretrained=True)  # keep pretrained head
    model.to(device)
    model.eval()
    models_list.append(model)
    loaded_model_keys.append(model_key)

if len(models_list) == 0:
    raise RuntimeError("No models were loaded; cannot run inference.")



## === cell 6
validation_scores = {
    "resnet18": 0.887,
    "seresnext50_32x4d": 0.777,  # 0.709,
    "seresnext101_32x4d": 0.9697,  # 0.951
}

validation_scores = {
    k: v for k, v in validation_scores.items() if k in loaded_model_keys
}

if len(validation_scores) == 0:
    weights = {k: 1.0 / len(loaded_model_keys) for k in loaded_model_keys}
else:
    total_score = sum(validation_scores.values())
    weights = {k: v / total_score for k, v in validation_scores.items()}



## === cell 7
_imagenet_idx = torch.arange(1000, dtype=torch.float32, device=device)[None, :]


def imagenet_probs_to_expected_index(probs_1000: torch.Tensor) -> torch.Tensor:
    b, c = probs_1000.shape
    if c != 1000:
        raise RuntimeError(
            f"Expected 1000-class logits/probs, got shape {probs_1000.shape}"
        )
    return (probs_1000 * _imagenet_idx).sum(dim=1)  # [B], in [0, 999]


def severity_to_label_with_thresholds(
    sev: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    t0, t1, t2, t3 = thresholds.tolist()
    return np.digitize(sev, bins=[t0, t1, t2, t3]).astype(np.int64)


def compute_thresholds_from_test_quantiles(sev: np.ndarray) -> np.ndarray:
    if sev is None or len(sev) == 0:
        return np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)

    qs = np.quantile(sev.astype(np.float32), [0.2, 0.4, 0.6, 0.8]).astype(np.float32)

    qs = np.maximum.accumulate(qs + np.array([0.0, 1e-3, 2e-3, 3e-3], dtype=np.float32))

    qs = np.clip(qs, 0.0, 999.0)
    return qs


all_sev = []

with torch.no_grad():
    for images in tqdm(test_loader):
        images = images.to(device, non_blocking=True)

        sevs = []
        for model_key, model in zip(loaded_model_keys, models_list):
            logits_1000 = model(images)  # [B, 1000]
            probs_1000 = nn.functional.softmax(logits_1000, dim=1)
            sev = imagenet_probs_to_expected_index(probs_1000)  # [B]
            w = weights.get(model_key, 1.0 / len(loaded_model_keys))
            sevs.append(w * sev)

        if len(sevs) == 0:
            raise RuntimeError("No model outputs produced for this batch.")

        sev_ens = torch.stack(sevs, dim=0).sum(dim=0)  # [B]
        all_sev.append(sev_ens.cpu().numpy())

severity = np.concatenate(all_sev, axis=0)  # [N]
thresholds = compute_thresholds_from_test_quantiles(severity)

final_predictions = (
    severity_to_label_with_thresholds(severity, thresholds).clip(0, 4).astype(int)
)



## === cell 8
test_ids = pd.read_csv(test_csv_file)["id_code"].values
if len(final_predictions) != len(test_ids):
    raise RuntimeError(
        f"Prediction length mismatch: got {len(final_predictions)} preds for {len(test_ids)} ids"
    )

submission_df = pd.DataFrame({"id_code": test_ids, "diagnosis": final_predictions})
submission_df.to_csv("submission.csv", index=False)

print(submission_df.head())
print("Thresholds used:", thresholds)
print("Wrote submission.csv with shape:", submission_df.shape)
