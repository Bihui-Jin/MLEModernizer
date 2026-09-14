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

3.10

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

0.6345175820343185

# 6. Current score

0.00981

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the unsupported `imgaug` dependency (it isn’t installed), and replace its usage with an equivalent minimal Albumentations pipeline so the code runs. I also fix the dataset `__getitem__` logic (Albumentations returns a dict; reshaping was incorrect and channel order wasn’t enforced), and ensure images are converted to proper float tensors in CHW format. Finally, I make the code robust to Kaggle path differences by using the provided `/kaggle/input/...` structure and add a safe fallback if the external weights file is missing, so a valid `submission.csv` is always produced.'
- What this solution (achieved -0.22648) has done: 'Your 0.0 score is consistent with the model effectively being random because the external weights file path likely doesn’t exist in your environment, so you’re submitting predictions from a randomly initialized ResNet18. The smallest change to move toward your target is to (1) correctly use the dataset’s `transform` argument (it’s currently ignored and always uses the global `train_transform`), and (2) load ImageNet pretrained weights when the competition-specific checkpoint is missing, so the submission becomes meaningfully predictive without changing architecture or training. I also fix the prediction head to output logits (remove `Softmax`) to match standard ResNet inference semantics; `argmax(softmax(logits))` is identical to `argmax(logits)`, so this is score-safe and avoids unnecessary numerical effects. These changes keep your pipeline inference-only and produce the same submission format.'
- What this solution (achieved -0.13665) has done: 'Your current negative kappa is consistent with a severe train/test preprocessing mismatch: you apply a strong brightness adjustment (+ sharpening/noise in the original pipeline) but you’re not using the same normalization that the ImageNet-pretrained ResNet18 expects, so the backbone features are effectively off-distribution. To move the score upward toward your target with minimal change and without altering the model/training approach, I (1) add a standard ImageNet normalization transform for test-time only (no augmentations), and (2) fix the dataset transform application so Albumentations is used correctly and not followed by a second manual transpose/scale that would double-handle the data. I also keep your existing weight-loading behavior, submission format, and inference loop intact, only adjusting the input preprocessing to be consistent and meaningful for the pretrained backbone.'
- What this solution (achieved -0.07183) has done: 'Your current negative kappa likely comes from a train/test mismatch caused by the hard-coded `adjust_brightness(image, 2.0)`, which shifts test images far off the ImageNet-pretrained ResNet18 distribution; removing that single step should move predictions toward sensible class ordering without changing the model or training approach. I also make `train_transform` safe (it currently lacks `ToTensorV2`/normalization and would break if you ever use it) while keeping it unused for your inference-only pipeline. Finally, I add a tiny safety clamp to ensure the submission contains valid integer labels in `[0,4]` and keep the exact submission format and paths unchanged.'
- What this solution (achieved -0.05377) has done: 'Your score is far below the target (gap ≈ -0.706), so we should improve it with the smallest changes that don’t alter your core “ResNet18 + linear head, inference-only” logic. The biggest issue is that when the competition checkpoint is missing you use an ImageNet backbone but a random (untrained) 5-class head, which makes predictions close to random and hurts kappa; we instead use the ImageNet-pretrained *classification* head to produce meaningful ordinal-ish predictions by mapping ImageNet logits to 5 bins (still inference-only, no new training). To keep evaluation semantics aligned with the competition, we also center-crop after resizing (common for ImageNet models) while keeping your normalization and tensor conversion unchanged. These changes are minimal, deterministic, and should move kappa upward toward the target band without changing your architecture/training loop when the competition weights are present.'
- What this solution (achieved -0.02834) has done: 'Your current score is far below the target, so we should improve it with the smallest change that makes predictions meaningfully correlated with DR severity. The main issue is the fallback path: mapping ImageNet logits to 5 classes via “top-5 mean + fixed bins” is essentially arbitrary and yields near-random ordinal labels, producing negative kappa. I keep your exact model/inference-only setup and preprocessing, but change only the fallback post-processing to a simple luminance-based ordinal rule (computed on the same already-loaded tensors) that is far more likely to correlate with DR severity than the current binning. This keeps runtime low, produces a valid submission, and should move kappa upward toward your target band without changing architecture/training logic.'
- What this solution (achieved 0.01134) has done: 'Your score is far below the target, and the current fallback (when the DR checkpoint is missing) ignores the model entirely and uses a luminance heuristic, which is weakly related to DR severity and can easily give negative kappa. I keep your exact ResNet18 inference-only setup and preprocessing, but change only the fallback path to use the ImageNet-pretrained model’s own logits in a deterministic way (no extra training) by converting them into a stable “severity index” and then mapping that to 5 classes by matching the training label distribution (a common minimal trick for kappa/ordinal tasks). This preserves core logic (single forward pass, argmax for the real checkpoint path) and only affects the fallback, which is the part currently causing near-random predictions. The submission format/paths remain unchanged and it still always write a valid `submission.csv`.'
- What this solution (achieved 0.00981) has done: 'Your current score (0.01134) is far below the target (0.6345), and the main reason is that the fallback path (when the DR checkpoint is missing) uses an arbitrary “ImageNet class index severity” that has little correlation with DR grade. Keeping your core inference-only ResNet18 logic intact, I change only the fallback to a more DR-relevant severity score computed from the model’s own features (global-average pooled penultimate layer), then map those scores to 0–4 using the same train-distribution quantile mapping you already use. This is minimal (no training, no architecture change, still one forward pass per batch) and should move kappa upward substantially versus near-random ordinal mapping. I also make the fallback deterministic and shape-safe while keeping the same paths and submission format.'
- What this solution (achieved 0.11082) has done: 'Your current score is far below the target, and the biggest issue is the fallback path: when the DR checkpoint is missing, the penultimate-feature L2 norm is a weak proxy for DR severity, so predictions remain close to random. I keep your exact ResNet18 inference-only setup, but change only the fallback severity score to something more DR-relevant computed from the same forward features: the mean per-image activation of the final conv feature map (after ReLU), which better correlates with overall lesion/structure signal than feature norm. I also standardize the fallback scores (z-score) before quantile mapping to make the thresholds numerically stable across batches/devices, without changing the quantile-to-label logic. Everything else (paths, transforms, dataloader, submission schema) stays the same and it still always write a valid `submission.csv`.'
- What this solution (achieved -0.08661) has done: 'Your score gap is large (0.11082 vs target 0.6345), and the current pipeline is almost certainly always taking the “no DR checkpoint found” fallback path, where the severity proxy is too weak. I keep the exact ResNet18 inference-only core, but make the fallback severity score more DR-relevant by using a simple “red-channel dominance” statistic computed from the same already-loaded (normalized) tensors, then map it to 0–4 using your existing train-distribution quantile mapping. This is a minimal, fast change (no training, no new files, same dataloader/transforms/submission), and it typically correlates better with fundus brightness/lesion presence than conv activation mean. I also make the score computation explicitly unnormalize to [0,1] RGB before computing the heuristic, to avoid artifacts from ImageNet normalization.'
- What this solution (achieved -0.0361) has done: 'Your score is far below the target, so we should increase it with the smallest change that fixes the main failure mode: you are almost certainly always in the “no DR checkpoint” fallback, and the current color/brightness heuristic is not reliably correlated with DR grade (hence negative kappa). I keep your exact ResNet18 inference-only pipeline, transforms, and quantile mapping, but change the fallback severity score to a more DR-relevant, still-fast statistic computed from the same tensors: the fraction/strength of bright lesions in the green channel (after unnormalizing), which is a classic simple proxy for exudates/bright pathology. I also blend in a mild focus/contrast proxy (Laplacian energy) to reduce random ranking on blurry/underexposed images, while keeping everything deterministic and producing the same submission schema. No training, no new files, same paths, same runtime class.'
- What this solution (achieved 0.00981) has done: 'Your current negative kappa strongly suggests the code is still using the fallback path (no DR checkpoint) and that the hand-crafted green-channel heuristic produces rankings that don’t align with true DR severity. To move the score upward toward your target with minimal change and without changing the ResNet18 architecture/training loop, I keep the same inference-only pipeline but replace the fallback heuristic with a pretrained-ResNet feature-based severity score (penultimate pooled embedding magnitude), which is generally more stable and correlated than color/brightness rules. I also make the fallback robust by explicitly extracting features via `model.avgpool`+`flatten` without modifying the model definition, and keep your existing train-quantile mapping so predictions match the label distribution. Submission writing, paths, transforms, and the checkpoint path behavior remain unchanged.'

# 9. Code solution

## === cell 0
import os
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2 as cv  # kept (original import), though not strictly needed
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image
import torchvision.transforms.functional as F

warnings.filterwarnings("ignore")



## === cell 1
SEED = 8
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)



## === cell 2
BASE = "/kaggle/input/aptos2019-blindness-detection"

test_path = os.path.join(BASE, "test.csv")
test_img_dir = os.path.join(BASE, "test_images")
sample_sub_path = os.path.join(BASE, "sample_submission.csv")
train_path = os.path.join(BASE, "train.csv")

test = pd.read_csv(test_path)
test.head()



## === cell 3
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

test_transform = A.Compose(
    [
        A.Resize(512, 512),
        A.CenterCrop(448, 448),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
        ToTensorV2(),
    ]
)

train_transform = A.Compose(
    [
        A.Resize(512, 512),
        A.CenterCrop(448, 448),
        A.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 1.5), p=1.0),
        A.GaussNoise(std_range=(0.0, 0.05), per_channel=True, p=1.0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
        ToTensorV2(),
    ]
)




## === cell 4
class dataset(Dataset):
    def __init__(self, data_path, img_dir, dt, transform):
        self.data_path = data_path
        self.img_dir = img_dir
        self.data = self.__get_data(self.data_path)
        self.dt = dt
        self.transform = transform

    def __get_data(self, path):
        return pd.read_csv(path)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        img_name = self.data["id_code"].iloc[idx]
        img_fp = os.path.join(self.img_dir, img_name + ".png")

        image = Image.open(img_fp).convert("RGB")
        image = np.array(image)  # HWC, uint8, RGB

        if self.transform is not None:
            out = self.transform(image=image)
            image = out["image"]
            return image

        image = cv.resize(image, (512, 512), interpolation=cv.INTER_LINEAR)
        image = image.transpose(2, 0, 1).astype(np.float32) / 255.0  # CHW, float32
        image = torch.from_numpy(image)
        return image

    def show(self, idx):
        img = self.__getitem__(idx)
        img = img.detach().cpu().numpy()

        if img.shape[0] == 3:
            if (img.min() < 0) or (img.max() > 1.5):
                mean = np.array(IMAGENET_MEAN)[:, None, None]
                std = np.array(IMAGENET_STD)[:, None, None]
                img = (img * std) + mean

        img = (img.transpose(1, 2, 0) * 255.0).clip(0, 255).astype("uint8")
        plt.imshow(img)
        plt.title(self.data["id_code"].iloc[idx])
        plt.axis("off")
        plt.show()




## === cell 5
batch = 64
test_data = dataset(test_path, test_img_dir, dt="test", transform=test_transform)
test_load = DataLoader(
    test_data,
    batch_size=batch,
    shuffle=False,
    num_workers=2,
    pin_memory=(device == "cuda"),
)

xb = next(iter(test_load))
print(len(test_data), xb.shape, xb.dtype)



## === cell 6
train_df = pd.read_csv(train_path)
label_counts = train_df["diagnosis"].value_counts().sort_index()
label_probs = (label_counts / label_counts.sum()).values  # for classes 0..4
label_cum = np.cumsum(label_probs)
print("Train label distribution:", label_counts.to_dict())


def map_scores_to_labels_by_train_quantiles(scores_1d: np.ndarray) -> np.ndarray:
    scores_1d = np.asarray(scores_1d, dtype=np.float64)
    qs = label_cum[:-1]  # 4 cut points
    qs = np.clip(qs, 1e-6, 1 - 1e-6)
    thr = np.quantile(scores_1d, qs, method="linear")
    pred = np.digitize(scores_1d, thr, right=False).astype(np.int64)
    return pred  # already 0..4




## === cell 7
weights_path = "/kaggle/input/aptosmodelweights12/Best_Model_NO_12.pth"

use_dr_checkpoint = os.path.exists(weights_path)

if use_dr_checkpoint:
    model = models.resnet18(weights=None)  # keep as-is when loading DR checkpoint
    model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.Linear(256, 5),
    )
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
else:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    print(
        f"WARNING: weights not found at {weights_path}. Using ImageNet pretrained ResNet18 fallback with a feature-based severity proxy + distribution-matched ordinal mapping."
    )

model.to(device)
model.eval()

predict = []
severity_scores = []  # only used in fallback


@torch.no_grad()
def extract_feature_severity_scores(
    model_: torch.nn.Module, x: torch.Tensor
) -> torch.Tensor:
    """
    Compute a scalar "severity" score per image using the pretrained backbone features.
    We do NOT change the model architecture; we just run the forward up to avgpool.

    Score = log(1 + mean(square(features)))  (smooth energy / magnitude proxy)
    Returns: [B] float tensor.
    """
    x = model_.conv1(x)
    x = model_.bn1(x)
    x = model_.relu(x)
    x = model_.maxpool(x)

    x = model_.layer1(x)
    x = model_.layer2(x)
    x = model_.layer3(x)
    x = model_.layer4(x)

    x = model_.avgpool(x)  # [B, 512, 1, 1]
    x = torch.flatten(x, 1)  # [B, 512]

    score = torch.log1p((x * x).mean(dim=1))
    return score


with torch.no_grad():
    for x in tqdm(test_load):
        x = x.to(device, non_blocking=True)

        if use_dr_checkpoint:
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            pred = pred.to("cpu").numpy().astype(np.int64)
            predict.extend(list(pred))
        else:
            score = extract_feature_severity_scores(model, x)
            severity_scores.append(score.detach().cpu().numpy())

if not use_dr_checkpoint:
    severity_scores = np.concatenate(severity_scores, axis=0).astype(np.float64)

    mu = float(np.mean(severity_scores))
    sigma = float(np.std(severity_scores)) + 1e-12
    severity_scores = (severity_scores - mu) / sigma

    predict = map_scores_to_labels_by_train_quantiles(severity_scores).tolist()



## === cell 8
pd.Series(predict).value_counts()



## === cell 9
sub = pd.read_csv(sample_sub_path)
if len(predict) != len(sub):
    raise ValueError(
        f"Prediction length {len(predict)} does not match submission length {len(sub)}."
    )

predict = np.asarray(predict, dtype=np.int64)
predict = np.clip(predict, 0, 4)

sub["diagnosis"] = predict
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
sub.head()
