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

0.006884960835171

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `imgaug` (not installed) and replace it with a minimal, equivalent-no-op transform so the pipeline runs. I also fix the dataset class to correctly use the passed `transform`, convert images to `float32` CHW tensors, and ensure test loading is deterministic (no shuffle) so predictions align with `sample_submission.csv` ordering. Finally, I make model-weight loading robust: if the external weights path doesn’t exist in your environment, the code fall back to an untrained model so it still produces a valid `submission.csv`. These changes are execution-focused and preserve the core ResNet18 inference logic and submission format.'
- What this solution (achieved 0.0) has done: 'Your score is 0.0 because the script is (likely) producing essentially random predictions due to missing external weights, and it also feeds the ResNet18 with unnormalized BGR 0–255 images that don’t match typical training preprocessing. To move the score upward toward the small target (0.00688), we make minimal inference-only fixes that preserve the same ResNet18 head and argmax logic: apply a deterministic, standard image normalization (BGR→RGB, scale to [0,1], ImageNet mean/std) and remove the terminal Softmax layer (argmax is unchanged, but this avoids extra saturation and better matches common checkpoint training). We also make weight loading slightly more robust (accept common checkpoint dict formats) without changing the architecture. These changes are small, execution-safe, and aimed at nudging predictions away from “pure random” toward a slightly better-than-zero kappa without redesigning the approach.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with producing essentially constant/near-random class predictions when the external checkpoint isn’t available, which makes QWK collapse to ~0. To move upward slightly toward the small target (0.00688) with minimal semantic change, I keep your exact ResNet18+FC head and argmax inference, but (1) load ImageNet pretrained weights as a safe fallback when your custom weights path is missing, and (2) apply the exact same normalization you already implemented to the test dataset (by passing `transform=train_transform` rather than `None`, even though it’s a no-op) to keep behavior consistent/deterministic. This keeps the core logic intact (same model family, same forward pass, same argmax labels) while making predictions non-degenerate and typically above 0.0 QWK. The script still writes a valid `submission.csv` with the required columns and ordering.'
- What this solution (achieved -0.26079) has done: 'Your 0.0 score is consistent with producing nearly constant/degenerate predictions because, when your custom checkpoint is missing, the randomly-initialized FC head dominates the output even if the ResNet18 backbone is ImageNet-pretrained. To move the score upward toward the small target (0.00688) with minimal semantic change, I keep the same ResNet18+FC architecture and argmax inference, but make the ImageNet-fallback path fully consistent by also initializing the FC head in a standard way (Kaiming/Xavier) and using the correct ImageNet input preprocessing (resize + center crop to 224 and torchvision’s official mean/std) so the pretrained backbone features are meaningful. I also ensure deterministic DataLoader worker seeding so test ordering and preprocessing are stable run-to-run without changing evaluation semantics. The script still runs end-to-end and writes a valid `submission.csv` with the required columns and ordering.'
- What this solution (achieved 0.0) has done: 'Your current negative kappa suggests the ImageNet-backbone fallback is producing a highly skewed (effectively “wrong” relative to the test distribution) class output because the randomly-initialized FC stack dominates. To move the score upward toward the small positive target with minimal semantic change, I keep the same ResNet18+FC architecture and the same argmax inference, but make the fallback path less degenerate by (1) using a single Linear(512→5) head so the pretrained backbone features map directly to classes and (2) bias-initializing that head to predict the empirical class prior from `train.csv` (a legitimate, non-leaky prior), which typically avoids strongly negative kappa. When your external checkpoint exists, nothing changes (the original FC stack is kept and weights are loaded as before). The rest of the pipeline (preprocessing, ordering, CSV writing) remains unchanged.'

# 9. Code solution

## === cell 0
import os
import random
import warnings

import numpy as np
import pandas as pd

import cv2 as cv
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models

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
test_path = "../input/aptos2019-blindness-detection/test.csv"
test_img_dir = "../input/aptos2019-blindness-detection/test_images"
sample_sub_path = "../input/aptos2019-blindness-detection/sample_submission.csv"
train_path = "../input/aptos2019-blindness-detection/train.csv"

test = pd.read_csv(test_path)
test.head()




## === cell 3
class _NoOpTransform:
    def __call__(self, image):
        return {"image": image}


train_transform = _NoOpTransform()



## === cell 4
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _resize_shorter_to(image: np.ndarray, shorter: int = 256) -> np.ndarray:
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return image
    if h < w:
        new_h = shorter
        new_w = int(round(w * (shorter / h)))
    else:
        new_w = shorter
        new_h = int(round(h * (shorter / w)))
    return cv.resize(image, (new_w, new_h), interpolation=cv.INTER_AREA)


def _center_crop(image: np.ndarray, size: int = 224) -> np.ndarray:
    h, w = image.shape[:2]
    top = max(0, (h - size) // 2)
    left = max(0, (w - size) // 2)
    crop = image[top : top + size, left : left + size]
    if crop.shape[0] != size or crop.shape[1] != size:
        crop = cv.resize(crop, (size, size), interpolation=cv.INTER_AREA)
    return crop


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
        img_path = os.path.join(self.img_dir, img_name + ".png")

        image = cv.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image at: {img_path}")

        if self.transform is not None:
            out = self.transform(image=image)
            image = out["image"] if isinstance(out, dict) and "image" in out else out

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = _resize_shorter_to(image, shorter=256)
        image = _center_crop(image, size=224)

        image = image.astype(np.float32) / 255.0
        image = (image - _IMAGENET_MEAN) / _IMAGENET_STD

        image = image.transpose(2, 0, 1).astype(np.float32)
        image = torch.from_numpy(image)

        if self.dt == "train":
            target = int(self.data["diagnosis"].iloc[idx])
            target = torch.tensor(target, dtype=torch.long)
            return image, target

        if self.dt == "test":
            return image

        raise ValueError(f"Unknown dt={self.dt}")

    def show(self, idx):
        img, target = self.__getitem__(idx)
        img = img.detach().cpu().numpy().transpose(1, 2, 0)
        img = img * _IMAGENET_STD + _IMAGENET_MEAN
        img = np.clip(img * 255.0, 0, 255).astype("uint8")
        plt.imshow(img)
        plt.title(str(int(target.detach().cpu().numpy())))
        plt.show()




## === cell 5
batch = 64

test_data = dataset(test_path, test_img_dir, dt="test", transform=train_transform)


def _seed_worker(worker_id):
    worker_seed = (SEED + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(SEED)

test_load = DataLoader(
    test_data,
    batch_size=batch,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    worker_init_fn=_seed_worker,
    generator=g,
)

len(test_data), next(iter(test_load)).shape



## === cell 6
weights_path = "../input/aptosmodelweights10/Best_Model_NO_10.pth"
use_imagenet_fallback = not os.path.exists(weights_path)

model = models.resnet18(
    weights=models.ResNet18_Weights.IMAGENET1K_V1 if use_imagenet_fallback else None
)

if use_imagenet_fallback:
    model.fc = nn.Linear(512, 5)
else:
    model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.Linear(256, 128),
        nn.Linear(128, 5),
    )

if use_imagenet_fallback:
    try:
        train_df = pd.read_csv(train_path)
        counts = (
            train_df["diagnosis"]
            .value_counts()
            .reindex([0, 1, 2, 3, 4], fill_value=1)
            .astype(np.float64)
        )
        prior = (counts / counts.sum()).to_numpy()
        prior = np.clip(prior, 1e-6, 1.0)
        bias = np.log(prior)  # softmax(log prior) = prior
        with torch.no_grad():
            nn.init.zeros_(model.fc.weight)
            model.fc.bias.copy_(torch.tensor(bias, dtype=model.fc.bias.dtype))
        print("Fallback head initialized to train-set class prior.")
    except Exception as e:
        print(
            "WARNING: Could not initialize prior bias, using default init. Error:",
            repr(e),
        )

if os.path.exists(weights_path):
    ckpt = torch.load(weights_path, map_location="cpu")
    if (
        isinstance(ckpt, dict)
        and "state_dict" in ckpt
        and isinstance(ckpt["state_dict"], dict)
    ):
        state = ckpt["state_dict"]
    elif (
        isinstance(ckpt, dict)
        and "model_state_dict" in ckpt
        and isinstance(ckpt["model_state_dict"], dict)
    ):
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            new_state[nk] = v
        state = new_state

    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) > 0 or len(unexpected) > 0:
        print(
            f"Loaded weights with strict=False. missing={len(missing)}, unexpected={len(unexpected)}"
        )
else:
    print(
        f"WARNING: Weights not found at {weights_path}. Using ImageNet-pretrained ResNet18 fallback (with prior-biased linear head) to produce less-degenerate predictions."
    )

model.to(device)
model.eval()



## === cell 7
predict = []
with torch.no_grad():
    for x in tqdm(test_load, total=len(test_load)):
        x = x.to(device, non_blocking=True)
        pred = model(x)
        pred = torch.argmax(pred, dim=1).to("cpu").numpy()
        predict.extend(list(pred))

len(predict), predict[:10]



## === cell 8
sub = pd.read_csv(sample_sub_path)
if len(sub) != len(predict):
    raise ValueError(
        f"Prediction length mismatch: len(sub)={len(sub)} vs len(predict)={len(predict)}"
    )

sub["diagnosis"] = predict
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
