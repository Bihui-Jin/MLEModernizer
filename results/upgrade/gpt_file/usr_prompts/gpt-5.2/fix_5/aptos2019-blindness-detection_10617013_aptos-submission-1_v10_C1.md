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

0.6400144551911864

# 6. Current score

0.34666

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.03645) has done: 'I fix the failure caused by the missing external weights file by loading ImageNet-pretrained ResNet-18 weights instead (same architecture, only weight source changes), so inference can run end-to-end in this environment. I also make weight loading robust (handling different checkpoint formats) and ensure the model head matches the saved format by keeping your existing classifier definition. Finally, I ensure `predict` is always created and a valid `submission.csv` with the required columns is written, matching the sample submission length and order.'
- What this solution (achieved 0.34666) has done: 'The timeout is dominated by CPU-side image decoding + heavy per-image PIL/Albumentations transforms and by running a full fine-tuning loop over 3295 images at 512×512, which is expensive in 10 minutes. I keep the exact same model, optimizer, loss, epochs, and transformations, but remove unnecessary work (e.g., plotting cell), cut repeated pandas indexing overhead, and make the input pipeline substantially faster by switching to OpenCV decoding/resizing/color conversion (equivalent RGB tensors), enabling persistent DataLoader workers with prefetching, and using faster, deterministic cuDNN/TF32-safe settings without changing semantics. I also enable channels-last + pinned memory + non-blocking transfers to reduce GPU input overhead while preserving numeric behavior. These changes are all performance-only and do not alter the algorithmic logic or training/inference semantics.'

# 9. Code solution

## === cell 0
import os
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image

import cv2  # faster image decode/resize than PIL for this workload

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import models
import torchvision.transforms as T
import torchvision.transforms.functional as F

import albumentations as A

warnings.filterwarnings("ignore")



## === cell 1
SEED = 8
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
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

train_path = "../input/aptos2019-blindness-detection/train.csv"
train_img_dir = "../input/aptos2019-blindness-detection/train_images"

test = pd.read_csv(test_path)
train = pd.read_csv(train_path)
print("train shape:", train.shape, "test shape:", test.shape)
test.head()



## === cell 3
train_transform = A.Compose(
    [
        A.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 1.5), p=1.0),
        A.GaussNoise(
            var_limit=(0.0, (0.05 * 255) ** 2), mean=0.0, per_channel=True, p=1.0
        ),
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

        self.ids = self.data["id_code"].to_numpy()
        if self.dt == "train":
            self.labels = self.data["diagnosis"].to_numpy(dtype=np.int64)
        else:
            self.labels = None

        self.size = (512, 512)  # keep exactly same target size

    def __get_data(self, path):
        return pd.read_csv(path)

    def __len__(self):
        return self.data.shape[0]

    @staticmethod
    def _adjust_brightness_saturation_rgb_uint8(
        img_rgb_u8, brightness=1.8, saturation=1.1
    ):
        img = img_rgb_u8.astype(np.float32) * float(brightness)

        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        img[..., 0] = img[..., 0] * float(saturation) + gray * (1.0 - float(saturation))
        img[..., 1] = img[..., 1] * float(saturation) + gray * (1.0 - float(saturation))
        img[..., 2] = img[..., 2] * float(saturation) + gray * (1.0 - float(saturation))

        img = np.clip(img, 0.0, 255.0).astype(np.uint8)
        return img

    def __getitem__(self, idx):
        img_name = self.ids[idx]
        path = os.path.join(self.img_dir, img_name + ".png")

        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            image = Image.open(path).convert("RGB")
            image = F.adjust_brightness(image, 1.8)
            image = F.adjust_saturation(image, 1.1)
            image = image.resize(self.size, resample=Image.BILINEAR)
            image = np.array(image, dtype=np.uint8)
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, self.size, interpolation=cv2.INTER_LINEAR)
            image = self._adjust_brightness_saturation_rgb_uint8(img_rgb, 1.8, 1.1)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
        image = torch.from_numpy(image)

        if self.dt == "train":
            y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
            return image, y

        return image

    def show(self, idx):
        if self.dt == "train":
            img, y = self.__getitem__(idx)
        else:
            img = self.__getitem__(idx)
            y = None
        img = img.detach().cpu().numpy()
        img = (img.transpose(1, 2, 0) * 255.0).clip(0, 255).astype("uint8")
        plt.imshow(img)
        plt.axis("off")
        if y is not None:
            plt.title(f"label={int(y)}")
        plt.show()




## === cell 5
batch = 64
num_workers = min(4, os.cpu_count() or 2)

train_data = dataset(train_path, train_img_dir, dt="train", transform=train_transform)
train_load = DataLoader(
    train_data,
    batch_size=batch,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)

test_data = dataset(test_path, test_img_dir, dt="test", transform=None)
test_load = DataLoader(
    test_data,
    batch_size=batch,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)

print("train batches:", len(train_load), "test batches:", len(test_load))



## === cell 6
pass



## === cell 7
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Sequential(
    nn.Linear(512, 256),
    nn.Linear(256, 128),
    nn.Linear(128, 5),
)

weights_path = "../input/aptos-model-18/Best_Model_NO_18.pth"
if os.path.exists(weights_path):
    ckpt = torch.load(weights_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("model."):
                nk = nk[len("model.") :]
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            new_state[nk] = v
        state = new_state
    model.load_state_dict(state, strict=False)
else:
    print(
        f"WARNING: Weights not found at {weights_path}. Using ImageNet pretrained ResNet18 weights and will fine-tune on train.csv."
    )

model.to(device)
if device == "cuda":
    model = model.to(memory_format=torch.channels_last)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

model.train()
epochs = 2  # unchanged
for ep in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in tqdm(train_load, desc=f"train epoch {ep+1}/{epochs}"):
        if device == "cuda":
            xb = xb.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        else:
            xb = xb.to(device)
        yb = yb.to(device, non_blocking=True) if device == "cuda" else yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == yb).sum().item()
        total += xb.size(0)

    print(f"epoch {ep+1}: loss={running_loss/total:.4f}, train_acc={correct/total:.4f}")

model.eval()



## === cell 8
predict = []
with torch.no_grad():
    for x in tqdm(test_load, desc="infer"):
        if device == "cuda":
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        else:
            x = x.to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).to("cpu").numpy()
        predict.extend(list(pred))

print(pd.Series(predict).value_counts().sort_index())



## === cell 9
sub = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")
if len(predict) != len(sub):
    raise ValueError(
        f"Prediction length {len(predict)} does not match submission length {len(sub)}"
    )

sub["diagnosis"] = np.asarray(predict, dtype=np.int64)
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
