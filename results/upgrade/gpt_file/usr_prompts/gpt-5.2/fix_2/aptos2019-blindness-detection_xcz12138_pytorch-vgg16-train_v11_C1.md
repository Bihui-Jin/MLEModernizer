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

3.7

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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

0.7219723542467201

# 6. Current score

-0.04372

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved -0.04372) has done: 'I fix the immediate runtime blockers so the notebook runs end-to-end and writes a valid `submission.csv`. The main issue is that the code tries to load a non-existent model file; I replace that with an equivalent VGG16-based classifier created in-code using `torchvision` so inference can run without external assets. I also fix several Python logic bugs (`is` vs `==`, invalid `num_workers=False`), make the train/eval split deterministic for stability, and ensure image loading is RGB and paths are resolved correctly. This should produce a valid submission file in the required format.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

DATA_DIR = "../input/aptos2019-blindness-detection"

print("Listing input dir:", DATA_DIR)
print(os.listdir(DATA_DIR)[:20])



## === cell 1
from PIL import Image
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchvision import transforms as tfs
from torch.utils.data import DataLoader, Dataset


class Config:
    data_dir = DATA_DIR
    crop_size = 224
    train_batch_size = 64
    test_batch_size = 1
    lr = 1e-3
    momentum = 0.9
    epochs = 20
    print_every = 5
    seed = 42


opt = Config()

np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)




## === cell 2
def read_file(data_dir, split="train"):
    csv_name = "test.csv" if split == "test" else "train.csv"
    file = os.path.join(data_dir, csv_name)
    dataset = pd.read_csv(file)

    if split == "test":
        data = [
            os.path.join(data_dir, f"test_images/{dataset.iloc[i].values[0]}.png")
            for i in range(len(dataset))
        ]
        label = None
        return data, label
    else:
        data = [
            os.path.join(data_dir, f"train_images/{dataset.iloc[i].values[0]}.png")
            for i in range(len(dataset))
        ]
        label = dataset.iloc[:, 1].values

        train_data, eval_data, train_label, eval_label = train_test_split(
            data, label, test_size=0.2, random_state=opt.seed, stratify=label
        )

        if split == "eval":
            return eval_data, eval_label
        else:
            return train_data, train_label


def transforms(img, crop_size):
    img_tfs = tfs.Compose(
        [
            tfs.RandomResizedCrop(crop_size),
            tfs.RandomHorizontalFlip(p=0.2),
            tfs.ToTensor(),
            tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    return img_tfs(img)


class APTOSSet(Dataset):
    def __init__(
        self, transform, split="train", data_dir=opt.data_dir, crop_size=opt.crop_size
    ):
        data_list, label = read_file(data_dir, split=split)
        self.transform = transform
        self.data_list = data_list
        self.label = label
        self.crop_size = crop_size
        self.split = split

    def __getitem__(self, idx):
        img_path = self.data_list[idx]
        img = Image.open(img_path).convert("RGB")
        img = transforms(img, self.crop_size)

        if self.split == "test":
            return img
        else:
            label = int(self.label[idx])
            return img, label

    def __len__(self):
        return len(self.data_list)


train_set = APTOSSet(split="train", transform=transforms)
eval_set = APTOSSet(split="eval", transform=transforms)
test_set = APTOSSet(split="test", transform=transforms)

APT_train = DataLoader(
    train_set, batch_size=opt.train_batch_size, shuffle=True, num_workers=0
)
APT_eval = DataLoader(
    eval_set, batch_size=opt.train_batch_size, shuffle=False, num_workers=0
)
APT_test = DataLoader(
    test_set, batch_size=opt.test_batch_size, shuffle=False, num_workers=0
)

print("Train/Eval/Test sizes:", len(train_set), len(eval_set), len(test_set))



## === cell 3
import torch.nn as nn
from torchvision import models


def build_model(num_classes=5):
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


model = build_model(num_classes=5)

for params in model.parameters():
    params.requires_grad = False
model.eval()



## === cell 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)
model.to(device)




## === cell 5
@torch.no_grad()
def test_predict(model):
    model.eval()
    prediction = []
    for data in APT_test:
        data = data.to(device)
        outputs = model(data)
        pred = outputs.argmax(dim=1)
        prediction.append(int(pred.item()))
    return prediction


sub = pd.read_csv(os.path.join(opt.data_dir, "sample_submission.csv"))
preds = test_predict(model)

assert len(preds) == len(sub), f"Pred length {len(preds)} != sub length {len(sub)}"

sub["diagnosis"] = preds
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
