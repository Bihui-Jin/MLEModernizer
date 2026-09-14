# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

3.9

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

0.9094321211240995

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -0.00925) has done: 'I add safe fall‑backs for the missing model files and undefined functions, and ensure the prediction variables always exist. The changes keep the original flow but replace unavailable pretrained checkpoints with torchvision’s built‑in ImageNet weights, so the notebook runs end‑to‑end and writes a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'I add a short fine‑tuning step that quickly trains the fallback densenet model on the available training images (using the same normalization as during inference). This inexpensive training should move the predictions from essentially random toward the true distribution, thereby improving the quadratic weighted kappa score and bringing it closer to the target. The rest of the pipeline and submission format remain unchanged.'
- What this solution (achieved 0.0) has done: 'Implemented data‑loading and inference speedups while keeping the exact model architecture, training loop, and prediction logic unchanged.

Key changes:
- Enabled cuDNN benchmarking and set a higher number of data‑loader workers with pin_memory for faster training data loading.
- Re‑written `make_predictions` to process test images in GPU‑friendly batches, use `torch.no_grad()`, and avoid per‑image device transfers, while still averaging original and flipped predictions as before.'
- What this solution (achieved 0.7546) has done: 'The update removes the implicit download of ImageNet weights when a pre‑trained checkpoint is unavailable. By initializing the fallback Densenet with `pretrained=False` we avoid a large network‑fetch and unnecessary weight loading, which was the primary source of the timeout while still preserving the original training loop and inference logic. No other behavior or model architecture is altered.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from glob import glob
import types, re




## === cell 1
__all__ = [
    "alexnet",
    "densenet121",
    "densenet169",
    "densenet201",
    "densenet161",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "inceptionv3",
    "squeezenet1_0",
    "squeezenet1_1",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19_bn",
    "vgg19",
]

model_urls = {
    "alexnet": "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth",
    "densenet121": "http://data.lip6.fr/cadene/pretrainedmodels/densenet121-fbdb23505.pth",
    "densenet169": "http://data.lip6.fr/cadene/pretrainedmodels/densenet169-f470b90a4.pth",
    "densenet201": "http://data.lip6.fr/cadene/pretrainedmodels/densenet201-5750cbb1e.pth",
    "densenet161": "http://data.lip6.fr/cadene/pretrainedmodels/densenet161-347e6b360.pth",
    "inceptionv3": "https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth",
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "squeezenet1_0": "https://download.pytorch.org/models/squeezenet1_0-a815701f.pth",
    "squeezenet1_1": "https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth",
    "vgg11": "https://download.pytorch.org/models/vgg11-bbd30ac9.pth",
    "vgg13": "https://download.pytorch.org/models/vgg13-c768596a.pth",
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
    "vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
}

input_sizes = {}
means = {}
stds = {}
for model_name in __all__:
    input_sizes[model_name] = [3, 224, 224]
    means[model_name] = [0.485, 0.456, 0.406]
    stds[model_name] = [0.229, 0.224, 0.225]
for model_name in ["inceptionv3"]:
    input_sizes[model_name] = [3, 299, 299]
    means[model_name] = [0.5, 0.5, 0.5]
    stds[model_name] = [0.5, 0.5, 0.5]

pretrained_settings = {}
for model_name in __all__:
    pretrained_settings[model_name] = {
        "imagenet": {
            "url": model_urls[model_name],
            "input_space": "RGB",
            "input_size": input_sizes[model_name],
            "input_range": [0, 1],
            "mean": means[model_name],
            "std": stds[model_name],
            "num_classes": 1000,
        }
    }
...




## === cell 2
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.item():.4f}, eps={self.eps})"


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


def get_se_resnet50_gem(pretrain):
    model = resnet50(
        num_classes=1000, pretrained="imagenet" if pretrain == "imagenet" else None
    )
    model.avg_pool = GeM()
    model.last_linear = nn.Linear(2048, 1)
    return model


def get_densenet121_gem(pretrain):
    model = densenet121(
        num_classes=1000, pretrained="imagenet" if pretrain == "imagenet" else None
    )
    model.avg_pool = GeM()
    model.last_linear = nn.Linear(1024, 1)
    return model




## === cell 3
TEST_IMAGE_PATH = "/kaggle/input/aptos2019-blindness-detection/test_images"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_images = glob(os.path.join(TEST_IMAGE_PATH, "*.png"))




## === cell 4
torch.backends.cudnn.benchmark = True
torch.set_num_threads(4)  # limit thread overhead on CPU‑only runs
torch.set_float32_matmul_precision("high")

MODEL_PATH_DENSENET = "../input/densenet121/model_densenet121_bs64_30.pth"

if os.path.exists(MODEL_PATH_DENSENET):
    model_densenet = get_densenet121_gem(pretrain=False)
    model_densenet.load_state_dict(torch.load(MODEL_PATH_DENSENET, map_location=device))
    model_densenet.eval()
    need_training = False
else:
    base = models.densenet121(pretrained=False)
    base.classifier = nn.Linear(1024, 1)
    model_densenet = base
    model_densenet.eval()
    need_training = True

norm_densenet = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

TRAIN_CSV_PATH = "/kaggle/input/aptos2019-blindness-detection/train.csv"
TRAIN_IMG_DIR = "/kaggle/input/aptos2019-blindness-detection/train_images"

train_df = pd.read_csv(TRAIN_CSV_PATH).dropna(subset=["diagnosis"])


class AptosDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['id_code']}.png")
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(row["diagnosis"], dtype=torch.float32)
        return img, label


train_dataset = AptosDataset(train_df, TRAIN_IMG_DIR, norm_densenet)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=min(4, os.cpu_count()),
    pin_memory=True,
)

model_densenet = model_densenet.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_densenet.parameters(), lr=1e-4)

if need_training:
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    model_densenet.train()
    for epoch in range(6):
        epoch_loss = 0.0
        for imgs, lbls in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model_densenet(imgs)
                    loss = criterion(outputs, lbls)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model_densenet(imgs)
                loss = criterion(outputs, lbls)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item() * imgs.size(0)
    model_densenet.eval()

predictions_densenet = []  # will be filled later if needed




## === cell 5
def make_predictions(
    model, test_images, transform, size=256, batch_size=32, device=device
):
    """
    Efficient batch inference with horizontal flip augmentation.
    Uses mixed‑precision when a CUDA device is available.
    """
    model.to(device)
    model.eval()
    preds = []
    resize_transform = transforms.Resize((size, size), interpolation=Image.BILINEAR)

    for i in range(0, len(test_images), batch_size):
        batch_paths = test_images[i : i + batch_size]

        batch_tensors = [
            transform(resize_transform(Image.open(p).convert("RGB")))
            for p in batch_paths
        ]
        batch_tensor = torch.stack(batch_tensors).to(device, non_blocking=True)

        with torch.no_grad():
            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    out = model(batch_tensor)
                    out_flip = model(torch.flip(batch_tensor, dims=(3,)))
            else:
                out = model(batch_tensor)
                out_flip = model(torch.flip(batch_tensor, dims=(3,)))
            avg = (out.squeeze() + out_flip.squeeze()) / 2.0

        preds.extend(
            (os.path.splitext(os.path.basename(p))[0], float(v))
            for p, v in zip(batch_paths, avg.cpu().numpy())
        )
    return preds




## === cell 6
if torch.cuda.is_available():
    MODEL_PATH_SERES = "../input/seresnet50testpseudo/model30.pth"
    seres_checkpoint_exists = os.path.exists(MODEL_PATH_SERES)
    try:
        model_seres = get_se_resnet50_gem(pretrain=False)
        if seres_checkpoint_exists:
            model_seres.load_state_dict(
                torch.load(MODEL_PATH_SERES, map_location=device)
            )
    except FileNotFoundError:
        model_seres = get_se_resnet50_gem(pretrain=False)

    norm_seres = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    if seres_checkpoint_exists:
        predictions_seresnet = make_predictions(
            model_seres, test_images, norm_seres, size=256, device=device
        )
    else:
        predictions_seresnet = []  # avoid using an un‑trained model
else:
    predictions_seresnet = []




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2848122832.py in <cell line: 0>()
      3     seres_checkpoint_exists = os.path.exists(MODEL_PATH_SERES)
      4     try:
----> 5         model_seres = get_se_resnet50_gem(pretrain=False)
      6         if seres_checkpoint_exists:
      7             model_seres.load_state_dict(

/tmp/ipykernel_55/2765391202.py in get_se_resnet50_gem(pretrain)
     17 
     18 def get_se_resnet50_gem(pretrain):
---> 19     model = resnet50(
     20         num_classes=1000, pretrained="imagenet" if pretrain == "imagenet" else None
     21     )

NameError: name 'resnet50' is not defined

## === cell 7
if not predictions_seresnet:
    predictions_densenet = make_predictions(
        model_densenet, test_images, norm_densenet, size=256, device=device
    )
final_predictions = (
    predictions_seresnet if predictions_seresnet else predictions_densenet
)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/813114181.py in <cell line: 0>()
----> 1 if not predictions_seresnet:
      2     predictions_densenet = make_predictions(
      3         model_densenet, test_images, norm_densenet, size=256, device=device
      4     )
      5 final_predictions = (

NameError: name 'predictions_seresnet' is not defined

## === cell 8

submission = pd.DataFrame(final_predictions, columns=["id_code", "diagnosis"])

submission["diagnosis"] = np.rint(submission["diagnosis"]).astype(int)

submission["diagnosis"] = submission["diagnosis"].clip(lower=0, upper=4)

submission = submission.sort_values("id_code").reset_index(drop=True)

submission.to_csv("submission.csv", index=False)

print(submission.head())

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2002964760.py in <cell line: 0>()
      5 
      6 # Convert list of tuples to DataFrame
----> 7 submission = pd.DataFrame(final_predictions, columns=["id_code", "diagnosis"])
      8 
      9 # Round predictions to nearest integer class

NameError: name 'final_predictions' is not defined
