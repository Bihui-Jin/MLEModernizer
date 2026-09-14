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

3.9

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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.9296087650720126

# 6. Current score

0.75675

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'Implemented fixes to resolve model initialization errors, added safe weight loading, handled missing images gracefully, and ensured a valid CSV submission is generated. The core model architecture and training logic remain unchanged.'
- What this solution (achieved -0.09406) has done: 'Implemented two minimal adjustments to move the model’s performance toward the target score:

1. **Use pretrained EfficientNet‑B4 weights** – switching `pretrained=False` to `pretrained=True` gives the backbone meaningful visual features without altering its architecture.
2. **Leverage the existing `combine3output` helper** – replaces the manual averaging of the three heads with the designed combination logic, which aligns better with the intended prediction strategy.

These changes keep the core pipeline intact while improving prediction quality and ensuring a valid `submission.csv` is written.'
- What this solution (achieved 0.0) has done: 'I make three small but impactful adjustments:   
1. Improve weight loading by checking several possible locations so the trained backbone is actually used (instead of falling back to random initialization).   
2. If no checkpoint is found, switch to a fully‑pretrained EfficientNet‑B4 model with a 5‑class head directly from timm, which gives reasonable predictions without additional training.   
3. Refine the ensemble step – `combine3output` now averages the three probability distributions (regression, classification, ordinal) and picks the class with highest averaged probability, which aligns better with the quadratic weighted‑kappa metric and should raise the score toward the target.'
- What this solution (achieved -0.1161) has done: 'I adjust the ensemble logic so that when only a pure pretrained EfficientNet classifier is used (i.e., the regression and ordinal outputs are zero tensors), the prediction relies solely on the classifier’s soft‑max output instead of diluting it with empty probability vectors. This small change keeps the core model unchanged but avoids unnecessary averaging that can hurt performance, moving the score closer to the target. The rest of the pipeline and file handling remain identical.'
- What this solution (achieved 0.13953) has done: 'I added a small utility `crop_image_from_gray` that safely handles missing images and simply returns the original image (or a cropped version for grayscale) so the pipeline no longer crashes. The function is defined early in the script, before it is used. With this fix the inference loop runs, populates the submission list, and the final CSV is written correctly. No core model logic was changed.'
- What this solution (achieved 0.75675) has done: 'I add a brief fine‑tuning step that trains the loaded EfficientNet (whether the custom three‑head version or the pure classifier) on a small subset of the training data for one epoch, and I expand the test‑time augmentation to include a vertical flip. These minimal changes keep the original architecture untouched while giving the model task‑specific signals, which should raise the quadratic weighted kappa toward the target score.'

# 9. Code solution

## === cell 0
import random
import math
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
from PIL import Image, ImageChops
from sklearn.metrics import cohen_kappa_score
import timm
from timm.models.efficientnet import EfficientNet  # for isinstance check

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")




## === cell 1
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, flatten=False):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.flatten = flatten

    def forward(self, x):
        x = gem(x, p=self.p, eps=self.eps)
        if self.flatten:
            x = x.flatten(1)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class Linear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if torch.jit.is_scripting():
            bias = self.bias.to(dtype=input.dtype) if self.bias is not None else None
            return F.linear(input, self.weight.to(dtype=input.dtype), bias=bias)
        else:
            return F.linear(input, self.weight, self.bias)


def crop_image_from_gray(img, tol=7):
    """
    Simple utility to handle images that may be None or need minimal cropping.
    If the image is grayscale, it crops out uniform borders; otherwise,
    it returns the original image unchanged.
    """
    if img is None:
        return None
    if len(img.shape) == 2:  # grayscale
        mask = img > tol
        if mask.any():
            img = img[np.ix_(mask.any(1), mask.any(0))]
        return img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray > tol
        if mask.any():
            img = img[np.ix_(mask.any(1), mask.any(0), range(3))]
        return img


class backboneNet_efficient(nn.Module):
    def __init__(self):
        super(backboneNet_efficient, self).__init__()
        net = timm.create_model("tf_efficientnet_b4_ns", pretrained=True)
        layers_to_train = ["blocks"]
        for name, parameter in net.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)
        self.num_features = 1792
        self.conv_stem = net.conv_stem
        self.bn1 = net.bn1
        self.act1 = getattr(net, "act1", nn.Identity())
        self.block0 = net.blocks[0]
        self.block1 = net.blocks[1]
        self.block2 = net.blocks[2]
        self.block3 = net.blocks[3]
        self.block4 = net.blocks[4]
        self.block5 = net.blocks[5]
        self.block6 = net.blocks[6]
        self.conv_head = net.conv_head
        self.bn2 = net.bn2
        self.act2 = getattr(net, "act2", nn.Identity())
        self.global_pool = net.global_pool
        self.drop_rate = 0.4

        self.rg_cls = Linear(self.num_features, 1, bias=True)
        self.cls_cls = Linear(self.num_features, 5, bias=True)
        self.ord_cls = Linear(self.num_features, 4, bias=True)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.global_pool(x)
        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        rg = self.rg_cls(x)
        cls = self.cls_cls(x)
        ord = self.ord_cls(x)
        return rg, cls, ord


threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = 0
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze().cpu().item()
    return prediction


def ordinal2class_prob(out):
    pred_prob = torch.zeros(out.size(0), 5).to(out.device)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    pred_prob = torch.zeros((out.size(0), 5)).to(out.device)
    for i in range(out.size(0)):
        if out[i] < 4.0:
            l1 = int(math.floor(out[i].item()))
            l2 = int(math.ceil(out[i].item()))
            pred_prob[i][l1] = 1 - (out[i] - l1)
            pred_prob[i][l2] = 1 - (l2 - out[i])
        else:
            pred_prob[i][4] = 1.0
    return pred_prob


def combine3output(r_out, c_out, o_out):
    """
    Produce a final class by averaging the three probability
    distributions (regression, classification, ordinal) and
    selecting the class with the highest averaged probability.
    If regression and ordinal outputs are zero (pure pretrained classifier),
    fall back to using only the classification probabilities.
    """
    if torch.all(r_out == 0) and torch.all(o_out == 0):
        cls_prob = F.softmax(c_out, dim=1)
        final_class = torch.argmax(cls_prob, dim=1).item()
        return int(final_class)

    reg_prob = regress2class_prob(r_out)
    cls_prob = F.softmax(c_out, dim=1)
    ord_prob = ordinal2class_prob(o_out)

    avg_prob = (reg_prob + cls_prob + ord_prob) / 3.0
    final_class = torch.argmax(avg_prob, dim=1).item()
    return int(final_class)




## === cell 2
test_ids = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")[
    "id_code"
].values.squeeze()

transform2 = transforms.Compose(
    [
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

net3 = backboneNet_efficient()

possible_paths = [
    "../input/weights/2.pth",
    "./weights/2.pth",
    "./2.pth",
    "../working/weights/2.pth",
]
loaded = False
for weights_path in possible_paths:
    if os.path.exists(weights_path):
        try:
            net3.load_state_dict(torch.load(weights_path, map_location=device))
            print(f"Weights loaded from {weights_path}.")
            loaded = True
            break
        except Exception as e:
            print(f"Failed to load weights from {weights_path}: {e}")

if not loaded:
    print(
        "No finetuned weights found – switching to a pure pretrained EfficientNet‑B4 classifier."
    )
    net3 = timm.create_model("tf_efficientnet_b4_ns", pretrained=True, num_classes=5)
else:
    print("Finetuned weights loaded; using custom three‑head backbone.")

net3 = net3.to(device)
net3.eval()



## === cell 3
train_df = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

subset_size = min(800, len(train_df))
train_subset = train_df.head(subset_size)


class DRDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['id_code']}.png")
        img = cv2.imread(img_path)
        img = crop_image_from_gray(img)
        if img is None:
            img = np.zeros((380, 380, 3), dtype=np.uint8)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        tensor = self.transform(img)
        label = int(row["diagnosis"])
        return tensor, label


train_dataset = DRDataset(
    train_subset,
    "../input/aptos2019-blindness-detection/train_images",
    transform2,
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, net3.parameters()), lr=1e-4, weight_decay=1e-5
)

net3.train()
epochs = 1
for epoch in range(epochs):
    epoch_loss = 0.0
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        if isinstance(net3, EfficientNet):
            outputs = net3(imgs)  # logits
        else:
            _, cls_logits, _ = net3(imgs)
            outputs = cls_logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * imgs.size(0)
    print(
        f"Fine‑tune epoch {epoch+1}/{epochs}, loss: {epoch_loss/len(train_loader.dataset):.4f}"
    )

net3.eval()
print("Fine‑tuning completed.")




## === cell 4
def forward_with_tta(model, img_tensor):
    """
    Run model on original, horizontally‑flipped and vertically‑flipped images,
    averaging the outputs. Handles both the pure classifier (EfficientNet)
    and the custom three‑head backbone.
    """
    h_flip = torch.flip(img_tensor, dims=[3])  # horizontal
    v_flip = torch.flip(img_tensor, dims=[2])  # vertical
    inputs = [img_tensor, h_flip, v_flip]

    if isinstance(model, EfficientNet):
        outs = [model(x) for x in inputs]
        avg_cls = torch.stack(outs, dim=0).mean(dim=0)
        rg = torch.zeros((1, 1), device=img_tensor.device)
        ord = torch.zeros((1, 4), device=img_tensor.device)
        return rg, avg_cls, ord
    else:
        rg_list, cls_list, ord_list = [], [], []
        for x in inputs:
            rg, cls, ord = model(x)
            rg_list.append(rg)
            cls_list.append(cls)
            ord_list.append(ord)
        rg_avg = torch.stack(rg_list, dim=0).mean(dim=0)
        cls_avg = torch.stack(cls_list, dim=0).mean(dim=0)
        ord_avg = torch.stack(ord_list, dim=0).mean(dim=0)
        return rg_avg, cls_avg, ord_avg




## === cell 5
submission = []
with torch.no_grad():
    for i, idx in enumerate(test_ids):
        print(f"Processing {i+1}/{len(test_ids)}: {idx}")
        image_path = f"../input/aptos2019-blindness-detection/test_images/{idx}.png"
        img = cv2.imread(image_path)
        img = crop_image_from_gray(img)
        if img is None:
            final_pred = 0
            submission.append([idx, final_pred])
            continue
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_tensor = transform2(img).unsqueeze(0).to(device)

        rg_out, cls_out, ord_out = forward_with_tta(net3, img_tensor)

        final_pred = combine3output(rg_out, cls_out, ord_out)
        final_pred = max(0, min(4, final_pred))

        submission.append([idx, final_pred])

submission = np.array(submission)



## === cell 6
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
assert not df.empty, "Submission DataFrame is empty!"
df.to_csv("submission.csv", index=False)
print("Saved submission.csv with", len(df), "rows.")
