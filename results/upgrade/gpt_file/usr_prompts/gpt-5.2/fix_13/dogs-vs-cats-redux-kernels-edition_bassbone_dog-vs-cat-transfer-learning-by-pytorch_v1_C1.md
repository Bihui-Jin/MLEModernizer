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
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

# 2. Python version

3.8

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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 5. Target score

0.06921

# 6. Current score

6.11513

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 3.18548) has done: 'The root cause of all downstream errors is that you extract `train.zip`/`test.zip` into `../data`, but then you look for images in `../data/train` and `../data/test` (those folders don’t exist after extraction), so `train_list`/`test_list` are empty and everything fails. I fix the paths by pointing `train_dir`/`test_dir` to the actual extracted folders and make the dataset robust to RGB conversion. I also ensure the submission ids match the competition exactly by reading `sample_submission.csv` and predicting in that id order, so Kaggle won’t reject the file due to id mismatch. These fixes are correctness-only (score-neutral aside from making a valid submission possible).'
- What this solution (achieved 3.10497) has done: 'Your score is far worse than the target (logloss 3.18548 vs 0.06921), so we should make a small, metric-aligned fix that improves correctness without changing the core model/training loop. The biggest issue for logloss here is usually probability calibration being destroyed by predicting on a *single deterministic center crop* at test time while training uses random resized crops; this mismatch can yield very overconfident wrong predictions. I keep the exact same VGG16 + “only last layer trainable” setup and epochs, but switch inference to a small, deterministic test-time augmentation (multi-crop/flip averaging) using the same resize/crop pipeline, which typically reduces logloss a lot while preserving core logic. I also clamp probabilities slightly away from 0/1 (numerically safe for logloss) and keep the submission id order exactly as in `sample_submission.csv`.'
- What this solution (achieved 3.10497) has done: 'Your score (3.10497) is far worse than the target (0.06921) on logloss, so the most likely issue is that the predicted probabilities are effectively “flipped” (i.e., you’re submitting p(dog) when the model’s class-1 is actually “cat”, or vice versa). This can happen because your dataset defines dog=1/cat=0, but the last layer is randomly initialized and trained very lightly, so the mapping can be inverted; logloss then explodes. I add a minimal, metric-aligned calibration step: compute validation logloss for both interpretations (p_dog vs 1-p_dog) and pick the one with lower validation logloss for the submission (no change to model/training/core inference). I also clamp probabilities a bit more safely for logloss.'
- What this solution (achieved 6.11505) has done: 'Your current logloss is catastrophically high for Dogs vs Cats, which usually means the submission probabilities are misaligned with the competition’s expected `id` rows (wrong file set / wrong ids / missing many test images) or the model is producing near-random probabilities. The folder listing you provided strongly suggests you’re using a *2500-image* “unknown” test subset (hence sample_submission has 2500 rows), so the main fix is to ensure we always predict exactly for those 2500 ids and never accidentally read a different test folder. Then, to move logloss materially toward the target without changing the core model/training loop, we add a minimal probability calibration step on the validation set: temperature scaling (1-parameter) plus the existing optional flip decision, both chosen by minimizing validation logloss. Finally, we keep TTA and add a slightly safer clip epsilon for logloss stability, producing a valid `submission.csv` with correct `id,label`.'
- What this solution (achieved 3.32233) has done: 'Your current logloss (6.11505) is far worse than the target, and with your existing VGG16 setup the most likely remaining “catastrophic” issue is that the probabilities are systematically inverted because the dog/cat class index in the model output is not guaranteed to correspond to your `dog=1` assumption. I keep your model, training loop, TTA, and temperature scaling intact, but I make the calibration use the *true dog probability from softmax* (class-1 probability) rather than the margin heuristic, and I pick whether to flip using validation logloss (as you already do). This is a minimal, metric-aligned fix that typically moves logloss from “very bad” to “reasonable” without changing core training. I also keep the submission strictly aligned to `sample_submission.csv` ids and clamp probabilities for numerical safety.'
- What this solution (achieved 6.32204) has done: 'Your logloss is still far above the target, and with this exact “freeze VGG16, train only last layer for 2 epochs” setup the biggest remaining correctness issue is the way you convert the 2-class logits into a dog probability: using `sigmoid(dog_logit/T)` ignores the competing cat logit and often be badly miscalibrated, exploding logloss. I keep your model/training loop/TTA/temperature-scan-and-optional-flip logic, but change the probability computation to be the metric-correct one for a 2-class softmax: `p_dog = softmax(logits/T)[dog_index]` averaged over crops. This is a minimal change (pure post-processing) that typically moves logloss from “catastrophic” toward “reasonable” without altering training. I also clip with a slightly safer epsilon for logloss and keep strict submission id alignment via `sample_submission.csv`.'
- What this solution (achieved 6.32204) has done: 'Your current logloss (6.32204) is still extremely far from the target (0.06921), so we should focus on a minimal, high-impact correctness fix rather than tuning. The most likely remaining issue is a label/probability mismatch: you assume “dog probability = softmax(logits)[1]”, but the model’s class-1 output is not guaranteed to correspond to dog, and picking flip based only on `1-p` can still fail if the class index mapping is wrong. I add a tiny calibration step on the validation set to choose the best dog class index (`dog_index` in {0,1}) together with temperature `T`, strictly by minimizing validation logloss, and then use that same mapping for test predictions (no model/training changes). This preserves your architecture/training/TTA, but fixes the semantic mapping that can otherwise explode logloss.'
- What this solution (achieved 6.11513) has done: 'Your score is far worse than the target (lower-is-better logloss 6.32 vs 0.069), so we need a small, high-impact correctness fix rather than tuning. The main remaining failure mode consistent with such a huge logloss is that the submission “id” row order and/or the “label” probability semantics don’t match what Kaggle evaluates; your code currently sorts by `id` at the end, which can silently desync from the exact `sample_submission.csv` order. I (1) force the submission rows to be in the *exact same order* as `sample_submission.csv` (no sorting), (2) add a tiny validation-based check to ensure the chosen `dog_index` and temperature `T` are evaluated using the *same aggregation pipeline as test* (currently val uses numpy softmax on mean logits while test uses torch softmax; make them identical), and (3) slightly increase numerical safety for logloss by clipping with `eps=1e-6` (still negligible). These are post-processing/alignment fixes only; the model, training loop, and feature pipeline remain unchanged.'
- What this solution (achieved 6.11513) has done: 'Your current logloss (6.11513) is far worse than the target (0.06921, lower is better), so we need a small correctness fix that can drastically reduce “catastrophic” logloss without changing the model/training. The highest-probability remaining issue is that the train labels used by `ImageFolder`-style directory structure (`train/cat`, `train/dog`) do not match your filename-based label parsing (which expects `cat.x.jpg`/`dog.x.jpg`), so you are effectively training on wrong labels or crashing depending on paths. I minimally fix the dataset labeling logic to use the parent folder name when available (cat/dog folders) and fall back to filename parsing otherwise, keeping architecture/training/TTA/temperature+dog_index calibration unchanged. This should move the score sharply toward the target by restoring correct supervision while preserving the rest of your pipeline and still writing a valid `submission.csv` in the exact `sample_submission.csv` id order.'
- What this solution (achieved 6.11513) has done: 'Your current logloss (6.11513) is catastrophically worse than the target (0.06921), so the most likely remaining issue is still semantic misalignment between the competition’s expected “probability of dog” and what we’re outputting. I keep your exact model/training loop/TTA, but add a minimal validation-based selection that jointly chooses (a) which model class corresponds to “dog” and (b) whether to interpret the label as p(dog) or 1−p(dog), both evaluated under the exact same TTA+temperature pipeline you use for test. This is a post-processing calibration-only change that can drastically reduce logloss when the probability semantics are inverted, without changing the core learning. I also ensure submission order remains exactly `sample_submission.csv` order and keep safe clipping for logloss stability.'
- What this solution (achieved 6.11513) has done: 'Your logloss is catastrophically far from the target, so the most likely remaining issue is a semantic mismatch between the model’s training label indices and what you submit as “probability of dog” (even with the current dog_index/invert scan). I make a minimal, metric-aligned fix by explicitly using the `ImageFolder`-style class mapping (`cat`/`dog`) that your extracted train directory actually implies, and then calibrate only temperature (and keep the existing safety clipping) using the *same* TTA pipeline used at test time. This preserves the exact VGG16 architecture, the “train only last layer for 2 epochs” loop, the same transforms, and still writes `submission.csv` in the exact `sample_submission.csv` id order. The only logic change is ensuring labels and “p(dog)” semantics are consistent end-to-end, which should move logloss sharply toward the target band.'
- What this solution (achieved 6.11513) has done: 'Your current logloss is catastrophically worse than the target, which strongly suggests a remaining semantic mismatch (which model class index corresponds to “dog”) and/or a temperature chosen under the wrong assumption. I keep your exact VGG16 setup, training loop, and TTA, but extend the existing calibration to *jointly* pick the best `(dog_class_index, temperature)` on the validation set by minimizing validation logloss under the same pipeline used for test inference. This is a post-processing/semantic alignment fix only (no training changes) and is the smallest high-impact change likely to move logloss sharply toward your target band. I then use the chosen `dog_class_index` and `T` for test predictions, keeping strict `sample_submission.csv` id order and safe clipping.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os, glob, time, copy, random, zipfile
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import models, transforms

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 1
torch.__version__



## === cell 2
base_dir = "../input/dogs-vs-cats-redux-kernels-edition"
os.listdir(base_dir)[:10]



## === cell 3
os.makedirs("../data", exist_ok=True)



## === cell 4
train_zip_path = os.path.join(base_dir, "train.zip")
test_zip_path = os.path.join(base_dir, "test.zip")

with zipfile.ZipFile(train_zip_path) as train_zip:
    train_zip.extractall("../data")

with zipfile.ZipFile(test_zip_path) as test_zip:
    test_zip.extractall("../data")

candidates_train = [
    "../data/train",
    "../data/dogs-vs-cats-redux-kernels-edition/train",
    "../data/train/train",
]
candidates_test = [
    "../data/test",
    "../data/dogs-vs-cats-redux-kernels-edition/test",
    "../data/test/test",
]


def pick_existing_dir(cands):
    for p in cands:
        if os.path.isdir(p):
            if len(glob.glob(os.path.join(p, "*.jpg"))) > 0:
                return p
    for p in cands:
        if os.path.isdir(p):
            sub = glob.glob(os.path.join(p, "*"))
            for s in sub:
                if os.path.isdir(s) and len(glob.glob(os.path.join(s, "*.jpg"))) > 0:
                    return s
    return None


train_dir = pick_existing_dir(candidates_train)
test_dir = pick_existing_dir(candidates_test)

print("Resolved train_dir:", train_dir)
print("Resolved test_dir :", test_dir)
assert train_dir is not None, "Could not find extracted train images directory."
assert test_dir is not None, "Could not find extracted test images directory."



## === cell 5
train_list = glob.glob(os.path.join(train_dir, "**", "*.jpg"), recursive=True)
test_list = glob.glob(os.path.join(test_dir, "**", "*.jpg"), recursive=True)

print("num train images:", len(train_list))
print("num test images :", len(test_list))
print("train sample:", train_list[:3])
print("test sample :", test_list[:3])



## === cell 6
img = Image.open(train_list[0]).convert("RGB")
plt.imshow(img)
plt.axis("off")
plt.show()



## === cell 7
img = Image.open(test_list[0]).convert("RGB")
plt.imshow(img)
plt.axis("off")
plt.show()



## === cell 8
CLASS_TO_IDX = {"cat": 0, "dog": 1}


def get_label_from_path(p):
    parent = os.path.basename(os.path.dirname(p)).lower()
    if parent in CLASS_TO_IDX:
        return CLASS_TO_IDX[parent]

    token = os.path.basename(p).split(".")[0].lower()
    if token in CLASS_TO_IDX:
        return CLASS_TO_IDX[token]

    raise ValueError(f"Could not infer label from path: {p}")


labels = [get_label_from_path(p) for p in train_list]
train_list, val_list = train_test_split(
    train_list, test_size=0.1, random_state=42, shuffle=True, stratify=labels
)

print(len(train_list), len(val_list))




## === cell 9
class ImageTransform:
    def __init__(self, resize, mean, std):
        self.data_transform = {
            "train": transforms.Compose(
                [
                    transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(resize),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
        }

    def __call__(self, img, phase):
        return self.data_transform[phase](img)




## === cell 10
class DogvsCatDataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase="train"):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")
        img_transformed = self.transform(img, self.phase)

        label = get_label_from_path(img_path)
        return img_transformed, label




## === cell 11
size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
batch_size = 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device



## === cell 12
train_dataset = DogvsCatDataset(
    train_list, transform=ImageTransform(size, mean, std), phase="train"
)
val_dataset = DogvsCatDataset(
    val_list, transform=ImageTransform(size, mean, std), phase="val"
)

print("Operation Check")
index = 0
x, y = train_dataset.__getitem__(index)
print(x.size(), y)



## === cell 13
train_dataloader = data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)
val_dataloader = data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

dataloader_dict = {"train": train_dataloader, "val": val_dataloader}

print("Operation Check")
batch_iterator = iter(train_dataloader)
inputs, label = next(batch_iterator)
print(inputs.size())
print(label[:10])



## === cell 14
use_pretrained = True
try:
    net = models.vgg16(
        weights=models.VGG16_Weights.IMAGENET1K_V1 if use_pretrained else None
    )
except TypeError:
    net = models.vgg16(pretrained=use_pretrained)

net.classifier[6] = nn.Linear(in_features=4096, out_features=2)
print("Model ready")



## === cell 15
params_to_update = []
update_params_name = ["classifier.6.weight", "classifier.6.bias"]

for name, param in net.named_parameters():
    if name in update_params_name:
        param.requires_grad = True
        params_to_update.append(param)
        print("Will update:", name)
    else:
        param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=params_to_update, lr=0.001, momentum=0.9)




## === cell 16
def train_model(net, dataloader_dict, criterion, optimizer, num_epoch):
    since = time.time()
    best_model_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0
    net = net.to(device)

    for epoch in range(num_epoch):
        print("Epoch {}/{}".format(epoch + 1, num_epoch))
        print("-" * 20)

        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            for inputs, labels in tqdm(dataloader_dict[phase], desc=phase, leave=False):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                epoch_loss += loss.item() * inputs.size(0)
                epoch_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_acc = epoch_corrects / len(dataloader_dict[phase].dataset)

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(net.state_dict())

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    net.load_state_dict(best_model_wts)
    return net




## === cell 17
num_epoch = 2
net = train_model(net, dataloader_dict, criterion, optimizer, num_epoch)




## === cell 18
def binary_logloss(y_true, p, eps=1e-6):
    p = np.clip(p, eps, 1.0 - eps)
    y_true = np.asarray(y_true).astype(np.float64)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


val_tta_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.TenCrop(size),
        transforms.Lambda(
            lambda crops: torch.stack(
                [
                    transforms.Normalize(mean, std)(transforms.ToTensor()(c))
                    for c in crops
                ]
            )
        ),
    ]
)

net = net.to(device)
net.eval()

val_true = []
val_logits_mean = []

with torch.no_grad():
    for img_path in tqdm(val_list, desc="calibrate(val)", leave=False):
        img = Image.open(img_path).convert("RGB")
        crops = val_tta_transform(img).to(device, non_blocking=True)
        outputs = net(crops).float()
        mean_logits = outputs.mean(dim=0).detach().cpu().numpy()

        y = get_label_from_path(img_path)
        val_true.append(y)
        val_logits_mean.append(mean_logits)

val_true = np.asarray(val_true, dtype=np.float64)
val_logits_mean = np.asarray(val_logits_mean, dtype=np.float64)

temps = np.exp(np.linspace(np.log(0.5), np.log(5.0), 31))

best = {"ll": 1e9, "T": 1.0, "dog_index": 1}

val_logits_mean_torch = torch.from_numpy(val_logits_mean).to(
    device=device, dtype=torch.float32
)

for dog_index in [0, 1]:
    for T in temps:
        logits_T = val_logits_mean_torch / float(T)
        probs_T = torch.softmax(logits_T, dim=1).detach().cpu().numpy()
        p = probs_T[:, dog_index]
        ll = binary_logloss(val_true, p, eps=1e-6)
        if ll < best["ll"]:
            best = {"ll": float(ll), "T": float(T), "dog_index": int(dog_index)}

print(
    f"Best val logloss: {best['ll']:.6f} at T={best['T']:.4f}, dog_index={best['dog_index']}"
)



## === cell 19
sample_path_candidates = [
    os.path.join(base_dir, "sample_submission.csv"),
    "../input/sample_submission.csv",
]
sample_path = None
for p in sample_path_candidates:
    if os.path.exists(p):
        sample_path = p
        break
assert sample_path is not None, "Could not locate sample_submission.csv"

sample = pd.read_csv(sample_path)
sample_ids = sample["id"].astype(int).tolist()

test_path_by_id = {}
for p in test_list:
    try:
        _id = int(os.path.basename(p).split(".")[0])
        test_path_by_id[_id] = p
    except Exception:
        pass

missing = [i for i in sample_ids if i not in test_path_by_id]
if len(missing) > 0:
    raise FileNotFoundError(
        f"Missing {len(missing)} test images referenced by sample_submission. Example missing id: {missing[0]}"
    )

tta_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.TenCrop(size),
        transforms.Lambda(
            lambda crops: torch.stack(
                [
                    transforms.Normalize(mean, std)(transforms.ToTensor()(c))
                    for c in crops
                ]
            )
        ),
    ]
)

id_list = []
pred_list = []

net.eval()
eps = 1e-6
T = best["T"]
DOG_CLASS_INDEX = best["dog_index"]

with torch.no_grad():
    for _id in tqdm(sample_ids, desc="predict"):
        test_path = test_path_by_id[_id]
        img = Image.open(test_path).convert("RGB")

        crops = tta_transform(img).to(device, non_blocking=True)
        outputs = net(crops).float()
        mean_logits = outputs.mean(dim=0)

        p_dog = torch.softmax(mean_logits / float(T), dim=0)[DOG_CLASS_INDEX].item()
        pred = float(np.clip(float(p_dog), eps, 1.0 - eps))

        id_list.append(_id)
        pred_list.append(pred)

res = pd.DataFrame({"id": id_list, "label": pred_list})

out_path = "submission.csv"
res.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", res.shape)
print(res.head())



## === cell 20
if len(res) > 0:
    class_ = {0: "cat", 1: "dog"}
    fig, axes = plt.subplots(2, 5, figsize=(20, 12), facecolor="w")
    for ax in axes.ravel():
        i = random.choice(res["id"].values.tolist())
        label_prob = res.loc[res["id"] == i, "label"].values[0]
        label = 1 if label_prob > 0.5 else 0
        img_path = test_path_by_id[int(i)]
        img = Image.open(img_path).convert("RGB")
        ax.set_title(f"{class_[label]} (p_dog={label_prob:.2f})")
        ax.imshow(img)
        ax.axis("off")
    plt.show()
