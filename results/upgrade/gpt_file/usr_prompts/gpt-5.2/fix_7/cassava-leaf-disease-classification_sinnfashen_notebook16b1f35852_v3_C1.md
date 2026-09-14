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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

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
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8375642187972197

# 6. Current score

0.52093

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the dataset/IO bugs that prevent the notebook from running: the test image folder contains a nested `test_images/` directory that must be filtered out, and `MyTestDataset.__len__` incorrectly references `self.labels`. I also fix the missing model weights path by using torchvision’s built-in pretrained SqueezeNet weights (keeping the same model architecture and inference logic) so the code can run end-to-end in this environment. Finally, I ensure inference runs in `eval()` mode with `torch.no_grad()` and writes a valid `submission.csv` with the required `image_id,label` columns.'
- What this solution (achieved 0.53587) has done: 'Your 0.0 score is because the current code uses an ImageNet-pretrained SqueezeNet (1000 classes) to predict cassava labels (0–4), so predictions are essentially random and won’t approach the target accuracy. To move the score toward 0.8376 with minimal changes while preserving the core approach (single pretrained CNN, no training loop), I keep SqueezeNet inference but add a lightweight “nearest-prototype” classifier: compute class prototypes from the provided `train_images/train.csv` using the same pretrained model, then classify each test image by nearest prototype in feature space. This keeps the model architecture and loss/training logic unchanged (still no training), but makes predictions aligned to the competition’s 5 classes. I also switch inference to use a DataLoader for correctness/performance and keep the submission format identical.'
- What this solution (achieved 0.53401) has done: 'To move your score up toward 0.8376 without changing the core “pretrained SqueezeNet + nearest-prototype (no training loop)” approach, I (1) build prototypes from a more representative subset of the training data by sampling uniformly per class instead of taking the first N rows (which can be order-biased), and (2) compute prototypes more efficiently by accumulating per-class sums without an inner loop. I also (3) use a slightly larger but still time-safe cap and (4) add lightweight test-time augmentation (simple horizontal flip) and average the similarities, which keeps the same inference semantics but typically improves accuracy for image classification. These are minimal, legitimate changes aimed at improving feature-prototype quality and stability, which should increase accuracy toward the target. The script still writes a valid `submission.csv` with `image_id,label`.'
- What this solution (achieved 0.49402) has done: 'Your current gap to the target is large (0.534 → 0.837), so we need a real accuracy lift while keeping the same core “pretrained SqueezeNet feature extractor + nearest-prototype, no training loop” approach. The biggest issue is that using only the mean prototype per class is too coarse; a minimal extension that preserves the same logic is to use multiple prototypes per class (k-means centroids) and classify by nearest centroid, which better represents intra-class variation. I also make the prototype computation use both original and hflip features (same TTA you already use at test time) so train/test feature distributions match better. Finally, I keep runtime under the 600s limit by capping the total prototype-build images and using a lightweight, torch-based k-means on the per-image feature vectors.'
- What this solution (achieved 0.53587) has done: 'Your current score (0.494) is far below the target (0.8376), so we should improve accuracy while keeping the same core “pretrained SqueezeNet features + (multi-)prototype nearest-centroid, no training loop” approach. The biggest likely issue is that the current k-means implementation is effectively “random init + cosine assignment” without k-means++ and can yield weak centroids; I switch to a minimal k-means++ init and a vectorized centroid update to get better, stabler prototypes without changing the overall method. I also compute centroids on CPU (as you do) but run assignment/update more efficiently, allowing a slightly higher K_PER_CLASS within the same time budget. Finally, I add a very small, metric-aligned calibration tweak: classify by the best centroid similarity but break ties by class-mean similarity (still prototype-based), which typically reduces noisy centroid effects without changing the evaluation semantics.'
- What this solution (achieved 0.52093) has done: 'Your current score (0.5359) is far below the target (0.8376), so we should improve accuracy while keeping the same core approach: SqueezeNet feature extractor + (multi-)prototype nearest-centroid with no training loop. The most impactful minimal change is to use a stronger pretrained feature extractor while preserving the same pipeline semantics; swapping SqueezeNet1_0 to SqueezeNet1_1 keeps the same architecture family and identical inference/prototype logic, but typically yields better ImageNet features. I also increase the per-class prototype data cap slightly and bump K_PER_CLASS modestly, which improves centroid coverage without changing the method and should move accuracy upward toward the target. Finally, I keep submission alignment stable by ordering predictions to exactly match `sample_submission.csv` image order.'

# 9. Code solution

## === cell 0
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

torch.manual_seed(0)
np.random.seed(0)



## === cell 1
TEST_PATH = "../input/cassava-leaf-disease-classification/test_images/"

files = sorted(
    [f for f in os.listdir(TEST_PATH) if os.path.isfile(os.path.join(TEST_PATH, f))]
)

assert len(files) > 0, (
    f"No test image files found in {TEST_PATH}. "
    f"Found entries: {os.listdir(TEST_PATH)[:10]}"
)



## === cell 2
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ]
)

preprocess_hflip = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        normalize,
    ]
)


def default_loader(path):
    img_pil = Image.open(path).convert("RGB")
    img_tensor = preprocess(img_pil)
    return img_tensor


def hflip_loader(path):
    img_pil = Image.open(path).convert("RGB")
    img_tensor = preprocess_hflip(img_pil)
    return img_tensor




## === cell 3
class MyTestDataset(Dataset):
    def __init__(self, files, loader=default_loader):
        self.files = files
        self.loader = loader

    def __getitem__(self, index):
        img_path = os.path.join(TEST_PATH, self.files[index])
        img = self.loader(img_path)
        return self.files[index], img

    def __len__(self):
        return len(self.files)




## === cell 4
testset = MyTestDataset(files)



## === cell 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = torchvision.models.SqueezeNet1_1_Weights.DEFAULT
model = torchvision.models.squeezenet1_1(weights=weights)
model = model.to(device)
model.eval()

feature_extractor = model.features
feature_extractor.eval()



## === cell 6
TRAIN_CSV = "../input/cassava-leaf-disease-classification/train.csv"
TRAIN_IMG_DIR = "../input/cassava-leaf-disease-classification/train_images/"

train_df = pd.read_csv(TRAIN_CSV)
assert {"image_id", "label"}.issubset(train_df.columns)


class MyTrainDataset(Dataset):
    def __init__(self, df, img_dir, loader=default_loader):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.loader = loader

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_id"])
        x = self.loader(img_path)
        y = int(row["label"])
        return x, y


num_classes = 5

CAP_TOTAL = min(18000, len(train_df))
per_class_cap = max(1, CAP_TOTAL // num_classes)

parts = []
for c in range(num_classes):
    df_c = train_df[train_df["label"] == c]
    take = min(per_class_cap, len(df_c))
    parts.append(df_c.sample(n=take, random_state=0, replace=False))
train_df_cap = (
    pd.concat(parts, axis=0).sample(frac=1.0, random_state=0).reset_index(drop=True)
)

trainset = MyTrainDataset(train_df_cap, TRAIN_IMG_DIR)
trainloader = DataLoader(
    trainset,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

trainset_hflip = MyTrainDataset(train_df_cap, TRAIN_IMG_DIR, loader=hflip_loader)
trainloader_hflip = DataLoader(
    trainset_hflip,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

feat_dim = 512  # SqueezeNet final conv channels

all_feats = []
all_labels = []

with torch.no_grad():
    for (xb, yb), (xb2, yb2) in zip(trainloader, trainloader_hflip):
        assert torch.equal(yb, yb2)

        xb = xb.to(device, non_blocking=True)
        xb2 = xb2.to(device, non_blocking=True)

        f1 = feature_extractor(xb).mean(dim=(2, 3))
        f1 = torch.nn.functional.normalize(f1, dim=1)

        f2 = feature_extractor(xb2).mean(dim=(2, 3))
        f2 = torch.nn.functional.normalize(f2, dim=1)

        f = 0.5 * (f1 + f2)
        f = torch.nn.functional.normalize(f, dim=1)

        all_feats.append(f.detach().cpu())
        all_labels.append(yb.detach().cpu())

all_feats = torch.cat(all_feats, dim=0)  # [N, 512] on CPU
all_labels = torch.cat(all_labels, dim=0).long()  # [N]

print("Collected features:", all_feats.shape, "labels:", all_labels.shape)
print("Per-class counts:", [(all_labels == c).sum().item() for c in range(num_classes)])


def kmeans_torch(x, k, iters=20, seed=0):
    """
    x: [N, D] float32 CPU tensor (already normalized)
    returns centroids: [k, D] normalized
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    n, d = x.shape
    if n == 0:
        return torch.empty((0, d), dtype=x.dtype)
    if n <= k:
        c = torch.nn.functional.normalize(x.clone(), dim=1)
        return c

    first = int(torch.randint(0, n, (1,), generator=g).item())
    centroids = [x[first].clone()]
    sims = x @ centroids[0].unsqueeze(1)  # [N,1]
    best_sim = sims.squeeze(1)  # [N]

    for _ in range(1, k):
        dist = (1.0 - best_sim).clamp_min(0.0)
        probs = dist / (dist.sum() + 1e-12)
        idx = int(torch.multinomial(probs, 1, generator=g).item())
        centroids.append(x[idx].clone())
        best_sim = torch.maximum(best_sim, (x @ centroids[-1].unsqueeze(1)).squeeze(1))

    centroids = torch.stack(centroids, dim=0)  # [k, D]
    centroids = torch.nn.functional.normalize(centroids, dim=1)

    for _ in range(iters):
        sims = x @ centroids.T  # [N, k]
        assign = torch.argmax(sims, dim=1)  # [N]

        new_centroids = torch.zeros((k, d), dtype=x.dtype)
        new_centroids.index_add_(0, assign, x)

        counts = torch.bincount(assign, minlength=k).to(torch.long)  # [k]
        empty = counts == 0
        if empty.any():
            num_empty = int(empty.sum().item())
            ridx = torch.randint(0, n, (num_empty,), generator=g)
            new_centroids[empty] = x[ridx]
            counts[empty] = 1

        new_centroids = new_centroids / counts.unsqueeze(1).to(new_centroids.dtype)
        centroids = torch.nn.functional.normalize(new_centroids, dim=1)

    return centroids


K_PER_CLASS = 16

centroids_list = []
centroids_labels = []
for c in range(num_classes):
    x_c = all_feats[all_labels == c].float()
    cents = kmeans_torch(x_c, k=K_PER_CLASS, iters=20, seed=0 + c)
    centroids_list.append(cents)
    centroids_labels.append(torch.full((cents.shape[0],), c, dtype=torch.long))

centroids = torch.cat(centroids_list, dim=0)  # [M, 512]
centroids_y = torch.cat(centroids_labels, dim=0)  # [M]
centroids = torch.nn.functional.normalize(centroids, dim=1)

print("Built centroids:", centroids.shape, "Centroid labels:", centroids_y.shape)

class_means = []
for c in range(num_classes):
    x_c = all_feats[all_labels == c].float()
    mu = x_c.mean(dim=0, keepdim=True)
    mu = torch.nn.functional.normalize(mu, dim=1)
    class_means.append(mu)
class_means = torch.cat(class_means, dim=0)  # [C, D]

centroids = centroids.to(device)
centroids_y = centroids_y.to(device)
class_means = class_means.to(device)



## === cell 7
testloader = DataLoader(
    testset,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

testset_hflip = MyTestDataset(files, loader=hflip_loader)
testloader_hflip = DataLoader(
    testset_hflip,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

pred_labels = []
pred_files = []

with torch.no_grad():
    for (file_batch, xb), (file_batch2, xb2) in zip(testloader, testloader_hflip):
        assert list(file_batch) == list(file_batch2)

        xb = xb.to(device, non_blocking=True)
        xb2 = xb2.to(device, non_blocking=True)

        feats1 = feature_extractor(xb).mean(dim=(2, 3))
        feats1 = torch.nn.functional.normalize(feats1, dim=1)

        feats2 = feature_extractor(xb2).mean(dim=(2, 3))
        feats2 = torch.nn.functional.normalize(feats2, dim=1)

        feats = 0.5 * (feats1 + feats2)
        feats = torch.nn.functional.normalize(feats, dim=1)

        sims = feats @ centroids.T  # [B, M]
        best_cent_sim, nn_idx = torch.max(sims, dim=1)  # [B], [B]
        yhat_primary = centroids_y[nn_idx]  # [B]

        mean_sims = feats @ class_means.T  # [B, C]
        alpha = 0.08
        combined = mean_sims
        combined.scatter_add_(
            1, yhat_primary.view(-1, 1), alpha * best_cent_sim.view(-1, 1)
        )

        yhat = torch.argmax(combined, dim=1).detach().cpu().numpy().astype(int).tolist()

        pred_labels.extend(yhat)
        pred_files.extend(list(file_batch))

submission = pd.DataFrame({"image_id": pred_files, "label": pred_labels})



## === cell 8
SAMPLE_SUB = "../input/cassava-leaf-disease-classification/sample_submission.csv"
sample = pd.read_csv(SAMPLE_SUB)
submission = sample[["image_id"]].merge(submission, on="image_id", how="left")
assert submission["label"].notna().all(), "Some test images were not predicted."
submission["label"] = submission["label"].astype(int)

submission



## === cell 9
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
